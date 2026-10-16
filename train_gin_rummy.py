"""
Train a Gin Rummy bot via NFSP self-play using RLCard.

NFSP (Neural Fictitious Self-Play) is designed for imperfect-information
games like gin rummy: each agent learns a best-response policy (DQN) and
an average policy (supervised), which converges toward Nash equilibrium.

Reward modes (--reward):
  points  (default) True per-hand score differential, zero-sum, /100:
                    gin = opponent deadwood + 25 bonus; knock = deadwood
                    difference, with a 25-point undercut bonus if the
                    defender ties or beats the knocker; dead hand = 0.
                    (No lay-offs: RLCard's engine doesn't implement them.)
  shaped            RLCard default (1 gin / 0.2 knock / -deadwood/100).
                    Denser signal; useful for pre-training from scratch.

Suggested curriculum:
  python train_gin_rummy.py --reward shaped --num_episodes 50000
  python train_gin_rummy.py --reward points --num_episodes 100000 \
      --load_model results/gin_rummy_nfsp/gin_rummy_nfsp.pth
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

import numpy as np
import torch
import rlcard
from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.games.gin_rummy.utils import melding
from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.scorers import GinRummyScorer
from rlcard.games.gin_rummy.utils.action_event import KnockAction, GinAction
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent

from shaping import shape_trajectory

GIN_BONUS = 25
UNDERCUT_BONUS = 25


def _deadwood(player) -> int:
    """Deadwood count of a player's hand under its best melding."""
    clusters = melding.get_best_meld_clusters(hand=player.hand)
    best = [] if not clusters else clusters[0]
    return gr_utils.get_deadwood_count(player.hand, best)


def real_points_payoff(player, game) -> float:
    """True gin rummy score differential for `player`, scaled by 1/100.

    Zero-sum: winner gets +points/100, loser gets -points/100.
    """
    going_out_action = game.round.going_out_action
    going_out_id = game.round.going_out_player_id
    if going_out_id is None:          # dead hand: nobody went out
        return 0.0

    me = player.player_id
    goer = game.round.players[going_out_id]
    defender = game.round.players[1 - going_out_id]

    if isinstance(going_out_action, GinAction):
        points = _deadwood(defender) + GIN_BONUS
        winner_id = going_out_id
    elif isinstance(going_out_action, KnockAction):
        knocker_dw, defender_dw = _deadwood(goer), _deadwood(defender)
        if knocker_dw < defender_dw:
            points = defender_dw - knocker_dw
            winner_id = going_out_id
        else:                          # undercut: defender ties or beats knocker
            points = knocker_dw - defender_dw + UNDERCUT_BONUS
            winner_id = 1 - going_out_id
    else:
        return 0.0

    return points / 100.0 if me == winner_id else -points / 100.0


def classify_outcome(game):
    """Label how a finished hand ended, plus which player won it.

    Returns (label, winner_id) with label in gin / knock / undercut / dead;
    winner_id is None for a dead hand.
    """
    action = game.round.going_out_action
    going_out_id = game.round.going_out_player_id
    if going_out_id is None or not isinstance(action, (GinAction, KnockAction)):
        return "dead", None
    if isinstance(action, GinAction):
        return "gin", going_out_id
    goer = game.round.players[going_out_id]
    defender = game.round.players[1 - going_out_id]
    if _deadwood(goer) < _deadwood(defender):
        return "knock", going_out_id
    return "undercut", 1 - going_out_id


def save_policy(agent, path):
    """Save just the average-policy network (~1.8 MB), not the whole agent.

    `torch.save(agent)` pickles the entire NFSPAgent including both replay
    buffers -- about 1.4 GB once they fill. Evaluation only ever touches the
    average policy (NFSP's `eval_step` ignores the DQN), so snapshots and the
    shipped model store only that. Full-agent saves are kept for resume alone.

    The policy net contains a BatchNorm layer, so its running statistics travel
    with the state_dict and must not be dropped.
    """
    torch.save(
        {
            "num_actions": agent._num_actions,
            "state_shape": agent._state_shape,
            "mlp_layers": agent._layer_sizes,   # hidden sizes + [num_actions]
            "state_dict": agent.policy_network.state_dict(),
        },
        path,
    )


class _QuietStdout(io.TextIOBase):
    """Drop RLCard's per-step 'INFO - Step N, rl-loss: ...' spam.

    Those are raw prints inside dqn_agent/nfsp_agent with end='' and a leading
    carriage return: readable on a live terminal, but they produce hundreds of
    MB of single-line garbage once a long run is redirected to a file.
    """

    def __init__(self, real):
        self._real = real

    def write(self, s):
        if s.startswith("\rINFO - Step"):
            return len(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()


def make_env(args, seed=None):
    """The project's one env factory. args.obs selects the observation:
    "base" (default, rlcard's 5x52) or "hist" (the v3 10x52 with belief and
    history planes -- see hist_env.py). rlcard.make() bypasses env subclasses,
    so the hist path instantiates directly."""
    config = {"seed": args.seed if seed is None else seed}
    if getattr(args, "obs", "base") == "hist":
        from hist_env import HistoryGinRummyEnv
        env = HistoryGinRummyEnv({"allow_step_back": False, **config})
    else:
        env = rlcard.make("gin-rummy", config=config)
    if args.reward == "points":
        if getattr(args, "rules", "classic") == "full":
            from fullrules import full_points_payoff
            env.game.judge.scorer = GinRummyScorer(
                name="FullRulesScorer", get_payoff=full_points_payoff)
        else:
            env.game.judge.scorer = GinRummyScorer(
                name="RealPointsScorer", get_payoff=real_points_payoff)
    return env


def build_agents(env, device, args):
    """Two NFSP agents that train against each other, or reload a checkpoint."""
    if args.load_model:
        print(f"Resuming from {args.load_model}")
        agents = [
            torch.load(args.load_model, map_location=device, weights_only=False)
            for _ in range(env.num_players)
        ]
        for agent in agents:
            agent.set_device(device)
            # Fine-tuning takes its update cadence from this run's CLI flags.
            agent._train_every = args.train_every
            agent._rl_agent.train_every = args.q_train_every
        return agents
    # NFSP counts epsilon decay in agent *steps*, not episodes; a gin rummy
    # hand feeds roughly 73 transitions per agent, so scale accordingly.
    decay_steps = min(int(args.num_episodes * 36), 4_000_000)
    return [
        NFSPAgent(
            num_actions=env.num_actions,          # 110 actions in gin rummy
            state_shape=env.state_shape[0],       # 5 x 52 card planes
            hidden_layers_sizes=[512, 512],       # average-policy network
            q_mlp_layers=[512, 512],              # best-response (DQN) network
            reservoir_buffer_capacity=int(2e5),
            q_replay_memory_size=int(2e5),
            q_epsilon_decay_steps=decay_steps,
            rl_learning_rate=1e-4,
            sl_learning_rate=1e-4,
            train_every=args.train_every,         # SL update every N transitions
            q_train_every=args.q_train_every,     # DQN update every N transitions
            device=device,
        )
        for _ in range(env.num_players)
    ]


def evaluate(eval_env, num_games, seed):
    """Average payoff for agent 0 plus the outcome mix over `num_games` hands.

    The deal sequence is reseeded every call (common random numbers), so
    successive points on the learning curve are comparable instead of each
    being an independent noisy sample.
    """
    np_state = np.random.get_state()
    eval_env.seed(seed)
    np.random.seed(seed)
    total = 0.0
    counts = {"gin": 0, "knock": 0, "undercut": 0, "dead": 0}
    wins = 0
    try:
        for _ in range(num_games):
            _, payoffs = eval_env.run(is_training=False)
            total += payoffs[0]
            label, winner = classify_outcome(eval_env.game)
            counts[label] += 1
            if winner == 0:
                wins += 1
    finally:
        np.random.set_state(np_state)
    stats = {k: v / num_games for k, v in counts.items()}
    stats["win_rate"] = wins / num_games
    return total / num_games, stats


def train(args):
    device = get_device()
    set_seed(args.seed)

    env = make_env(args)                           # self-play training env
    eval_env = make_env(args, seed=args.seed + 1)  # evaluation vs. a random bot
    rule_env = make_env(args, seed=args.seed + 2)  # evaluation vs. the rule agent

    agents = build_agents(env, device, args)
    env.set_agents(agents)
    eval_env.set_agents([agents[0], RandomAgent(num_actions=env.num_actions)])
    # rlcard ships a novice rule agent: gins/knocks at the first legal chance and
    # otherwise discards to minimise deadwood. Much more informative than random,
    # which the bot saturates against early. Eval only -- training stays self-play.
    rule_env.set_agents([agents[0], GinRummyNoviceRuleAgent()])

    os.makedirs(args.log_dir, exist_ok=True)
    stats_path = os.path.join(args.log_dir, "outcomes.csv")
    stats_file = open(stats_path, "w", newline="")
    stats_writer = csv.DictWriter(
        stats_file,
        fieldnames=["episode", "reward", "win_rate", "gin", "knock", "undercut",
                    "dead", "reward_vs_rule", "win_rate_vs_rule", "elapsed_s"],
    )
    stats_writer.writeheader()

    resume_path = os.path.join(args.log_dir, "resume.pth")
    start = time.time()
    real_stdout = sys.stdout

    def record(episode, score, stats, rule=None):
        stats_writer.writerow({
            "episode": episode, "reward": round(score, 5),
            "win_rate": round(stats["win_rate"], 4),
            "gin": round(stats["gin"], 4), "knock": round(stats["knock"], 4),
            "undercut": round(stats["undercut"], 4), "dead": round(stats["dead"], 4),
            "reward_vs_rule": "" if rule is None else round(rule[0], 5),
            "win_rate_vs_rule": "" if rule is None else round(rule[1]["win_rate"], 4),
            "elapsed_s": round(time.time() - start, 1),
        })
        stats_file.flush()

    with Logger(args.log_dir) as logger:
        with contextlib.redirect_stdout(_QuietStdout(real_stdout)):
            for episode in range(args.num_episodes):
                # NFSP samples between best-response and average policy each episode
                for agent in agents:
                    agent.sample_episode_policy()

                # Play one full self-play hand and learn from it
                trajectories, payoffs = env.run(is_training=True)
                trajectories = reorganize(trajectories, payoffs)
                if args.shaping == "deadwood":
                    # Potential-based: dense signal, provably same optimal policy.
                    for traj in trajectories:
                        shape_trajectory(traj, args.gamma)
                for agent, traj in zip(agents, trajectories):
                    for ts in traj:
                        agent.feed(ts)

                # Periodically measure performance against a random opponent.
                # In points mode this is avg score differential per hand (x100 pts).
                if episode % args.evaluate_every == 0:
                    score, stats = evaluate(eval_env, args.num_eval_games, args.seed + 7)
                    rule = None
                    if args.rule_eval_every and episode % args.rule_eval_every == 0:
                        rule = evaluate(rule_env, args.num_rule_eval_games,
                                        args.seed + 11)
                    logger.log_performance(episode, score)
                    logger.csv_file.flush()
                    record(episode, score, stats, rule)
                    elapsed = time.time() - start
                    rate = (episode + 1) / max(elapsed, 1e-9)
                    eta = (args.num_episodes - episode - 1) / max(rate, 1e-9) / 3600
                    extra = "" if rule is None else (
                        f" | vs rule {rule[0] * 100:+.2f} pts/hand "
                        f"(win {rule[1]['win_rate']:.1%})")
                    print(
                        f"  outcomes: gin {stats['gin']:.1%} knock {stats['knock']:.1%} "
                        f"undercut {stats['undercut']:.1%} dead {stats['dead']:.1%} | "
                        f"win {stats['win_rate']:.1%} | {rate:.1f} eps/s | "
                        f"ETA {eta:.2f}h{extra}",
                        file=real_stdout, flush=True,
                    )

                # Small policy-only snapshots form the checkpoint ladder.
                if args.policy_snapshot_every and episode > 0 \
                        and episode % args.policy_snapshot_every == 0:
                    save_policy(agents[0],
                                os.path.join(args.log_dir, f"snap_ep{episode}.policy.pt"))

                # One rolling full-agent file: crash recovery + stage handoff only.
                if args.full_checkpoint_every and episode > 0 \
                        and episode % args.full_checkpoint_every == 0:
                    torch.save(agents[0], resume_path)
                    print(f"  resume checkpoint @ episode {episode}",
                          file=real_stdout, flush=True)

            # Final evaluation point so the curve ends on the finished model.
            score, stats = evaluate(eval_env, args.num_eval_games, args.seed + 7)
            rule = evaluate(rule_env, args.num_rule_eval_games, args.seed + 11)
            logger.log_performance(args.num_episodes, score)
            record(args.num_episodes, score, stats, rule)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    stats_file.close()
    plot_curve(csv_path, fig_path, "nfsp")

    final_policy = os.path.join(args.log_dir, "final.policy.pt")
    save_policy(agents[0], final_policy)   # the deliverable: ~1.8 MB
    torch.save(agents[0], resume_path)     # full agent, for the next stage
    print(f"Final policy saved to {final_policy}")
    print(f"Resume checkpoint saved to {resume_path}")
    print(f"Outcome log: {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser("Gin Rummy NFSP self-play trainer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_episodes", type=int, default=100_000,
        help="Self-play hands to train on (more is better; expect 100k+)",
    )
    parser.add_argument(
        "--reward", choices=["points", "shaped"], default="points",
        help="points = true score differential; shaped = RLCard default",
    )
    parser.add_argument(
        "--load_model", type=str, default=None,
        help="Path to a saved .pth to continue training from (curriculum)",
    )
    parser.add_argument(
        "--shaping", choices=["none", "deadwood"], default="deadwood",
        help="potential-based deadwood shaping; terminal-only reward does not learn",
    )
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount used by the shaping term (match the agent)")
    parser.add_argument("--evaluate_every", type=int, default=5_000)
    parser.add_argument("--num_eval_games", type=int, default=300)
    parser.add_argument(
        "--rule_eval_every", type=int, default=25_000,
        help="Also benchmark against rlcard's novice rule agent this often (0 = never)",
    )
    parser.add_argument("--num_rule_eval_games", type=int, default=300)
    parser.add_argument(
        "--train_every", type=int, default=4,
        help="SL (average policy) gradient step every N transitions",
    )
    parser.add_argument(
        "--q_train_every", type=int, default=4,
        help="DQN (best response) gradient step every N transitions",
    )
    parser.add_argument(
        "--policy_snapshot_every", type=int, default=10_000,
        help="Save a small policy-only snapshot every N episodes (0 disables)",
    )
    parser.add_argument(
        "--full_checkpoint_every", type=int, default=50_000,
        help="Overwrite the rolling full-agent resume checkpoint every N episodes",
    )
    parser.add_argument("--log_dir", type=str, default="results/gin_rummy_nfsp")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

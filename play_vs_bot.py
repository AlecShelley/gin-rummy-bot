"""
Play gin rummy against a bot in your terminal.

    python play_vs_bot.py v3search                              # the v3 search bot
    python play_vs_bot.py results/v2_final/gin_rummy_bot.policy.pt
    python play_vs_bot.py v3search --hands 5

`v3search` is the strongest bot in this project: the belief-state rollout
searcher (expert.py). It thinks for a moment on each move. Any other argument
is treated as a network .policy.pt path.

Scoring is the true points scorer from train_gin_rummy.py (gin = opponent
deadwood + 25, knock = deadwood difference, undercut = difference + 25 to the
defender). Note that RLCard's engine has no lay-offs, and dead hands score 0,
which is why strong play here holds for gin far more than paper strategy would.
"""

import argparse

from rlcard.games.gin_rummy.utils import melding
from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import ActionEvent

from gin_rummy_bot import GinRummyBot
from train_gin_rummy import make_env, classify_outcome

SUIT_ORDER = {"S": 0, "H": 1, "D": 2, "C": 3}


class _Args:
    reward = "points"
    seed = 0


def sort_hand(hand):
    return sorted(hand, key=lambda c: (SUIT_ORDER.get(c.suit, 9),
                                       gr_utils.get_rank_id(c)))


def unmelded(hand, cluster):
    meld_cards = [c for meld in cluster for c in meld]
    return [c for c in hand if c not in meld_cards]


def best_melding(hand):
    """Lowest-deadwood meld cluster for a hand of any size, with its count.

    Both `get_best_meld_clusters` and `get_deadwood_count` hard-require exactly
    10 cards, but you hold 11 between drawing and discarding -- precisely when
    you most want to see your melds. `get_meld_clusters` has no such
    restriction, so score its clusters here with the per-card deadwood values.
    """
    best, best_count = [], sum(gr_utils.get_deadwood_value(c) for c in hand)
    for cluster in melding.get_meld_clusters(hand=hand):
        count = sum(gr_utils.get_deadwood_value(c)
                    for c in unmelded(hand, cluster))
        if count < best_count:
            best, best_count = cluster, count
    return best, best_count


def show_hand(hand):
    """Print the hand split into its best melds and the leftover deadwood."""
    best, count = best_melding(hand)
    deadwood = unmelded(hand, best)
    if best:
        parts = ["[" + " ".join(str(c) for c in sort_hand(meld)) + "]" for meld in best]
        print("  melds    : " + "  ".join(parts))
    else:
        print("  melds    : (none)")
    line = ("  deadwood : "
            + (" ".join(str(c) for c in sort_hand(deadwood)) or "(none)")
            + f"   = {count} points")
    if len(hand) == 11 and deadwood:
        # You still owe a discard, so what matters is the count after dropping
        # your worst deadwood card -- that's the number you can knock at.
        worst = max(gr_utils.get_deadwood_value(c) for c in deadwood)
        line += f"  (discard your worst and you knock at {count - worst})"
    print(line)


def show_table(game, human_id):
    round_ = game.round
    discard = round_.dealer.discard_pile
    top = str(discard[-1]) if discard else "(empty)"
    opp = round_.players[1 - human_id]
    print(f"\n  stock: {len(round_.dealer.stock_pile)} cards left"
          f"   |   top of discard: {top}")
    if opp.known_cards:
        print("  bot is known to hold: "
              + " ".join(str(c) for c in sort_hand(opp.known_cards)))
    print("  your hand:")
    show_hand(round_.players[human_id].hand)


def _option_order(item):
    """Going-out and draw options first; discards last, worst card first."""
    action_id, label = item
    event = ActionEvent.decode_action(action_id)
    card = getattr(event, "card", None)
    if label.startswith("discard"):
        return (2, -gr_utils.get_deadwood_value(card), gr_utils.get_card_id(card))
    if label.startswith("knock"):
        return (1, -gr_utils.get_deadwood_value(card), gr_utils.get_card_id(card))
    return (0, action_id, 0)


def choose_action(state, game, human_id):
    legal = list(state["legal_actions"].keys())
    if len(legal) == 1:                      # forced (e.g. the scoring steps)
        return legal[0]
    show_table(game, human_id)
    labels = sorted((( a, str(ActionEvent.decode_action(a))) for a in legal),
                    key=_option_order)
    print("\n  your options:")
    for i, (_, label) in enumerate(labels):
        print(f"    {i:>2}. {label}")
    while True:
        raw = input("  choose > ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if raw.isdigit() and 0 <= int(raw) < len(labels):
            return labels[int(raw)][0]
        print(f"  enter a number from 0 to {len(labels) - 1}, or 'q' to quit.")


def describe_bot_moves(game, seen, human_id):
    """Narrate whatever the bot did since we last looked."""
    sheet = game.round.move_sheet
    for move in sheet[seen:]:
        player = getattr(move, "player", None)
        if player is None or player.player_id == human_id:
            continue
        action = getattr(move, "action", None)
        if action is None:
            continue
        text = str(action)
        # Don't leak which card the bot drew from the stock.
        if text.startswith("draw_card"):
            text = "draw a card from the stock"
        print(f"  bot: {text}")
    return len(sheet)


def play_hand(env, bot, human_id, hand_no):
    print(f"\n{'=' * 62}\n  HAND {hand_no}   (you are player {human_id})\n{'=' * 62}")
    state, player_id = env.reset()
    seen = 0
    while not env.is_over():
        if player_id == human_id:
            seen = describe_bot_moves(env.game, seen, human_id)
            action = choose_action(state, env.game, human_id)
        else:
            action, _ = bot.eval_step(state)
        state, player_id = env.step(action)
    describe_bot_moves(env.game, seen, human_id)

    payoffs = env.get_payoffs()
    label, winner = classify_outcome(env.game)
    print(f"\n  --- hand over: {label} ---")
    for pid in (human_id, 1 - human_id):
        who = "you" if pid == human_id else "bot"
        print(f"  {who}:")
        show_hand(env.game.round.players[pid].hand)
    points = payoffs[human_id] * 100
    if winner is None:
        print("\n  dead hand - no score.")
    else:
        who = "You win" if winner == human_id else "Bot wins"
        print(f"\n  {who} {abs(points):.0f} points.")
    return points


def main():
    ap = argparse.ArgumentParser("Play gin rummy against a bot")
    ap.add_argument("model",
                    help="'v7' (strongest), 'v6', 'v5', 'v4', 'v3search', "
                         "or a .policy.pt path")
    ap.add_argument("--hands", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seat", type=int, choices=[0, 1], default=0,
                    help="which seat you play")
    ap.add_argument("--greedy", action="store_true",
                    help="network bots only: always play the most likely action")
    args = ap.parse_args()

    _Args.seed = args.seed if args.seed is not None else 0
    env = make_env(_Args)
    if args.seed is not None:
        env.seed(args.seed)
    if args.model in ("v7", "v6"):
        # v7 (promoted 2026-08-31, results/v8_gates.log): v6's search +
        # filter with the redesigned opponent-action likelihood
        # (binary-rule pickup model + mh_all_takes rejuvenation -- the
        # hidden-5S inverted-likelihood fix). Ties v6 head-to-head;
        # promoted on posterior quality + clean probe. "v6" here pins
        # the frozen 2026-08-28 config (gain-logistic pickup) exactly
        # as it was gated: beats v5 +1.92 +/- 0.97 (3984 fresh hands),
        # first bot to beat the patient teacher head-to-head
        # (+2.56 +/- 1.65), unexploited. results/v6_gates.log.
        from expert import ExpertAgent
        from pf_expert import ParticleBelief, PFExpert, V5_CFG
        # Both registries are FROZEN at their gated configs since the
        # v10 promotion moved DEFAULT_PF (eps 0.05->0.02, tau_po=0.0;
        # results/v10_gates.log) -- pin the pre-v10 knobs explicitly.
        pin_cfg = dict(eps=0.05, tau_po=None) \
            if args.model == "v7" else dict(
                pickup_mode="logistic", tau_p=1.0, mh_all_takes=False,
                eps=0.05, tau_po=None)

        class _PFAgent(ExpertAgent):
            def _belief_for(self, game):
                rnd = game.round
                if not hasattr(rnd, "_v3_token"):
                    rnd._v3_token = object()
                if rnd._v3_token is not self._round_key:
                    self._round_key = rnd._v3_token
                    self._belief = ParticleBelief(rnd.current_player_id,
                                                  cfg=pin_cfg)
                return self._belief

        bot = _PFAgent(PFExpert(veto=True, opp_model=(10, 1.0),
                                sample_boost=0.0, **V5_CFG), env)
    elif args.model == "v5":
        # The 2026-08-26 candidate: adaptive staged world budget (16/48/192)
        # + 2-ply close-call rollouts on the corrected serial loop. Beats v4
        # +1.20 +/- 0.93 (3984 fresh hands) at ~1.7x its wall-clock.
        from adaptive import AdaptiveExpert
        from expert import ExpertAgent
        bot = ExpertAgent(AdaptiveExpert(stages=(16, 48, 192), deep=True,
                                         truncate=16, z=1.5,
                                         opp_model=(10, 1.0),
                                         sample_boost=2.5), env)
    elif args.model == "v4":
        # The 2026-08-23 promotion: v3search + opponent-model conditioning
        # (knocks-at-10 continuation + world weighting) + correlated worlds.
        from expert import ExpertAgent, RolloutExpert
        bot = ExpertAgent(RolloutExpert(worlds=48, truncate=16, z=1.5,
                                        opp_model=(10, 1.0),
                                        sample_boost=2.5), env)
    elif args.model == "v3search":
        from expert import ExpertAgent, RolloutExpert
        bot = ExpertAgent(RolloutExpert(worlds=48, truncate=16, z=1.5), env)
    else:
        bot = GinRummyBot(args.model, greedy=args.greedy)
    env.set_agents([None, None])   # we drive the loop ourselves

    print(f"\nPlaying against {args.model}")
    print("Type the number of the action you want, or 'q' to quit.")

    total = 0.0
    try:
        for i in range(1, args.hands + 1):
            total += play_hand(env, bot, args.seat, i)
            if args.hands > 1:
                print(f"\n  running total: {total:+.0f} points "
                      f"after {i} hand{'s' if i > 1 else ''}")
    except (KeyboardInterrupt, EOFError):
        print("\n\n  quit.")
    print(f"\nFinal: {total:+.0f} points over the hands played.\n")


if __name__ == "__main__":
    main()

"""
Loadable wrapper around a trained gin rummy policy.

The network is a policy-only file (`*.policy.pt`, ~1.8 MB) holding the
average-policy network produced by the behaviour-cloning + DAgger pipeline.

The bot has two parts:

  * the network, for discard / knock / gin decisions;
  * an explicit heuristic for the draw decision (see `should_pick_up`).

The second part is legacy insurance. It exists because the *original* teacher
chose its draw at random -- `GinRummyNoviceRuleAgent.step` has no branch for that
decision, so it falls through to `np.random.choice` -- and the clone faithfully
learned the noise; overriding it was worth about 11 points/hand for the v1 bot.

The current bot (`results/v2_final/`) was cloned from the deterministic teacher in
`teacher.py`, so the network learned the draw rule itself and the override is now
worth only +0.41 points/hand. It stays on by default because it measures
marginally better, but the network alone beats the smart-draw rule agent
(+1.89 +/- 1.11 over 2000 hands).

`GinRummyBot` is API-compatible with an RLCard agent (`use_raw`, `step`,
`eval_step`), so it can be dropped straight into `env.set_agents([...])`.

    from gin_rummy_bot import GinRummyBot
    bot = GinRummyBot("results/v2_final/gin_rummy_bot.policy.pt")
    env.set_agents([bot, opponent])
"""

import numpy as np
import torch
from rlcard.agents.nfsp_agent import AveragePolicyNetwork
from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import (
    ActionEvent, DrawCardAction, PickUpDiscardAction, DeclareDeadHandAction)
from rlcard.utils.utils import remove_illegal

from shaping import _min_deadwood


def draw_decision(state):
    """Return an action id for a draw decision, or None if this isn't one.

    Takes the upcard only when it strictly lowers the hand's deadwood. Picking
    up otherwise is doubly bad: it fails to help, and it tells the opponent
    exactly which card you hold.
    """
    legal = list(state["legal_actions"].keys())
    events = [ActionEvent.decode_action(a) for a in legal]
    pickup = [e.action_id for e in events if isinstance(e, PickUpDiscardAction)]
    other = [e.action_id for e in events
             if isinstance(e, (DrawCardAction, DeclareDeadHandAction))]
    if not pickup or not other:
        return None                      # not a draw decision

    hand = gr_utils.decode_cards(state["obs"][0])
    top = gr_utils.decode_cards(state["obs"][1])
    if not top or len(hand) != 10:
        return other[0]
    candidate = hand + top
    # The engine forbids discarding the picked-up card immediately
    # (is_allowed_to_discard_picked_up_card=False), so only the original ten are
    # discard candidates.
    after = min(_min_deadwood([c for c in candidate if c != drop]) for drop in hand)
    return pickup[0] if after < _min_deadwood(hand) else other[0]


class GinRummyBot:
    """A trained gin rummy policy, usable anywhere an RLCard agent is."""

    use_raw = False

    def __init__(self, path, device=None, greedy=True, smart_draw=True):
        """
        Args:
            path: a `*.policy.pt` from save_policy(), or a full pickled NFSPAgent.
            device: torch device; defaults to CUDA when available.
            greedy: take the argmax legal action instead of sampling.
                Defaults to True because it measured 2-3 points/hand stronger
                against every opponent tested. The caveat is that a
                deterministic policy is more exploitable by an *adaptive*
                opponent; pass greedy=False for a mixed strategy.
            smart_draw: use `draw_decision` instead of the network for the
                draw/pick-up choice. Defaults to True, but for a bot cloned from
                the deterministic teacher it is worth only +0.41 points/hand --
                the network has learned the same rule. It was worth ~11 for the
                v1 bot, whose teacher drew at random. Pass False for the pure
                network.
        """
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.smart_draw = smart_draw
        self.path = path

        blob = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            net = AveragePolicyNetwork(
                num_actions=blob["num_actions"],
                state_shape=blob["state_shape"],
                mlp_layers=blob["mlp_layers"],
            )
            net.load_state_dict(blob["state_dict"])
            self.num_actions = blob["num_actions"]
        else:
            # A full pickled NFSPAgent: lift out just the average policy.
            net = blob.policy_network
            self.num_actions = blob._num_actions
        self.policy_network = net.to(self.device)
        # eval() matters: the network starts with a BatchNorm layer, which would
        # otherwise use batch statistics on a batch of one.
        self.policy_network.eval()

    def _probs(self, state):
        obs = np.expand_dims(state["obs"], axis=0)
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            log_probs = self.policy_network(obs).cpu().numpy()
        probs = np.exp(log_probs)[0]
        return remove_illegal(probs, list(state["legal_actions"].keys()))

    def _choose(self, probs):
        if self.greedy:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def step(self, state):
        if self.smart_draw:
            drawn = draw_decision(state)
            if drawn is not None:
                return drawn
        return self._choose(self._probs(state))

    def eval_step(self, state):
        if self.smart_draw:
            drawn = draw_decision(state)
            if drawn is not None:
                return drawn, {}
        probs = self._probs(state)
        action = self._choose(probs)
        legal = list(state["legal_actions"].keys())
        info = {"probs": {state["raw_legal_actions"][i]: float(probs[legal[i]])
                          for i in range(len(legal))}}
        return action, info

    def __repr__(self):
        return (f"GinRummyBot({self.path!r}, greedy={self.greedy}, "
                f"smart_draw={self.smart_draw})")


def load_opponent(spec, num_actions, device=None, greedy=False, smart_draw=True):
    """Resolve a CLI opponent spec to an agent and a display name.

    'random', 'rule' (rlcard's novice agent), 'smart' (that agent with its
    coin-flip draw replaced -- the opponent this bot is measured against),
    'strong' (the deterministic teacher in teacher.py), or a path to a model.
    """
    spec_lower = spec.lower()
    if spec_lower == "random":
        from rlcard.agents import RandomAgent
        return RandomAgent(num_actions=num_actions), "random"
    if spec_lower == "rule":
        from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
        return GinRummyNoviceRuleAgent(), "novice-rule"
    if spec_lower in ("smart", "strong"):
        from teacher import make_teacher
        return make_teacher(spec_lower), ("smart-draw-rule" if spec_lower == "smart"
                                          else "strong-teacher")
    return (GinRummyBot(spec, device=device, greedy=greedy,
                        smart_draw=smart_draw),
            spec if smart_draw else spec + " (network only)")

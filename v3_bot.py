"""
V3Bot: the deployable history-aware policy.

Self-contained: it carries its own Belief and builds its own (10, 52)
observation from the live game via belief.build_obs, so it plays in ANY
rlcard gin rummy env (the env's own observation is ignored). The one
requirement is access to the game object -- pass the env at construction or
call attach(env) before play, exactly like expert.ExpertAgent.

Loads the same {num_actions, state_shape, mlp_layers, state_dict} blob format
the rest of the project uses; state_shape [10, 52] marks a v3 policy.
"""

import numpy as np
import torch
from rlcard.agents.nfsp_agent import AveragePolicyNetwork

from belief import Belief, build_obs


class V3Bot:

    use_raw = False

    def __init__(self, path, env=None, device=None, greedy=True,
                 teacher_floor=None):
        """teacher_floor: if set (0..1), play FastTeacher's action unless the
        network's top choice has at least this much probability mass -- keeps
        the teacher's reliability, admits only confident learned deviations."""
        self.path = path
        self.env = env
        self.greedy = greedy
        self.teacher_floor = teacher_floor
        if teacher_floor is not None:
            from sim import FastTeacher, fast_legal_actions
            self._teacher = FastTeacher()
            self._legal = fast_legal_actions
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        blob = torch.load(path, map_location=self.device, weights_only=False)
        assert blob["state_shape"] == [10, 52], \
            f"not a v3 policy: state_shape {blob['state_shape']}"
        net = AveragePolicyNetwork(num_actions=blob["num_actions"],
                                   state_shape=blob["state_shape"],
                                   mlp_layers=blob["mlp_layers"])
        net.load_state_dict(blob["state_dict"])
        self.net = net.to(self.device)
        self.net.eval()                  # leading BatchNorm needs eval mode
        self._round_key = None
        self._belief = None

    def attach(self, env):
        self.env = env
        return self

    def _belief_for(self, game):
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = Belief(rnd.current_player_id)
        return self._belief

    def _probs(self, obs, legal_ids):
        x = torch.from_numpy(obs.reshape(1, -1).astype(np.float32)) \
            .to(self.device)
        with torch.no_grad():
            log_probs = self.net(x).cpu().numpy()[0]
        probs = np.exp(log_probs)
        mask = np.zeros_like(probs)
        mask[legal_ids] = probs[legal_ids]
        total = mask.sum()
        if total <= 0:
            mask[legal_ids] = 1.0 / len(legal_ids)
            return mask
        return mask / total

    def step(self, state):
        legal_ids = list(state["legal_actions"].keys())
        if len(legal_ids) == 1:
            return legal_ids[0]
        game = self.env.game
        belief = self._belief_for(game)
        obs = build_obs(game, game.round.current_player_id, belief, legal_ids)
        probs = self._probs(obs, legal_ids)
        if self.teacher_floor is not None and \
                float(probs.max()) < self.teacher_floor:
            return self._teacher.act(game, self._legal(game)).action_id
        if self.greedy:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def eval_step(self, state):
        return self.step(state), {}

    def __repr__(self):
        return f"V3Bot({self.path!r}, greedy={self.greedy})"

"""
V12 A2: the PHASE-CONDITIONAL ORACLE (registered
results/v12_gates.log). HomeV8Policy machinery, but when the stock
is at or below --stockmax the belief's sampled worlds are the TRUE
opponent hidden hand (Richards & Amir oracle pattern; world veto
disabled in the oracle phase so the true world is never rejected).
K=99 = the full home oracle bound for v10home; K=10 / K=4 measure
the share of the oracle gap that late-phase information alone
captures — the power budget for any endgame solver.

    .venv/Scripts/python.exe phase_oracle.py --stockmax 99 --deals 1500
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from home_variant import HomeV8Policy
from sim import cid, fast_legal_actions, step_fast

ROWS_PATH = "results/phase_oracle_rows.jsonl"


class PhaseOracleBelief:
    """Wraps ParticleBelief; oracle worlds when stock <= stock_max."""

    def __init__(self, hero_id, seed=0, stock_max=99):
        from pf_expert import ParticleBelief
        self.inner = ParticleBelief(hero_id, seed=seed)
        self.hero_id = hero_id
        self.stock_max = stock_max
        self.oracle_on = False
        self._true_hidden = []

    def update(self, game):
        self.inner.update(game)
        rnd = game.round
        opp = rnd.players[1 - self.hero_id]
        self._true_hidden = [cid(c) for c in opp.hand
                             if c not in opp.known_cards]
        self.oracle_on = len(rnd.dealer.stock_pile) <= self.stock_max

    def sample_free_cards(self, unseen_ids, num_hidden, rng, boost=0.0):
        if self.oracle_on:
            return list(self._true_hidden)
        return self.inner.sample_free_cards(unseen_ids, num_hidden,
                                            rng, boost)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _make_expert(threshold, seed, belief):
    from pf_expert import PFExpert, V5_CFG

    class PhaseOracleExpert(PFExpert):
        def _world_weight(self, opp_hand_ids, opp_discard_count):
            if belief.oracle_on:
                return 1.0               # never veto the true world
            return super()._world_weight(opp_hand_ids,
                                         opp_discard_count)
    return PhaseOracleExpert(veto=True, opp_model=(threshold, 1.0),
                             sample_boost=0.0, seed=seed, **V5_CFG)


class PhaseOraclePolicy(HomeV8Policy):
    """HomeV8Policy with the phase-oracle belief/expert pair."""

    def __init__(self, seed=0, stock_max=99):
        super().__init__(seed=seed)
        self.stock_max = stock_max

    def act(self, game, legal):
        # Run the parent's per-hand reset ourselves, then pre-install
        # the oracle belief/expert pair so the parent's lazy
        # construction block (which would build the standard pair)
        # finds them already present. Book paths are untouched.
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._n += 1
            self._belief = None
            self._expert = None
            self._t2_used = False
        opening = getattr(rnd, "_home_opening", 0)
        if self._belief is None and not opening:
            threshold = game.settings.going_out_deadwood_count
            self._belief = PhaseOracleBelief(
                int(rnd.current_player_id),
                seed=self.seed * 100003 + self._n,
                stock_max=self.stock_max)
            self._expert = _make_expert(threshold,
                                        self.seed + self._n,
                                        self._belief)
        return super().act(game, legal)


def _play(deal_seed, hero_oracle, ps, stock_max):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = int(rnd.current_player_id)
    pols = {hero: (PhaseOraclePolicy(seed=ps, stock_max=stock_max)
                   if hero_oracle else HomeV8Policy(seed=ps)),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[int(rnd.current_player_id)].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return pts if hero == 0 else -pts


def _shard(job):
    n, seed, stock_max = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 15733 + k * 37
        rows.append(dict(
            deal_seed=deal_seed, ps=ps, stock_max=stock_max,
            oracle=_play(deal_seed, True, ps, stock_max),
            base=_play(deal_seed, False, ps, stock_max)))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    rows = list(uniq.values())
    a = np.array([r["oracle"] for r in rows])
    b = np.array([r["base"] for r in rows])
    d = a - b
    n = len(rows)
    K = rows[0]["stock_max"] if rows else "?"
    print(f"=== phase oracle K={K}, {n} paired deals ===")
    print(f"oracle-hero {a.mean():+.2f}  v10-hero {b.mean():+.2f}  "
          f"paired diff {d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}")


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("phase-conditional oracle gate")
    ap.add_argument("--stockmax", type=int, default=99)
    ap.add_argument("--deals", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=140000)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ROWS_PATH = args.out or \
        f"results/phase_oracle_k{args.stockmax}_rows.jsonl"
    if args.report:
        report([json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")])
        return
    open(ROWS_PATH, "w").close()
    jobs = [(args.shard, args.seed + 977 * i, args.stockmax)
            for i in range(max(1, args.deals // args.shard))]
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_shard, jobs), 1):
            with open(ROWS_PATH, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            rows.extend(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({len(rows)} deals) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    report(rows)


if __name__ == "__main__":
    main()

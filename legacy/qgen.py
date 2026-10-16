import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
V5 data generation: per-candidate EV tables from RolloutExpert.analyze().

Adapts exit_gen.py's skeleton (roll-in mix, small-shard fan-out, worker
plumbing) with the label swapped from the argmax action id to the full
analyze() EV table -- dense regression targets instead of classification
labels (HANDOFF_V5_TRAINING.md; the v3-clone lesson).

Differences from exit_gen that MATTER (all deliberate):

  * rules ride the job tuple and every worker calls sim.set_rules(rules)
    first -- exit_gen silently generated CLASSIC-rules labels (Windows spawn
    inherits nothing);
  * sample_boost=2.5 (v4's gated world realism) instead of the 0.0 default;
  * opp_model is TRUTHFUL per roll-in opponent -- (0, 1.0) for teacher-like
    opponents that only ever gin, None for random -- never v4's deployed
    (10, 1.0) robustness assumption: a model that is false of the actual
    opponent biases the world weights and therefore the labels.

The label expert runs K=128 worlds (halves per-candidate SE vs the playing
config's K=48: fewer hands, cleaner labels -- the right trade for
regression) and PLAYS its own chosen action (free -- it comes with the
analysis). Decisions whose analyze() returned no EV table (forced /
class-scoped / no-worlds) are played but not recorded.

Shard fields (padded to CMAX=8 candidates; true max is 6):
  obs float16 (N,10,52)  build_obs -- the imperfect-info hero view (GA1)
  cand int16 (N,8)       candidate action ids, -1 pad
  ev float32 (N,8)       payoff_means (points/100, hero view), 0 pad
  se float32 (N,8)       SE of the paired diff vs prior (se[t_idx]==0), 0 pad
  t_idx/n_cand/chosen/reason/hero int8 (N,)   worlds int16   n_eff float32
  (reason: 0=prior 1=deviate; chosen = index into cand)

    .venv/Scripts/python.exe qgen.py --hands 2000 --workers 12 --worlds 128 \
        --out results/v5_q/smoke --seed 90001
    .venv/Scripts/python.exe qgen.py --check results/v5_q/smoke
    .venv/Scripts/python.exe qgen.py --selftest
    .venv/Scripts/python.exe qgen.py --dump 3      # positions for evaluate.py
"""

import argparse
import glob
import multiprocessing as mp
import os
import random
import time

import numpy as np

import sim
from belief import Belief, build_obs
from expert import RolloutExpert
from sim import FastTeacher, cid, fast_legal_actions, step_fast

CMAX = 8
REASON_CODE = {"prior": 0, "deviate": 1}

# Truthful opponent models for the label expert (see module docstring). The
# teacher-family opponents (and the expert's own prior) knock only at gin,
# which the engine forces anyway; random has no coherent knock model.
OPP_MODEL_V5 = {"self": (0, 1.0), "strong": (0, 1.0), "v2": (0, 1.0),
                "random": None}

# The v2-measured roll-in mix (exit_gen.SHARES): self-play alone never shows
# the long hands weak opponents produce.
SHARES = (("self", 0.40), ("strong", 0.20), ("v2", 0.20), ("random", 0.20))


def gen_chunk(job):
    """Worker: play `hands` games, record analyze() tables at hero decisions,
    write one shard."""
    (opponent, hands, seed, out_path, worlds, truncate, z, boost, rules) = job
    sim.set_rules(rules)
    assert sim.get_rules() == rules
    rng = random.Random(seed)
    expert = RolloutExpert(worlds=worlds, truncate=truncate, z=z,
                           opp_model=OPP_MODEL_V5[opponent],
                           sample_boost=boost, seed=seed)
    if opponent == "self":
        opp_policy = None                # the other hero
    elif opponent == "strong":
        opp_policy = FastTeacher()
    elif opponent == "v2":
        from exit_gen import _V2Clone
        opp_policy = _V2Clone("results/v2_final/gin_rummy_bot.policy.pt")
    else:
        from exit_gen import _RandomPolicy
        opp_policy = _RandomPolicy(rng)

    rows = {k: [] for k in ("obs", "cand", "ev", "se", "t_idx", "n_cand",
                            "chosen", "reason", "hero", "worlds", "n_eff")}
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        hero_seats = (0, 1) if opponent == "self" else (rng.randrange(2),)
        beliefs = {s: Belief(s) for s in hero_seats}
        while game.round.going_out_action is None and not game.round.is_over:
            pid = game.round.current_player_id
            legal = fast_legal_actions(game)
            if len(legal) == 1:
                step_fast(game, legal[0])
                continue
            if pid in beliefs:
                res = expert.analyze(game, beliefs[pid])
                action = res["chosen"]
                if res["reason"] in REASON_CODE:
                    cands = res["candidates"]
                    n = len(cands)
                    assert n <= CMAX
                    obs = build_obs(game, pid, beliefs[pid],
                                    [e.action_id for e in legal])
                    cand = np.full(CMAX, -1, np.int16)
                    cand[:n] = [a.action_id for a in cands]
                    ev = np.zeros(CMAX, np.float32)
                    ev[:n] = res["payoff_means"]
                    se = np.zeros(CMAX, np.float32)
                    se[:n] = res["se"]
                    rows["obs"].append(obs.astype(np.float16))
                    rows["cand"].append(cand)
                    rows["ev"].append(ev)
                    rows["se"].append(se)
                    rows["t_idx"].append(res["t_idx"])
                    rows["n_cand"].append(n)
                    rows["chosen"].append(next(
                        k for k, a in enumerate(cands)
                        if a.action_id == action.action_id))
                    rows["reason"].append(REASON_CODE[res["reason"]])
                    rows["hero"].append(pid)
                    rows["worlds"].append(res["worlds"])
                    rows["n_eff"].append(res["n_eff"])
            else:
                action = opp_policy.act(game, legal)
            step_fast(game, action)
    n = len(rows["obs"])
    np.savez_compressed(
        out_path,
        obs=(np.stack(rows["obs"]) if n else
             np.zeros((0, 10, 52), np.float16)),
        cand=np.array(rows["cand"], np.int16).reshape(n, CMAX),
        ev=np.array(rows["ev"], np.float32).reshape(n, CMAX),
        se=np.array(rows["se"], np.float32).reshape(n, CMAX),
        t_idx=np.array(rows["t_idx"], np.int8),
        n_cand=np.array(rows["n_cand"], np.int8),
        chosen=np.array(rows["chosen"], np.int8),
        reason=np.array(rows["reason"], np.int8),
        hero=np.array(rows["hero"], np.int8),
        worlds=np.array(rows["worlds"], np.int16),
        n_eff=np.array(rows["n_eff"], np.float32))
    return n, out_path


def generate(hands, workers, out_prefix, worlds=128, truncate=16, z=1.5,
             boost=2.5, rules="full", seed=80000, splits=8):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # Small jobs: shards land incrementally and stragglers even out.
    jobs = []
    per_worker = max(1, hands // workers)
    for w in range(workers):
        offset = 0
        for opponent, share in SHARES:
            for split in range(splits):
                n = max(1, int(per_worker * share / splits))
                out_path = f"{out_prefix}_w{w}_{opponent}_{split}.npz"
                jobs.append((opponent, n, seed + 100 * w + 10 * split
                             + offset, out_path, worlds, truncate, z, boost,
                             rules))
            offset += 1
    t0 = time.time()
    total, done = 0, 0
    with mp.Pool(workers) as pool:
        for n, path in pool.imap_unordered(gen_chunk, jobs):
            total += n
            done += 1
            dt = time.time() - t0
            print(f"[{done}/{len(jobs)}] +{n} rows ({total} total, "
                  f"{total / dt:.1f}/s) {os.path.basename(path)}", flush=True)
    dt = time.time() - t0
    print(f"generated {total} labeled decisions from ~{hands} hands in "
          f"{dt:.0f}s ({total / dt:.1f}/s) -> {out_prefix}_*.npz", flush=True)
    return total


# ---------------------------------------------------------------------------
# --check: shard invariants and label statistics.
# ---------------------------------------------------------------------------

def check(prefix):
    files = sorted(glob.glob(f"{prefix}_*.npz"))
    assert files, f"no shards match {prefix}_*.npz"
    tot = dev = draw = knock_nodes = 0
    ev_abs, se_all, worlds_all = [], [], []
    per_opp = {}
    for f in files:
        d = np.load(f)
        n = len(d["t_idx"])
        if n == 0:
            continue
        cand, ev, se = d["cand"], d["ev"], d["se"]
        t_idx, n_cand = d["t_idx"].astype(int), d["n_cand"].astype(int)
        rows = np.arange(n)
        assert np.all(se[rows, t_idx] == 0.0), f"{f}: se[t_idx] != 0"
        assert n_cand.min() >= 2 and n_cand.max() <= 6, f"{f}: n_cand range"
        assert np.all(t_idx < n_cand) and np.all(
            d["chosen"].astype(int) < n_cand), f"{f}: index out of range"
        mask = np.arange(CMAX)[None, :] < n_cand[:, None]
        assert cand[mask].min() >= 2 and cand[mask].max() <= 109, \
            f"{f}: candidate id range"
        assert np.all(cand[~mask] == -1), f"{f}: padding"
        # deviate rows must actually choose a non-teacher candidate
        is_dev = d["reason"] == 1
        assert np.all((d["chosen"] != t_idx)[is_dev]), f"{f}: deviate=chosen"
        assert np.all((d["chosen"] == t_idx)[~is_dev]), f"{f}: prior!=teacher"
        tot += n
        dev += int(is_dev.sum())
        draw += int((n_cand == 2).sum())
        knock_nodes += int(np.any((cand >= 58) & mask, axis=1).sum())
        ev_abs.append(np.abs(ev[mask]) * 100.0)
        se_pts = se[mask] * 100.0
        se_all.append(se_pts[se_pts > 0])
        worlds_all.append(d["worlds"])
        opp = os.path.basename(f).rsplit("_", 2)[-2]
        per_opp[opp] = per_opp.get(opp, 0) + n
    ev_abs = np.concatenate(ev_abs)
    se_all = np.concatenate(se_all)
    worlds_all = np.concatenate(worlds_all)
    dev_rate = dev / tot
    print(f"{len(files)} shards, {tot} rows  {per_opp}")
    print(f"deviation rate {dev_rate:.3f}  draw-node frac {draw / tot:.3f}  "
          f"knock-candidate frac {knock_nodes / tot:.3f}")
    print(f"|EV| pts: mean {ev_abs.mean():.2f}  p95 "
          f"{np.percentile(ev_abs, 95):.2f}  max {ev_abs.max():.2f}")
    print(f"label SE pts (nonzero): mean {se_all.mean():.2f}  p95 "
          f"{np.percentile(se_all, 95):.2f}")
    print(f"worlds: min {worlds_all.min()}  mean {worlds_all.mean():.1f}")
    # v4-config experts deviate ~13-18% (measured K=48 0.132 / K=128 0.152 vs
    # strong, smoke 0.181 overall; v3search-era was ~16% at T=16) -- the
    # handoff's "~4%" was the v3-clone era. Band is a structural sanity check.
    assert 0.02 <= dev_rate <= 0.30, f"deviation rate {dev_rate:.3f} " \
        "outside the plausible 2-30% regime"
    print("check: all invariants pass")


def selftest(worlds=32):
    """Same job tuple twice -> identical arrays (worker determinism)."""
    import tempfile
    sim.set_rules("full")
    tmp = tempfile.mkdtemp()
    paths = [os.path.join(tmp, f"st{i}.npz") for i in (0, 1)]
    for p in paths:
        gen_chunk(("strong", 12, 4242, p, worlds, 16, 1.5, 2.5, "full"))
    a, b = np.load(paths[0]), np.load(paths[1])
    assert a["obs"].shape == b["obs"].shape and len(a["t_idx"]) > 0
    for k in a.files:
        assert np.array_equal(a[k], b[k]), f"selftest: {k} differs"
    print(f"selftest: {len(a['t_idx'])} rows bit-identical across two runs")


# ---------------------------------------------------------------------------
# --dump: write evaluate.py position files from live labeled decisions, so
# the recorded EV table can be cross-checked against the evaluator (the same
# analyze() code path driven through an independent state reconstruction).
# ---------------------------------------------------------------------------

def _card_tok(card):
    return card.rank + card.suit


def _dump_position(game, pid, opp_model, boost, path):
    from rlcard.games.gin_rummy.utils.action_event import PickUpDiscardAction
    from rlcard.games.gin_rummy.utils.move import (
        DealHandMove, DiscardMove, DrawCardMove, PickupDiscardMove)
    rnd = game.round
    hero, opp = rnd.players[pid], rnd.players[1 - pid]
    lines = [f"me: {' '.join(_card_tok(c) for c in hero.hand)}"]
    if opp.known_cards:
        lines.append(
            f"opp-known: {' '.join(_card_tok(c) for c in opp.known_cards)}")
    pile = rnd.dealer.discard_pile
    if pile:
        lines.append(f"pile: {' '.join(_card_tok(c) for c in pile)}")
    if len(hero.hand) == 11 and isinstance(game.actions[-1],
                                           PickUpDiscardAction):
        lines.append(f"picked: {_card_tok(rnd.move_sheet[-1].card)}")
    if hero.known_cards:
        lines.append(
            f"my-known: {' '.join(_card_tok(c) for c in hero.known_cards)}")
    # Event extraction mirrors Belief.update exactly (replayed public pile).
    replay = []
    for move in rnd.move_sheet:
        if isinstance(move, DealHandMove):
            continue
        is_opp = move.player.player_id != pid
        if isinstance(move, DrawCardMove):
            if is_opp and replay:
                lines.append(f"event: opp-decline {DECK_TOK[replay[-1]]}")
        elif isinstance(move, PickupDiscardMove):
            taken = replay.pop()
            if is_opp:
                lines.append(f"event: opp-pickup {DECK_TOK[taken]}")
        elif isinstance(move, DiscardMove):
            card_id = cid(move.action.card)
            replay.append(card_id)
            if is_opp:
                lines.append(f"event: opp-discard {DECK_TOK[card_id]}")
            else:
                lines.append(f"event: my-discard {DECK_TOK[card_id]}")
    if opp_model is not None:
        lines.append(f"opp-model: {opp_model[0]},{opp_model[1]}")
    lines.append(f"boost: {boost}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


DECK_TOK = [c.rank + c.suit for c in sim.DECK]


def dump(count, out_dir="results/v5_q", worlds=128, seed=77001, every=6):
    """Single-process mini-generation vs the teacher; every `every`-th
    labeled decision becomes a position file + a printed EV table."""
    sim.set_rules("full")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    opp_model, boost = OPP_MODEL_V5["strong"], 2.5
    expert = RolloutExpert(worlds=worlds, truncate=16, z=1.5,
                           opp_model=opp_model, sample_boost=boost, seed=seed)
    opp_policy = FastTeacher()
    labeled = written = 0
    while written < count:
        game = sim.new_game(rng.randrange(1 << 30))
        hero = rng.randrange(2)
        belief = Belief(hero)
        while game.round.going_out_action is None and not game.round.is_over:
            pid = game.round.current_player_id
            legal = fast_legal_actions(game)
            if len(legal) == 1:
                step_fast(game, legal[0])
                continue
            if pid == hero:
                res = expert.analyze(game, belief)
                action = res["chosen"]
                if res["reason"] in REASON_CODE:
                    labeled += 1
                    if labeled % every == 0 and written < count:
                        written += 1
                        path = os.path.join(out_dir, f"pos_{written}.txt")
                        _dump_position(game, pid, opp_model, boost, path)
                        print(f"--- {path}  (hero seat {pid}, "
                              f"reason {res['reason']})")
                        for k, a in enumerate(res["candidates"]):
                            mark = "*" if k == res["candidates"].index(
                                res["chosen"]) else " "
                            prior = " (prior)" if k == res["t_idx"] else ""
                            print(f" {mark} id {a.action_id:3d}  "
                                  f"EV {res['payoff_means'][k] * 100:+7.2f}  "
                                  f"diff {res['diff_means'][k] * 100:+6.2f} "
                                  f"+/-{res['se'][k] * 100:5.2f}{prior}")
            else:
                action = opp_policy.act(game, legal)
            step_fast(game, action)
    print(f"\ncross-check: .venv/Scripts/python.exe evaluate.py "
          f"{out_dir}/pos_1.txt --worlds {worlds}")


def main():
    ap = argparse.ArgumentParser("v5 EV-label generation")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/v5_q/smoke")
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--boost", type=float, default=2.5)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=80000)
    ap.add_argument("--check", default=None, metavar="PREFIX")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump", type=int, default=0, metavar="N")
    args = ap.parse_args()
    if args.check:
        check(args.check)
    elif args.selftest:
        selftest()
    elif args.dump:
        dump(args.dump, worlds=args.worlds)
    else:
        generate(args.hands, args.workers, args.out, args.worlds,
                 args.truncate, args.z, args.boost, args.rules, args.seed)


if __name__ == "__main__":
    main()

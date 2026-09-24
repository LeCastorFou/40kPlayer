#!/usr/bin/env python3
"""Mesure la vitesse du moteur : parties aléatoires complètes, clonage, et profil optionnel.

    python3 scripts/bench.py                 # 20 parties aléatoires
    python3 scripts/bench.py --games 50 --profile
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import load_catalog  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.actions import EndPhaseAction  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402


def random_playout(engine: Engine, state, rng: random.Random) -> int:
    n = 0
    while not engine.is_over(state):
        d = engine.decision(state)
        opts = d.options
        if d.kind == "select_unit":
            opts = [o for o in opts if not isinstance(o, EndPhaseAction)] or opts
        engine.step(state, rng.choice(opts), validate=False)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="garder le journal (plus lent)")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    cat = load_catalog()
    engine = Engine()
    template = new_toy_state(cat, seed=args.seed)
    template.recording = args.record
    # chauffe (imports, caches)
    s = template.clone(seed=10_000)
    engine.start(s)
    random_playout(engine, s, random.Random(10_000))

    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    t0 = time.perf_counter()
    decisions = 0
    for g in range(args.games):
        s = template.clone(seed=args.seed + g)
        engine.start(s)
        decisions += random_playout(engine, s, random.Random(args.seed + g))
    dt = time.perf_counter() - t0
    if prof:
        prof.disable()

    mid = template.clone(seed=1)
    engine.start(mid)
    random_playout_steps = 20
    rng = random.Random(1)
    for _ in range(random_playout_steps):
        d = engine.decision(mid)
        engine.step(mid, rng.choice(d.options), validate=False)
    t1 = time.perf_counter()
    for _ in range(2000):
        mid.clone(record=False)
    clone_us = (time.perf_counter() - t1) / 2000 * 1e6

    print(f"{args.games} parties en {dt:.2f} s : {dt / args.games * 1000:.0f} ms/partie, "
          f"{decisions / args.games:.0f} décisions/partie, {dt / decisions * 1000:.2f} ms/décision ; clone : {clone_us:.0f} µs")
    los = template.los
    print(f"cache lignes de vue : {len(los.cache)} paires, {los.hits} lectures, {los.misses} calculs")
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()

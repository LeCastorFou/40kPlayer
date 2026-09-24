#!/usr/bin/env python3
"""Joue une partie complète bot contre bot et affiche le journal.

    python3 scripts/play.py                          # aléatoire vs aléatoire, seed 0
    python3 scripts/play.py --seed 3 --png out/      # + un PNG de la table après chaque tour
    python3 scripts/play.py --quiet --games 20       # statistiques sur 20 parties
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.agents import RandomAgent  # noqa: E402
from fortyk.data import load_catalog  # noqa: E402
from fortyk.engine.game import Game  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402

SIDE_COLORS = {"attacker": (200, 60, 60, 255), "defender": (60, 110, 200, 255)}


def snapshot(state, out_dir: Path, tag: str) -> None:
    models = []
    for u in state.units.values():
        for m in u.alive_models:
            models.append((m.x, m.y, m.radius, SIDE_COLORS[u.side]))
    state.layout.render(out_dir / f"{tag}.png", scale=12, models=models)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--quiet", action="store_true", help="n'affiche pas le journal")
    p.add_argument("--png", help="dossier où écrire un PNG de la table après chaque tour")
    args = p.parse_args(argv)

    cat = load_catalog()
    wins = Counter()
    scores = Counter()
    t0 = time.perf_counter()
    for g in range(args.games):
        seed = args.seed + g
        state = new_toy_state(cat, seed=seed)
        agents = {"attacker": RandomAgent("Aléatoire A", seed * 2 + 1), "defender": RandomAgent("Aléatoire D", seed * 2 + 2)}
        game = Game(state, agents)
        if args.png:
            out = Path(args.png)
            out.mkdir(parents=True, exist_ok=True)
            counter = {"n": 0}

            def on_turn_end(s, out=out, counter=counter, seed=seed):
                counter["n"] += 1
                snapshot(s, out, f"game{seed}_r{s.battle_round}_{counter['n']:02d}_{s.side_to_move}")

            game.on_turn_end = on_turn_end
        result = game.play()
        wins[result.winner or "égalité"] += 1
        scores["attacker"] += result.scores["attacker"]
        scores["defender"] += result.scores["defender"]
        if not args.quiet:
            print("\n".join(result.log))
        print(f"Partie {seed} : {result}")
    dt = time.perf_counter() - t0
    if args.games > 1:
        print(f"\n{args.games} parties en {dt:.1f} s ({dt / args.games:.2f} s/partie)")
        print("Victoires :", dict(wins))
        print(f"Score moyen : attaquant {scores['attacker'] / args.games:.1f}, défenseur {scores['defender'] / args.games:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

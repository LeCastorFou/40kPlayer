#!/usr/bin/env python3
"""Joue une partie contre le bot, dans le terminal.

    python3 scripts/play_human.py                       # tu joues l'attaquant (Space Marines), bot aléatoire en face
    python3 scripts/play_human.py --side defender       # tu joues les Emperor's Children
    python3 scripts/play_human.py --board /tmp/board.png --seed 4

Le plateau est redessiné dans le PNG indiqué à chaque décision : ouvre-le dans Aperçu et garde-le
à côté du terminal, il se met à jour tout seul. Ton unité en cours de décision est en jaune.
Tape « ? » à n'importe quel moment pour l'aide sur la saisie libre, « q » pour abandonner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.agents import HumanAgent, Quit, RandomAgent  # noqa: E402
from fortyk.data import load_catalog  # noqa: E402
from fortyk.engine.game import Game  # noqa: E402
from fortyk.engine.state import other_side  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["attacker", "defender"], default="attacker", help="ton camp (attaquant = Space Marines, défenseur = Emperor's Children)")
    p.add_argument("--seed", type=int, default=0, help="graine des dés")
    p.add_argument("--board", default="board.png", help="PNG du plateau, redessiné à chaque décision")
    p.add_argument("--bot", choices=["random"], default="random", help="adversaire")
    args = p.parse_args(argv)

    print("Chargement des fiches Wahapedia…")
    cat = load_catalog()
    state = new_toy_state(cat, seed=args.seed)
    human = HumanAgent("Valentin", board_png=args.board)
    bot = RandomAgent("Bot aléatoire", seed=args.seed + 99)
    agents = {args.side: human, other_side(args.side): bot}
    game = Game(state, agents)
    game.on_log = print
    print(f"\nTu joues {args.side} ({'Space Marines' if args.side == 'attacker' else 'Emperor’s Children'}). Plateau : {args.board}\n")
    try:
        result = game.play()
    except Quit:
        print("\nPartie abandonnée.")
        return 1
    print(f"\n{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

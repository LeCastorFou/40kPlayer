#!/usr/bin/env python3
"""Lance le service web 40kPlayer : parties à deux joueurs (liens secrets) ou contre le bot, sauvegardées.

    python3 scripts/serve.py                                  # http://127.0.0.1:8040/ — accueil
    python3 scripts/serve.py --host 0.0.0.0 --port 8040       # accessible depuis le réseau
    python3 scripts/serve.py --data-dir ~/40k/parties
    python3 scripts/serve.py --vs-bot --side defender --attacker-list ec_mercurial_host_2000

Sur l'accueil : créer une partie (ta liste, ton camp ; contre le bot, ou contre un humain qui rejoint avec
le code de partie et choisit sa liste), rejoindre avec un code, reprendre une partie, importer une liste NewRecruit. Chaque partie est un fichier JSON du
dossier de données (``data/games`` par défaut, ou ``FORTYK_DATA_DIR``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.web import serve  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("FORTYK_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8040")))
    p.add_argument("--data-dir", help="dossier des parties (défaut : FORTYK_DATA_DIR ou data/games)")
    p.add_argument("--no-browser", action="store_true", help="ne pas ouvrir le navigateur automatiquement")
    p.add_argument("--vs-bot", action="store_true", help="crée tout de suite une partie contre le bot et ouvre son lien")
    p.add_argument("--side", choices=["attacker", "defender"], default="attacker", help="avec --vs-bot : ton camp")
    p.add_argument("--attacker-list", help="avec --vs-bot : liste de l'attaquant (nom dans data/lists ; sans : toy model)")
    p.add_argument("--defender-list", help="avec --vs-bot : liste du défenseur")
    args = p.parse_args(argv)
    print("Chargement des fiches Wahapedia…")
    quick = None
    if args.vs_bot or args.attacker_list or args.defender_list:
        quick = {"side": args.side, "attacker_list": args.attacker_list, "defender_list": args.defender_list}
    serve(host=args.host, port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir, quick=quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

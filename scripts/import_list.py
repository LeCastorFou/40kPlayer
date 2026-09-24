#!/usr/bin/env python3
"""Importe une liste d'armée (export texte NewRecruit) et affiche la fiche d'armée résolue contre
l'export Wahapedia : unités, figurines, armes, personnages attachés, améliorations, détachements,
règle choisie, stratagèmes disponibles, contrôle des points, et ce que le moteur en joue déjà.

    python3 scripts/import_list.py data/lists/ec_mercurial_host_2000.txt
    python3 scripts/import_list.py ec_mercurial_host_2000                # nom d'une liste de data/lists
    pbpaste | python3 scripts/import_list.py - --save ma_liste          # depuis le presse-papiers (Mac)
    python3 scripts/import_list.py ma_liste.txt --json ma_liste.json     # export JSON
    python3 scripts/import_list.py ma_liste.txt --no-coverage --no-stratagems

``--save NOM`` range la liste dans data/lists/NOM.txt : elle se joue ensuite par son nom
(``scripts/serve.py --attacker-list NOM``) ou depuis le menu « Listes / nouvelle partie » du navigateur.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import load_catalog  # noqa: E402
from fortyk.data.list_library import list_path, resolve_text, save_list  # noqa: E402
from fortyk.engine.coverage import coverage, format_coverage  # noqa: E402


def weapon_line(w) -> str:
    rng = "Mêlée" if w.is_melee else f'{w.range_in:g}"' if w.range_in else "—"
    skill = "N/A" if w.skill is None else f"{w.skill}+"
    kws = f" [{', '.join((getattr(k, 'raw', '') or k.name).strip() for k in w.keywords)}]" if w.keywords else ""
    return f"{w.name} : {rng} A{w.attacks} {'CC' if w.is_melee else 'CT'}{skill} F{w.strength} PA{w.ap} D{w.damage}{kws}"


def _group(name: str) -> str:
    return name.split(" – ")[0].split(" - ")[0]


def gear_labels(weapons) -> list:
    """« Heavy reaper autocannon » ×2 → « 2× Heavy reaper autocannon » ; les profils d'une même arme
    (strike / sweep, standard / supercharge) ne comptent qu'une fois."""
    labels = []
    for g in dict.fromkeys(_group(w) for w in weapons):
        n = sum(1 for w in weapons if _group(w) == g)
        profiles = len({w for w in weapons if _group(w) == g})
        copies = n // max(1, profiles)
        labels.append(f"{copies}× {g}" if copies > 1 else g)
    return labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="fichier texte, nom d'une liste de data/lists, ou « - » pour lire l'entrée standard")
    ap.add_argument("--save", metavar="NOM", help="enregistre la liste dans data/lists/NOM.txt")
    ap.add_argument("--overwrite", action="store_true", help="avec --save : remplace une liste du même nom")
    ap.add_argument("--json", help="écrit la liste résolue en JSON")
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--no-stratagems", action="store_true")
    ap.add_argument("--weapons", action="store_true", help="détaille les profils d'arme")
    args = ap.parse_args(argv)

    cat = load_catalog()
    if args.path == "-":
        text = sys.stdin.read()
    else:
        p = Path(args.path)
        text = (p if p.exists() else list_path(args.path)).read_text(encoding="utf-8")
    al = resolve_text(text, cat)
    if args.save:
        name, al = save_list(text, cat, name=args.save, overwrite=args.overwrite)
        print(f"Liste enregistrée : data/lists/{name}.txt\n")
    d = al.doc
    print(f"{al.faction_name} [{al.faction_id}] — {al.points} pts recalculés (liste : {d.total_points} pts) — {len(al.units)} unités")
    print("Détachements : " + ", ".join(f"{x.name} ({x.dp} DP, {x.force_disposition})" for x in al.detachments) + f" — total {al.dp} DP")
    print(f"Force Disposition : {d.force_disposition or '—'}")
    for a in al.active_rules:
        print(f"Règle de détachement : {a.name} ({a.detachment}) — {a.text}")
    wl = al.warlord
    print(f"Warlord : {wl.key if wl else '—'}")
    if d.secondaries:
        print(f"Secondaires : {d.secondaries}")
    print()
    by_key = {u.key: u for u in al.units}
    for u in al.units:
        if u.leading:
            continue
        leaders = [by_key[k] for k in u.leaders]
        title = u.key + "".join(f" + {l.key}" for l in leaders)
        pts = sum(x.points_computed or 0 for x in [u] + leaders)
        p = u.datasheet.models[0]
        print(f"■ {title} — {pts} pts — M{p.move_in:g}\" T{p.toughness} Sv{p.save}+" + (f" Inv{p.invuln}+" if p.invuln else "") + f" W{p.wounds} Ld{p.leadership}+ OC{p.oc}")
        for member in leaders + [u]:
            extra = []
            if member.warlord:
                extra.append("Warlord")
            if member.enhancement:
                extra.append(f"amélioration {member.enhancement.name} (+{member.enhancement.cost} pts)")
            counts = {}
            for m in member.models:
                k = (m.name, tuple(w.name for w in m.weapons), m.wargear_abilities)
                counts[k] = counts.get(k, 0) + 1
            for (name, weapons, wargear), n in counts.items():
                print(f"    {n}× {name} : {', '.join(gear_labels(weapons) + [f'{w} (équipement)' for w in wargear])}" + (f"   [{', '.join(extra)}]" if extra else ""))
                extra = []
        if args.weapons:
            shown = set()
            for member in leaders + [u]:
                for m in member.models:
                    for w in m.weapons:
                        if id(w) not in shown:
                            shown.add(id(w))
                            print(f"        · {weapon_line(w)}")
    if al.issues:
        print("\nPoints à vérifier :")
        for i in al.issues:
            print(f"  ! {i}")
    else:
        print("\nListe valide : points, effectifs, équipement et attachements vérifiés.")
    if not args.no_stratagems:
        print(f"\nStratagèmes disponibles ({len(al.stratagems)}) :")
        for s in al.stratagems:
            print(f"  {s.name} ({s.cp} CP, {s.detachment or 'core'}) — {s.turn} · {s.phase}")
            if s.when:
                print(f"      QUAND : {s.when}")
            if s.target:
                print(f"      CIBLE : {s.target}")
            if s.effect:
                print(f"      EFFET : {s.effect}")
    if not args.no_coverage:
        print()
        print(format_coverage(coverage(al)))
    if args.json:
        Path(args.json).write_text(json.dumps(al.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nJSON écrit dans {args.json}")
    return 0 if not al.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

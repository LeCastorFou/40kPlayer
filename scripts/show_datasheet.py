#!/usr/bin/env python3
"""Affiche une fiche d'unité telle que le chargeur la comprend (vérification visuelle).

    python3 scripts/show_datasheet.py "Intercessor Squad"
    python3 scripts/show_datasheet.py "Leman Russ Commander" --faction AM
    python3 scripts/show_datasheet.py --find "intercessor"          # recherche par regex
    python3 scripts/show_datasheet.py "Infractors" --json > infractors.json
    python3 scripts/show_datasheet.py --faction EC --list           # toutes les fiches d'une faction
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import Catalog, CatalogError, Datasheet, load_catalog  # noqa: E402


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"non sérialisable : {type(obj).__name__}")


def render(ds: Datasheet, cat: Catalog, full: bool = False) -> str:
    out = []
    out.append(f"{ds.name}  [{ds.faction_name} / {ds.faction_id}]  id={ds.id}" + ("  (virtuelle)" if ds.virtual else ""))
    out.append(f"  {ds.link}")
    out.append("")
    out.append("  PROFILS")
    for m in ds.models:
        mv = "-" if m.move_in is None else f'{m.move_in:g}"'
        inv = f"  Inv {m.invuln}+" if m.invuln else ""
        base = m.base.raw if m.base else "(socle non précisé)"
        out.append(
            f"    {m.name:32s} M {mv:>4s}  T {m.toughness:<2d} Sv {m.save}+{inv}  W {m.wounds:<2d} Ld {m.leadership}+  OC {m.oc:<2d} | {base}"
        )
    out.append("")
    out.append("  ARMES DE TIR")
    for w in ds.ranged_weapons:
        out.append(f"    {w}")
    out.append("  ARMES DE MÊLÉE")
    for w in ds.melee_weapons:
        out.append(f"    {w}")
    unknown = sorted({str(k) for w in ds.weapons for k in w.unknown_keywords})
    if unknown:
        out.append(f"    ! mots-clés d'arme hors règles de base : {', '.join(unknown)}")
    out.append("")
    out.append("  CAPACITÉS")
    for a in ds.abilities:
        param = f" {a.parameter}" if a.parameter else ""
        who = f" ({a.model})" if a.model else ""
        head = f"    [{a.type}] {a.name}{param}{who}"
        if full or a.type not in ("Core", "Faction"):
            text = a.text.replace("\n", "\n        ")
            out.append(f"{head}\n        {text}")
        else:
            out.append(head)
    out.append("")
    out.append(f"  MOTS-CLÉS   {', '.join(ds.keywords)}")
    out.append(f"  FACTION     {', '.join(ds.faction_keywords)}")
    out.append(f"  COMPOSITION {' ; '.join(ds.composition)}")
    if ds.loadout:
        out.append(f"  ÉQUIPEMENT  {ds.loadout}")
    out.append("  COÛTS")
    for c in ds.costs:
        span = ""
        if c.kind == "unit" and (c.from_copy != 1 or c.to_copy is not None):
            span = f"  (exemplaires {c.from_copy}" + (f" à {c.to_copy})" if c.to_copy else "+)")
        ctx = f"  [{c.context}]" if c.context else ""
        out.append(f"    {c.kind:8s} {c.description:40s} {c.points:4d} pts{span}{ctx}")
    if ds.options:
        out.append("  OPTIONS")
        for o in ds.options:
            out.append("    - " + o.replace("\n", " / "))
    if ds.leader_ids:
        names = sorted(cat.datasheets[i].name for i in ds.leader_ids if i in cat.datasheets)
        out.append(f"  PEUT ÊTRE MENÉE PAR  {', '.join(names)}")
    if ds.can_lead_ids:
        names = sorted(cat.datasheets[i].name for i in ds.can_lead_ids if i in cat.datasheets)
        out.append(f"  PEUT MENER  {', '.join(names)}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", help="nom exact de la fiche (insensible à la casse)")
    p.add_argument("--faction", "-f", help="id (SM, EC…) ou nom de faction pour lever une ambiguïté")
    p.add_argument("--find", help="regex : liste les fiches dont le nom correspond")
    p.add_argument("--list", action="store_true", help="liste les fiches (de la faction si --faction)")
    p.add_argument("--json", action="store_true", help="sortie JSON complète de la fiche")
    p.add_argument("--full", action="store_true", help="afficher aussi le texte des capacités Core / Faction")
    p.add_argument("--raw-dir", help="dossier des CSV (défaut : data/wahapedia/raw ou $FORTYK_WAHAPEDIA_DIR)")
    args = p.parse_args(argv)

    cat = load_catalog(args.raw_dir)

    if args.list or args.find:
        rows = cat.find(args.find or ".", args.faction)
        for d in rows:
            cost = f"{d.min_cost} pts" if d.min_cost is not None else "?"
            print(f"  {d.faction_id:4s} {d.name:45s} {cost:>8s}  {d.id}")
        print(f"{len(rows)} fiche(s) — export Wahapedia du {cat.last_update}", file=sys.stderr)
        return 0

    if not args.name:
        p.error("donne un nom de fiche, ou --find / --list")
    try:
        ds = cat.get(args.name, args.faction)
    except CatalogError as err:
        print(err, file=sys.stderr)
        hits = cat.find(args.name, args.faction)
        if hits:
            print("Fiches proches : " + ", ".join(f"{d.name} [{d.faction_id}]" for d in hits[:12]), file=sys.stderr)
        return 1

    if args.json:
        json.dump(ds, sys.stdout, default=_json_default, ensure_ascii=False, indent=2)
        print()
    else:
        print(render(ds, cat, full=args.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

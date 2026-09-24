"""Bibliothèque de listes d'armée : les exports texte NewRecruit rangés dans ``data/lists``.

Importer une nouvelle liste = coller le texte (interface navigateur, ou ``scripts/import_list.py -``
depuis le presse-papiers) : elle est résolue contre Wahapedia, le rapport (points recalculés, points à
vérifier, couverture des règles) est affiché, puis elle est enregistrée sous un nom court et devient
jouable par son nom (``--attacker-list mon_nom`` ou le menu « Nouvelle partie »).
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .army_list import ArmyList, parse_army_list, resolve_army_list

__all__ = ["LISTS_DIR", "BUILTIN_LISTS_DIR", "slugify", "list_path", "list_text", "saved_lists", "load_named_list", "resolve_text",
           "save_list", "list_report"]

#: listes livrées avec le dépôt (lecture seule sur un serveur)
BUILTIN_LISTS_DIR = Path(__file__).resolve().parents[2] / "data" / "lists"
#: dossier où l'on enregistre les listes importées (``FORTYK_LISTS_DIR`` sur un serveur : un volume persistant)
LISTS_DIR = Path(os.environ["FORTYK_LISTS_DIR"]).expanduser() if os.environ.get("FORTYK_LISTS_DIR") else BUILTIN_LISTS_DIR


def _dirs(directory: Optional[Path] = None) -> List[Path]:
    """Dossiers consultés : ``directory`` seul s'il est donné, sinon les listes enregistrées puis celles du dépôt."""
    if directory is not None:
        return [directory]
    out = [LISTS_DIR]
    if BUILTIN_LISTS_DIR.resolve() != Path(LISTS_DIR).resolve():
        out.append(BUILTIN_LISTS_DIR)
    return out

_CACHE: Dict[Path, Tuple[float, dict]] = {}


def slugify(text: str) -> str:
    """« Mercurial Host 2000 pts » → « mercurial_host_2000_pts »."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t[:60] or "liste"


def list_path(name: str, directory: Optional[Path] = None) -> Path:
    """Chemin d'une liste par son nom court (ou un chemin de fichier existant) : la première trouvée
    parmi les listes enregistrées puis celles du dépôt ; à défaut, là où elle serait enregistrée."""
    p = Path(name)
    if p.suffix == ".txt" and p.exists():
        return p
    fname = f"{slugify(p.stem if p.suffix == '.txt' else name)}.txt"
    dirs = _dirs(directory)
    for d in dirs:
        if (d / fname).exists():
            return d / fname
    return dirs[0] / fname


def list_text(name: str, directory: Optional[Path] = None) -> str:
    """Texte NewRecruit d'une liste enregistrée (gardé tel quel dans les parties sauvegardées)."""
    path = list_path(name, directory)
    if not path.exists():
        raise FileNotFoundError(f"liste inconnue : {name}")
    return path.read_text(encoding="utf-8")


def resolve_text(text: str, cat) -> ArmyList:
    """Texte NewRecruit → liste résolue (lève ValueError si ce n'est pas une liste lisible)."""
    if not text or not text.strip():
        raise ValueError("liste vide")
    doc = parse_army_list(text)
    if not doc.entries:
        raise ValueError("aucune unité reconnue : colle l'export texte de NewRecruit (« Copy to clipboard » / format texte)")
    return resolve_army_list(doc, cat)


def default_name(al: ArmyList) -> str:
    det = al.detachments[0].name if al.detachments else ""
    return slugify(f"{al.faction_id} {det} {al.doc.total_points or al.points}")


def save_list(text: str, cat, name: Optional[str] = None, directory: Optional[Path] = None, overwrite: bool = False) -> Tuple[str, ArmyList]:
    """Résout puis enregistre la liste ; retourne (nom court, liste). Sans ``overwrite``, un nom déjà
    pris reçoit un suffixe (_2, _3…)."""
    al = resolve_text(text, cat)
    directory = directory or LISTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    base = slugify(name) if name else default_name(al)
    slug, k = base, 1
    while (directory / f"{slug}.txt").exists() and not overwrite:
        k += 1
        slug = f"{base}_{k}"
    (directory / f"{slug}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    return slug, al


def load_named_list(name: str, cat, directory: Optional[Path] = None) -> ArmyList:
    path = list_path(name, directory)
    if not path.exists():
        raise FileNotFoundError(f"liste inconnue : {name} (cherchée dans {path.parent})")
    return resolve_text(path.read_text(encoding="utf-8"), cat)


def list_report(al: ArmyList) -> dict:
    """Résumé affichable d'une liste résolue : points, détachements, unités, points à vérifier, couverture."""
    from ..engine.coverage import coverage

    items = coverage(al)
    counts = {k: sum(1 for it in items if it.status == k) for k in ("joué", "partiel", "non joué")}
    by_key = {u.key: u for u in al.units}
    units = []
    for u in al.units:
        if u.leading:
            continue
        leaders = [by_key[k].datasheet.name for k in u.leaders]
        pts = (u.points_computed or 0) + sum(by_key[k].points_computed or 0 for k in u.leaders)
        units.append({"name": u.datasheet.name + "".join(f" + {n}" for n in leaders), "models": len(u.models) + sum(len(by_key[k].models) for k in u.leaders),
                      "points": pts, "transport": bool(u.datasheet.transport)})
    return {
        "faction": al.faction_name,
        "points": al.points,
        "points_listed": al.doc.total_points,
        "detachments": [d.name for d in al.detachments],
        "rules": [a.name for a in al.active_rules],
        "units": units,
        "issues": list(al.issues),
        "coverage": counts,
        "not_played": [f"{it.name} ({it.category})" for it in items if it.status == "non joué" and not it.category.startswith("stratagème")][:40],
    }


def saved_lists(cat, directory: Optional[Path] = None) -> List[dict]:
    """Listes enregistrées (résumé), triées par nom ; les listes illisibles sont signalées. Sans
    ``directory`` : les listes enregistrées puis celles du dépôt (``builtin``), un nom n'apparaissant qu'une fois."""
    out = []
    seen = set()
    paths = []
    for k, d in enumerate(_dirs(directory)):
        if not d.exists():
            continue
        for path in sorted(d.glob("*.txt")):
            if path.stem not in seen:
                seen.add(path.stem)
                paths.append((path, k > 0))
    paths.sort(key=lambda x: x[0].stem)
    for path, builtin in paths:
        mtime = path.stat().st_mtime
        hit = _CACHE.get(path)
        if hit is None or hit[0] != mtime:
            try:
                al = resolve_text(path.read_text(encoding="utf-8"), cat)
                info = {"name": path.stem, "faction": al.faction_name, "points": al.points,
                        "detachments": [d.name for d in al.detachments], "units": sum(1 for u in al.units if not u.leading), "issues": len(al.issues)}
            except Exception as err:  # noqa: BLE001
                info = {"name": path.stem, "error": f"{type(err).__name__}: {err}"}
            _CACHE[path] = (mtime, info)
            hit = _CACHE[path]
        item = dict(hit[1])
        if builtin:
            item["builtin"] = True
        out.append(item)
    return out

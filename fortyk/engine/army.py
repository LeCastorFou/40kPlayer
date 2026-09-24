"""Construction d'unités jouables à partir des fiches Wahapedia.

Une :class:`UnitSpec` décrit ce qu'on veut aligner (fiche, effectif, camp) ;
:func:`build_unit` fabrique l':class:`~fortyk.engine.state.Unit` avec ses figurines
équipées selon l'équipement par défaut de la fiche (« Every model is equipped with:
bolt pistol; bolt rifle; close combat weapon. »). Les options d'équipement peuvent
être forcées figurine par figurine avec ``weapon_overrides``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.catalog import Catalog
from ..data.models import Datasheet, Weapon
from ..data.parse import clean
from .geometry import Point
from .state import Model, Unit

__all__ = ["UnitSpec", "build_unit", "build_army", "formation_offsets", "default_loadout", "TOY_ROSTERS", "TOY_ROSTERS_INFANTRY",
           "build_units_from_list", "unit_code", "footprint_for", "apply_footprint", "HULLS_PATH"]


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    datasheet: str
    n_models: int
    faction: Optional[str] = None
    weapon_overrides: Dict[int, Tuple[str, ...]] = field(default_factory=dict)  #: index de figurine → armes


# ------------------------------------------------------------ composition


_COMPO_RE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s+(.+?)\s*$")
_LOADOUT_RE = re.compile(r"^(?:The\s+|Every\s+|Each\s+|All\s+)?(?P<who>.+?)\s+(?:is|are)\s+equipped\s+with:\s*(?P<items>.+?)\.?$", re.IGNORECASE)


def _composition(ds: Datasheet) -> List[Tuple[int, int, str]]:
    """[(min, max, nom)] d'après les lignes « 1 Intercessor Sergeant », « 4-9 Intercessors »."""
    out = []
    for line in ds.composition:
        m = _COMPO_RE.match(line)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            out.append((lo, hi, clean(m.group(3))))
    return out


def _singular(name: str) -> str:
    n = name.lower()
    return n[:-1] if n.endswith("s") and not n.endswith("ss") else n


def model_names(ds: Datasheet, n_models: int) -> List[str]:
    """Noms des figurines d'une unité de ``n_models`` : les lignes à effectif fixe d'abord (chef),
    puis le reste avec le type variable."""
    compo = _composition(ds)
    if not compo:
        return [ds.name] * n_models
    names: List[str] = []
    for lo, hi, name in compo:
        if lo == hi:
            names.extend([_singular_display(name)] * lo)
    variable = [(lo, hi, name) for lo, hi, name in compo if lo != hi]
    filler = _singular_display(variable[0][2]) if variable else _singular_display(compo[-1][2])
    while len(names) < n_models:
        names.append(filler)
    return names[:n_models]


def _singular_display(name: str) -> str:
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _match_weapons(ds: Datasheet, item: str) -> Tuple[Weapon, ...]:
    """Trouve les profils d'arme correspondant à un libellé d'équipement (« plasma pistol » → 2 profils)."""
    key = clean(item).lower().rstrip(".")
    exact = tuple(w for w in ds.weapons if w.name.lower() == key)
    if exact:
        return exact
    grouped = tuple(w for w in ds.weapons if w.group.lower() == key)
    if grouped:
        return grouped
    # tolérance : singulier/pluriel, article
    key2 = re.sub(r"^(a|an|one|1)\s+", "", key)
    for w in ds.weapons:
        if w.group.lower() in (key2, key2 + "s") or _singular(w.group) == _singular(key2):
            return tuple(x for x in ds.weapons if x.group == w.group)
    raise LookupError(f"{ds.name} : arme {item!r} introuvable (armes : {', '.join(sorted({w.group for w in ds.weapons}))})")


def default_loadout(ds: Datasheet, names: Sequence[str]) -> List[Tuple[Weapon, ...]]:
    """Armes par figurine d'après le texte d'équipement de la fiche."""
    rules: List[Tuple[Optional[str], Tuple[Weapon, ...]]] = []
    for line in ds.loadout.split("\n"):
        m = _LOADOUT_RE.match(clean(line))
        if not m:
            continue
        who = clean(m.group("who")).lower()
        weapons: Tuple[Weapon, ...] = ()
        for item in m.group("items").split(";"):
            item = clean(item)
            if item:
                weapons += _match_weapons(ds, item)
        rules.append((None if who in ("model", "models") else who, weapons))
    if not rules:
        raise LookupError(f"{ds.name} : équipement par défaut illisible : {ds.loadout!r}")
    out = []
    for name in names:
        chosen = None
        for who, weapons in rules:
            if who is not None and (_singular(who) == _singular(name) or _singular(who) in _singular(name)):
                chosen = weapons
                break
        if chosen is None:
            generic = [w for who, w in rules if who is None]
            if generic:
                chosen = generic[0]
            else:
                # « Every Infractor » sert de règle générale pour les figurines non nommées
                chosen = rules[-1][1]
        out.append(chosen)
    return out


# ------------------------------------------------------------ formation


def formation_offsets(n: int, radius: float, gap: float = 0.25, per_row: int = 3) -> List[Point]:
    """Positions relatives (centrées) d'une formation compacte en rangées de ``per_row``."""
    step = 2 * radius + gap
    rows = int(math.ceil(n / per_row))
    offsets = []
    for i in range(n):
        r, c = divmod(i, per_row)
        in_row = min(per_row, n - r * per_row)
        x = (c - (in_row - 1) / 2) * step
        y = (r - (rows - 1) / 2) * step
        offsets.append((x, y))
    return offsets


# ------------------------------------------------------------ construction


def build_unit(cat: Catalog, spec: UnitSpec, side: str, center: Point = (0.0, 0.0)) -> Unit:
    ds = cat.get(spec.datasheet, spec.faction)
    if not ds.models:
        raise ValueError(f"{ds.name} : pas de profil de figurine")
    profile = ds.models[0]
    names = model_names(ds, spec.n_models)
    loadout = default_loadout(ds, names)
    for idx, override in spec.weapon_overrides.items():
        weapons: Tuple[Weapon, ...] = ()
        for item in override:
            weapons += _match_weapons(ds, item)
        loadout[idx] = weapons
    models = []
    for i, (name, weapons) in enumerate(zip(names, loadout)):
        m = Model(id=f"{spec.unit_id}.{i + 1}", unit_id=spec.unit_id, name=name, profile=profile, weapons=weapons, wounds=profile.wounds)
        if profile.base is None:
            apply_footprint(m, footprint_for(ds.name, ds.has_keyword("Vehicle"))[0])
        models.append(m)
    offsets = formation_offsets(spec.n_models, max(m.extent for m in models))
    for m, (dx, dy) in zip(models, offsets):
        m.move_to(center[0] + dx, center[1] + dy)
    return Unit(id=spec.unit_id, side=side, datasheet=ds, models=models)


def build_army(cat: Catalog, specs: Sequence[UnitSpec], side: str) -> List[Unit]:
    return [build_unit(cat, s, side) for s in specs]


#: Rosters du toy model : 2 × 5 Intercessors + 1 Redemptor Dreadnought (Space Marines) contre
#: 2 × 5 Infractors + 1 Daemon Prince of Slaanesh (Emperor's Children). Le véhicule et le monstre
#: servent à tester les règles qui leur sont propres (Hazardous 3 BM, Big Guns Never Tire, Deadly
#: Demise, pas de Hidden, Duty Eternal, Lone Operative conditionnel, aura Excessive Vigour).
TOY_ROSTERS = {
    "attacker": (
        UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),
        UnitSpec("SM2", "Intercessor Squad", 5, faction="SM"),
        UnitSpec("SM3", "Redemptor Dreadnought", 1, faction="SM"),
    ),
    "defender": (
        UnitSpec("EC1", "Infractors", 5, faction="EC"),
        UnitSpec("EC2", "Infractors", 5, faction="EC"),
        UnitSpec("EC3", "Daemon Prince of Slaanesh", 1, faction="EC"),
    ),
}

#: Ancien duo d'infanterie seule (tests et comparaisons).
TOY_ROSTERS_INFANTRY = {
    "attacker": TOY_ROSTERS["attacker"][:2],
    "defender": TOY_ROSTERS["defender"][:2],
}


# ------------------------------------------------------------ listes importées

#: Empreintes des figurines sans socle (« Use model ») : coques de véhicule, motos… (mm, éditable)
HULLS_PATH = Path(__file__).resolve().parents[2] / "data" / "hulls.json"
#: orientation des coques au déploiement : axe long vers l'adversaire (les zones sont en haut et en bas)
HULL_DEFAULT_ANGLE = math.pi / 2


def footprint_for(name: str, is_vehicle: bool, path: Optional[Path] = None) -> Tuple[dict, str]:
    """Empreinte d'une fiche sans socle : (spécification, provenance « exact » / « famille » / « défaut »).
    Le fichier est relu à chaque appel : on peut corriger une coque sans relancer quoi que ce soit."""
    try:
        table = json.loads((path or HULLS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        table = {}
    exact = table.get("exact", {})
    if name in exact:
        return exact[name], "exact"
    for pattern, spec in table.get("families", []):
        if pattern.lower() in name.lower():
            return spec, "famille"
    default = table.get("default_vehicle" if is_vehicle else "default_other") or {"shape": "rect", "length_mm": 120, "width_mm": 90}
    return default, "défaut"


def apply_footprint(model: Model, spec: dict) -> None:
    length = float(spec.get("length_mm", spec.get("width_mm", 32))) / 25.4
    width = float(spec.get("width_mm", spec.get("length_mm", 32))) / 25.4
    model.set_footprint(spec.get("shape", "rect"), length, width)
    if not model.is_round:
        model.angle = HULL_DEFAULT_ANGLE


def unit_code(name: str, index: int) -> str:
    """Identifiant court d'unité pour le plateau : « Flawless Blades » n°2 → « FB2 », « Fulgrim » → « FUL1 »."""
    words = [w for w in re.split(r"[^A-Za-z]+", name) if w]
    if len(words) >= 2:
        code = "".join(w[0] for w in words[:3]).upper()
    else:
        code = (words[0][:3] if words else "U").upper()
    return f"{code}{index}"


def build_units_from_list(army_list, side: str, taken_ids=()) -> List[Unit]:
    """Unités jouables d'une liste importée (:class:`fortyk.data.army_list.ArmyList`).

    Chaque personnage qui mène une unité (« Leading … ») rejoint ses gardes du corps : ses figurines
    sont ajoutées à l'unité (en tête de liste, protégées à l'allocation des blessures), l'unité porte
    ses capacités et mots-clés. L'équipement est celui de la liste, figurine par figurine (une arme en
    deux exemplaires apparaît deux fois). Les figurines sans socle (« Use model » : Rhino, Land Raider,
    motos…) reçoivent l'empreinte de ``data/hulls.json`` (coque rectangulaire orientée vers l'adversaire)."""
    taken = set(taken_ids)
    counters: Dict[str, int] = {}
    by_key = {u.key: u for u in army_list.units}
    out: List[Unit] = []
    for ru in army_list.units:
        if ru.leading:  # le personnage est construit avec l'unité qu'il mène
            continue
        base_code = unit_code(ru.datasheet.name, 1)[:-1]
        counters[base_code] = counters.get(base_code, 0) + 1
        uid = f"{base_code}{counters[base_code]}"
        while uid in taken:
            counters[base_code] += 1
            uid = f"{base_code}{counters[base_code]}"
        taken.add(uid)
        members = [(by_key[k], True) for k in ru.leaders] + [(ru, False)]
        models: List[Model] = []
        for member, is_leader in members:
            for rm in member.models:
                m = Model(
                    id=f"{uid}.{len(models) + 1}",
                    unit_id=uid,
                    name=rm.name,
                    profile=rm.profile,
                    weapons=rm.weapons,
                    wounds=rm.profile.wounds,
                    is_leader=is_leader,
                )
                if rm.profile.base is None:
                    apply_footprint(m, footprint_for(member.datasheet.name, member.datasheet.has_keyword("Vehicle"))[0])
                models.append(m)
        # formation compacte autour de l'origine (le déploiement place ensuite l'unité)
        r = max(m.extent for m in models)
        for m, (dx, dy) in zip(models, formation_offsets(len(models), r)):
            m.move_to(dx, dy)
        leaders = tuple(by_key[k].datasheet for k in ru.leaders)
        enh = tuple(u.enhancement.name for u, _ in members if u.enhancement is not None)
        unit = Unit(id=uid, side=side, datasheet=ru.datasheet, models=models, leaders=leaders, enhancements=enh,
                    points=sum(u.points_computed or 0 for u, _ in members), is_warlord=any(u.warlord for u, _ in members),
                    list_key=ru.key)
        out.append(unit)
    return out

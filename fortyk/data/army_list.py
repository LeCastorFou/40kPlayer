"""Import d'une liste d'armée au format texte de NewRecruit (et des exports « style GW » proches).

Deux étapes :

1. :func:`parse_army_list` lit le texte tel quel (en-tête ``+ CLÉ: valeur``, blocs d'unités,
   équipement par figurine, ``Leading …`` / ``Attached to …``, améliorations) sans rien savoir des
   règles ;
2. :func:`resolve_army_list` rattache chaque élément à l'export Wahapedia : fiches, profils,
   armes, détachements, règle de détachement choisie, stratagèmes disponibles (core V11 + ceux des
   détachements), améliorations, et vérifie la liste (points recalculés avec les paliers
   d'exemplaires et le coût de l'équipement, attachements permis, nombre de figurines).

Rien ici ne dépend du moteur : la construction des unités jouables est dans
:func:`fortyk.engine.army.build_units_from_list`.

Format reconnu (exemple NewRecruit v36) ::

    + FACTION KEYWORD: Chaos - Emperor's Children
    + DETACHMENT: Mercurial Host, Spectacle of Slaughter (Quicksilver Grace)
    + FORCE DISPOSITION: Reconnaissance
    Char1: 1x Fulgrim (340 pts): Warlord, Daemonic blades, Malefic lash, Serpentine tail
    Char3: 1x Lord Exultant (90 pts): Bolt pistol, …
    Leading Infractors[1]
    5x Infractors (85 pts)
    • 1x Obsessionist: Power sword, Bolt pistol
    • 4x Infractor: 4 with Bolt pistol, Duelling sabre
      Attached to Lord Exultant[1]
    3x Flawless Blades (110 pts)
      Enhancement: Beguiling Grotesquerie (+15 pts)
      3 with Blissblade, Bolt pistol
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detachments import Detachment, DetachmentAbility, Enhancement, Stratagem, normalize_name
from .models import Ability, Datasheet, ModelProfile, Weapon
from .parse import clean

__all__ = [
    "GearGroup",
    "ListEntry",
    "ArmyListDoc",
    "ResolvedModel",
    "ResolvedUnit",
    "ArmyList",
    "parse_army_list",
    "resolve_army_list",
    "load_army_list",
]

KNOWN_TAGS = {"warlord"}

_UNIT_RE = re.compile(r"^(?:(?P<label>[A-Za-z]+\d+)\s*:\s*)?(?P<n>\d+)\s*x\s+(?P<name>.+?)\s*\((?P<pts>\d+)\s*pts?\)\s*(?::\s*(?P<gear>.*))?$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[•·\-\*]\s*(?P<n>\d+)\s*x\s+(?P<model>[^:]+?)\s*:\s*(?P<gear>.*)$")
_WITH_RE = re.compile(r"(?:^|,\s*)(\d+)\s+with\s+", re.IGNORECASE)
_ENH_RE = re.compile(r"^Enhancements?\s*:\s*(?P<name>.+?)\s*(?:\(\s*\+?\s*(?P<pts>\d+)\s*pts?\s*\))?$", re.IGNORECASE)
_LEADING_RE = re.compile(r"^Leading\s+(?P<ref>.+)$", re.IGNORECASE)
_ATTACHED_RE = re.compile(r"^Attached\s+to\s+(?P<ref>.+)$", re.IGNORECASE)
_REF_RE = re.compile(r"^(?P<name>.+?)\s*(?:\[(?P<i>\d+)\])?$")
_QTY_RE = re.compile(r"^(?P<q>\d+)\s*x\s+(?P<name>.+)$", re.IGNORECASE)


# ------------------------------------------------------------------ texte brut


@dataclass
class GearGroup:
    """``count`` figurines (``model_name`` si précisé) équipées de ``items`` = [(quantité, nom)]."""

    count: int
    model_name: str = ""
    items: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class ListEntry:
    name: str
    count: int
    points: Optional[int]
    label: str = ""
    tags: List[str] = field(default_factory=list)
    groups: List[GearGroup] = field(default_factory=list)
    enhancement: Optional[str] = None
    enhancement_points: Optional[int] = None
    leading: Optional[str] = None
    attached_to: Optional[str] = None
    key: str = ""  #: « Infractors[2] » : n-ième unité de ce nom dans la liste
    lines: List[str] = field(default_factory=list)


@dataclass
class ArmyListDoc:
    faction: str = ""
    detachments: List[str] = field(default_factory=list)
    detachment_rules: List[str] = field(default_factory=list)  #: « (Quicksilver Grace) » après les détachements (indication seulement)
    force_disposition: str = ""
    total_points: Optional[int] = None
    warlord: str = ""
    header_enhancements: List[str] = field(default_factory=list)
    secondaries: str = ""
    number_of_units: Optional[int] = None
    source: str = ""
    header: Dict[str, str] = field(default_factory=dict)
    entries: List[ListEntry] = field(default_factory=list)


def _split_items(text: str) -> List[Tuple[int, str]]:
    items = []
    for raw in text.split(","):
        item = clean(raw)
        if not item:
            continue
        m = _QTY_RE.match(item)
        items.append((int(m.group("q")), clean(m.group("name"))) if m else (1, item))
    return items


def _parse_gear(text: str, default_count: int, model_name: str = "") -> Tuple[List[GearGroup], List[str]]:
    """« 4 with A, B, 1 with C » ou « A, 2x B » → groupes ; les étiquettes connues (Warlord) à part."""
    text = clean(text)
    tags: List[str] = []
    if not text:
        return [], tags
    parts = _WITH_RE.split(text)
    groups: List[GearGroup] = []
    if len(parts) > 1:
        # parts = [avant, n1, texte1, n2, texte2, ...]
        lead = parts[0]
        if clean(lead):
            groups.append(GearGroup(default_count, model_name, _split_items(lead)))
        for i in range(1, len(parts) - 1, 2):
            groups.append(GearGroup(int(parts[i]), model_name, _split_items(parts[i + 1])))
    else:
        groups.append(GearGroup(default_count, model_name, _split_items(text)))
    for g in groups:
        kept = []
        for q, name in g.items:
            if normalize_name(name) in KNOWN_TAGS:
                tags.append(name)
            else:
                kept.append((q, name))
        g.items = kept
    return [g for g in groups if g.items or g.model_name], tags


def parse_army_list(text: str) -> ArmyListDoc:
    doc = ArmyListDoc()
    last_key: Optional[str] = None
    entry: Optional[ListEntry] = None
    last_character: Optional[ListEntry] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= {"+"}:
            continue
        if line.startswith("+") or line.startswith("&"):
            body = line.lstrip("+&").strip()
            if not body:
                continue
            if line.startswith("&") and last_key:
                doc.header[last_key] += " & " + body
                if last_key == "ENHANCEMENT":
                    doc.header_enhancements.append(body)
                continue
            if ":" in body:
                k, v = body.split(":", 1)
                last_key = clean(k).upper()
                v = clean(v)
                doc.header[last_key] = v
                if last_key in ("FACTION KEYWORD", "FACTION"):
                    doc.faction = v
                elif last_key in ("DETACHMENT", "DETACHMENTS"):
                    m = re.match(r"^(?P<dets>.*?)\s*(?:\((?P<rules>[^)]*)\))?\s*$", v)
                    doc.detachments = [clean(d) for d in m.group("dets").split(",") if clean(d)]
                    if m.group("rules"):
                        doc.detachment_rules = [clean(r) for r in m.group("rules").split(",") if clean(r)]
                elif last_key == "FORCE DISPOSITION":
                    doc.force_disposition = v
                elif last_key == "TOTAL ARMY POINTS":
                    m = re.search(r"\d+", v)
                    doc.total_points = int(m.group()) if m else None
                elif last_key == "WARLORD":
                    doc.warlord = clean(v.split(":", 1)[-1])
                elif last_key == "ENHANCEMENT":
                    doc.header_enhancements.append(v)
                elif last_key == "NUMBER OF UNITS":
                    m = re.search(r"\d+", v)
                    doc.number_of_units = int(m.group()) if m else None
                elif last_key.startswith("SECONDAR"):
                    doc.secondaries = v
            continue
        if line.lower().startswith("created with"):
            doc.source = clean(line[len("created with"):])
            continue
        m = _UNIT_RE.match(line)
        if m:
            entry = ListEntry(name=clean(m.group("name")), count=int(m.group("n")), points=int(m.group("pts")), label=m.group("label") or "")
            entry.lines.append(line)
            if m.group("gear"):
                groups, tags = _parse_gear(m.group("gear"), entry.count)
                entry.groups.extend(groups)
                entry.tags.extend(tags)
            doc.entries.append(entry)
            if entry.label:
                last_character = entry
            continue
        if entry is None:
            continue
        entry.lines.append(line)
        m = _BULLET_RE.match(line)
        if m:
            groups, tags = _parse_gear(m.group("gear"), int(m.group("n")), clean(m.group("model")))
            if len(groups) == 1:
                groups[0].count = int(m.group("n"))
            entry.groups.extend(groups)
            entry.tags.extend(tags)
            continue
        m = _ENH_RE.match(line)
        if m:
            entry.enhancement = clean(m.group("name"))
            entry.enhancement_points = int(m.group("pts")) if m.group("pts") else None
            continue
        m = _LEADING_RE.match(line)
        if m:
            entry.leading = clean(m.group("ref"))
            continue
        m = _ATTACHED_RE.match(line)
        if m:
            entry.attached_to = clean(m.group("ref"))
            continue
        if re.match(r"^\d+\s+with\s+", line, re.IGNORECASE):
            groups, tags = _parse_gear(line, entry.count)
            entry.groups.extend(groups)
            entry.tags.extend(tags)
            continue
    # clés « Nom[i] »
    seen: Dict[str, int] = {}
    for e in doc.entries:
        k = normalize_name(e.name)
        seen[k] = seen.get(k, 0) + 1
        e.key = f"{e.name}[{seen[k]}]"
    return doc


# ------------------------------------------------------------------ résolution


@dataclass
class ResolvedModel:
    name: str
    profile: ModelProfile
    weapons: Tuple[Weapon, ...]
    wargear_abilities: Tuple[str, ...] = ()


@dataclass
class ResolvedUnit:
    key: str
    entry: ListEntry
    datasheet: Datasheet
    models: List[ResolvedModel]
    copy_index: int
    points_listed: Optional[int]
    points_computed: Optional[int]
    enhancement: Optional[Enhancement] = None
    warlord: bool = False
    leading: Optional[str] = None  #: clé de l'unité menée (pour un personnage)
    leaders: List[str] = field(default_factory=list)  #: clés des personnages qui mènent cette unité

    @property
    def name(self) -> str:
        return self.datasheet.name


@dataclass
class ArmyList:
    doc: ArmyListDoc
    faction_id: str
    faction_name: str
    detachments: List[Detachment]
    active_rules: List[DetachmentAbility]
    stratagems: List[Stratagem]
    enhancements_available: List[Enhancement]
    units: List[ResolvedUnit]
    issues: List[str] = field(default_factory=list)

    @property
    def points(self) -> int:
        return sum(u.points_computed or 0 for u in self.units)

    @property
    def dp(self) -> int:
        return sum(d.dp or 0 for d in self.detachments)

    def unit(self, key: str) -> ResolvedUnit:
        for u in self.units:
            if u.key == key:
                return u
        raise KeyError(key)

    @property
    def warlord(self) -> Optional[ResolvedUnit]:
        return next((u for u in self.units if u.warlord), None)

    def to_dict(self) -> dict:
        return {
            "faction": {"id": self.faction_id, "name": self.faction_name},
            "detachments": [{"name": d.name, "dp": d.dp, "force_disposition": d.force_disposition} for d in self.detachments],
            "force_disposition": self.doc.force_disposition,
            "active_rules": [{"name": a.name, "detachment": a.detachment, "text": a.text} for a in self.active_rules],
            "points": {"listed": self.doc.total_points, "computed": self.points},
            "units": [
                {
                    "key": u.key,
                    "datasheet": u.datasheet.name,
                    "datasheet_id": u.datasheet.id,
                    "points": u.points_computed,
                    "warlord": u.warlord,
                    "enhancement": u.enhancement.name if u.enhancement else None,
                    "leading": u.leading,
                    "leaders": u.leaders,
                    "models": [{"name": m.name, "weapons": [w.name for w in m.weapons], "wargear": list(m.wargear_abilities)} for m in u.models],
                }
                for u in self.units
            ],
            "stratagems": [{"name": s.name, "cp": s.cp, "detachment": s.detachment or "core", "turn": s.turn, "phase": s.phase,
                            "when": s.when, "target": s.target, "effect": s.effect, "restrictions": s.restrictions} for s in self.stratagems],
            "issues": list(self.issues),
            "secondaries": self.doc.secondaries,
            "source": self.doc.source,
        }


def _singular(name: str) -> str:
    n = normalize_name(name)
    return n[:-1] if n.endswith("s") and not n.endswith("ss") else n


def _match_weapons(ds: Datasheet, item: str) -> Tuple[Weapon, ...]:
    """Libellé d'équipement → profils de la fiche (« Daemonic blades » → strike + sweep)."""
    key = normalize_name(item)
    exact = tuple(w for w in ds.weapons if normalize_name(w.name) == key)
    if exact:
        return exact
    grouped = tuple(w for w in ds.weapons if normalize_name(w.group) == key)
    if grouped:
        return grouped
    sk = _singular(item)
    return tuple(w for w in ds.weapons if _singular(w.group) == sk or _singular(w.name) == sk)


def _match_profile(ds: Datasheet, model_name: str) -> ModelProfile:
    if len(ds.models) == 1 or not model_name:
        return ds.models[0]
    sk = _singular(model_name)
    for p in ds.models:
        if _singular(p.name) == sk:
            return p
    return ds.models[0]


def _composition_names(ds: Datasheet) -> List[Tuple[int, int, str]]:
    out = []
    for line in ds.composition:
        m = re.match(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s+(.+?)(?:\s+[–-]\s+.*)?\s*$", line)
        if m:
            lo = int(m.group(1))
            out.append((lo, int(m.group(2)) if m.group(2) else lo, clean(m.group(3))))
    return out


def _display_singular(name: str) -> str:
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _resolve_ref(ref: str, keys: Dict[str, str]) -> Optional[str]:
    m = _REF_RE.match(clean(ref))
    if not m:
        return None
    name, idx = m.group("name"), m.group("i") or "1"
    return keys.get(f"{normalize_name(name)}[{idx}]")


def resolve_army_list(doc: ArmyListDoc, cat) -> ArmyList:
    """Rattache la liste à l'export Wahapedia (``cat`` : :class:`~fortyk.data.catalog.Catalog`)."""
    issues: List[str] = []
    rules = cat.rules
    faction = cat.faction_by_name(doc.faction) if doc.faction else None
    fid = faction.id if faction else ""

    detachments: List[Detachment] = []
    for name in doc.detachments:
        try:
            detachments.append(rules.detachment(name, fid or None))
        except LookupError as err:
            issues.append(str(err))
    det_ids = [d.id for d in detachments]
    # V11 (validé avec Valentin) : toutes les règles des détachements pris s'appliquent ; la parenthèse
    # de NewRecruit (« (Quicksilver Grace) ») n'est qu'une indication, on vérifie seulement qu'elle existe.
    active = [a for d in detachments for a in d.abilities]
    for r in doc.detachment_rules:
        if not any(normalize_name(a.name) == normalize_name(r) for a in active):
            issues.append(f"règle de détachement inconnue : {r!r}")
    if doc.force_disposition and detachments and not any(normalize_name(d.force_disposition) == normalize_name(doc.force_disposition) for d in detachments):
        issues.append(f"Force Disposition {doc.force_disposition!r} : aucun détachement de la liste ne la porte")
    stratagems = rules.core_stratagems() + [s for d in detachments for s in d.stratagems]
    enh_available = [e for d in detachments for e in d.enhancements]

    units: List[ResolvedUnit] = []
    copies: Dict[str, int] = {}
    for e in doc.entries:
        try:
            ds = cat.get(e.name, fid or None)
        except LookupError:
            try:
                ds = cat.get(e.name)
                issues.append(f"{e.key} : fiche trouvée hors de la faction ({ds.faction_id})")
            except LookupError as err:
                issues.append(f"{e.key} : {err}")
                continue
        copies[ds.id] = copies.get(ds.id, 0) + 1
        compo = _composition_names(ds)
        models: List[ResolvedModel] = []
        remaining = e.count
        for g in e.groups:
            weapons: List[Weapon] = []
            wargear: List[str] = []
            for q, item in g.items:
                ws = _match_weapons(ds, item)
                if ws:
                    weapons.extend(list(ws) * q)
                    continue
                ab = ds.ability(item)
                if ab is not None:
                    wargear.append(ab.name)
                    continue
                issues.append(f"{e.key} : équipement inconnu {item!r} (armes de la fiche : {', '.join(sorted({w.group for w in ds.weapons}))})")
            model_name = g.model_name
            if not model_name:
                # figurine « anonyme » : le type variable de la composition (ou le nom de la fiche)
                variable = [c for c in compo if c[0] != c[1]] or compo
                model_name = _display_singular(variable[-1][2]) if variable else ds.name
            profile = _match_profile(ds, model_name)
            for _ in range(g.count):
                models.append(ResolvedModel(_display_singular(model_name), profile, tuple(weapons), tuple(wargear)))
            remaining -= g.count
        if remaining > 0:
            if models:
                issues.append(f"{e.key} : {remaining} figurine(s) sans équipement précisé — copiées sur la dernière")
                models.extend([models[-1]] * remaining)
            else:
                issues.append(f"{e.key} : aucun équipement précisé")
        if len(models) != e.count:
            issues.append(f"{e.key} : {len(models)} figurines lues pour {e.count} annoncées")
        if compo:
            lo = sum(c[0] for c in compo)
            hi = sum(c[1] for c in compo)
            if not lo <= e.count <= hi:
                issues.append(f"{e.key} : {e.count} figurines, la fiche en autorise {lo} à {hi}")
        # points : barème de l'exemplaire, équipement payant, amélioration
        pts = ds.cost_for(e.count, copies[ds.id])
        if pts is None:
            issues.append(f"{e.key} : pas de coût pour {e.count} figurines (exemplaire n°{copies[ds.id]})")
            pts = 0
        for c in ds.wargear_costs:
            m = re.match(r"^per\s+(.+)$", c.description or "", re.IGNORECASE)
            if m:
                target = normalize_name(m.group(1))
                n = sum(1 for rm in models for w in rm.weapons if normalize_name(w.group) == target and w.profile_index <= 1)
                pts += n * c.points
        enh = None
        if e.enhancement:
            try:
                enh = rules.enhancement(e.enhancement, fid or None, det_ids)
                pts += enh.cost or 0
                if enh.detachment_id not in det_ids:
                    issues.append(f"{e.key} : amélioration {enh.name!r} d'un détachement absent de la liste ({enh.detachment})")
                if enh.eligible_datasheet_ids and ds.id not in enh.eligible_datasheet_ids:
                    issues.append(f"{e.key} : {enh.name!r} ne peut pas aller sur {ds.name}")
            except LookupError as err:
                issues.append(f"{e.key} : {err}")
        if e.points is not None and pts != e.points:
            issues.append(f"{e.key} : {e.points} pts dans la liste, {pts} pts recalculés")
        warlord = any(normalize_name(t) == "warlord" for t in e.tags) or (bool(doc.warlord) and normalize_name(doc.warlord) == normalize_name(ds.name) and not any(u.warlord for u in units))
        units.append(ResolvedUnit(e.key, e, ds, models, copies[ds.id], e.points, pts, enh, warlord))

    # attachements
    keys = {normalize_name(u.key.rsplit("[", 1)[0]) + "[" + u.key.rsplit("[", 1)[1]: u.key for u in units}
    by_key = {u.key: u for u in units}
    for u in units:
        if u.entry.leading:
            target = _resolve_ref(u.entry.leading, keys)
            if target is None:
                issues.append(f"{u.key} : unité menée introuvable {u.entry.leading!r}")
                continue
            u.leading = target
            if u.key not in by_key[target].leaders:
                by_key[target].leaders.append(u.key)
        if u.entry.attached_to:
            leader = _resolve_ref(u.entry.attached_to, keys)
            if leader is None:
                issues.append(f"{u.key} : personnage introuvable {u.entry.attached_to!r}")
                continue
            if by_key[leader].leading not in (None, u.key):
                issues.append(f"{u.key} : {leader} mène déjà {by_key[leader].leading}")
                continue
            by_key[leader].leading = u.key
            if leader not in u.leaders:
                u.leaders.append(leader)
    for u in units:
        if u.leading:
            bodyguard = by_key[u.leading]
            if bodyguard.datasheet.id not in u.datasheet.can_lead_ids:
                issues.append(f"{u.key} ne peut pas mener {bodyguard.datasheet.name}")
    if doc.total_points is not None and sum(u.points_computed or 0 for u in units) != doc.total_points:
        issues.append(f"total : {doc.total_points} pts annoncés, {sum(u.points_computed or 0 for u in units)} pts recalculés")
    if doc.number_of_units is not None and doc.number_of_units != len(units):
        issues.append(f"{doc.number_of_units} unités annoncées, {len(units)} lues")
    if not any(u.warlord for u in units):
        issues.append("aucun Warlord désigné")

    return ArmyList(doc, fid, faction.name if faction else doc.faction, detachments, active, stratagems, enh_available, units, issues)


def load_army_list(path, cat) -> ArmyList:
    from pathlib import Path

    return resolve_army_list(parse_army_list(Path(path).read_text(encoding="utf-8")), cat)

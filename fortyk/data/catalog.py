"""Assemblage des tables Wahapedia en objets :class:`~fortyk.data.models.Datasheet`."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .models import Ability, Datasheet, Faction, KeywordEntry, ModelProfile, UnitCost, Weapon
from .parse import (
    ParseError,
    clean,
    parse_base_size,
    parse_dice,
    parse_inches,
    parse_int,
    parse_save,
    parse_skill,
    parse_weapon_keywords,
    strip_html,
)
from .wahapedia import RawTables

__all__ = ["Catalog", "load_catalog", "CatalogError"]


class CatalogError(LookupError):
    pass


_COST_MODELS_RE = re.compile(r"^\s*(\d+)\s+models?\b", re.IGNORECASE)
_ORD = r"(\d+)(?:ST|ND|RD|TH)"
_COST_RANGE_RE = re.compile(rf"^YOUR\s+{_ORD}\s+TO\s+{_ORD}\s+UNITS?\s+COSTS?$")  # YOUR 1ST TO 3RD UNITS COST
_COST_FROM_RE = re.compile(rf"^YOUR\s+{_ORD}\s*\+\s*UNITS?\s+COSTS?$")  # YOUR 4TH + UNIT COSTS
_COST_ONLY_RE = re.compile(rf"^YOUR\s+{_ORD}\s+UNITS?\s+COSTS?$")  # YOUR 1ST UNIT COSTS
_COST_PLAIN_RE = re.compile(r"^YOUR\s+UNITS?\s+COSTS?$")  # YOUR UNIT COSTS
_PROFILE_SEP = re.compile(r"\s+[–-]\s+")  # « Plasma pistol – supercharge »


def _bool(text: str) -> bool:
    return clean(text).lower() == "true"


def _int_or(text: str, default: int) -> int:
    value = parse_int(text)
    return default if value is None else value


class Catalog:
    """Toutes les fiches d'unité de l'export, indexées par identifiant et par nom.

    >>> cat = Catalog()                       # lit data/wahapedia/raw
    >>> ds = cat.get("Intercessor Squad")     # ou cat.get("intercessor squad", faction="SM")
    >>> ds.models[0].base.radius_in, ds.weapon("Bolt rifle").attacks.mean
    """

    def __init__(self, raw: Optional[RawTables] = None):
        self.raw = raw if raw is not None else RawTables()
        self.last_update = self.raw.last_update()
        self.factions: Dict[str, Faction] = {
            r["id"]: Faction(r["id"], clean(r["name"]), r.get("link", "")) for r in self.raw.read("Factions")
        }
        self.datasheets: Dict[str, Datasheet] = {}
        self._by_name: Dict[str, List[str]] = {}
        self.warnings: List[str] = []
        self._build()

    # ------------------------------------------------------------------ accès

    def get(self, name: str, faction: Optional[str] = None) -> Datasheet:
        """Fiche par nom exact (insensible à la casse). ``faction`` : id (« SM ») ou nom."""
        ids = self._by_name.get(clean(name).lower(), [])
        if faction:
            f = clean(faction).lower()
            ids = [i for i in ids if self.datasheets[i].faction_id.lower() == f or self.datasheets[i].faction_name.lower() == f]
        if not ids:
            raise CatalogError(f"aucune fiche nommée {name!r}" + (f" pour la faction {faction!r}" if faction else ""))
        if len(ids) > 1:
            found = ", ".join(f"{self.datasheets[i].name} [{self.datasheets[i].faction_id}]" for i in ids)
            raise CatalogError(f"nom ambigu {name!r} : {found} — précise la faction")
        return self.datasheets[ids[0]]

    def find(self, pattern: str, faction: Optional[str] = None) -> List[Datasheet]:
        """Fiches dont le nom contient ``pattern`` (regex, insensible à la casse)."""
        rx = re.compile(pattern, re.IGNORECASE)
        out = [d for d in self.datasheets.values() if rx.search(d.name)]
        if faction:
            f = clean(faction).lower()
            out = [d for d in out if d.faction_id.lower() == f or d.faction_name.lower() == f]
        return sorted(out, key=lambda d: (d.faction_id, d.name))

    def by_faction(self, faction: str) -> List[Datasheet]:
        f = clean(faction).lower()
        return sorted(
            (d for d in self.datasheets.values() if d.faction_id.lower() == f or d.faction_name.lower() == f),
            key=lambda d: d.name,
        )

    @property
    def rules(self):
        """Détachements, stratagèmes et améliorations (:class:`~fortyk.data.detachments.ArmyRules`),
        chargés à la première demande."""
        if getattr(self, "_rules", None) is None:
            from .detachments import ArmyRules

            self._rules = ArmyRules(self.raw)
        return self._rules

    def by_id(self, ds_id: str) -> Datasheet:
        return self.datasheets[ds_id]

    def faction_by_name(self, name: str) -> Faction:
        """Faction par id (« EC ») ou par nom, y compris « Chaos - Emperor's Children » (apostrophes
        typographiques ignorées) ; la dernière partie après « - » suffit."""
        from .detachments import normalize_name

        key = normalize_name(name)
        candidates = [key] + [normalize_name(part) for part in re.split(r"\s+-\s+", name)[1:]]
        for c in candidates:
            for f in self.factions.values():
                if normalize_name(f.id) == c or normalize_name(f.name) == c:
                    return f
        raise CatalogError(f"faction inconnue : {name!r}")

    def __len__(self) -> int:
        return len(self.datasheets)

    def __iter__(self):
        return iter(self.datasheets.values())

    def __repr__(self) -> str:
        return f"Catalog({len(self)} datasheets, {len(self.factions)} factions, export {self.last_update})"

    # ------------------------------------------------------------ construction

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def _build(self) -> None:
        raw = self.raw
        models = raw.by_key("Datasheets_models", "datasheet_id")
        wargear = raw.by_key("Datasheets_wargear", "datasheet_id")
        abilities = raw.by_key("Datasheets_abilities", "datasheet_id")
        keywords = raw.by_key("Datasheets_keywords", "datasheet_id")
        composition = raw.by_key("Datasheets_unit_composition", "datasheet_id")
        options = raw.by_key("Datasheets_options", "datasheet_id")
        costs = raw.by_key("Datasheets_models_cost", "datasheet_id")
        ability_index = raw.index("Abilities")
        leaders_of: Dict[str, List[str]] = {}
        can_lead: Dict[str, List[str]] = {}
        for r in raw.read("Datasheets_leader"):
            leaders_of.setdefault(r["attached_id"], []).append(r["leader_id"])
            can_lead.setdefault(r["leader_id"], []).append(r["attached_id"])

        for r in raw.read("Datasheets"):
            ds_id = r["id"]
            name = clean(r["name"])
            faction = self.factions.get(r["faction_id"])
            ds = Datasheet(
                id=ds_id,
                name=name,
                faction_id=r["faction_id"],
                faction_name=faction.name if faction else r["faction_id"],
                models=self._models(ds_id, name, models.get(ds_id, [])),
                weapons=self._weapons(ds_id, name, wargear.get(ds_id, [])),
                abilities=self._abilities(abilities.get(ds_id, []), ability_index),
                keyword_entries=tuple(
                    KeywordEntry(clean(k["keyword"]), _bool(k["is_faction_keyword"]), clean(k["model"]))
                    for k in keywords.get(ds_id, [])
                    if clean(k["keyword"])  # l'export contient quelques mots-clés vides (ex. Emperor's Children)
                ),
                composition=tuple(strip_html(c["description"]) for c in composition.get(ds_id, [])),
                options=tuple(strip_html(o["description"]) for o in options.get(ds_id, [])),
                costs=self._costs(costs.get(ds_id, [])),
                loadout=strip_html(r["loadout"]),
                role=clean(r["role"]),
                transport=strip_html(r["transport"]),
                legend=strip_html(r["legend"]),
                virtual=_bool(r["virtual"]),
                is_support=_bool(r["is_support"]),
                damaged_wounds=clean(r["damaged_w"]),
                damaged_text=strip_html(r["damaged_description"]),
                leader_ids=tuple(leaders_of.get(ds_id, [])),
                can_lead_ids=tuple(can_lead.get(ds_id, [])),
                link=r["link"],
                source_id=r["source_id"],
            )
            self.datasheets[ds_id] = ds
            self._by_name.setdefault(name.lower(), []).append(ds_id)

    def _models(self, ds_id: str, ds_name: str, rows: Iterable[dict]) -> tuple:
        out = []
        for r in rows:
            try:
                save = parse_save(r["Sv"])
                leadership = parse_save(r["Ld"])
                out.append(
                    ModelProfile(
                        name=clean(r["name"]),
                        move_in=parse_inches(r["M"]),
                        toughness=_int_or(r["T"], 0),
                        save=7 if save is None else save,  # 7+ = aucune sauvegarde
                        invuln=parse_save(r["inv_sv"]),
                        wounds=_int_or(r["W"], 0),
                        leadership=7 if leadership is None else leadership,
                        oc=_int_or(r["OC"], 0),
                        base=parse_base_size(r["base_size"]),
                        invuln_note=strip_html(r["inv_sv_descr"]),
                        base_note=strip_html(r["base_size_descr"]),
                        line=_int_or(r["line"], 0),
                    )
                )
            except ParseError as err:
                self._warn(f"{ds_name} [{ds_id}] profil {r.get('name')!r} ignoré : {err}")
        return tuple(out)

    def _weapons(self, ds_id: str, ds_name: str, rows: Iterable[dict]) -> tuple:
        out = []
        for r in sorted(rows, key=lambda x: (_int_or(x["line"], 0), _int_or(x["line_in_wargear"], 0))):
            kind = clean(r["type"]).lower()
            if kind not in ("ranged", "melee"):
                continue  # ligne d'équipement sans profil d'arme
            try:
                attacks = parse_dice(r["A"])
                damage = parse_dice(r["D"])
                if attacks is None or damage is None:
                    raise ParseError("A ou D manquant")
                name = clean(r["name"])
                out.append(
                    Weapon(
                        name=name,
                        kind=kind,
                        range_in=None if kind == "melee" else parse_inches(r["range"]),
                        attacks=attacks,
                        skill=parse_skill(r["BS_WS"]),
                        strength=_int_or(r["S"], 0),
                        ap=_int_or(r["AP"], 0),
                        damage=damage,
                        keywords=parse_weapon_keywords(r["description"]),
                        group=_PROFILE_SEP.split(name, maxsplit=1)[0],
                        line=_int_or(r["line"], 0),
                        profile_index=_int_or(r["line_in_wargear"], 1),
                    )
                )
            except ParseError as err:
                self._warn(f"{ds_name} [{ds_id}] arme {r.get('name')!r} ignorée : {err}")
        return tuple(out)

    @staticmethod
    def _abilities(rows: Iterable[dict], ability_index: Dict[str, dict]) -> tuple:
        out = []
        for r in sorted(rows, key=lambda x: _int_or(x["line"], 0)):
            ref = ability_index.get(r["ability_id"]) if r["ability_id"] else None
            name = clean(r["name"]) or (clean(ref["name"]) if ref else "")
            html = r["description"] or (ref["description"] if ref else "")
            out.append(
                Ability(
                    name=name,
                    type=clean(r["type"]),
                    text=strip_html(html),
                    parameter=clean(r["parameter"]),
                    model=clean(r["model"]),
                    html=html,
                    ability_id=r["ability_id"],
                )
            )
        return tuple(out)

    @staticmethod
    def _cost_header(header: str):
        """Interprète un en-tête de bloc de coûts → (kind, from_copy, to_copy, context)."""
        h = clean(header).upper()
        if "WARGEAR" in h:
            return "wargear", 1, None, ""
        m = _COST_RANGE_RE.match(h)
        if m:
            return "unit", int(m.group(1)), int(m.group(2)), ""
        m = _COST_FROM_RE.match(h)
        if m:
            return "unit", int(m.group(1)), None, ""
        m = _COST_ONLY_RE.match(h)
        if m:
            return "unit", int(m.group(1)), int(m.group(1)), ""
        if _COST_PLAIN_RE.match(h):
            return "unit", 1, None, ""
        # En-tête de contexte (Agents of the Imperium : « AGENTS OF THE IMPERIUM Detachment »,
        # « Assigned Agent ») : il précède un bloc « YOUR UNIT COSTS » ordinaire et s'y applique.
        return "unit", 1, None, clean(header)

    @classmethod
    def _costs(cls, rows: Iterable[dict]) -> tuple:
        """Les lignes de coût sont organisées en blocs introduits par un en-tête sans coût
        (« YOUR UNIT COSTS », « YOUR 1ST TO 2ND UNITS COST », « YOUR 3RD + UNIT COSTS »,
        « WARGEAR OPTIONS »…). L'export répète parfois les blocs : on déduplique."""
        seen = set()
        out = []
        kind, from_copy, to_copy, context = "unit", 1, None, ""
        for r in sorted(rows, key=lambda x: _int_or(x["line"], 0)):
            desc = strip_html(r["description"])
            cost = parse_int(r["cost"]) if clean(r["cost"]) else None
            if cost is None:
                kind, from_copy, to_copy, new_context = cls._cost_header(desc)
                if new_context:
                    context = new_context  # un en-tête de contexte reste actif pour les blocs suivants
                continue
            key = (kind, from_copy, to_copy, context, desc, cost)
            if key in seen:
                continue
            seen.add(key)
            m = _COST_MODELS_RE.match(desc)
            out.append(
                UnitCost(
                    models=int(m.group(1)) if m else None,
                    points=cost,
                    description=desc,
                    from_copy=from_copy if kind == "unit" else 1,
                    to_copy=to_copy if kind == "unit" else None,
                    kind=kind,
                    context=context if kind == "unit" else "",
                )
            )
        return tuple(out)


def load_catalog(raw_dir=None) -> Catalog:
    """Raccourci : ``Catalog(RawTables(raw_dir))``."""
    return Catalog(RawTables(raw_dir))

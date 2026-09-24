"""Structures de données immuables décrivant une fiche d'unité (datasheet) V11.

Ces objets sont *purement descriptifs* : ils ne portent aucun état de partie.
Le moteur de jeu construira ses propres objets (unités déployées, figurines
avec position et blessures restantes) à partir de ces descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .parse import BaseSize, DiceExpr, WeaponKeyword

__all__ = [
    "Faction",
    "ModelProfile",
    "Weapon",
    "Ability",
    "KeywordEntry",
    "UnitCost",
    "Datasheet",
]


@dataclass(frozen=True)
class Faction:
    id: str
    name: str
    link: str = ""


@dataclass(frozen=True)
class ModelProfile:
    """Ligne de profil (M, T, Sv, W, Ld, OC) d'un type de figurine de la fiche."""

    name: str
    move_in: Optional[float]  #: None pour les fortifications (M = "-")
    toughness: int
    save: int
    invuln: Optional[int]
    wounds: int
    leadership: int
    oc: int
    base: Optional[BaseSize]
    invuln_note: str = ""
    base_note: str = ""
    line: int = 0

    @property
    def base_radius_in(self) -> Optional[float]:
        return self.base.radius_in if self.base else None


@dataclass(frozen=True)
class Weapon:
    """Un profil d'arme. Une arme à plusieurs profils (plasma standard / surcharge)
    donne plusieurs :class:`Weapon` partageant le même ``group``."""

    name: str
    kind: str  #: "ranged" ou "melee"
    range_in: Optional[float]  #: None en mêlée
    attacks: DiceExpr
    skill: Optional[int]  #: BS ou WS ; None pour Torrent (touche automatiquement)
    strength: int
    ap: int  #: 0, -1, -2… (valeur signée telle que sur la fiche)
    damage: DiceExpr
    keywords: tuple = ()
    group: str = ""  #: nom de l'arme sans le suffixe de profil (« Plasma pistol »)
    line: int = 0
    profile_index: int = 1

    @property
    def is_melee(self) -> bool:
        return self.kind == "melee"

    def has(self, keyword: str) -> bool:
        return any(k.name == keyword for k in self.keywords)

    def keyword(self, name: str) -> Optional[WeaponKeyword]:
        for k in self.keywords:
            if k.name == name:
                return k
        return None

    @property
    def unknown_keywords(self) -> tuple:
        return tuple(k for k in self.keywords if not k.known)

    def __str__(self) -> str:
        rng = "Melee" if self.is_melee else f'{self.range_in:g}"'
        skill_name = "WS" if self.is_melee else "BS"
        skill = "N/A" if self.skill is None else f"{self.skill}+"
        kws = ", ".join(str(k) for k in self.keywords)
        core = (
            f"{self.name}: {rng} | A {self.attacks} {skill_name} {skill} "
            f"S {self.strength} AP {self.ap} D {self.damage}"
        )
        return f"{core} [{kws}]" if kws else core


@dataclass(frozen=True)
class Ability:
    """Capacité listée sur la fiche. ``type`` : Core, Faction, Datasheet, Wargear…"""

    name: str
    type: str
    text: str  #: description en texte brut (HTML retiré)
    parameter: str = ""  #: ex. "D3" pour Deadly Demise D3, '6"' pour Scouts 6"
    model: str = ""  #: figurine concernée si la capacité ne s'applique qu'à une seule
    html: str = ""
    ability_id: str = ""

    @property
    def is_core(self) -> bool:
        return self.type == "Core"


@dataclass(frozen=True)
class KeywordEntry:
    name: str
    is_faction: bool = False
    model: str = ""  #: vide = toute l'unité


@dataclass(frozen=True)
class UnitCost:
    """Une ligne de coût en points.

    En V11 le prix d'une fiche peut dépendre du nombre d'exemplaires déjà pris
    (« YOUR 1ST UNIT COSTS » / « YOUR 2ND + UNIT COSTS », « YOUR 1ST TO 3RD UNITS COST »
    / « YOUR 4TH + UNIT COSTS »…) : ``from_copy`` et ``to_copy`` bornent les exemplaires
    concernés (``to_copy`` None = sans limite ; le barème unique est (1, None)).
    ``kind`` vaut ``"unit"`` ou ``"wargear"`` (bloc « WARGEAR OPTIONS », coût par arme,
    ``models`` est alors None). ``context`` porte un éventuel en-tête particulier
    (Agents of the Imperium : « Agents of the Imperium Detachment » / « Assigned Agent »).
    """

    models: Optional[int]  #: nombre de figurines, None si non déductible du libellé
    points: int
    description: str = ""
    from_copy: int = 1
    to_copy: Optional[int] = None
    kind: str = "unit"
    context: str = ""

    def applies_to_copy(self, copy_index: int) -> bool:
        return self.from_copy <= copy_index and (self.to_copy is None or copy_index <= self.to_copy)


@dataclass(frozen=True)
class Datasheet:
    id: str
    name: str
    faction_id: str
    faction_name: str
    models: tuple = ()
    weapons: tuple = ()
    abilities: tuple = ()
    keyword_entries: tuple = ()
    composition: tuple = ()
    options: tuple = ()
    costs: tuple = ()
    loadout: str = ""
    role: str = ""
    transport: str = ""
    legend: str = ""
    virtual: bool = False
    is_support: bool = False
    damaged_wounds: str = ""
    damaged_text: str = ""
    leader_ids: tuple = ()  #: fiches pouvant mener cette unité
    can_lead_ids: tuple = ()  #: fiches que celle-ci peut mener
    link: str = ""
    source_id: str = ""

    # ----------------------------------------------------------- raccourcis

    @property
    def keywords(self) -> tuple:
        return tuple(k.name for k in self.keyword_entries if not k.is_faction and not k.model)

    @property
    def faction_keywords(self) -> tuple:
        return tuple(k.name for k in self.keyword_entries if k.is_faction)

    def has_keyword(self, name: str) -> bool:
        target = name.lower()
        return any(k.name.lower() == target for k in self.keyword_entries)

    @property
    def ranged_weapons(self) -> tuple:
        return tuple(w for w in self.weapons if not w.is_melee)

    @property
    def melee_weapons(self) -> tuple:
        return tuple(w for w in self.weapons if w.is_melee)

    def weapon(self, name: str) -> Optional[Weapon]:
        target = name.lower()
        for w in self.weapons:
            if w.name.lower() == target:
                return w
        return None

    def ability(self, name: str) -> Optional[Ability]:
        target = name.lower()
        for a in self.abilities:
            if a.name.lower() == target:
                return a
        return None

    @property
    def core_abilities(self) -> tuple:
        return tuple(a for a in self.abilities if a.is_core)

    def unit_costs(self, copy_index: int = 1, context: Optional[str] = None) -> tuple:
        """Barème applicable au ``copy_index``-ième exemplaire de la fiche (1 = premier).
        ``context`` filtre un en-tête particulier (voir :class:`UnitCost`) ; None = tous."""
        return tuple(
            c
            for c in self.costs
            if c.kind == "unit" and c.applies_to_copy(copy_index) and (context is None or c.context == context)
        )

    def cost_for(self, n_models: int, copy_index: int = 1, context: Optional[str] = None) -> Optional[int]:
        for c in self.unit_costs(copy_index, context):
            if c.models == n_models:
                return c.points
        return None

    @property
    def min_cost(self) -> Optional[int]:
        pts = [c.points for c in self.unit_costs(1)]
        return min(pts) if pts else None

    @property
    def wargear_costs(self) -> tuple:
        return tuple(c for c in self.costs if c.kind == "wargear")

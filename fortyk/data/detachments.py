"""Règles d'armée V11 de l'export Wahapedia : détachements, règles de détachement, stratagèmes,
améliorations (enhancements).

En V11 une armée combine un ou plusieurs détachements (chacun coûte des « DP » et porte une
Force Disposition) ; chaque détachement apporte sa règle, ses stratagèmes et ses améliorations.
Les stratagèmes « core » V11 sont ceux de type exactement ``Core Stratagem`` : l'export contient
aussi des reliquats V10 (« Core – Strategic Ploy Stratagem »…) et les modes Boarding Actions /
Challenger, qu'on écarte.

Le texte des stratagèmes est découpé en WHEN / TARGET / EFFECT / RESTRICTIONS : c'est ce qui
servira à brancher chaque stratagème sur une fenêtre de décision du moteur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .parse import clean, parse_int, strip_html
from .wahapedia import RawTables

__all__ = [
    "Stratagem",
    "DetachmentAbility",
    "Enhancement",
    "Detachment",
    "ArmyRules",
    "normalize_name",
    "V11_CORE_STRATAGEM_TYPE",
]

V11_CORE_STRATAGEM_TYPE = "Core Stratagem"

_SECTION_RE = re.compile(r"\b(WHEN|TARGET|EFFECT|RESTRICTIONS?|COST):\s*", re.IGNORECASE)


def normalize_name(text: str) -> str:
    """Clé de comparaison des noms : casse, apostrophes typographiques, espaces."""
    return clean(text).replace("’", "'").replace("‘", "'").lower()


def _sections(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    parts = _SECTION_RE.split(text)
    # parts = [avant, NOM, texte, NOM, texte, ...]
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].upper().rstrip("S") if parts[i].upper().startswith("RESTRICTION") else parts[i].upper()
        out[key] = clean(parts[i + 1])
    return out


@dataclass(frozen=True)
class Stratagem:
    id: str
    name: str
    cp: Optional[int]
    type: str
    turn: str  #: « Your turn », « Opponent’s turn », « Either player’s turn »
    phase: str
    faction_id: str = ""
    detachment: str = ""
    detachment_id: str = ""
    when: str = ""
    target: str = ""
    effect: str = ""
    restrictions: str = ""
    text: str = ""

    @property
    def is_core(self) -> bool:
        return not self.faction_id and self.type == V11_CORE_STRATAGEM_TYPE

    @property
    def is_reactive(self) -> bool:
        """Jouable pendant le tour adverse (réaction) — ces stratagèmes demandent des fenêtres de
        décision pour l'adversaire dans la machine à états."""
        t = normalize_name(self.turn)
        return t.startswith("opponent") or t.startswith("either")


@dataclass(frozen=True)
class DetachmentAbility:
    id: str
    name: str
    detachment: str
    detachment_id: str
    text: str
    faction_id: str = ""


@dataclass(frozen=True)
class Enhancement:
    id: str
    name: str
    cost: Optional[int]
    detachment: str
    detachment_id: str
    text: str
    faction_id: str = ""
    upgrade: bool = False  #: V11 : amélioration d'unité (« FLAWLESS BLADES unit only »), pas d'un personnage
    eligible_datasheet_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Detachment:
    id: str
    name: str
    faction_id: str
    dp: Optional[int]
    force_disposition: str
    type: str = ""  #: vide = partie standard ; « Boarding Actions »…
    abilities: Tuple[DetachmentAbility, ...] = ()
    stratagems: Tuple[Stratagem, ...] = ()
    enhancements: Tuple[Enhancement, ...] = ()


class ArmyRules:
    """Index des détachements, stratagèmes et améliorations de l'export."""

    def __init__(self, raw: RawTables):
        self.raw = raw
        self.stratagems: List[Stratagem] = []
        for r in raw.read("Stratagems"):
            text = strip_html(r.get("description", "")).replace("\n", " ")
            sec = _sections(text)
            self.stratagems.append(Stratagem(
                id=r.get("id", ""), name=clean(r.get("name", "")), cp=parse_int(r.get("cp_cost", "")), type=clean(r.get("type", "")),
                turn=clean(r.get("turn", "")), phase=clean(r.get("phase", "")), faction_id=clean(r.get("faction_id", "")),
                detachment=clean(r.get("detachment", "")), detachment_id=clean(r.get("detachment_id", "")),
                when=sec.get("WHEN", ""), target=sec.get("TARGET", ""), effect=sec.get("EFFECT", ""),
                restrictions=sec.get("RESTRICTION", ""), text=text,
            ))
        eligible: Dict[str, List[str]] = {}
        for r in raw.read("Datasheets_enhancements"):
            eligible.setdefault(r["enhancement_id"], []).append(r["datasheet_id"])
        self.enhancements: List[Enhancement] = [
            Enhancement(
                id=r.get("id", ""), name=clean(r.get("name", "")), cost=parse_int(r.get("cost", "")), detachment=clean(r.get("detachment", "")),
                detachment_id=clean(r.get("detachment_id", "")), text=strip_html(r.get("description", "")).replace("\n", " "),
                faction_id=clean(r.get("faction_id", "")), upgrade=clean(r.get("upgrade", "")).lower() == "true",
                eligible_datasheet_ids=tuple(eligible.get(r.get("id", ""), ())),
            )
            for r in raw.read("Enhancements")
        ]
        abilities = [
            DetachmentAbility(
                id=r.get("id", ""), name=clean(r.get("name", "")), detachment=clean(r.get("detachment", "")), detachment_id=clean(r.get("detachment_id", "")),
                text=strip_html(r.get("description", "")).replace("\n", " "), faction_id=clean(r.get("faction_id", "")),
            )
            for r in raw.read("Detachment_abilities")
        ]
        self.detachments: Dict[str, Detachment] = {}
        for r in raw.read("Detachments"):
            did = r.get("id", "")
            self.detachments[did] = Detachment(
                id=did, name=clean(r.get("name", "")), faction_id=clean(r.get("faction_id", "")), dp=parse_int(r.get("dp", "")),
                force_disposition=clean(r.get("force_disposition", "")), type=clean(r.get("type", "")),
                abilities=tuple(a for a in abilities if a.detachment_id == did),
                stratagems=tuple(s for s in self.stratagems if s.detachment_id == did),
                enhancements=tuple(e for e in self.enhancements if e.detachment_id == did),
            )

    # ----------------------------------------------------------- accès

    def core_stratagems(self) -> List[Stratagem]:
        """Stratagèmes core V11 (un exemplaire par nom)."""
        seen, out = set(), []
        for s in self.stratagems:
            if s.is_core and s.name not in seen:
                seen.add(s.name)
                out.append(s)
        return out

    def detachment(self, name: str, faction_id: Optional[str] = None, standard_only: bool = True) -> Detachment:
        key = normalize_name(name)
        found = [d for d in self.detachments.values()
                 if normalize_name(d.name) == key and (faction_id is None or d.faction_id == faction_id) and (not standard_only or not d.type)]
        if not found:
            raise LookupError(f"détachement inconnu : {name!r}" + (f" ({faction_id})" if faction_id else ""))
        return found[0]

    def detachments_of(self, faction_id: str, standard_only: bool = True) -> List[Detachment]:
        return sorted((d for d in self.detachments.values() if d.faction_id == faction_id and (not standard_only or not d.type)), key=lambda d: d.name)

    def enhancement(self, name: str, faction_id: Optional[str] = None, detachment_ids: Optional[List[str]] = None) -> Enhancement:
        key = normalize_name(name)
        found = [e for e in self.enhancements if normalize_name(e.name) == key and (faction_id is None or e.faction_id == faction_id)]
        if detachment_ids:
            preferred = [e for e in found if e.detachment_id in detachment_ids]
            found = preferred or found
        if not found:
            raise LookupError(f"amélioration inconnue : {name!r}")
        return found[0]

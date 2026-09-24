"""Couverture des règles : ce qu'une liste d'armée utilise, et ce que le moteur en joue vraiment.

Le registre ci-dessous est la liste, tenue à la main, des règles que le moteur applique. Un test
vérifie que chaque nom enregistré apparaît bien dans le code du moteur (pas de registre qui ment).
Tout le reste est « pas encore joué » : l'IA n'en tiendra pas compte, ce qui est acceptable pour une
règle absente, mais à garder en tête avant un long entraînement (voir le README).

:func:`coverage` classe chaque élément d'une liste importée en ``joué`` / ``partiel`` / ``non joué``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from .combat import _DAMAGE_REDUCTION_RE

__all__ = [
    "IMPLEMENTED_ABILITIES",
    "PARTIAL_ABILITIES",
    "IMPLEMENTED_WEAPON_KEYWORDS",
    "IMPLEMENTED_STRATAGEMS",
    "IMPLEMENTED_DETACHMENT_RULES",
    "IMPLEMENTED_ENHANCEMENTS",
    "CoverageItem",
    "coverage",
    "format_coverage",
]

#: capacités appliquées par le moteur (nom exact de l'export) → ce qui est joué
IMPLEMENTED_ABILITIES: Dict[str, str] = {
    "Deadly Demise": "D6 à la destruction ; sur 6, X blessures mortelles aux unités à 6\"",
    "Feel No Pain": "jet de FNP par point de dégât",
    "Lone Operative": "ciblable au tir seulement à 12\" ou moins",
    "Oath of Moment": "relance des touches contre la cible désignée en phase de commandement",
    "Thrill Seekers": "tir et charge après Advance / Fall Back, avec les restrictions de cible",
    "Objective Secured": "objectif collant",
    "Hail of Bolts": "+2 A aux bolt rifles",
    "Excessive Assault": "relance des blessures de 1 (toutes à portée d'objectif)",
    "Lord of Excess": "Lone Operative à 3\" d'une infanterie Slaanesh",
    "Excessive Vigour (Aura)": "+1 PA en mêlée pour une unité Slaanesh qui a chargé, à 6\"",
    "Leader": "le personnage rejoint l'unité qu'il mène",
    "Scouts": "mouvement de X\" avant le round 1, fin à plus de 9\" de l'ennemi ; le transport dédié en profite si tous ses passagers l'ont",
}

#: capacités jouées avec une approximation connue
PARTIAL_ABILITIES: Dict[str, str] = {
    "Leader": "sauvegarde et invulnérable des gardes du corps pour toute l'unité ; capacités « while leading » non jouées",
}

#: mots-clés d'arme appliqués par la séquence d'attaque (voir combat.py)
IMPLEMENTED_WEAPON_KEYWORDS = frozenset({
    "assault", "heavy", "pistol", "close-quarters", "blast", "ignores cover", "hazardous", "torrent", "lethal hits", "sustained hits",
    "devastating wounds", "anti", "twin-linked", "rapid fire", "lance", "extra attacks",
})

IMPLEMENTED_STRATAGEMS: frozenset = frozenset()  #: aucun pour l'instant (CP et fenêtres de réaction à venir)
IMPLEMENTED_DETACHMENT_RULES: frozenset = frozenset()
IMPLEMENTED_ENHANCEMENTS: frozenset = frozenset()

#: règles structurelles tirées des mots-clés / champs de la fiche
_STRUCTURAL = [
    ("Transport", lambda ds: bool(ds.transport), "embarquer (déploiement, fin de mouvement à 3\"), débarquer (règles V10 validées), capacité lue dans la fiche ; "
                                                 "transport détruit : défaut V10 à confirmer ; Firing Deck non joué"),
    ("Empreinte sans socle (coque)", lambda ds: any(m.base is None for m in ds.models), "coque rectangulaire de data/hulls.json (dimensions approximatives, éditables), pivot libre mesuré au coin le plus loin"),
    ("Damaged (profil dégradé)", lambda ds: bool(ds.damaged_wounds), "malus des figurines endommagées non joué"),
    ("Deep Strike / réserves", lambda ds: ds.ability("Deep Strike") is not None, "l'unité se déploie sur la table"),
    ("Fly", lambda ds: ds.has_keyword("Fly"), "sans effet : tous les décors sont franchissables et il n'y a pas d'étages"),
]


_STRUCTURAL_STATUS = {"Fly": "partiel", "Transport": "partiel", "Empreinte sans socle (coque)": "partiel"}


@dataclass
class CoverageItem:
    category: str  #: capacité, mot-clé d'arme, stratagème, règle de détachement, amélioration, structure
    name: str
    status: str  #: « joué », « partiel », « non joué »
    units: List[str] = field(default_factory=list)
    note: str = ""


def _ability_status(ability) -> tuple:
    if ability.name in PARTIAL_ABILITIES:
        return "partiel", PARTIAL_ABILITIES[ability.name]
    if ability.name in IMPLEMENTED_ABILITIES:
        return "joué", IMPLEMENTED_ABILITIES[ability.name]
    if _DAMAGE_REDUCTION_RE.search(ability.text or ""):
        return "joué", "-1 aux dégâts alloués (minimum 1), lu dans le texte"
    return "non joué", ""


def coverage(army_list) -> List[CoverageItem]:
    """Inventaire des règles d'une liste importée et de leur prise en charge par le moteur."""
    items: Dict[tuple, CoverageItem] = {}

    def add(category: str, name: str, status: str, unit: str = "", note: str = "") -> None:
        key = (category, name)
        it = items.get(key)
        if it is None:
            it = items[key] = CoverageItem(category, name, status, [], note)
        if unit and unit not in it.units:
            it.units.append(unit)

    for u in army_list.units:
        ds = u.datasheet
        for a in ds.abilities:
            status, note = _ability_status(a)
            add(f"capacité ({a.type or '?'})", a.name, status, u.key, note)
        for m in u.models:
            for w in m.weapons:
                for k in w.keywords:
                    ok = k.name in IMPLEMENTED_WEAPON_KEYWORDS
                    add("mot-clé d'arme", k.name, "joué" if ok else "non joué", u.key)
            for wa in m.wargear_abilities:
                add("équipement", wa, "non joué", u.key)
        for label, test, note in _STRUCTURAL:
            if test(ds):
                add("structure", label, _STRUCTURAL_STATUS.get(label, "non joué"), u.key, note)
        if u.enhancement is not None:
            add("amélioration", u.enhancement.name, "joué" if u.enhancement.name in IMPLEMENTED_ENHANCEMENTS else "non joué", u.key, u.enhancement.text[:140])
    for a in army_list.active_rules:
        add("règle de détachement", a.name, "joué" if a.name in IMPLEMENTED_DETACHMENT_RULES else "non joué", "", a.text[:140])
    for s in army_list.stratagems:
        note = f"{s.cp} CP, {s.turn}, {s.phase}" + (" — réactif" if s.is_reactive else "")
        add(f"stratagème ({s.detachment or 'core'})", s.name, "joué" if s.name in IMPLEMENTED_STRATAGEMS else "non joué", "", note)
    order = {"non joué": 0, "partiel": 1, "joué": 2}
    return sorted(items.values(), key=lambda it: (it.category.startswith("stratagème"), it.category, order[it.status], it.name))


def format_coverage(items: List[CoverageItem]) -> str:
    lines = []
    total = len(items)
    done = sum(1 for it in items if it.status == "joué")
    part = sum(1 for it in items if it.status == "partiel")
    lines.append(f"Couverture du moteur : {done} règles jouées, {part} partielles, {total - done - part} non jouées (sur {total})")
    current = None
    for it in items:
        if it.category != current:
            current = it.category
            lines.append(f"\n  {current}")
        mark = {"joué": "✔", "partiel": "≈", "non joué": "✘"}[it.status]
        where = f" — {', '.join(it.units)}" if it.units else ""
        note = f"  ({it.note})" if it.note else ""
        lines.append(f"    {mark} {it.name}{where}{note}")
    return "\n".join(lines)

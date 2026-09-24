"""Assemblage du toy model : catalogue Wahapedia + Layout A + rosters + mission."""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

from .data import Catalog, load_catalog
from .engine.army import TOY_ROSTERS, UnitSpec, build_army
from .engine.layout import load_layout
from .engine.mission import UnstoppableForce
from .engine.rules import DEFAULT_RULES, RulesConfig
from .engine.state import GameState

__all__ = ["new_toy_state", "new_state_from_lists"]


def new_toy_state(
    cat: Optional[Catalog] = None,
    layout: str = "layout_a",
    seed: int = 0,
    rosters: Optional[Dict[str, Sequence[UnitSpec]]] = None,
    rules: RulesConfig = DEFAULT_RULES,
) -> GameState:
    cat = cat or load_catalog()
    rosters = rosters or TOY_ROSTERS
    units = {}
    for side, specs in rosters.items():
        for u in build_army(cat, specs, side):
            units[u.id] = u
    return GameState(
        layout=load_layout(layout),
        units=units,
        rules=rules,
        mission=UnstoppableForce(),
        rng=random.Random(seed),
    )


def new_state_from_lists(
    attacker,
    defender,
    cat: Optional[Catalog] = None,
    layout: str = "layout_a",
    seed: int = 0,
    rules: RulesConfig = DEFAULT_RULES,
) -> GameState:
    """Partie entre deux listes d'armée importées : chemins de fichiers texte, noms de listes
    enregistrées dans ``data/lists`` ou :class:`~fortyk.data.army_list.ArmyList`. ``None`` pour un
    camp = roster du toy model pour ce camp."""
    from pathlib import Path

    from .data.army_list import ArmyList, load_army_list
    from .data.list_library import load_named_list
    from .engine.army import build_units_from_list

    cat = cat or load_catalog()

    def resolve(a):
        if a is None or isinstance(a, ArmyList):
            return a
        p = Path(a)
        return load_army_list(p, cat) if p.exists() else load_named_list(str(a), cat)

    lists = [resolve(a) for a in (attacker, defender)]
    units = {}
    for side, al in zip(("attacker", "defender"), lists):
        if al is None:
            for u in build_army(cat, TOY_ROSTERS[side], side):
                units[u.id] = u
            continue
        for u in build_units_from_list(al, side, taken_ids=units.keys()):
            units[u.id] = u
    state = GameState(layout=load_layout(layout), units=units, rules=rules, mission=UnstoppableForce(), rng=random.Random(seed))
    state.army_lists = {side: al for side, al in zip(("attacker", "defender"), lists) if al is not None}
    return state

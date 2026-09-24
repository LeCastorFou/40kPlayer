"""Couche données : lecture de l'export Wahapedia et fiches d'unité typées.

Point d'entrée habituel ::

    from fortyk.data import load_catalog
    cat = load_catalog()
    intercessors = cat.get("Intercessor Squad")
"""

from .catalog import Catalog, CatalogError, load_catalog
from .models import Ability, Datasheet, Faction, KeywordEntry, ModelProfile, UnitCost, Weapon
from .parse import BaseSize, DiceExpr, ParseError, WeaponKeyword
from .wahapedia import RawTables, default_raw_dir

__all__ = [
    "Catalog",
    "CatalogError",
    "load_catalog",
    "Ability",
    "Datasheet",
    "Faction",
    "KeywordEntry",
    "ModelProfile",
    "UnitCost",
    "Weapon",
    "BaseSize",
    "DiceExpr",
    "ParseError",
    "WeaponKeyword",
    "RawTables",
    "default_raw_dir",
]

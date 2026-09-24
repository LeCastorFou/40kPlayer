"""Moteur de jeu : géométrie 2D, règles V11, séquence d'attaque, carte, mission, état de partie,
mouvement, combat, machine à états du jeu (Engine) et pilote de partie (Game)."""

from .attack import (
    AttackOutcome,
    AttackProfile,
    Defender,
    ExpectedOutcome,
    estimate_models_slain,
    expected_attacks,
    expected_hazardous_mortal_wounds,
    hazardous_test,
    resolve_attacks,
)
from .army import TOY_ROSTERS, UnitSpec, build_army, build_unit
from .engine import Engine, IllegalAction
from .game import Game, GameResult
from .geometry import Disk, Polygon, Terrain
from .layout import Layout, ObjectivePoint, TerrainPiece, load_layout
from .mission import Scoreboard, ScoringEvent, UnstoppableForce, controller, level_of_control
from .rules import DEFAULT_RULES, RulesConfig, save_needed, wound_roll_needed
from .state import GameState, Model, Unit

__all__ = [
    "AttackOutcome",
    "AttackProfile",
    "Defender",
    "ExpectedOutcome",
    "estimate_models_slain",
    "expected_attacks",
    "expected_hazardous_mortal_wounds",
    "hazardous_test",
    "resolve_attacks",
    "TOY_ROSTERS",
    "UnitSpec",
    "build_army",
    "build_unit",
    "Engine",
    "IllegalAction",
    "Game",
    "GameResult",
    "GameState",
    "Model",
    "Unit",
    "Disk",
    "Polygon",
    "Terrain",
    "Layout",
    "ObjectivePoint",
    "TerrainPiece",
    "load_layout",
    "Scoreboard",
    "ScoringEvent",
    "UnstoppableForce",
    "controller",
    "level_of_control",
    "DEFAULT_RULES",
    "RulesConfig",
    "save_needed",
    "wound_roll_needed",
]

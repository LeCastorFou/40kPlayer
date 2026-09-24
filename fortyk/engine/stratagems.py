"""Stratagèmes de base V11 (15) : coûts, restrictions (15.01) et effets en cours.

Le moteur ouvre une *fenêtre* (décision ``stratagem``) au moment précis où un stratagème peut être
joué, et **seulement s'il est utilisable** : assez de CP, pas déjà utilisé dans la phase, une cible
éligible (et utile : relancer un 6 d'Advance ne sert à rien). Sinon la partie continue sans rien
demander — c'est le « passe automatique ». Les stratagèmes de son propre tour qui ne réagissent à
rien (Explosives) sont proposés parmi les options de la décision en cours.

Restrictions (15.01) : un joueur n'utilise pas deux fois le même stratagème dans la même phase, ne
cible pas la même unité avec deux stratagèmes dans la même phase, et ne cible jamais une unité
battle-shocked (01.07). Insane Bravery : une fois par bataille.

Pas encore couverts : Rapid Ingress (pas de réserves stratégiques dans le moteur) et Command Re-roll
sur les jets de touche, blessure, sauvegarde, dégâts, danger et nombre d'attaques (il faudrait
interrompre la résolution des attaques) ; il relance ici les jets d'Advance et de charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .state import GameState, Unit

__all__ = ["Stratagem", "CORE_STRATAGEMS", "WINDOWS", "unavailable", "record_use", "effect_of", "used_this_battle", "cost_of"]


@dataclass(frozen=True)
class Stratagem:
    key: str
    name: str
    cost: int
    ref: str
    summary: str  #: effet, en une ligne (affiché au joueur)


CORE_STRATAGEMS: Dict[str, Stratagem] = {s.key: s for s in (
    Stratagem("command_reroll", "Command Re-roll", 1, "15.02", "relance le jet (la charge : les deux dés)"),
    Stratagem("epic_challenge", "Epic Challenge", 1, "15.03", "les armes de mêlée d'une figurine PERSONNAGE gagnent [PRECISION] jusqu'à la fin de la phase"),
    Stratagem("insane_bravery", "Insane Bravery", 1, "15.04", "le test de battle-shock est réussi d'office (une fois par bataille)"),
    Stratagem("explosives", "Explosives", 1, "15.05", "une unité ennemie désengagée à 8\" et visible : 6D6, chaque 4+ = 1 blessure mortelle"),
    Stratagem("crushing_impact", "Crushing Impact", 1, "15.06", "D6 = E de la figurine : chaque 1 = 1 BM pour ton unité, chaque 5+ = 1 BM pour l'ennemi (6 au plus chacun)"),
    Stratagem("fire_overwatch", "Fire Overwatch", 1, "15.08", "tir d'opportunité : une unité ennemie visible à 24\", touche seulement sur un 6 non modifié, sans relance"),
    Stratagem("smokescreen", "Smokescreen", 1, "15.10", "jusqu'à la fin de la phase, ton unité SMOKE a le couvert contre les attaques qui la visent"),
    Stratagem("heroic_intervention", "Heroic Intervention", 1, "15.11", "ton unité charge à son tour : Leap to Defend (cibles qui ont chargé) ou Into the Fray (+1 CP : jet plafonné à 6, ennemis à 6\")"),
    Stratagem("counteroffensive", "Counteroffensive", 2, "15.12", "ton unité gagne Fights First et combat tout de suite"),
)}

#: fenêtres de réaction : clé → (stratagème, moment, en clair)
WINDOWS: Dict[str, tuple] = {
    "reroll_advance": ("command_reroll", "juste après ton jet d'Advance"),
    "reroll_charge": ("command_reroll", "juste après ton jet de charge"),
    "insane_bravery": ("insane_bravery", "juste avant un test de battle-shock"),
    "epic_challenge": ("epic_challenge", "ton unité PERSONNAGE vient d'être choisie pour combattre"),
    "crushing_impact": ("crushing_impact", "ton MONSTER / VEHICLE vient de finir sa charge"),
    "fire_overwatch": ("fire_overwatch", "fin de la phase de mouvement adverse"),
    "smokescreen": ("smokescreen", "début de la phase de tir adverse"),
    "heroic_intervention": ("heroic_intervention", "fin de la phase de charge adverse"),
    "counteroffensive": ("counteroffensive", "une unité ennemie vient de combattre (étape Fights First)"),
}


def cost_of(key: str, mode: Optional[str] = None) -> int:
    """Coût en CP (Into the Fray : +1 CP, 15.01 étape 2)."""
    base = CORE_STRATAGEMS[key].cost
    return base + (1 if key == "heroic_intervention" and mode == "fray" else 0)


def _this_phase(state: GameState, entry: tuple) -> bool:
    return entry[3] == state.turn_counter and entry[4] == state.phase


def used_this_battle(state: GameState, side: str, key: str) -> bool:
    return any(e[0] == side and e[1] == key for e in state.strat_used)


def unavailable(state: GameState, side: str, key: str, unit: Optional[Unit] = None, mode: Optional[str] = None) -> Optional[str]:
    """Pourquoi ``side`` ne peut pas utiliser ce stratagème (sur ``unit``) maintenant ; None = possible."""
    if not state.stratagems:
        return "stratagèmes désactivés pour cette partie"
    cost = cost_of(key, mode)
    if state.cp.get(side, 0) < cost:
        return f"{cost} CP requis ({state.cp.get(side, 0)} disponible(s))"
    if any(e[0] == side and e[1] == key and _this_phase(state, e) for e in state.strat_used):
        return "déjà utilisé dans cette phase"
    if key == "insane_bravery" and used_this_battle(state, side, key):
        return "une fois par bataille"
    if unit is not None:
        if unit.battle_shocked:
            return "unité battle-shocked : aucun stratagème ne peut la cibler"
        if any(e[0] == side and e[2] == unit.id and _this_phase(state, e) for e in state.strat_used):
            return "unité déjà ciblée par un stratagème dans cette phase"
    return None


def record_use(state: GameState, side: str, key: str, unit_id: Optional[str], detail=None, mode: Optional[str] = None) -> int:
    """Dépense les CP et note l'utilisation (restrictions, effets jusqu'à la fin de la phase)."""
    cost = cost_of(key, mode)
    state.cp[side] = state.cp.get(side, 0) - cost
    state.strat_used.append((side, key, unit_id, state.turn_counter, state.phase, detail))
    return cost


def effect_of(state: GameState, key: str, unit_id: str):
    """Détail d'un stratagème ``key`` qui cible ``unit_id`` et dure jusqu'à la fin de la phase en
    cours (Epic Challenge : la figurine ; Smokescreen : True) ; None s'il n'est pas actif."""
    for e in reversed(state.strat_used):
        if e[1] == key and e[2] == unit_id and _this_phase(state, e):
            return e[5] if e[5] is not None else True
    return None

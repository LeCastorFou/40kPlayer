"""Joueur humain en terminal.

À chaque décision : rendu du plateau en PNG (à garder ouvert dans Aperçu, il se met à jour),
état de l'unité concernée, liste numérotée des options légales, et pour le déploiement et le
mouvement une saisie libre validée par le moteur :

* déploiement : ``x y`` (centre de la formation, en pouces) ;
* mouvement : ``m ANGLE DIST`` (cap en degrés : 0 = vers la droite, 90 = vers le bas, 180 = vers la
  gauche, 270 = vers le haut ; distance en pouces), ``v DX DY`` (vecteur), ``a ANGLE`` (Advance),
  ``r ANGLE DIST`` (Fall Back), ``s`` (immobile) ;
* ``i`` affiche toutes les unités, ``?`` l'aide, ``q`` abandonne.
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Optional

from ..engine.actions import Action, ChargeAction, Decision, DeployAction, FightAction, MoveAction, OathAction, SelectUnitAction, ShootAction
from ..engine.geometry import dist
from ..engine.movement import FormationMove, MoveKind
from ..engine.state import GameState, Unit

SIDE_COLORS = {"attacker": (200, 60, 60, 255), "defender": (60, 110, 200, 255)}
SIDE_FR = {"attacker": "attaquant", "defender": "défenseur"}


class Quit(Exception):
    pass


class HumanAgent:
    interactive = True  #: le moteur redemande au lieu de lever une erreur sur une action illégale

    def __init__(self, name: str = "Humain", board_png: Optional[str] = "board.png", input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print):
        self.name = name
        self.board_png = board_png
        self.input_fn = input_fn
        self.out = output_fn

    # ------------------------------------------------------------ affichage

    def render(self, state: GameState, highlight: Optional[str] = None) -> None:
        if not self.board_png:
            return
        try:
            models, labels = [], []
            for u in state.units.values():
                if u.is_destroyed or not self._is_deployed(state, u):
                    continue
                for m in u.alive_models:
                    col = SIDE_COLORS[u.side]
                    if highlight and u.id == highlight:
                        col = (250, 200, 40, 255)
                    models.append((m.x, m.y, m.radius, col))
                c = u.centroid
                labels.append((c[0] + 1.6, c[1] - 1.6, u.id))
            state.layout.render(self.board_png, scale=12, models=models, labels=labels)
        except Exception as err:  # Pillow absent, etc. : on continue sans image
            self.out(f"(pas de rendu du plateau : {err})")
            self.board_png = None

    @staticmethod
    def _is_deployed(state: GameState, u: Unit) -> bool:
        """Avant déploiement, les figurines sont autour de l'origine, hors table."""
        return state.phase != "deployment" or all(state.on_board(m.disk) for m in u.alive_models)

    def describe_unit(self, state: GameState, u: Unit) -> str:
        c = u.centroid
        flags = []
        if u.battle_shocked:
            flags.append("battle-shock")
        if u.advanced:
            flags.append("a avancé")
        if u.fell_back:
            flags.append("s'est repliée")
        if state.is_engaged(u):
            flags.append("ENGAGÉE")
        weapons = sorted({w.name for m in u.alive_models for w in m.weapons})
        if not self._is_deployed(state, u):
            zone = state.layout.deployment_zones[u.side].bbox
            return (f"{u.name} ({SIDE_FR[u.side]}) — {u.strength} fig. à déployer ; zone x {zone[0]:g}–{zone[2]:g}, y {zone[1]:g}–{zone[3]:g} "
                    f"(profonde de 20\" côté {'gauche' if u.side == 'attacker' else 'droit'}, 12\" de l'autre)\n      armes : {', '.join(weapons)}")
        return (f"{u.name} ({SIDE_FR[u.side]}) — {u.strength}/{u.starting_strength} fig., centre ({c[0]:.1f}, {c[1]:.1f}), "
                f"M {u.move_in:g}\"  {' · '.join(flags)}\n      armes : {', '.join(weapons)}")

    def situation(self, state: GameState, side: str) -> str:
        lines = [f"Round {state.battle_round} — {state.phase} — {SIDE_FR[side]} joue. Score : attaquant {state.scoreboard.total('attacker')} / défenseur {state.scoreboard.total('defender')}"]
        for u in state.units.values():
            if u.is_destroyed:
                lines.append(f"   {u.name} : détruite")
                continue
            if not self._is_deployed(state, u):
                lines.append(f"   {u.name} ({SIDE_FR[u.side]}) {u.strength} fig. : pas encore déployée")
                continue
            c = u.centroid
            nearest = None
            for e in state.enemies_of(u.side):
                g = u.min_gap_to(e)
                if nearest is None or g < nearest[1]:
                    nearest = (e, g)
            near = f", ennemi le plus proche {nearest[0].id} à {nearest[1]:.1f}\"" if nearest else ""
            lines.append(f"   {u.name} ({SIDE_FR[u.side]}) {u.strength}/{u.starting_strength} en ({c[0]:.1f}, {c[1]:.1f}){near}")
        ctrl = {o.id: state.objective_controller(o) or "-" for o in state.layout.objectives}
        lines.append("   objectifs : " + ", ".join(f"{k} → {SIDE_FR.get(v, v)}" for k, v in ctrl.items()))
        return "\n".join(lines)

    # ------------------------------------------------------------ décision

    def choose(self, state: GameState, decision: Decision) -> Action:
        unit = state.unit(decision.unit_id) if decision.unit_id else None
        self.render(state, highlight=decision.unit_id)
        self.out("")
        self.out("=" * 78)
        self.out(self.situation(state, decision.side))
        if unit is not None:
            self.out("   → " + self.describe_unit(state, unit))
        self.out(self._prompt_title(decision, state))
        shown = decision.options[:12] if decision.kind == "deploy" else decision.options
        for i, opt in enumerate(shown, 1):
            self.out(f"  {i:2d}. {self._label(opt, state)}")
        if len(shown) < len(decision.options):
            self.out(f"  … ({len(decision.options) - len(shown)} autres positions ; ou tape « x y »)")
        while True:
            try:
                raw = self.input_fn("> ").strip()
            except EOFError:
                raise Quit()
            if not raw:
                continue
            low = raw.lower()
            if low == "q":
                raise Quit()
            if low == "?":
                self.out(__doc__)
                continue
            if low == "i":
                self.out(self.situation(state, decision.side))
                continue
            if low.isdigit():
                k = int(low)
                if 1 <= k <= len(decision.options):
                    return decision.options[k - 1]
                self.out("numéro hors liste")
                continue
            free = self._parse_free(decision, unit, raw)
            if free is not None:
                return free
            self.out("saisie non comprise (? pour l'aide)")

    def observe(self, state: GameState, decision: Decision) -> None:
        """Action adverse (mode pas à pas) : on l'affiche et on attend Entrée."""
        self.render(state)
        self.out(f"[adversaire] {decision.note}")
        try:
            self.input_fn("(Entrée pour continuer) ")
        except EOFError:
            raise Quit()

    def _prompt_title(self, decision: Decision, state: GameState) -> str:
        return {
            "select_unit": "Quelle unité activer ? (numéro ; la dernière option termine la phase)",
            "deploy": "Déploiement : choisis un numéro ou tape « x y » (centre de la formation).",
            "oath": "Oath of Moment : quelle unité ennemie ?",
            "move": "Mouvement : un numéro, ou « m ANGLE DIST », « v DX DY », « a ANGLE » (Advance), « r ANGLE DIST » (repli), « s » (immobile).",
            "advance_move": "Advance déclarée (voir le jet) : un numéro, ou « m ANGLE DIST » jusqu'à la distance annoncée, « s » (immobile).",
            "shoot": "Tir : quelle cible ?",
            "charge": "Charge : quelle cible ? (2D6 à lancer ensuite)",
            "fight": "Combat : quelle unité active-t-on, et contre qui ?",
        }.get(decision.kind, decision.kind)

    def _label(self, opt: Action, state: GameState) -> str:
        if isinstance(opt, MoveAction):
            mv = opt.move
            if mv.kind == MoveKind.STATIONARY:
                return "reste immobile"
            return f"{mv.label} — {mv.kind}, ({mv.dx:+.1f}, {mv.dy:+.1f})"
        if isinstance(opt, (ShootAction, ChargeAction)):
            if opt.target_id is None:
                return "ne rien faire"
            t = state.unit(opt.target_id)
            u = state.unit(opt.unit_id)
            return f"{t.name} ({t.strength} fig., à {u.min_gap_to(t):.1f}\")"
        if isinstance(opt, FightAction):
            return f"{state.unit(opt.unit_id).name} frappe {state.unit(opt.target_id).name}"
        if isinstance(opt, OathAction):
            return state.unit(opt.target_id).name if opt.target_id else "aucun"
        if isinstance(opt, SelectUnitAction):
            u = state.unit(opt.unit_id)
            return f"{u.name} [{u.id}] — {u.strength}/{u.starting_strength} fig."
        return str(opt)

    def _parse_free(self, decision: Decision, unit: Optional[Unit], raw: str) -> Optional[Action]:
        parts = raw.replace(",", " ").split()
        try:
            if decision.kind == "deploy" and len(parts) == 2:
                return DeployAction(decision.unit_id, float(parts[0]), float(parts[1]))
            if decision.kind == "move" and unit is not None:
                cmd = parts[0].lower()
                if cmd == "s":
                    return MoveAction(unit.id, FormationMove(MoveKind.STATIONARY, label="reste immobile"))
                if cmd == "m" and len(parts) == 3:
                    ang, d = math.radians(float(parts[1])), float(parts[2])
                    return MoveAction(unit.id, FormationMove(MoveKind.NORMAL, d * math.cos(ang), d * math.sin(ang), f"cap {parts[1]}° ({d:g}\")"))
                if cmd == "v" and len(parts) == 3:
                    dx, dy = float(parts[1]), float(parts[2])
                    return MoveAction(unit.id, FormationMove(MoveKind.NORMAL, dx, dy, f"vecteur ({dx:+g}, {dy:+g})"))
                if cmd == "a" and len(parts) == 2:
                    ang = math.radians(float(parts[1]))
                    d = unit.move_in + 6
                    return MoveAction(unit.id, FormationMove(MoveKind.ADVANCE, d * math.cos(ang), d * math.sin(ang), f"advance cap {parts[1]}°"))
                if cmd == "r" and len(parts) == 3:
                    ang, d = math.radians(float(parts[1])), float(parts[2])
                    return MoveAction(unit.id, FormationMove(MoveKind.FALL_BACK, d * math.cos(ang), d * math.sin(ang), f"repli cap {parts[1]}° ({d:g}\")"))
        except ValueError:
            return None
        return None

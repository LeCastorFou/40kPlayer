"""Parties sauvegardées à deux joueurs : stockage JSON, salles de jeu, annulation.

Une partie est un document JSON (format ``fortyk-game/2``, un fichier par partie dans le dossier de
données) qui suffit à la reconstruire à l'identique :

* ``config`` : carte, graine des dés, listes d'armée **en texte** (la partie ne dépend pas d'un fichier
  de liste qui changerait ensuite) ;
* ``players`` : pour chaque camp, humain (nom, jeton secret du lien) ou bot ;
* ``history`` : chaque décision prise, dans l'ordre (camp, action JSON, genre de décision, round,
  phase, libellé lisible, horodatage, humain ou bot) ;
* ``reseeds`` : points de l'historique où les dés ont été re-tirés (après une annulation, les dés
  suivants sont nouveaux : on ne « relance » pas en annulant, mais on ne rejoue pas non plus les mêmes) ;
* ``undos`` : chaque annulation (qui, quand, retour à quel point, actions annulées) — ces branches
  abandonnées restent dans le journal, utiles pour l'entraînement.

La :class:`Room` garde la partie en mémoire (état courant + un instantané avant chaque décision, pour
annuler instantanément), fait jouer les bots, et réécrit le fichier après chaque changement. Au
chargement, la partie est rejouée depuis l'état initial en réappliquant les re-tirages.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agents import RandomAgent
from ..data import Catalog
from ..engine import Engine
from ..engine.actions import (
    DeployAction,
    AutoChargeMoveAction,
    ChargeAction,
    ContinueAction,
    DeclareAdvanceAction,
    DeclareChargeAction,
    Decision,
    DeployModelsAction,
    DisembarkModelsAction,
    EndPhaseAction,
    FightAction,
    ModelMoveAction,
    MoveAction,
    OathAction,
    SelectUnitAction,
    ManualAction,
    ShootAction,
    StratagemAction,
    UseStratagemAction,
)
from ..engine.movement import FormationMove, MoveKind
from ..engine.record import action_from_json, action_to_json
from ..engine.state import SIDES, GameState, other_side
from ..training import build_initial_state, iter_replay
from .serialize import action_label, decision_to_json, factions_of, layout_to_json, rules_menu, state_to_json, stratagem_book_json

__all__ = ["FORMAT", "GameStore", "Room", "Rooms", "RoomError", "RoomWaiting", "RoomDeleted", "decode_action", "build_initial_state", "public_record",
           "waiting_snapshot", "normalize_code", "format_code", "new_record"]

FORMAT = "fortyk-game/2"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,40}$")
SIDE_FR = {"attacker": "attaquant", "defender": "défenseur"}
PHASE_FR = {"deployment": "déploiement", "scouts": "scouts", "command": "commandement", "movement": "mouvement", "shooting": "tir",
            "charge": "charge", "fight": "combat", "end_turn": "fin de tour", "end_battle": "fin de bataille"}


class RoomError(Exception):
    """Refus d'une requête (message affichable au joueur)."""


class RoomDeleted(RoomError):
    """La partie a été supprimée (le fichier est rangé dans ``deleted/``, récupérable)."""

    def __init__(self, info: Dict[str, Any]):
        who = info.get("name") or SIDE_FR.get(info.get("by"), "un joueur")
        side = f" ({SIDE_FR[info['by']]})" if info.get("by") in SIDE_FR else ""
        super().__init__(f"Cette partie a été supprimée par {who}{side}.")
        self.info = info


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ actions JSON de l'interface


def _decode_positions(raw: Dict[str, Any]) -> tuple:
    """{model_id: [x, y] ou [x, y, angle]} → ((model_id, x, y[, angle]), …)."""
    return tuple((mid, *(float(v) for v in xy[:3])) for mid, xy in raw.items())


def decode_action(decision: Decision, payload: Dict[str, Any]):
    """Traduit une action envoyée par le navigateur (voir le protocole dans :mod:`fortyk.web.server`)."""
    kind = payload.get("type")
    uid = payload.get("unit_id") or decision.unit_id
    if kind == "option":
        return decision.options[int(payload["index"])]
    if kind == "continue":
        return ContinueAction()
    if kind == "select_unit":
        for opt in decision.options:
            if isinstance(opt, SelectUnitAction) and opt.unit_id == payload["unit_id"]:
                return opt
        raise ValueError(f"{payload['unit_id']} n'est pas activable maintenant")
    if kind == "end_phase":
        for opt in decision.options:
            if isinstance(opt, EndPhaseAction):
                return opt
        raise ValueError("impossible de terminer la phase ici")
    if kind == "deploy_models":
        return DeployModelsAction(uid, _decode_positions(payload["positions"]))
    if kind == "disembark_models":
        return DisembarkModelsAction(uid, _decode_positions(payload["positions"]))
    if kind == "model_move":
        return ModelMoveAction(uid, payload.get("kind", MoveKind.NORMAL), _decode_positions(payload["positions"]))
    if kind == "declare_advance":
        return DeclareAdvanceAction(uid)
    if kind == "declare_charge":
        return DeclareChargeAction(uid)
    if kind == "auto_charge":
        for opt in decision.options:
            if isinstance(opt, AutoChargeMoveAction):
                return opt
        raise ValueError("pas de placement automatique ici")
    if kind == "stationary":
        return MoveAction(uid, FormationMove(MoveKind.STATIONARY, label="reste immobile"))
    if kind == "skip":
        for opt in decision.options:
            if isinstance(opt, (ShootAction, ChargeAction)) and opt.target_id is None:
                return opt
        raise ValueError("pas d'option « ne rien faire » ici")
    if kind == "target":
        tid = payload["target_id"]
        for opt in decision.options:
            if isinstance(opt, (ShootAction, ChargeAction, OathAction)) and opt.target_id == tid:
                return opt
        raise ValueError(f"{tid} n'est pas une cible possible")
    if kind == "stratagem":  # {"type": "stratagem", "stratagem": clé ou null, "unit_id", "target_id", "model_id", "mode"}
        want = payload.get("stratagem")
        for opt in decision.options:
            if isinstance(opt, StratagemAction) and opt.stratagem == want and (want is None or all(
                    payload.get(k) in (None, getattr(opt, k)) for k in ("unit_id", "target_id", "model_id", "mode"))):
                return opt
        raise ValueError("ce stratagème n'est pas utilisable maintenant")
    if kind == "fight":
        for opt in decision.options:
            if isinstance(opt, FightAction) and opt.unit_id == payload["unit_id"] and opt.target_id == payload["target_id"]:
                return opt
        raise ValueError("combinaison unité / cible impossible")
    raise ValueError(f"type d'action inconnu : {kind}")


def decode_free(payload: Dict[str, Any]):
    """Action libre envoyée par le navigateur : {"type": "use_stratagem", stratagem_id, unit_id, target_id,
    model_id, choice, note} ou {"type": "manual", kind, unit_id, model_ids, value, positions, effect,
    effect_value, until, scope, vs, rule, note}."""
    t = payload.get("type")

    def opt_str(k, n=120):
        v = payload.get(k)
        return None if v in (None, "") else str(v)[:n]

    if t == "use_stratagem":
        return UseStratagemAction(str(payload["stratagem_id"]), opt_str("unit_id"), opt_str("target_id"), opt_str("model_id"), opt_str("choice"),
                                  (payload.get("note") or "")[:300])
    if t == "manual":
        value = payload.get("value")
        return ManualAction(
            kind=str(payload.get("kind", "note")), unit_id=opt_str("unit_id"), model_ids=tuple(str(m) for m in (payload.get("model_ids") or ())),
            value=None if value in (None, "") else int(value), positions=_decode_positions(payload.get("positions") or {}),
            effect=opt_str("effect"), effect_value=opt_str("effect_value"), until=str(payload.get("until") or "phase"),
            scope=str(payload.get("scope") or "all"), vs=opt_str("vs"), rule=(payload.get("rule") or "")[:120], note=(payload.get("note") or "")[:300],
        )
    raise ValueError(f"action libre inconnue : {t}")


# ------------------------------------------------------------------ document


JOIN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  #: sans 0/O ni 1/I : se dicte sans ambiguïté


def new_join_code() -> str:
    return "".join(secrets.choice(JOIN_ALPHABET) for _ in range(6))


def normalize_code(code: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def format_code(code: str) -> str:
    return f"{code[:3]}-{code[3:]}" if len(code) == 6 else code


def new_record(store: Optional["GameStore"], lists: Dict[str, Optional[Dict[str, str]]], players: Dict[str, Dict[str, Any]],
               title: Optional[str] = None, seed: Optional[int] = None, layout: str = "layout_a") -> Dict[str, Any]:
    """Document d'une nouvelle partie (jetons des joueurs humains compris)."""
    gid = store.new_id() if store is not None else secrets.token_urlsafe(6)
    seed = secrets.randbelow(2**31) if seed is None else int(seed)
    pl = {}
    for side in SIDES:
        p = players.get(side) or {"kind": "human"}
        kind = p.get("kind", "human")
        if kind not in ("human", "bot"):
            raise RoomError(f"type de joueur inconnu : {kind}")
        name = (p.get("name") or "").strip()[:40] or ("Bot aléatoire" if kind == "bot" else SIDE_FR[side].capitalize())
        pl[side] = {"kind": kind, "name": name, "token": secrets.token_urlsafe(18) if kind == "human" else None, "step_mode": True}
    if not any(p["kind"] == "human" for p in pl.values()):
        raise RoomError("il faut au moins un joueur humain")
    title = (title or "").strip()[:80] or None
    return {
        "format": FORMAT,
        "id": gid,
        "title": title,
        "title_auto": title is None,
        "created": _now(),
        "updated": _now(),
        "config": {"layout": layout, "seed": seed, "deploy": True, "dice_after_undo": "new", "rev": 3, "stratagems": True,
                   "lists": {side: (dict(lists[side]) if lists.get(side) else None) for side in SIDES}},
        "players": pl,
        "history": [],
        "reseeds": [],
        "undos": [],
        "timeline": 0,
        "status": "active",
        "result": None,
        "summary": {},
    }


def public_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Le document de partie sans les jetons secrets (téléchargement, liste des parties)."""
    doc = json.loads(json.dumps(record))
    for p in doc.get("players", {}).values():
        p.pop("token", None)
    doc.pop("join", None)  # le code de partie ne sert qu'à l'invité
    return doc


# ------------------------------------------------------------------ stockage


class GameStore:
    """Un fichier JSON par partie dans ``directory`` ; écriture atomique (fichier temporaire + rename)."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._summaries: Dict[str, tuple] = {}

    def path(self, gid: str) -> Path:
        if not _ID_RE.match(gid or ""):
            raise RoomError("identifiant de partie invalide")
        return self.directory / f"{gid}.json"

    def exists(self, gid: str) -> bool:
        return self.path(gid).exists()

    def new_id(self) -> str:
        while True:
            gid = secrets.token_urlsafe(6).replace("-", "x").replace("_", "y")
            if not self.path(gid).exists():
                return gid

    def save(self, record: Dict[str, Any]) -> None:
        path = self.path(record["id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    @property
    def trash(self) -> Path:
        return self.directory / "deleted"

    def delete(self, record: Dict[str, Any], by: str) -> Dict[str, Any]:
        """Supprime la partie : le fichier est déplacé dans ``deleted/`` (hors des listes et des exports,
        mais récupérable à la main), avec qui l'a supprimée et quand."""
        info = {"t": _now(), "by": by, "name": (record["players"].get(by) or {}).get("name")}
        doc = json.loads(json.dumps(record))
        doc["deleted"] = info
        self.trash.mkdir(parents=True, exist_ok=True)
        dest = self.trash / f"{record['id']}.json"
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, dest)
        path = self.path(record["id"])
        if path.exists():
            path.unlink()
        self._summaries.pop(record["id"], None)
        return info

    def load(self, gid: str) -> Dict[str, Any]:
        path = self.path(gid)
        if not path.exists():
            gone = self.trash / f"{gid}.json"
            if gone.exists():
                try:
                    raise RoomDeleted(json.loads(gone.read_text(encoding="utf-8")).get("deleted") or {})
                except ValueError:
                    raise RoomDeleted({}) from None
            raise RoomError("partie introuvable")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("format") != FORMAT:
            raise RoomError(f"format de partie inconnu : {doc.get('format')}")
        return doc

    def find_join(self, code: str) -> Optional[Dict[str, Any]]:
        """Partie en attente d'un adversaire dont le code de partie est ``code`` (déjà normalisé)."""
        if len(code) != 6:
            return None
        for path in self.directory.glob("*.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            join = doc.get("join") or {}
            if doc.get("status") == "waiting" and join.get("code") and hmac.compare_digest(join["code"], code):
                return doc
        return None

    def summaries(self) -> List[Dict[str, Any]]:
        """Résumé de chaque partie (sans jeton), la plus récente d'abord ; lu une fois par version du fichier."""
        out = []
        for path in self.directory.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
                hit = self._summaries.get(path.stem)
                if hit is None or hit[0] != mtime:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    hit = (mtime, _summary_of(doc))
                    self._summaries[path.stem] = hit
                out.append(dict(hit[1]))
            except (OSError, ValueError, KeyError):
                continue
        out.sort(key=lambda d: d.get("updated") or "", reverse=True)
        return out


def _summary_of(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc["id"],
        "title": doc.get("title"),
        "created": doc.get("created"),
        "updated": doc.get("updated"),
        "status": doc.get("status"),
        "players": {side: {"kind": p.get("kind"), "name": p.get("name") or None} for side, p in doc.get("players", {}).items()},
        "open_side": (doc.get("join") or {}).get("side"),
        "lists": {side: (l or {}).get("name") for side, l in (doc.get("config", {}).get("lists") or {}).items()},
        "actions": len(doc.get("history", [])),
        "undos": len(doc.get("undos", [])),
        **(doc.get("summary") or {}),
    }


# ------------------------------------------------------------------ salle de jeu


class Room:
    """Une partie en mémoire. Toutes les méthodes publiques prennent le verrou de la salle."""

    def __init__(self, cat: Catalog, store: Optional[GameStore], record: Dict[str, Any]):
        self.cat = cat
        self.store = store
        self.record = record
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)  #: réveille les clients en attente (long-polling)
        self.engine = Engine(listener=self._on_event)
        self.sim = Engine()  #: moteur sans écouteur pour les simulations (vérification en direct)
        self.initial = build_initial_state(cat, record["config"])
        self.layout_json = layout_to_json(self.initial.layout, self.initial.rules)
        self.version = 0
        self.snaps: List[GameState] = []  #: snaps[i] = état juste avant la décision history[i]
        self.state: Optional[GameState] = None
        self.observe: Optional[Dict[str, str]] = None  #: pause « pas à pas » après une action du bot
        self.error: Optional[str] = None
        self.deleted: Optional[Dict[str, Any]] = None  #: renseigné quand la partie est supprimée
        self._replaying = False
        self._cache: Dict[Optional[str], Dict[str, Any]] = {}
        seed = int(record["config"]["seed"])
        self.bots = {side: RandomAgent("Bot aléatoire", seed=seed + 99 + k) for k, side in enumerate(SIDES)
                     if record["players"][side]["kind"] == "bot"}
        self._rebuild()
        with self.lock:
            n = len(record["history"])
            self._drive()  # une partie rechargée où c'était au bot de jouer
            self._changed(save=len(record["history"]) != n)

    # ------------------------------------------------------------ création / relecture

    @classmethod
    def create(cls, cat: Catalog, store: Optional[GameStore], lists: Dict[str, Optional[Dict[str, str]]], players: Dict[str, Dict[str, Any]],
               title: Optional[str] = None, seed: Optional[int] = None, layout: str = "layout_a") -> "Room":
        """Nouvelle partie dont les deux listes sont connues. ``lists[side]`` = {"name", "text"} ou None
        (toy model) ; ``players[side]`` = {"kind": "human" | "bot", "name"}. Les humains reçoivent un
        jeton secret (leur lien)."""
        record = new_record(store, lists, players, title, seed, layout)
        room = cls(cat, store, record)
        if record.get("title_auto"):
            f = factions_of(room.state)
            record["title"] = f"{f.get('attacker', '?')} vs {f.get('defender', '?')}"
        room._changed()
        return room

    def _rebuild(self) -> None:
        """Rejoue l'historique depuis l'état initial (instantané avant chaque décision)."""
        snaps = []
        self._replaying = True
        try:
            it = iter_replay(self.engine, self.initial, self.record)
            while True:
                try:
                    _i, s, _d, _h = next(it)
                except StopIteration as stop:
                    final = stop.value
                    break
                snaps.append(s.clone())
        except ValueError as err:
            raise RoomError(str(err)) from err
        finally:
            self._replaying = False
        self.state, self.snaps = final, snaps

    # ------------------------------------------------------------ événements du moteur

    def _on_event(self, state: GameState, event: dict) -> None:
        if self._replaying or event.get("kind") != "notify":
            return
        acting = event["side"]
        other = other_side(acting)
        if acting in self.bots and other not in self.bots and self.record["players"][other].get("step_mode", True):
            self.observe = {"side": other, "message": event["message"]}

    # ------------------------------------------------------------ identité

    def side_of(self, token: Optional[str]) -> Optional[str]:
        """Camp du joueur qui présente ce jeton ; None = spectateur."""
        if not token:
            return None
        for side, p in self.record["players"].items():
            t = p.get("token")
            if t and hmac.compare_digest(t, token):
                return side
        return None

    def _require_player(self, token: Optional[str]) -> str:
        if self.deleted is not None:
            raise RoomDeleted(self.deleted)
        side = self.side_of(token)
        if side is None:
            raise RoomError("lien de spectateur : seuls les joueurs peuvent agir")
        return side

    # ------------------------------------------------------------ jeu

    def _apply(self, side: str, decision: Decision, action, by: str, **flags) -> None:
        """Enregistre et applique une action ; en cas d'erreur du moteur, revient à l'état d'avant.
        ``by`` : human, bot ou auto (réglage « actions automatiques » du joueur) ; ``flags`` :
        ``implicit=True`` pour une sélection d'unité faite en agissant directement sur elle."""
        s = self.state
        entry = {
            "i": len(self.record["history"]),
            "side": side,
            "by": by,
            "decision": decision.kind,
            "round": s.battle_round,
            "phase": s.phase,
            "unit_id": getattr(action, "unit_id", None) or decision.unit_id,
            "label": action_label(s, decision, action),
            "action": action_to_json(action),
            "t": _now(),
            **flags,
        }
        before = s.clone()
        try:
            self.engine.step(s, action, validate=False)
        except Exception as err:  # noqa: BLE001 — une erreur moteur ne doit pas corrompre la partie
            traceback.print_exc()
            self.state = before
            raise RoomError(f"erreur du moteur : {type(err).__name__}: {err}") from err
        self.snaps.append(before)
        self.record["history"].append(entry)

    def _drive(self) -> None:
        """Fait jouer les bots jusqu'à une décision humaine (ou une pause « pas à pas »)."""
        guard = 0
        while self.observe is None:
            try:
                d = self.engine.decision(self.state)
            except Exception as err:  # noqa: BLE001
                traceback.print_exc()
                self.error = f"{type(err).__name__}: {err}"
                return
            if d is None:
                return
            bot = self.bots.get(d.side)
            if bot is None:
                auto = self._auto_choice(d)
                if auto is None:
                    return
                self._apply(d.side, d, auto, by="auto")
            else:
                self._apply(d.side, d, bot.choose(self.state, d), by="bot")
            guard += 1
            if guard > 5000:
                self.error = "le bot ne rend pas la main"
                return

    # ------------------------------------------------------------ réglages et automatismes

    SETTINGS = {"step_mode": True, "auto_actions": True, "auto_deploy": False, "stratagems": True}

    def setting(self, side: str, key: str) -> bool:
        return bool(self.record["players"][side].get(key, self.SETTINGS[key]))

    def _auto_choice(self, d: Decision):
        """Action jouée d'office pour un joueur humain, selon ses réglages ; None = on lui demande."""
        side = d.side
        if self.record["players"][side]["kind"] != "human":
            return None
        if d.kind == "stratagem" and not self.setting(side, "stratagems"):
            return next(o for o in d.options if isinstance(o, StratagemAction) and o.stratagem is None)  # fenêtres coupées : on passe
        if self.setting(side, "auto_deploy") and self.state.phase == "deployment":
            if d.kind == "select_unit":
                units = [o for o in d.options if isinstance(o, SelectUnitAction)]
                return units[0] if units else None
            if d.kind == "deploy":
                return self._auto_deploy_spot(d)
        if not self.setting(side, "auto_actions"):
            return None
        if d.kind == "fight" and len(d.options) == 1:
            return d.options[0]
        if d.kind == "shoot":
            targets = [o for o in d.options if isinstance(o, ShootAction) and o.target_id is not None]
            if len(targets) == 1:
                return targets[0]
        if d.kind in ("oath", "charge_target"):
            real = [o for o in d.options if getattr(o, "target_id", None) is not None]
            if len(real) == 1 and len(d.options) == 1:
                return real[0]
        return None

    def _auto_deploy_spot(self, d: Decision):
        """Position de déploiement automatique : parmi celles que propose le moteur, la plus proche du
        centre de la zone (le moteur écarte déjà les chevauchements et la sortie de zone)."""
        spots = [o for o in d.options if isinstance(o, DeployAction)]
        if not spots:
            return None
        zone = self.state.layout.deployment_zones[d.side].vertices
        cx = sum(v[0] for v in zone) / len(zone)
        cy = sum(v[1] for v in zone) / len(zone)
        return min(spots, key=lambda o: (o.x - cx) ** 2 + (o.y - cy) ** 2)

    def set_settings(self, token: Optional[str], values: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            side = self._require_player(token)
            p = self.record["players"][side]
            for key in self.SETTINGS:
                if key in values:
                    p[key] = bool(values[key])
            if not self.setting(side, "step_mode") and self.observe and self.observe["side"] == side:
                self.observe = None
            self._drive()
            self._changed()
            return {"ok": True, **{k: self.setting(side, k) for k in self.SETTINGS}}

    # ------------------------------------------------------------ actions des joueurs

    _DIRECT = {"model_move", "declare_advance", "stationary", "declare_charge", "disembark_models"}

    def _decide(self, side: str, d: Decision, payload: Dict[str, Any]):
        """Décode et valide une action ; lève RoomError si elle est refusée."""
        try:
            action = decode_action(d, payload)
        except (KeyError, ValueError, IndexError, TypeError) as err:
            raise RoomError(str(err)) from err
        if action not in d.options:
            err = self.engine.free_action_error(self.state, d, action)
            if err is not None:
                raise RoomError(f"Action illégale : {err}")
        return action

    def _implicit_select(self, d: Decision, payload: Dict[str, Any]):
        """Action directe sur une unité pendant « quelle unité ? » : la sélection qu'elle implique."""
        if d.kind != "select_unit" or payload.get("type") not in self._DIRECT:
            return None
        uid = payload.get("unit_id")
        return next((o for o in d.options if isinstance(o, SelectUnitAction) and o.unit_id == uid), None)

    def act(self, token: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            side = self._require_player(token)
            if payload.get("type") == "continue":
                if self.observe and self.observe["side"] == side:
                    self.observe = None
                    self._drive()
                    self._changed(save=False)
                return {"ok": True}
            if self.observe and self.observe["side"] == side:
                raise RoomError("valide d'abord l'action adverse (« Continuer »)")
            d = self.engine.decision(self.state)
            if d is None:
                raise RoomError("la partie est terminée")
            if d.side != side:
                raise RoomError("ce n'est pas à toi de jouer")
            select = self._implicit_select(d, payload)
            if select is not None:
                # glisser directement une unité : on la sélectionne, puis on joue l'action ; si
                # l'action est refusée, la sélection est défaite (rien n'a bougé)
                self._apply(side, d, select, by="human", implicit=True)
                d = self.engine.decision(self.state)
                try:
                    if d is None or d.side != side:
                        raise RoomError("cette unité ne peut pas faire cela maintenant")
                    action = self._decide(side, d, payload)
                except RoomError:
                    self.state = self.snaps.pop()
                    self.record["history"].pop()
                    raise
            else:
                action = self._decide(side, d, payload)
            self._apply(side, d, action, by="human")
            self._drive()
            self._changed()
            return {"ok": True}

    def free(self, token: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Action libre (effet manuel, stratagème du panneau) : l'un ou l'autre joueur, à tout moment."""
        with self.lock:
            side = self._require_player(token)
            try:
                action = decode_free(payload)
            except (KeyError, ValueError, TypeError) as err:
                raise RoomError(str(err)) from err
            err = self.engine.free_error(self.state, side, action)
            if err is not None:
                raise RoomError(err)
            s = self.state
            entry = {"i": len(self.record["history"]), "side": side, "by": "human", "decision": "free", "round": s.battle_round, "phase": s.phase,
                     "unit_id": getattr(action, "unit_id", None), "label": action_label(s, None, action), "action": action_to_json(action), "t": _now()}
            before = s.clone()
            try:
                self.engine.apply_free(s, side, action, validate=False)
                self.engine.decision(s)  # la décision en cours doit rester possible
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.state = before
                raise RoomError(f"action impossible ici : {type(exc).__name__}: {exc}") from exc
            self.snaps.append(before)
            self.record["history"].append(entry)
            self.observe = None if self.observe and self.observe["side"] == side else self.observe
            self._drive()
            self._changed()
            return {"ok": True}

    def check(self, token: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Vérification à blanc d'un placement (pendant le glissement) : rien n'est joué ni enregistré.
        Retourne {ok, error, model_id} (la figurine fautive quand le message la nomme)."""
        if payload.get("type") in ("manual", "use_stratagem"):
            with self.lock:
                side = self._require_player(token)
                try:
                    err = self.engine.free_error(self.state, side, decode_free(payload))
                except (KeyError, ValueError, TypeError) as exc:
                    err = str(exc)
            m = re.match(r"^([A-Za-z0-9]+\.\d+)\b", err or "")
            return {"ok": err is None, "error": err, "model_id": m.group(1) if m else None}
        with self.lock:
            side = self._require_player(token)
            d = self.engine.decision(self.state)
            if d is None or d.side != side:
                return {"ok": False, "error": "ce n'est pas à toi de jouer", "model_id": None}
            state = self.state
            select = self._implicit_select(d, payload)
            if select is not None:
                state = self.state.clone(record=False)
                self.sim.step(state, select, validate=False)
                d = self.sim.decision(state)
            try:
                action = decode_action(d, payload)
            except (KeyError, ValueError, IndexError, TypeError) as err:
                return {"ok": False, "error": str(err), "model_id": None}
            err = None if action in d.options else self.sim.free_action_error(state, d, action)
        if err is None:
            return {"ok": True, "error": None, "model_id": None}
        m = re.match(r"^([A-Za-z0-9]+\.\d+)\b", err)
        return {"ok": False, "error": err, "model_id": m.group(1) if m else None}

    def frames(self, start: int, limit: int = 150) -> Dict[str, Any]:
        """Positions des figurines avant l'action ``start`` puis après chacune des suivantes (rejeu animé
        de « ce qui s'est passé pendant ton absence ») ; seules les figurines qui changent sont listées."""
        with self.lock:
            n = len(self.record["history"])
            start = max(0, min(int(start), n), n - limit)

            def pose(st: GameState) -> Dict[str, list]:
                return {m.id: [round(m.x, 3), round(m.y, 3), round(m.angle, 4), 1 if m.alive else 0, m.wounds, 1 if (u.embarked_in is None and not u.in_reserve) else 0]
                        for u in st.units.values() for m in u.models}

            first = self.snaps[start] if start < n else self.state
            prev = pose(first)
            frames = []
            for i in range(start, n):
                after = self.snaps[i + 1] if i + 1 < n else self.state
                cur = pose(after)
                h = self.record["history"][i]
                frames.append({"i": i, "side": h["side"], "by": h.get("by", "human"), "label": h.get("label"), "round": h.get("round"),
                               "phase": h.get("phase"),
                               "models": {mid: v for mid, v in cur.items() if prev.get(mid) != v}})
                prev = cur
            return {"start": start, "initial": pose(first), "frames": frames, "timeline": self.record.get("timeline", 0)}

    def unit_details(self, uid: str) -> Dict[str, Any]:
        """Fiche complète d'une unité pour le panneau latéral."""
        with self.lock:
            try:
                u = self.state.unit(uid)
            except KeyError as err:
                raise RoomError(f"unité inconnue : {uid}") from err
            profiles: Dict[str, Dict[str, Any]] = {}
            for m in u.models:
                p = m.profile
                row = profiles.setdefault(p.name, {"name": p.name, "M": p.move_in, "T": p.toughness, "Sv": p.save, "InSv": p.invuln, "W": p.wounds,
                                                   "Ld": p.leadership, "OC": p.oc, "alive": 0, "total": 0})
                row["total"] += 1
                row["alive"] += 1 if m.alive else 0
            weapons: Dict[tuple, Dict[str, Any]] = {}
            for m in u.models:
                if not m.alive:
                    continue
                for w in m.weapons:
                    key = (w.name, w.kind)
                    row = weapons.setdefault(key, {"name": w.name, "kind": w.kind, "range": None if w.is_melee else w.range_in, "A": str(w.attacks),
                                                   "skill": w.skill, "S": w.strength, "AP": w.ap, "D": str(w.damage),
                                                   "keywords": [str(k) for k in w.keywords], "count": 0})
                    row["count"] += 1
            sheets = [u.datasheet, *u.leaders]
            abilities = []
            seen = set()
            for ds in sheets:
                for a in ds.abilities:
                    key = (a.name, a.parameter)
                    if key in seen:
                        continue
                    seen.add(key)
                    abilities.append({"name": a.name + (f" {a.parameter}" if a.parameter else ""), "type": a.type,
                                      "text": (a.text or "")[:600], "from": ds.name if ds is not u.datasheet else None})
            return {
                "id": u.id, "side": u.side, "name": u.datasheet.name, "leaders": [l.name for l in u.leaders],
                "enhancements": list(u.enhancements), "points": u.points, "warlord": u.is_warlord,
                "keywords": list(u.datasheet.keywords), "faction_keywords": list(u.datasheet.faction_keywords),
                "profiles": list(profiles.values()), "weapons": sorted(weapons.values(), key=lambda w: (w["kind"] != "ranged", w["name"])),
                "abilities": abilities, "transport": u.datasheet.transport or None,
            }

    def wait_change(self, version: int, timeout: float = 25.0) -> None:
        """Long-polling : attend que la partie change (ou ``timeout``) si le client a déjà ``version``."""
        with self.cond:
            if self.version == version and self.deleted is None:
                self.cond.wait(timeout)

    def set_step_mode(self, token: Optional[str], on: bool) -> Dict[str, Any]:
        return self.set_settings(token, {"step_mode": on})

    # ------------------------------------------------------------ annulation

    def undo(self, token: Optional[str], to: Optional[int] = None) -> Dict[str, Any]:
        """Revient juste avant l'action ``to`` (par défaut : la dernière action d'un joueur humain,
        pour ne pas laisser le bot rejouer aussitôt). Les dés suivants sont re-tirés."""
        with self.lock:
            side = self._require_player(token)
            hist = self.record["history"]
            if not hist:
                raise RoomError("rien à annuler")
            if to is None:
                humans = [i for i, h in enumerate(hist) if h.get("by", "human") == "human"]
                if not humans:
                    raise RoomError("aucune action de joueur à annuler")
                k = humans[-1]
                if k > 0 and hist[k - 1].get("implicit") and not hist[k].get("implicit"):
                    k -= 1  # l'action directe sur une unité compte pour une seule action
            else:
                k = int(to)
                if not 0 <= k < len(hist):
                    raise RoomError("point d'historique invalide")
            undone = hist[k:]
            target = hist[k]
            new_seed = secrets.randbits(48)
            state = self.snaps[k].clone()
            state.rng.seed(new_seed)
            self.state = state
            self.snaps = self.snaps[:k]
            self.record["history"] = hist[:k]
            self.record["reseeds"] = [r for r in self.record.get("reseeds", []) if int(r["at"]) < k] + [{"at": k, "seed": new_seed}]
            self.record["timeline"] = int(self.record.get("timeline", 0)) + 1
            self.record["undos"].append({
                "n": len(self.record["undos"]) + 1,
                "by": side,
                "name": self.record["players"][side]["name"],
                "t": _now(),
                "to": k,
                "count": len(undone),
                "where": f"round {target['round']}, {PHASE_FR.get(target['phase'], target['phase'])}" if target["round"] else PHASE_FR.get(target["phase"], target["phase"]),
                "first": target["label"],
                "undone": undone,
            })
            self.record["status"] = "active"
            self.record["result"] = None
            self.observe = None
            self.error = None
            self._drive()
            self._changed()
            return {"ok": True, "undone": len(undone), "to": k}

    # ------------------------------------------------------------ état → JSON

    def _changed(self, save: bool = True) -> None:
        with self.cond:
            self._changed_locked(save)
            self.cond.notify_all()

    def _changed_locked(self, save: bool) -> None:
        if self.deleted is not None:
            return
        s = self.state
        over = self.engine.is_over(s)
        rec = self.record
        rec["updated"] = _now()
        if over and rec["status"] != "finished":
            res = self.engine.result(s)
            rec["status"], rec["result"] = "finished", {"winner": res.winner, "scores": res.scores, "rounds": res.rounds_played}
        try:
            d = self.engine.decision(s)
        except Exception:  # noqa: BLE001
            d = None
        rec["summary"] = {"round": s.battle_round, "phase": s.phase, "scores": s.scoreboard.totals(), "to_move": d.side if d else None,
                          "factions": factions_of(s)}
        self.version += 1
        self._cache = {}
        if save and self.store is not None:
            self.store.save(rec)

    def history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [{k: h.get(k) for k in ("i", "side", "by", "decision", "round", "phase", "label", "t", "implicit")} for h in self.record["history"]]

    def snapshot(self, token: Optional[str], since: int = 0) -> Dict[str, Any]:
        with self.lock:
            viewer = self.side_of(token)
            snap = self._cache.get(viewer)
            if snap is None:
                snap = self._build_snapshot(viewer)
                self._cache[viewer] = snap
            snap = dict(snap)
        log = snap.pop("_log")
        snap["log"] = log[since:]
        snap["log_since"] = since
        return snap

    def _build_snapshot(self, viewer: Optional[str]) -> Dict[str, Any]:
        s = self.state
        rec = self.record
        try:
            d = self.engine.decision(s)
        except Exception as err:  # noqa: BLE001
            d, self.error = None, f"{type(err).__name__}: {err}"
        if self.observe and self.observe["side"] == viewer:
            d = Decision("observe", viewer, [ContinueAction()], note=self.observe["message"], phase=s.phase)
        over = self.engine.is_over(s)
        players = {side: {"kind": p["kind"], "name": p["name"]} for side, p in rec["players"].items()}
        undo = rec["undos"][-1] if rec["undos"] else None
        extra = {
            "game": {"id": rec["id"], "title": rec.get("title"), "created": rec.get("created")},
            "players": players,
            "you": viewer,
            "vs_bot": bool(self.bots),
            "game_over": over,
            "result": rec.get("result") if over else None,
            "engine_error": self.error,
            "last_error": None,
            "waiting_for_human": d is not None and viewer is not None and d.side == viewer,
            "step_mode": bool(viewer and self.setting(viewer, "step_mode")),
            "settings": {k: self.setting(viewer, k) for k in self.SETTINGS} if viewer else None,
            "timeline": rec.get("timeline", 0),
            "history_length": len(rec["history"]),
            "can_undo": viewer is not None and any(h.get("by", "human") == "human" for h in rec["history"]),
            "last_undo": None if undo is None else {k: undo[k] for k in ("n", "by", "name", "t", "to", "count", "where", "first")},
        }
        if viewer is not None:
            from ..engine.effects import KINDS
            from ..engine.free import MANUAL_KINDS

            extra["stratagem_book"] = stratagem_book_json(s, viewer) if s.stratagems else []
            extra["rules_menu"] = rules_menu(s, viewer)
            extra["manual_kinds"] = MANUAL_KINDS
            extra["effect_kinds"] = KINDS
        snap = state_to_json(s, d, viewer, since=0, extra=extra)
        snap["_log"] = snap.pop("log")
        snap["version"] = self.version
        return snap


# ------------------------------------------------------------------ registre des salles


class RoomWaiting(RoomError):
    """La partie attend encore son second joueur (pas encore de plateau)."""

    def __init__(self, record: Dict[str, Any]):
        super().__init__("la partie attend son second joueur")
        self.record = record


def waiting_snapshot(record: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    """État d'une partie en attente, vu par le détenteur de ``token`` (créateur, invité ou spectateur)."""
    join = record.get("join") or {}
    open_side = join.get("side")
    viewer = None
    for side, p in record["players"].items():
        if token and p.get("token") and hmac.compare_digest(p["token"], token):
            viewer = side
    creator = other_side(open_side) if open_side else None
    snap = {
        "ok": True,
        "waiting": True,
        "game": {"id": record["id"], "title": record.get("title"), "created": record.get("created")},
        "players": {side: {"kind": p["kind"], "name": p.get("name") or None} for side, p in record["players"].items()},
        "lists": {side: (l or {}).get("name") for side, l in record["config"]["lists"].items()},
        "you": viewer,
        "open_side": open_side,
        "can_join": viewer is not None and viewer == open_side,
        "join": None,
        "timeline": 0,
        "version": 0,
    }
    if viewer is not None and viewer == creator:
        snap["join"] = {"code": format_code(join["code"]), "invite": f"/g/{record['id']}?t={record['players'][open_side]['token']}"}
    return snap


class Rooms:
    """Salles chargées à la demande depuis le stockage (une seule instance par partie)."""

    def __init__(self, cat: Catalog, store: GameStore):
        self.cat = cat
        self.store = store
        self._rooms: Dict[str, Room] = {}
        self._lock = threading.RLock()

    def get(self, gid: str) -> Room:
        """La salle de la partie ; :class:`RoomWaiting` si elle attend encore son second joueur."""
        with self._lock:
            room = self._rooms.get(gid)
            if room is None:
                record = self.store.load(gid)
                if record.get("status") == "waiting":
                    raise RoomWaiting(record)
                room = Room(self.cat, self.store, record)
                self._rooms[gid] = room
            return room

    def delete(self, gid: str, token: Optional[str]) -> Dict[str, Any]:
        """Un joueur de la partie la supprime (le créateur seulement, tant que l'adversaire n'a pas rejoint)."""
        with self._lock:
            room = self._rooms.get(gid)
            record = room.record if room is not None else self.store.load(gid)
            by = None
            for side, p in record["players"].items():
                if token and p.get("token") and hmac.compare_digest(p["token"], token):
                    by = side
            if by is None:
                raise RoomError("seuls les joueurs de la partie peuvent la supprimer")
            if record.get("status") == "waiting" and by == (record.get("join") or {}).get("side"):
                raise RoomError("seul le créateur peut supprimer une partie en attente")
            if room is not None:
                with room.cond:
                    info = self.store.delete(room.record, by)
                    room.deleted = info
                    room.cond.notify_all()  # réveille les clients en attente : ils voient la suppression
            else:
                info = self.store.delete(record, by)
            self._rooms.pop(gid, None)
            return {"ok": True, "deleted": info}

    def create(self, **kwargs) -> Room:
        room = Room.create(self.cat, self.store, **kwargs)
        with self._lock:
            self._rooms[room.record["id"]] = room
        return room

    def open_game(self, side: str, name: str, list_ref: Optional[Dict[str, str]], title: Optional[str] = None,
                  seed: Optional[int] = None, layout: str = "layout_a") -> Dict[str, Any]:
        """Partie contre un adversaire humain qui n'a pas encore rejoint : le créateur joue ``side``
        avec sa liste ; l'adversaire rejoindra avec le code de partie (ou le lien d'invitation) et
        choisira alors sa propre liste. Le plateau est construit à ce moment-là."""
        from ..data.list_library import resolve_text

        if side not in SIDES:
            raise RoomError(f"camp inconnu : {side}")
        other = other_side(side)
        if list_ref and list_ref.get("text"):
            try:
                faction = resolve_text(list_ref["text"], self.cat).faction_name
            except Exception as err:  # noqa: BLE001
                raise RoomError(f"liste illisible : {err}") from err
        else:
            faction = "Space Marines" if side == "attacker" else "Emperor’s Children"  # rosters du toy model
        record = new_record(self.store, {side: list_ref, other: None}, {side: {"kind": "human", "name": name}, other: {"kind": "human", "name": ""}},
                            title=title, seed=seed, layout=layout)
        record["players"][other]["name"] = ""
        record["status"] = "waiting"
        record["config"]["awaiting"] = other
        record["join"] = {"code": new_join_code(), "side": other}
        if record["title_auto"]:
            record["title"] = f"{faction} vs ?" if side == "attacker" else f"? vs {faction}"
        self.store.save(record)
        return record

    def join(self, record: Dict[str, Any], name: str, list_ref: Optional[Dict[str, str]]) -> Tuple[str, str, Room]:
        """L'adversaire rejoint une partie en attente : son nom, sa liste ; la partie démarre.
        Retourne (camp, jeton, salle)."""
        with self._lock:
            current = self.store.load(record["id"])  # relu sous verrou : deux adversaires ne peuvent pas rejoindre
            if current.get("status") != "waiting":
                raise RoomError("cette partie a déjà ses deux joueurs")
            rec = json.loads(json.dumps(current))
            side = rec["join"]["side"]
            rec["players"][side]["name"] = (name or "").strip()[:40] or SIDE_FR[side].capitalize()
            rec["config"]["lists"][side] = dict(list_ref) if list_ref else None
            rec["config"].pop("awaiting", None)
            rec.pop("join", None)
            rec["status"] = "active"
            rec["joined"] = _now()
            try:
                room = Room(self.cat, self.store, rec)
            except RoomError:
                raise
            except Exception as err:  # noqa: BLE001 — liste illisible, déploiement impossible…
                traceback.print_exc()
                raise RoomError(f"{type(err).__name__}: {err}") from err
            if rec.get("title_auto"):
                f = factions_of(room.state)
                rec["title"] = f"{f.get('attacker', '?')} vs {f.get('defender', '?')}"
            room._changed()
            self._rooms[rec["id"]] = room
            return side, rec["players"][side]["token"], room

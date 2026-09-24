"""Service web : parties à deux joueurs (ou contre le bot) sauvegardées, avec annulation.

Bibliothèque standard uniquement (``http.server``). Chaque partie est un fichier JSON dans le dossier
de données (:class:`~fortyk.web.rooms.GameStore`) ; chaque joueur humain a un lien secret
``/g/<partie>?t=<jeton>`` (sans jeton : spectateur). Le navigateur interroge l'état (polling) et poste
ses actions.

Pages : ``/`` (accueil : parties, création, listes d'armée — ``static/lobby.html``) et ``/g/<id>``
(plateau — ``static/index.html``).

API (JSON) :

* ``GET /api/games`` — résumé des parties ; ``POST /api/games`` — nouvelle partie
  ``{title, lists: {attacker, defender} (noms de listes, null = toy model), players: {side: {kind:
  "human" | "bot", name}}, access_code}`` → liens des joueurs ;
* ``GET /api/g/<id>/layout`` · ``GET /api/g/<id>/state?t=&since=`` · ``GET /api/g/<id>/history`` ;
* ``POST /api/g/<id>/action?t=`` — une action (protocole ci-dessous) ;
* ``POST /api/g/<id>/undo?t=`` ``{to: index | null}`` — revenir avant l'action ``to`` (null : la
  dernière action d'un joueur) ; les dés suivants sont re-tirés ;
* ``POST /api/g/<id>/settings?t=`` ``{step_mode}`` — pas à pas contre le bot ;
* ``GET /api/g/<id>/record`` — le document de la partie (sans jetons) ; ``GET /api/g/<id>/export`` —
  export d'entraînement (états avant chaque décision) ;
* ``GET /api/lists`` · ``POST /api/lists/preview`` · ``POST /api/lists/save`` ; ``GET /healthz``.

Protocole des actions : ``{"type": "option", "index": i}`` (une option proposée par le moteur) ;
``deploy_models`` / ``model_move`` (``kind`` : normal, fall_back, desperate, advance_move, scout,
charge) / ``disembark_models`` avec ``positions: {model_id: [x, y] ou [x, y, angle]}`` ;
``declare_advance``, ``stationary``, ``declare_charge``, ``auto_charge`` ; ``target`` (``target_id``),
``skip`` ; ``select_unit`` (``unit_id``), ``end_phase`` ; ``fight`` (``unit_id``, ``target_id``) ;
``continue`` (pas à pas).

Variables d'environnement : ``FORTYK_DATA_DIR`` (parties), ``FORTYK_LISTS_DIR`` (listes importées),
``FORTYK_ACCESS_CODE`` (code demandé pour créer une partie ou enregistrer une liste), ``PORT``.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from ..data import Catalog, load_catalog
from .rooms import GameStore, Room, RoomError, Rooms, public_record

__all__ = ["App", "make_handler", "serve", "STATIC_DIR"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "games"
_GAME_RE = re.compile(r"^/api/g/([A-Za-z0-9_-]{4,40})/([a-z]+)$")
_PAGE_RE = re.compile(r"^/g/([A-Za-z0-9_-]{4,40})/?$")


class App:
    """État du service : catalogue Wahapedia, stockage des parties, salles chargées, code d'accès."""

    def __init__(self, cat: Catalog, data_dir=None, access_code: Optional[str] = None):
        self.cat = cat
        self.store = GameStore(data_dir or os.environ.get("FORTYK_DATA_DIR") or DEFAULT_DATA_DIR)
        self.rooms = Rooms(cat, self.store)
        self.access_code = access_code if access_code is not None else (os.environ.get("FORTYK_ACCESS_CODE") or None)

    def check_code(self, code: Optional[str]) -> None:
        if self.access_code and not hmac.compare_digest(str(code or ""), self.access_code):
            raise RoomError("code d'accès incorrect")

    def create_game(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ..data.list_library import list_text

        self.check_code(payload.get("access_code"))
        lists = {}
        for side in ("attacker", "defender"):
            name = (payload.get("lists") or {}).get(side)
            if name:
                try:
                    lists[side] = {"name": name, "text": list_text(name)}
                except FileNotFoundError as err:
                    raise RoomError(str(err)) from err
            else:
                lists[side] = None
        try:
            room = self.rooms.create(lists=lists, players=payload.get("players") or {}, title=payload.get("title"), seed=payload.get("seed"))
        except RoomError:
            raise
        except Exception as err:  # noqa: BLE001 — liste illisible, déploiement impossible…
            traceback.print_exc()
            raise RoomError(f"{type(err).__name__}: {err}") from err
        rec = room.record
        gid = rec["id"]
        links = {side: f"/g/{gid}?t={p['token']}" for side, p in rec["players"].items() if p.get("token")}
        return {"ok": True, "id": gid, "title": rec["title"], "links": links, "spectate": f"/g/{gid}",
                "players": {side: {"kind": p["kind"], "name": p["name"]} for side, p in rec["players"].items()}}


def make_handler(app: App):
    return type("Handler", (_Handler,), {"app": app})


class _Handler(BaseHTTPRequestHandler):
    server_version = "fortyk/0.2"
    app: App = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # silence
        pass

    # ------------------------------------------------------------ réponses

    def _json(self, data: Any, status: int = 200, download: Optional[str] = None) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _room(self, gid: str) -> Room:
        return self.app.rooms.get(gid)

    # ------------------------------------------------------------ GET

    def do_GET(self) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        token = qs.get("t", [None])[0]
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC_DIR / "lobby.html", "text/html; charset=utf-8")
            if _PAGE_RE.match(path):
                return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            if path == "/healthz":
                return self._json({"ok": True})
            if path == "/api/config":
                return self._json({"access_code_required": bool(self.app.access_code)})
            if path == "/api/games":
                return self._json({"games": self.app.store.summaries()})
            if path == "/api/lists":
                from ..data.list_library import saved_lists

                return self._json({"lists": saved_lists(self.app.cat)})
            m = _GAME_RE.match(path)
            if m:
                gid, what = m.groups()
                room = self._room(gid)
                if what == "layout":
                    return self._json(room.layout_json)
                if what == "state":
                    return self._json(room.snapshot(token, int(qs.get("since", ["0"])[0])))
                if what == "history":
                    return self._json({"history": room.history(), "timeline": room.record.get("timeline", 0)})
                if what == "record":
                    with room.lock:
                        doc = public_record(room.record)
                    return self._json(doc, download=f"40k_{gid}.json")
                if what == "export":
                    from ..training import training_export

                    with room.lock:
                        rec = json.loads(json.dumps(room.record))
                    states = qs.get("states", ["1"])[0] != "0"
                    return self._json(training_export(self.app.cat, rec, states=states), download=f"40k_{gid}_training.json")
            self.send_error(404)
        except RoomError as err:
            self._json({"ok": False, "error": str(err)}, 404)
        except Exception as err:  # noqa: BLE001
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(err).__name__}: {err}"}, 500)

    # ------------------------------------------------------------ POST

    def do_POST(self) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        token = qs.get("t", [None])[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "JSON invalide"}, 400)
        path = url.path
        try:
            if path == "/api/games":
                return self._json(self.app.create_game(payload))
            if path in ("/api/lists/preview", "/api/lists/save"):
                return self._json(self._lists(path, payload))
            m = _GAME_RE.match(path)
            if m:
                gid, what = m.groups()
                room = self._room(gid)
                if what == "action":
                    return self._json(room.act(token, payload))
                if what == "undo":
                    to = payload.get("to")
                    return self._json(room.undo(token, None if to is None else int(to)))
                if what == "settings":
                    return self._json(room.set_step_mode(token, bool(payload.get("step_mode", True))))
            self.send_error(404)
        except RoomError as err:
            self._json({"ok": False, "error": str(err)})
        except Exception as err:  # noqa: BLE001
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(err).__name__}: {err}"}, 500)

    def _lists(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ..data.list_library import default_name, list_report, resolve_text, save_list

        text = payload.get("text", "")
        try:
            if path.endswith("save"):
                self.app.check_code(payload.get("access_code"))
                name, al = save_list(text, self.app.cat, name=payload.get("name") or None, overwrite=bool(payload.get("overwrite")))
            else:
                al = resolve_text(text, self.app.cat)
                name = default_name(al)
        except RoomError:
            raise
        except Exception as err:  # noqa: BLE001
            return {"ok": False, "error": f"{type(err).__name__}: {err}"}
        return {"ok": True, "name": name, "report": list_report(al)}


def serve(host: str = "127.0.0.1", port: int = 8040, cat: Optional[Catalog] = None, open_browser: bool = True, data_dir=None,
          access_code: Optional[str] = None, quick: Optional[Dict[str, Any]] = None):
    """Lance le service. ``quick`` : crée tout de suite une partie (``{"side", "attacker_list",
    "defender_list"}``) contre le bot et ouvre son lien."""
    cat = cat or load_catalog()
    app = App(cat, data_dir=data_dir, access_code=access_code)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}/"
    if quick:
        side = quick.get("side") or "attacker"
        other = "defender" if side == "attacker" else "attacker"
        res = app.create_game({"access_code": app.access_code, "lists": {"attacker": quick.get("attacker_list"), "defender": quick.get("defender_list")},
                               "players": {side: {"kind": "human", "name": "Moi"}, other: {"kind": "bot"}}})
        url = f"http://{shown}:{port}{res['links'][side]}"
    print(f"40kPlayer — parties dans {app.store.directory}")
    print(f"Service ouvert sur {url}  (Ctrl-C pour arrêter)" + ("  — code d'accès requis pour créer une partie" if app.access_code else ""))
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        httpd.server_close()
    return httpd

"""Service web : parties sauvegardées à deux joueurs (ou contre le bot), plateau canvas, annulation."""

from .rooms import GameStore, Room, RoomError, Rooms
from .server import App, make_handler, serve

__all__ = ["App", "GameStore", "Room", "RoomError", "Rooms", "make_handler", "serve"]

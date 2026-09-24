"""Agents : tout objet exposant ``name`` et ``choose(state, decision) -> action``."""

from .base import Agent
from .human_agent import HumanAgent, Quit
from .random_agent import RandomAgent

__all__ = ["Agent", "HumanAgent", "Quit", "RandomAgent"]

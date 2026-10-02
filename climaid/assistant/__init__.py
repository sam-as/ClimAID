"""ClimAID assistant: a built-in, offline conversational guide (``climaid ai``).

Rule-based (no language model): it understands common requests, asks for what is missing,
runs ClimAID, and explains results and documentation. See ``Assistant``.
"""
from .core import Assistant
from .actions import Runner

__all__ = ["Assistant", "Runner"]

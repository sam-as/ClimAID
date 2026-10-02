"""Terminal front end for the assistant (``climaid ai``)."""
from __future__ import annotations

import textwrap

from .core import Assistant


def _print(message: str):
    for line in message.splitlines() or [""]:
        # keep tables and indented lines as they are; wrap long prose lines
        if line.startswith(" ") or len(line) <= 100:
            print(line)
        else:
            print(textwrap.fill(line, width=100))
    print()


def run_terminal(assistant: Assistant | None = None):
    assistant = assistant or Assistant()
    print()
    _print(assistant.greet())
    while not assistant.finished:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not text:
            continue
        print()
        for message in assistant.respond(text):
            _print(message)

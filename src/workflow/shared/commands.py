# commands.py — human-gate command classification.
#
# Canonical home of the exit/continue keyword logic previously private in
# wait_for_user.py. Pure functions, table-tested.

from __future__ import annotations

# Anything matching these (case-insensitive, stripped) ends the run.
# Bare words match exactly; phrases match as prefix ("exit the loop", "quit now"...).
EXIT_KEYWORDS = frozenset(
    {
        "exit",
        "quit",
        "q",
        "stop",
        "end",
        "finish",
        "done",
        "bye",
        "close",
        "shutdown",
    }
)

EXIT_PHRASES = (
    "exit the loop",
    "quit the loop",
    "end the loop",
    "stop the loop",
    "exit loop",
    "quit loop",
)

# Inputs that mean "no new info, just carry on with the current goal".
CONTINUE_KEYWORDS = frozenset({"continue", "go on", "proceed", "resume", ""})


def is_exit_command(text: str) -> bool:
    """True only when the user explicitly asks to exit the loop."""
    normalized = (text or "").strip().lower().rstrip(" .!！")
    if not normalized:
        return False
    if normalized in EXIT_KEYWORDS:
        return True
    return any(normalized.startswith(phrase) for phrase in EXIT_PHRASES)


def is_continue_command(text: str) -> bool:
    """True for bare 'continue'-style inputs that carry no new task info."""
    return (text or "").strip().lower() in CONTINUE_KEYWORDS


__all__ = [
    "EXIT_KEYWORDS",
    "EXIT_PHRASES",
    "CONTINUE_KEYWORDS",
    "is_exit_command",
    "is_continue_command",
]

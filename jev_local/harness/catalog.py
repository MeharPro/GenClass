"""Closed vocabularies for the Mac harness: intents, keys, folders, risk lexicon.

These strings are part of the model's input. The synthetic-data generator and the runtime both
import them from here, so changing a description here means retraining the fast model.
"""

from __future__ import annotations

import re

# label -> description shown to the model as the option text
INTENTS: dict[str, str] = {
    "open_app": "open, launch, start, or switch to an application",
    "quit_app": "quit or close a whole application",
    "click": "click, press, select, check, or choose something visible on screen",
    "type_text": "type, write, enter, or dictate text into the current field",
    "search_web": "search the web or google for something",
    "open_url": "go to or open a website address",
    "press_key": "press a keyboard key or shortcut such as enter, escape, tab, copy, paste, save",
    "scroll_down": "scroll or page down",
    "scroll_up": "scroll or page up",
    "open_folder": "open a folder such as downloads, documents, or desktop",
    "go_back": "go back to the previous page or screen",
    "new_tab": "open a new tab",
    "close_tab": "close the current tab or window",
    "undo": "undo or revert the last thing that was done",
    "confirm": "yes, confirm, go ahead, or do it, answering a pending question",
    "cancel": "no, cancel, stop, or never mind",
    "wait": "the words so far are not yet enough to know which action is meant",
    "none": "not a command for the computer: talking to someone else, thinking aloud, or chit-chat",
}

# Intents that carry a free-text payload taken verbatim from the transcript.
PAYLOAD_INTENTS = frozenset({"type_text", "search_web"})
# Intents that need an on-screen target element.
TARGET_INTENTS = frozenset({"click"})
NON_ACTIONS = frozenset({"wait", "none"})

KEYS: dict[str, str] = {
    "return": "enter or return key",
    "escape": "escape key",
    "tab": "tab key",
    "space": "space bar",
    "delete": "delete or backspace key",
    "up": "up arrow",
    "down": "down arrow",
    "left": "left arrow",
    "right": "right arrow",
    "page_down": "page down key",
    "page_up": "page up key",
    "cmd+a": "select all",
    "cmd+c": "copy",
    "cmd+v": "paste",
    "cmd+x": "cut",
    "cmd+z": "undo",
    "cmd+shift+z": "redo",
    "cmd+s": "save",
    "cmd+f": "find",
    "cmd+n": "new window or new document",
    "cmd+r": "reload or refresh",
    "none": "no key is mentioned",
}

FOLDERS: dict[str, str] = {
    "desktop": "Desktop folder",
    "documents": "Documents folder",
    "downloads": "Downloads folder",
    "applications": "Applications folder",
    "home": "home folder",
    "pictures": "Pictures folder",
    "music": "Music folder",
    "movies": "Movies folder",
    "none": "no folder is mentioned",
}

FOLDER_PATHS: dict[str, str] = {
    "desktop": "~/Desktop",
    "documents": "~/Documents",
    "downloads": "~/Downloads",
    "applications": "/Applications",
    "home": "~",
    "pictures": "~/Pictures",
    "music": "~/Music",
    "movies": "~/Movies",
}

SCROLL_LEVELS: list[str] = [
    "a little, a few lines",
    "about one page or screen",
    "all the way to the top or bottom",
]

# Words that make an action HIGH risk when they appear in the target label, payload or key.
RISK_WORDS = (
    "delete", "remove", "trash", "erase", "empty", "format", "wipe", "destroy", "discard",
    "send", "submit", "post", "publish", "share", "reply all", "forward",
    "pay", "buy", "purchase", "order", "checkout", "check out", "subscribe", "transfer",
    "sign out", "log out", "logout", "unsubscribe", "uninstall", "reset", "overwrite", "replace all",
    "accept", "agree", "allow", "approve", "install", "confirm", "quit", "close",
)
RISK_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in RISK_WORDS) + r")\b", re.IGNORECASE)

# Apps the harness never acts inside unless the user allow-lists them (substring of the app name or
# bundle id, case-insensitive). Terminals and apps that run what is typed into them (a typed line +
# "press enter" / cmd+r executes code), password managers, and money apps.
DENY_APP_PATTERNS = (
    "system settings", "system preferences", "keychain",
    # terminals
    "terminal", "iterm", "warp", "ghostty", "kitty", "alacritty", "wezterm", "tabby", "termius",
    # agents, script runners and consoles
    "script editor", "scripteditor", "automator", "console",
    # password managers
    "1password", "bitwarden", "lastpass", "dashlane", "passwords", "keepass", "keeper", "proton pass",
    "protonpass", "nordpass", "enpass", "roboform", "strongbox", "secrets",
    # money
    "wallet", "bank", "trading", "robinhood", "coinbase", "binance", "kraken",
)
# Short names that would match too much as substrings ("hyper" in "HyperDock", "code" in many
# apps): denied only as the whole app name (or bundle id).
DENY_APP_NAMES = frozenset({
    "hyper", "co.zeit.hyper", "rio", "com.raphaelamorim.rio", "claude", "com.anthropic.claudefordesktop", "code", "visual studio code",
    "com.microsoft.vscode", "cursor", "com.todesktop.230313mzl4w4u92", "windsurf", "zed",
})

SECURE_FIELD_RE = re.compile(r"password|passcode|\bpin\b|cvv|cvc|card number|security code|\bssn\b|social security", re.I)

FILLERS = frozenset({"um", "uh", "uhm", "erm", "hmm", "like", "please", "okay", "ok", "so", "hey", "just"})

# Word that separates chained commands ("open notes and type hello").
CHAIN_WORDS = ("and then", "then", "and", "also", "after that")

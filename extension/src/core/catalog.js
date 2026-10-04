// Closed vocabularies (port of jev_local/harness/catalog.py). These strings are MODEL INPUT: the model
// was trained on exactly these descriptions, so do not edit them.

export const INTENTS = {
  open_app: "open, launch, start, or switch to an application",
  quit_app: "quit or close a whole application",
  click: "click, press, select, check, or choose something visible on screen",
  type_text: "type, write, enter, or dictate text into the current field",
  search_web: "search the web or google for something",
  open_url: "go to or open a website address",
  press_key: "press a keyboard key or shortcut such as enter, escape, tab, copy, paste, save",
  scroll_down: "scroll or page down",
  scroll_up: "scroll or page up",
  open_folder: "open a folder such as downloads, documents, or desktop",
  go_back: "go back to the previous page or screen",
  new_tab: "open a new tab",
  close_tab: "close the current tab or window",
  undo: "undo or revert the last thing that was done",
  confirm: "yes, confirm, go ahead, or do it, answering a pending question",
  cancel: "no, cancel, stop, or never mind",
  wait: "the words so far are not yet enough to know which action is meant",
  none: "not a command for the computer: talking to someone else, thinking aloud, or chit-chat",
};

// Browser-only intents GenClass appends to the trained set. Options are scored independently of each other
// (block-masked), so adding one does not change how the trained options are scored, only the softmax total.
export const BROWSER_EXTRA_INTENTS = {
  go_forward: "go forward to the next page",
};

export const PAYLOAD_INTENTS = new Set(["type_text", "search_web"]);
export const TARGET_INTENTS = new Set(["click"]);
export const NON_ACTIONS = new Set(["wait", "none"]);

export const KEYS = {
  return: "enter or return key",
  escape: "escape key",
  tab: "tab key",
  space: "space bar",
  delete: "delete or backspace key",
  up: "up arrow",
  down: "down arrow",
  left: "left arrow",
  right: "right arrow",
  page_down: "page down key",
  page_up: "page up key",
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
  none: "no key is mentioned",
};

export const FOLDERS = {
  desktop: "Desktop folder",
  documents: "Documents folder",
  downloads: "Downloads folder",
  applications: "Applications folder",
  home: "home folder",
  pictures: "Pictures folder",
  music: "Music folder",
  movies: "Movies folder",
  none: "no folder is mentioned",
};

export const FOLDER_PATHS = {
  desktop: "~/Desktop",
  documents: "~/Documents",
  downloads: "~/Downloads",
  applications: "/Applications",
  home: "~",
  pictures: "~/Pictures",
  music: "~/Music",
  movies: "~/Movies",
};

export const SCROLL_LEVELS = [
  "a little, a few lines",
  "about one page or screen",
  "all the way to the top or bottom",
];

export const RISK_WORDS = [
  "delete", "remove", "trash", "erase", "empty", "format", "wipe", "destroy", "discard",
  "send", "submit", "post", "publish", "share", "reply all", "forward",
  "pay", "buy", "purchase", "order", "checkout", "check out", "subscribe", "transfer",
  "sign out", "log out", "logout", "unsubscribe", "uninstall", "reset", "overwrite", "replace all",
  "accept", "agree", "allow", "approve", "install", "confirm", "quit", "close",
];

export const reEscape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const RISK_RE = new RegExp("\\b(" + RISK_WORDS.map(reEscape).join("|") + ")\\b", "i");

export const DENY_APP_PATTERNS = [
  "system settings", "system preferences", "keychain",
  "terminal", "iterm", "warp", "ghostty", "kitty", "alacritty", "wezterm", "tabby", "termius",
  "script editor", "scripteditor", "automator", "console",
  "1password", "bitwarden", "lastpass", "dashlane", "passwords", "keepass", "keeper", "proton pass",
  "protonpass", "nordpass", "enpass", "roboform", "strongbox", "secrets",
  "wallet", "bank", "trading", "robinhood", "coinbase", "binance", "kraken",
];
export const DENY_APP_NAMES = new Set([
  "hyper", "co.zeit.hyper", "rio", "com.raphaelamorim.rio", "claude", "com.anthropic.claudefordesktop", "code",
  "visual studio code", "com.microsoft.vscode", "cursor", "com.todesktop.230313mzl4w4u92", "windsurf", "zed",
]);

export const SECURE_FIELD_RE =
  /password|passcode|\bpin\b|cvv|cvc|card number|security code|\bssn\b|social security/i;

export const FILLERS = new Set(["um", "uh", "uhm", "erm", "hmm", "like", "please", "okay", "ok", "so", "hey", "just"]);
// Python iterates a frozenset in hash order; only the membership and the longest-first regex order matter.
export const FILLERS_LIST = [...FILLERS];

export const CHAIN_WORDS = ["and then", "then", "and", "also", "after that"];

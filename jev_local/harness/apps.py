"""Installed and running macOS applications, by display name.

The harness offers these names as the options of the `app` question, and the executor maps the
name the model picked back to a bundle path. Both directions go through this module so a name
the model sees is always a name `app_path` can open.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

APP_DIRS = (
    "/Applications",
    "/System/Applications",
    "/System/Applications/Utilities",
    "~/Applications",
)
# Apps that live outside the standard folders but that people ask for by name.
EXTRA_APPS = ("/System/Library/CoreServices/Finder.app",)
# Plain folders inside an app dir ("Utilities", "RealVNC", "Chrome Apps.localized") are scanned one
# level deep, because vendors put their apps there.
_SUBDIR_DEPTH = 1
_CACHE_TTL_S = 300.0

_lock = threading.Lock()
_index: dict[str, str] | None = None  # casefolded name -> bundle path
_names: list[str] = []
_built_at = 0.0


@dataclass(frozen=True)
class RunningApp:
    name: str
    bundle_id: str
    pid: int
    path: str | None
    active: bool


def _display_name(path: str) -> str:
    stem = os.path.basename(path)[: -len(".app")]
    try:
        from AppKit import NSFileManager

        name = str(NSFileManager.defaultManager().displayNameAtPath_(path) or "")
    except Exception:  # AppKit missing or the path vanished mid-scan
        name = ""
    name = name[: -len(".app")] if name.endswith(".app") else name
    return name.strip() or stem


def _scan_dir(d: str, depth: int, out: list[str]) -> None:
    try:
        entries = sorted(os.scandir(d), key=lambda e: e.name.lower())
    except OSError:
        return
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.name.endswith(".app"):
            out.append(e.path)
        elif depth > 0 and e.is_dir(follow_symlinks=False):
            _scan_dir(e.path, depth - 1, out)


def _build() -> tuple[dict[str, str], list[str]]:
    paths: list[str] = []
    for d in APP_DIRS:
        _scan_dir(os.path.expanduser(d), _SUBDIR_DEPTH, paths)
    paths += [p for p in EXTRA_APPS if os.path.isdir(p)]
    index: dict[str, str] = {}
    names: list[str] = []
    for p in paths:
        name = _display_name(p)
        key = name.casefold()
        if key not in index:  # first dir wins: /Applications shadows ~/Applications duplicates
            index[key] = p
            names.append(name)
        # Also accept the file name ("VoiceMemos") when it differs from the display name.
        stem = os.path.basename(p)[: -len(".app")].casefold()
        index.setdefault(stem, p)
    return index, names


_refreshing = False


def _rebuild() -> None:
    global _index, _names, _built_at, _refreshing
    try:
        index, names = _build()
        with _lock:
            _index, _names, _built_at = index, names, time.monotonic()
    finally:
        _refreshing = False


def _ensure(refresh: bool = False) -> dict[str, str]:
    global _refreshing
    with _lock:
        have = _index is not None
        expired = time.monotonic() - _built_at > _CACHE_TTL_S
    if refresh or not have:
        _rebuild()  # the first scan is synchronous (~0.1-0.7 s); callers need an answer
    elif expired and not _refreshing:
        # Serve the stale list and rescan off-thread, so a decision never waits on the disk.
        _refreshing = True
        threading.Thread(target=_rebuild, name="jev-apps-rescan", daemon=True).start()
    with _lock:
        return _index or {}


def installed_apps(refresh: bool = False) -> list[str]:
    """Display names of installed apps (cached for 5 minutes), in folder order."""
    _ensure(refresh)
    return list(_names)


def running(regular_only: bool = True) -> list[RunningApp]:
    """Running apps, frontmost first. Regular = has a Dock icon (activation policy 0)."""
    from AppKit import NSWorkspace

    ws = NSWorkspace.sharedWorkspace()
    out: list[RunningApp] = []
    for a in ws.runningApplications():
        if regular_only and a.activationPolicy() != 0:
            continue
        url = a.bundleURL()
        out.append(
            RunningApp(
                name=str(a.localizedName() or ""),
                bundle_id=str(a.bundleIdentifier() or ""),
                pid=int(a.processIdentifier()),
                path=str(url.path()) if url is not None else None,
                active=bool(a.isActive()),
            )
        )
    # stable sort: the active app first, the rest in launch order
    out.sort(key=lambda r: not r.active)
    return out


def running_apps() -> list[str]:
    """Names of running regular apps, frontmost first, without duplicates."""
    return list(dict.fromkeys(r.name for r in running() if r.name))


def app_path(name: str) -> str | None:
    """Bundle path for an app display name (case-insensitive), or None.

    Order: installed index, running apps, then LaunchServices. No fuzzy matching: the model picks
    names from the options we built from these same sources, and a fuzzy guess could open the
    wrong app.
    """
    key = name.strip().removesuffix(".app").strip().casefold()
    if not key or key == "none":
        return None
    hit = _ensure().get(key)
    if hit:
        return hit
    try:
        for r in running(regular_only=False):
            if r.path and (r.name.casefold() == key or r.bundle_id.casefold() == key):
                return r.path
        from AppKit import NSWorkspace

        ws = NSWorkspace.sharedWorkspace()
        p = ws.fullPathForApplication_(name.strip())
        if p:
            return str(p)
        if "." in key:  # a bundle id such as com.apple.Notes
            url = ws.URLForApplicationWithBundleIdentifier_(name.strip())
            if url is not None:
                return str(url.path())
    except Exception:
        return None
    return None


def bundle_id(path: str) -> str | None:
    try:
        from AppKit import NSBundle

        b = NSBundle.bundleWithPath_(path)
        bid = b.bundleIdentifier() if b is not None else None
        return str(bid) if bid else None
    except Exception:
        return None


def find_running(name: str) -> RunningApp | None:
    """The running app with this display name, bundle id or bundle path."""
    key = name.strip().removesuffix(".app").strip().casefold()
    path = app_path(name)
    for r in running(regular_only=False):
        if r.name.casefold() == key or r.bundle_id.casefold() == key or (path and r.path == path):
            return r
    return None

from __future__ import annotations

import os

import pytest

from jev_local.harness import apps


def test_scan_dir_finds_apps_one_level_deep(tmp_path):
    (tmp_path / "Foo.app").mkdir()
    (tmp_path / "Vendor" / "Bar.app").mkdir(parents=True)
    (tmp_path / "Vendor" / "Deeper" / "Baz.app").mkdir(parents=True)
    (tmp_path / ".Hidden.app").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    found: list[str] = []
    apps._scan_dir(str(tmp_path), 1, found)
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["Bar.app", "Foo.app"]


def test_build_index_dedupes_and_accepts_file_names(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "Notes.app").mkdir(parents=True)
    (a / "VoiceMemos.app").mkdir(parents=True)
    (b / "Notes.app").mkdir(parents=True)  # shadowed by the first dir
    monkeypatch.setattr(apps, "APP_DIRS", (str(a), str(b)))
    monkeypatch.setattr(apps, "EXTRA_APPS", ())
    monkeypatch.setattr(apps, "_display_name", lambda p: {"VoiceMemos": "Voice Memos"}.get(
        os.path.basename(p)[:-4], os.path.basename(p)[:-4]))
    index, names = apps._build()
    assert names == ["Notes", "Voice Memos"]
    assert index["notes"] == str(a / "Notes.app")
    assert index["voice memos"] == index["voicememos"] == str(a / "VoiceMemos.app")


def test_app_path_rejects_empty_and_none():
    assert apps.app_path("") is None
    assert apps.app_path("none") is None
    assert apps.app_path("   ") is None


@pytest.mark.macos
def test_installed_and_running_on_this_mac():
    names = apps.installed_apps()
    assert "Finder" in names  # added from CoreServices
    assert len(names) == len(set(n.casefold() for n in names))
    assert apps.installed_apps() == names  # cached
    assert apps.app_path("finder") == "/System/Library/CoreServices/Finder.app"
    assert apps.app_path("FINDER.app") == "/System/Library/CoreServices/Finder.app"
    assert apps.app_path("com.apple.finder") == "/System/Library/CoreServices/Finder.app"
    assert apps.app_path("Definitely Not An App 9431") is None
    running = apps.running_apps()
    assert "Finder" in running
    assert len(running) == len(set(running))
    r = apps.find_running("finder")
    assert r is not None and r.bundle_id == "com.apple.finder" and r.pid > 0
    assert apps.bundle_id("/System/Library/CoreServices/Finder.app") == "com.apple.finder"

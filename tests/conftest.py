"""Shared pytest configuration.

`macos`-marked tests need macOS (Accessibility / AppKit / installed apps); on any other platform they are skipped
instead of failing, so the full suite (`pytest -q tests`) is runnable on the Linux VMs (CONTRACT: heavy tests
run on VM train, pure-Python tests on the Mac).
"""

from __future__ import annotations

import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform == "darwin":
        return
    skip = pytest.mark.skip(reason="macos-marked test: needs macOS Accessibility / AppKit")
    for item in items:
        if "macos" in item.keywords:
            item.add_marker(skip)

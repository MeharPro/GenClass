"""`jev-local` command line. Only the subcommands that exist so far are wired up.

    jev-local demo [...]     voice computer-use demo (jev_local.demo; see docs/DEMO.md)
    jev-local serve [...]    the Jev-compatible API server on 127.0.0.1
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "demo":
        from jev_local.demo import main as demo_main

        return demo_main(rest)
    if cmd == "serve":
        import argparse

        from jev_local.server.app import serve

        ap = argparse.ArgumentParser(prog="jev-local serve")
        ap.add_argument("--port", type=int, default=8765)
        a = ap.parse_args(rest)
        serve(port=a.port)
        return 0
    print(f"unknown command {cmd!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

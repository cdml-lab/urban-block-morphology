"""Command line: blockmorph run <config.yaml> | blockmorph dashboard <output_dir>."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="blockmorph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the analysis described by a configuration file")
    r.add_argument("config")
    r.add_argument("--areas", nargs="*", help="only these area ids")
    r.add_argument("--no-dashboard", action="store_true")
    d = sub.add_parser("dashboard", help="build the HTML dashboard of an output folder")
    d.add_argument("output_dir")
    d.add_argument("--title", default=None)
    d.add_argument("--colours", default=None, help="JSON {model: {type: hex}}")
    d.add_argument("--no-building-forms", action="store_true",
                   help="leave per-building form predictions out of the page")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        out = run(a.config, a.areas)
        if not a.no_dashboard:
            from .dashboard import build
            print("dashboard:", build(out))
    else:
        from .dashboard import build
        print("dashboard:", build(a.output_dir, a.title, a.colours, not a.no_building_forms))


if __name__ == "__main__":
    sys.exit(main())

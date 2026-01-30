from __future__ import annotations

import argparse
import importlib
import pkgutil
import benchmarks
from tqdm import tqdm


def list_benches() -> list[str]:
    names = []
    for m in pkgutil.iter_modules(benchmarks.__path__):
        if m.name in {"run", "__init__", "common"}:
            continue
        names.append(m.name)
    return sorted(names)


def run_one(name: str, rest: list[str]) -> int:
    mod = importlib.import_module(f"benchmarks.{name}")
    if not hasattr(mod, "main"):
        raise SystemExit(f"benchmarks.{name} has no main(argv) function")
    return int(mod.main(rest) or 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bench", nargs="?", default="all", help="benchmark name or 'all'")
    p.add_argument("--list", action="store_true", help="list available benchmarks")
    args, rest = p.parse_known_args(argv)

    if args.list:
        for n in list_benches():
            print(n)
        return 0

    if args.bench == "all":
        rc = 0
        for n in list_benches():
            print(f"\n=== {n} ===")
            rc |= run_one(n, rest)
        return rc

    return run_one(args.bench, rest)


if __name__ == "__main__":
    raise SystemExit(main())
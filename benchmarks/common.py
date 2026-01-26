import argparse
from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    T: float = 1.0
    steps: int = 10000
    N: int = 3
    samples: int = 10
    gap: int = 1
    repeats: int = 10
    warmup: int = 2
    seed: int = 0

DEFAULTS = Defaults()

def add_common_args(p: argparse.ArgumentParser, d: Defaults = DEFAULTS) -> None:
    p.add_argument("--T", type=float, default=d.T)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--N", type=int, default=d.N)
    p.add_argument("--samples", type=int, default=d.samples)
    p.add_argument("--gap", type=int, default=d.gap)
    p.add_argument("--repeats", type=int, default=d.repeats)
    p.add_argument("--warmup", type=int, default=d.warmup)
    p.add_argument("--seed", type=int, default=d.seed)
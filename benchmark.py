"""Benchmark our PIPs model's forward pass.

For comparison, a PyTorch version is implemented here:
    https://github.com/brentyi/pips/blob/main/benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import numpy as onp
import tyro
from jax import numpy as jnp

from pips_jax.model import Pips


class Timer:
    """Timing helper.

    Usage:
        with Timer() as t:
            ... # Do something.
        print(t.elapsed) # seconds
    """

    elapsed: float

    def __enter__(self) -> Timer:
        self._start = time.time()
        return self

    def __exit__(self, *_unused_args):
        self.elapsed = time.time() - self._start


@dataclass(frozen=True)
class BenchmarkConfig:
    b: int = 1
    n: int = 256
    h: int = 360
    w: int = 640
    iters: int = 6
    num_forward_trials: int = 20


def benchmark(
    config: BenchmarkConfig,
    model: Pips = Pips(stride=4),
) -> None:
    print(config)
    print(model)
    print(">>>>")

    # Initialize parameters.
    print("Initializing parameters...")
    with Timer() as t:
        params = model.init_params(seed=0)
        jax.block_until_ready(params)
    print(f"\t{t.elapsed} seconds")

    # Set up inputs.
    rgbs = jnp.zeros((config.b, model.S, config.h, config.w, 3), dtype=jnp.uint8)
    xys = jnp.zeros((config.b, config.n, 2), dtype=jnp.float32)

    # Compile forward pass.
    model_apply = jax.jit(
        lambda rgbs, xys: model.apply(
            params, xys=xys, rgbs=rgbs, iters=config.iters, train=False
        )
    )
    print("Compiling forward pass...")
    with Timer() as t:
        model_apply_compiled = model_apply.lower(rgbs, xys).compile()
    print(f"\t{t.elapsed} seconds")

    # Run network!
    print("Forward pass...")
    runtimes = []
    for _ in range(config.num_forward_trials):
        with Timer() as t:
            jax.block_until_ready(model_apply_compiled(rgbs, xys))
        runtimes.append(t.elapsed)

    mean = onp.mean(runtimes)
    std_err = onp.std(runtimes, ddof=1) / onp.sqrt(len(runtimes))
    print(f"\t{mean:.05f} ± {std_err:.05f} seconds")


if __name__ == "__main__":
    tyro.cli(benchmark, description=__doc__)

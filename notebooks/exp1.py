#!/usr/bin/env python3
# %% [markdown]
# Exp1 notebook (script-mode)
#
# Demonstrates computing volume, EHZ capacity, and Minkowski billiard cycles
# for simple 2x2 product polytopes. Figures are saved to `artefacts/`.

# %%
from __future__ import annotations

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from viterbo.exp1.capacity_ehz import capacity_and_cycle
from viterbo.exp1.examples import regular_ngon2d
from viterbo.exp1.polytopes import lagrangian_product
from viterbo.exp1.volume import volume

os.makedirs("artefacts", exist_ok=True)

# %%
# Build a 4D product: square x hexagon
left = regular_ngon2d(4)
right = regular_ngon2d(6)
prod = lagrangian_product(left, right)

# %%
# Volume via fast estimator (from vertices)
vol = float(volume(prod, method="fast"))

# %%
# Minkowski billiard capacity and cycle (default geometry == table)
cap, cycle_pts = capacity_and_cycle(prod, method="minkowski_reference")

# %%
# Plot 2D projections of the cycle onto left/right coordinate blocks
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
V1 = left.v
V2 = right.v

ax1.plot(jnp.concatenate([V1[:, 0], V1[:1, 0]]), jnp.concatenate([V1[:, 1], V1[:1, 1]]), "k-")
ax1.set_aspect("equal")
ax1.set_title("Left factor (proj)")
ax1.plot(
    jnp.concatenate([cycle_pts[:, 0], cycle_pts[:1, 0]]),
    jnp.concatenate([cycle_pts[:, 1], cycle_pts[:1, 1]]),
    "r-",
)

ax2.plot(jnp.concatenate([V2[:, 0], V2[:1, 0]]), jnp.concatenate([V2[:, 1], V2[:1, 1]]), "k-")
ax2.set_aspect("equal")
ax2.set_title("Right factor (proj)")
ax2.plot(
    jnp.concatenate([cycle_pts[:, 2], cycle_pts[:1, 2]]),
    jnp.concatenate([cycle_pts[:, 3], cycle_pts[:1, 3]]),
    "r-",
)

fig.tight_layout()
fig.savefig("artefacts/exp1_minkowski_cycle_square_x_hexagon.png", dpi=150)
plt.close(fig)

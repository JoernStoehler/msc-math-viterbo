# %% [markdown]
"""\
# Minimal Action Cycle on the Viterbo Counterexample (Placeholder)

This notebook outlines the plots we want to build for the minimal action cycle on the
standard Viterbo counterexample. We keep all dependencies abstract: nothing here should
be treated as a committed API, and most helper functions referenced below do not exist
yet. The goal is to document the workflow and expected data requirements so we can
implement the supporting library pieces over the weekend.
"""

# %%
from __future__ import annotations

# NOTE: These imports are aspirational. Replace them with the real modules once the
# plotting helpers land in the codebase.
import matplotlib.pyplot as plt
import numpy as np

from viterbo.visualization import four_d_cycle_projection  # placeholder
from viterbo.datasets import load_counterexample_cycle  # placeholder


# %% [markdown]
"""\
## 1. Load the counterexample trajectory

We assume a helper that returns the discrete trajectory with both the $p$ and $q$
components, possibly as a structured array or a small dataclass.
"""

# %%
cycle = load_counterexample_cycle(
    name="viterbo_counterexample_default",  # placeholder signature
    sample_density="weekend-highres",  # placeholder: we might allow presets
)

print("Loaded cycle stats (placeholder):", cycle)


# %% [markdown]
"""\
## 2. Prepare the 2D projections

We expect to slice the 4D trajectory into two 2D components. The helper below should
encapsulate the projection logic once we implement it.
"""

# %%
projection = four_d_cycle_projection(
    cycle,
    scheme="piecewise-linear",
    normalize=True,
)  # placeholder return type

left_component = projection.left  # placeholder attribute
right_component = projection.right  # placeholder attribute


# %% [markdown]
"""\
## 3. Plot the piecewise-linear trajectory with matching gradients

Each segment should have a consistent gradient between the $p$ and $q$ components so
that the viewer can track corresponding points. We will likely need a custom colormap to
keep both plots synchronised.
"""

# %%
fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(10, 4))

# Placeholder: assume the projection object has a `segments` iterator that yields both
# components, along with IDs for the labelled points.
for segment in projection.iter_segments():  # placeholder method
    color = plt.cm.viridis(segment.t_normalized)  # placeholder attribute
    ax_p.plot(segment.p[:, 0], segment.p[:, 1], color=color)
    ax_q.plot(segment.q[:, 0], segment.q[:, 1], color=color)

    for point in segment.points:  # placeholder attribute
        label = f"{point.index}"  # placeholder attribute
        ax_p.text(point.p[0], point.p[1], label, color=color)
        ax_q.text(point.q[0], point.q[1], label, color=color)

ax_p.set_title("p-component (placeholder)")
ax_q.set_title("q-component (placeholder)")

plt.suptitle("Minimal action cycle projection (placeholder)")
plt.show()


# %% [markdown]
"""\
## 4. Checklist and follow-ups

- [ ] Finalise the data loader for counterexample trajectories.
- [ ] Define the projection helper that keeps the $p$ and $q$ components aligned.
- [ ] Implement consistent colour gradients shared across subplots.
- [ ] Add annotations for action values at key points if useful.
"""

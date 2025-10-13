---
title: "Future: probe connectivity of {sys <> 1}"
created: 2025-10-13
status: idea
owner: TBD
priority: backlog
labels: [future]
---

## Summary

We can check for whether {sys>,<1} is connected, i.e. whether any pair of examples can be transformed into each other. Potentially very simple transforms work and don't cross sys=1. Relatedly, we can look for local minima and maxima of sys, and the polytopes that traject into them, given some optimizer (e.g. GD with fixed vertices, GD + random extra vertices on facets, etc).

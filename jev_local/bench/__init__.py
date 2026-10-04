"""jevbench v1: the held-out yardstick for jev-local v2 (docs/research/v2/PLAN.md §1).

`registry` is the single source of truth for which datasets are benchmark test/dev sets and which
training sources must be excluded (decontamination stage 1). It is pure stdlib so every data agent can
import it on any machine.
"""

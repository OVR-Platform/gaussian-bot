# 0001. Pluggable renderer and VLM protocols

- Status: accepted
- Date: 2026-06-18

## Context

We are starting an R&D project (robot navigation inside a Gaussian Splat via
VLM perception) and have **not** yet decided on a concrete renderer (gsplat?
web viewer?) nor a concrete VLM (vLLM? OpenAI API?). We expect to experiment
with several of each.

Forcing an early choice would either:

- couple the navigation logic to a specific backend and force rewrites when we
  swap, or
- block all progress until we've evaluated options.

Neither is acceptable for research velocity.

## Decision

Define both external dependencies — **renderer** and **VLM** — as small
`typing.Protocol` interfaces in `gaussian_robot.render.base.Renderer` and
`gaussian_robot.vlm.client.VLMClient`, plus the shared value types they produce
(`RenderResult`, `VLMResponse`).

The navigation and orchestration code (`nav.Navigator`, `nav.Robot`) depends
**only** on these protocols, never on concrete implementations.

Concrete backends (a gsplat adapter, an OpenAI vLLM client, etc.) will live in
their own modules and be passed in via dependency injection.

## Consequences

- ✅ We can build and test the core loop today using fakes/doubles without
  committing to heavy deps (torch, gsplat) — they're optional extras.
- ✅ Swapping a backend = writing one new class, not editing the planner.
- ✅ Clear seam for future work (e.g. a learned policy replacing the VLM).
- ⚠️ Slight indirection: every new backend must honour the protocol's shapes.
  We mitigate with `runtime_checkable` and shape validation in `__post_init__`.
- ⚠️ The protocol may need to evolve (e.g. depth support, structured actions).
  When it does, we bump it and add an ADR noting the change.

# dialectical-games

Game-agnostic dialectical argumentation engine core, extracted from
[dialectical-checkers](https://github.com/ctoth/dialectical-checkers) and
[dialectical-chess](https://github.com/ctoth/dialectical-chess).

A game built on this core supplies a **cartridge** (board substrate, move
generation, witness producers, static prior, value vocabulary, search backend,
protocol harness); the core supplies the argumentation machinery (typed
evidence, the crisp Dung-AF filter, the graded `doxa.BipolarOpinionGraph`
layer, the lexicographic FACT-then-graded decider, the engine orchestrator,
the loss-mining turning-point algorithm).

## Status

Phase 2 of the core extraction. Scope of this initial release: the foundational
typed-evidence taxonomy — `Tier`, `Value`, `CriticalQuestion`,
`ArgumentEvidence`, `to_argument_evidence`. The orchestrator, crisp/graded
layers, decider, and loss-mining algorithm remain in the source engines until
their cartridge seams have been cut; subsequent extraction cycles will move
them here once that seam work is done.

## Dependencies

- [`formal-argumentation`](https://github.com/ctoth/argumentation) — the Dung
  argumentation-framework library, supplying `ArgumentationFramework` and
  `grounded_extension`.
- [`doxa`](https://github.com/ctoth/doxa) — subjective-logic opinions and the
  bipolar opinion-graph evaluator the graded layer uses.

## Install

This package is intended to be a **git-pinned** dependency of the game
engines that build on it — never a path dependency. From a game repo's
`pyproject.toml`:

```toml
dependencies = [
    "dialectical-games @ git+https://github.com/ctoth/dialectical-games.git@<commit-sha>",
]
```

## License

MIT.

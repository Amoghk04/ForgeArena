---
description: "Add a new grader checkpoint or modify an existing one. Use when: extending the explanation grader rubric, adjusting checkpoint weights, or adding a new grader component."
---

# Add or Modify a Grader Checkpoint

## Context

The explanation grader (`src/forge_arena/graders/explanation.py`) uses a four-checkpoint rubric. Any change to checkpoint weights must be reflected in `docs/reward_design.md` and the composite formula in `composite.py`.

## Constraints

1. **No LLM calls.** All graders must be fully deterministic. Use string matching, regex, ROUGE-L, or pre-trained sklearn classifiers only.
2. **Checkpoint weights must sum to ≤ 1.0.** The hallucination penalty is a deduction, not a weight.
3. **50 ms execution budget.** Each grader call including all checkpoints must complete in under 50 ms.
4. **Unit test coverage required.** Every checkpoint must have tests for full credit, zero credit, and edge cases (see `tests/test_graders.py`).

## Current Checkpoint Summary

| Checkpoint | Weight | Method |
|---|---|---|
| Type naming | 0.40 | `any(phrase in explanation.lower() for phrase in corruption_type.trigger_vocab)` |
| Citation specificity | 0.30 | Quotation detection + token overlap with `worker_output` |
| Mechanism proposal | 0.20 | Binary sklearn classifier (`graders/data/mechanism_clf.pkl`) |
| Hallucination penalty | −0.10 max | ROUGE-1 overlap for quoted claims vs `worker_output` |

## Steps to Add a New Checkpoint

1. Define the checkpoint as a method `_score_<name>(self, explanation: str, context: GraderContext) -> float` on `ExplanationGrader`.
2. Add it to the `score()` method with its weight.
3. Update `ExplanationScore` Pydantic model in `models/rewards.py` to include the new component.
4. Update the composite formula in `composite.py` if the new checkpoint affects the total reward.
5. Add unit tests in `tests/test_graders.py`.
6. Update `docs/reward_design.md` with the new checkpoint rationale.

## Steps to Change Checkpoint Weights

1. Update the constants in `explanation.py`.
2. Verify all weights still sum to ≤ 1.0.
3. Update `ExplanationScore` model defaults.
4. Update `docs/reward_design.md` — include the reason for the change.
5. Re-run the full test suite to catch any hardcoded expected values: `pytest tests/test_graders.py -v`.

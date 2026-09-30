# Methodology

## Gradient factorization

For one fully connected layer, let

- `S ∈ {0,1}^{M×L}` be the presynaptic binary spike matrix,
- `G ∈ R^{N×L}` be the temporal gradient factor,
- `L = B T` for mini-batch size `B` and temporal horizon `T`.

The observed gradients satisfy

\[
\nabla W = G S^\top,\qquad
\nabla b = G \mathbf{1}_L.
\]

Using bias augmentation,

\[
D=[\nabla W \mid \nabla b]
  = G\widetilde S^\top,
\]

with

\[
\widetilde S =
\begin{bmatrix}
S\\
\mathbf{1}_L^\top
\end{bmatrix}.
\]

## Stage I: algebraic candidate recovery

Stage I operates only on the observed FC gradients.

The implementation performs:

1. bias augmentation;
2. SVD-based row-space extraction;
3. coordinate compression using zero, one, and duplicate gradient columns;
4. guided MILP enumeration of binary vectors constrained to the recovered row space;
5. no-good cuts to prevent repeated candidate solutions.

Main tolerances:

- equality tolerance: `1e-8`;
- projection tolerance: `1e-7`;
- zero-column tolerance: `1e-12`.

Candidate enumeration is unique-vector based. Consequently, candidate-pool
coverage is reported over **distinct true binary patterns**, not temporal
multiplicity.

## Stage II: temporal ordering and batch separation

The fast configuration uses:

```text
target: x0
cover:  s1
```

For each candidate `x_t`, Stage II propagates only

```text
x_t → fc1 → LIF1 → predicted s1
```

and checks the predicted `s1` against the remaining recovered `s1`
multiset. A matched downstream pattern is consumed; backtracking restores it.

The sequential search recovers one complete length-`T` sequence at a time
and then continues to the next batch member.

Current search budget:

- maximum recursive sequence nodes: `2,000,000`;
- maximum complete sequence hypotheses retained per example: `5`.

A `node_limit` failure means the recursive sequence-node budget was exceeded.
A `search_exhausted` failure means no complete reconstruction was found under
the configured search/branching limits.

## Evaluation

For `B > 1`, reconstructed sequences are compared with the ground truth up
to a single global batch permutation using Hungarian assignment. Metrics
include exact reconstruction, bit accuracy, Hamming distance, precision,
recall, and F1.

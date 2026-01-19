# Tools Policy

This directory exists to make visible **only validated outcomes** — including boundary failures that sharpen falsifiability, constructiveness, or regime clarity of Dual-Frame Theory (DFT), without contaminating the canonical core theory chapters.

A “tool” in this repository means: a **diagnostic instrument** or **constructive bridge** that enforces scope boundaries, yields checkable invariants, and supports refusal logic.

## Admission rule (hard gate)

A new tool (or new item within an existing tool) may be added only if at least one of the following is true:

1. **Reproduction with fewer assumptions**  
   It reproduces a benchmarked behavior using strictly fewer, or strictly cleaner, assumptions than prior DFT work (no added knobs).

2. **Regime separation / boundary discovery**  
   It demonstrates a clear, checkable separation between regimes (e.g., diffusive vs long-memory envelopes) and provides explicit breakpoints.

3. **Falsification hooks / no-go constraints**  
   It introduces a diagnostic that can fail in a crisp way (e.g., higher cumulants persist beyond stated mixing bounds), with a documented interpretation of failure.

4. **Contract-bound reproducibility**  
   - It ships with a minimal reproducibility path and stable identifiers (inputs, outputs, and fixed audit artifacts).
   - Reproducibility requires regenerability from fixed inputs and code; large generated outputs may be excluded from version control if deterministically reproducible.

Items that are exploratory, speculative, or parameter-fishing do **not** belong here.

## Scope labeling (required)

Every tool directory must include a `README.md` that states:

- what success is being claimed,
- what is not being claimed,
- what would falsify the construction,
- and the minimal reproduction path.

## Canonical vs tools

- Canonical chapters (01–05) contain the **stable theory spine**.
- `tools/` contains **earned but not yet canonized** instruments.
- Promoting a tool into the canonical spine requires a separate editorial pass
  to remove scaffolding, freeze semantics, and align with chapter numbering.

## No hype clause

Tools must avoid promotional language. Any claim must be:

- bounded,
- testable,
- and linked to explicit artifacts or scripts.

## Versioning

Versioning is not permitted until all current work is fully checked in and verified.

Only after the repository state is stable and acknowledged may versioning actions be taken.

If a tool’s outputs or semantics then change materially, either:

- introduce a new versioned artifact ID (preferred), or
- create a new subdirectory version.

[![CI](https://github.com/hobbybaschtler/sciml-teleological-taxonomy/actions/workflows/ci.yml/badge.svg)](https://github.com/hobbybaschtler/sciml-teleological-taxonomy/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22671128.svg)](https://doi.org/10.5281/zenodo.22671128)

# A Teleological Taxonomy for SciML Architecture Selection

Reference implementation and machine-readable schema accompanying:

> J. Wiens and C. Reich, "From Descriptive Classification to Prescriptive
> Selection: A Goal-Oriented Taxonomy, Decision Protocol and Machine-Readable
> Schema for Scientific Machine Learning Architectures".

Extended version of:

> J. Wiens and C. Reich, "Scientific Machine Learning (SciML): A Teleological
> Taxonomy for Decision Support", in *Proceedings of the 16th International
> Conference on Simulation and Modeling Methodologies, Technologies and
> Applications (SIMULTECH 2026)*, SciTePress, Setúbal, Portugal.

---

## What this is

Choosing a SciML architecture is normally a matter of expert intuition, because
the reference classifications of the field organise models by *how* physical
knowledge enters the learner. Those attributes are properties of a finished
model, so they cannot be evaluated on an unsolved problem. This repository
contains a selection instrument built on the practitioner's epistemic goal
instead — an attribute of the task, observable before any modelling commitment.

It provides:

| Artefact | File | Article |
|---|---|---|
| Complete taxonomy schema (11 decision nodes, 16 leaves) | `taxonomy.yaml` | Listing 1, §8 |
| Task descriptor and constraint classes | `sciml_taxonomy/descriptor.py` | Def. 1, Def. 2 |
| Single-class selection with declared failure modes | `sciml_taxonomy/selection.py` | Algorithm 1, §5 |
| Composition across macro-classes | `sciml_taxonomy/selection.py` | Algorithm 2, §7.2 |
| Structural metrics | `sciml_taxonomy/structural.py` | Table 5, §9.2 |
| Case-study descriptors | `cases/cases.yaml` | Table 7, §9.4 |

Listing 1 in the article is an excerpt; `taxonomy.yaml` here is the complete
file it is drawn from.

## Quick start

```bash
pip install -e .
python -m sciml_taxonomy replay        # replay every case study of §9.4
```

Other commands:

```bash
python -m sciml_taxonomy replay case7  # one case
python -m sciml_taxonomy structural    # reproduce Table 5
python -m sciml_taxonomy validate      # check the schema
python -m sciml_taxonomy leaves        # list the corpus of Table 4
pytest                                 # full regression suite
```

`replay` prints, for each case, the node-by-node provenance trace and compares
the result against the outcome reported in Table 7. Every traversal in §9 can
therefore be checked rather than taken on the authors' word.

## Using it on your own problem

```python
from sciml_taxonomy import Constraints, DataRegime, Kappa, Knowledge, Task, Taxonomy, Unknown, select

task = Task(
    name="Subsurface state estimation",
    objective="Reconstruct a field from sparse boreholes for a drill/no-drill decision",
    unknowns=frozenset({Unknown.STATE}),
    knowledge=Knowledge(kappa=Kappa.COMPLETE),
    data=DataRegime(n=40, noise=0.15, moderate_state_dimension=True),
    constraints=Constraints(c_unc=True),
)

result = select(task, Taxonomy.load())
print(result.render())
```

The output is a candidate set together with a provenance trace, never a bare
recommendation. Three outcomes are possible, and the last two are declared
rather than hidden:

- **resolved** — one admissible family.
- **under_determined** — several survivors, reported with the criterion that
  failed to discriminate, rather than tie-broken arbitrarily.
- **relaxation_requested** — no leaf satisfies the stated constraints. The
  instrument names the binding constraints, names which leaves satisfy each one
  separately, proposes the weakest for relaxation, and flags for expert review.
  It does not return the nearest miss as though it were a recommendation.

## Two interpretive decisions

The article states the protocol in prose and pseudocode; making it executable
forced two readings to be fixed. Both are recorded here so that a reader can
disagree with them explicitly.

### Node N2.5 and Proposition 1

Proposition 1 shows that the driver attribute of Pateras et al. is not a priori
observable except when `kappa = none`. Node N2.5 nevertheless refines the
simulation branch by something driver-like. The schema states the criterion over
**operator availability and dynamical regime**, both properties of the task, and
not over the driver attribute itself. The operational test in Table 3 already
reads this way ("is a PDE/ODE statement available in the domain literature"), so
the implementation follows the test rather than the label. This is what keeps
`omega = 11/11` consistent with Proposition 1.

### Reading of Algorithm 1, line 13

Line 13 filters the candidate set by the dominant constraint `c*`. Taken
literally, filtering on `c*` alone would return a leaf that violates a second
asserted requirement — and Case 7 asserts two. Since §9.5 reports that Case 7
reaches the `C = empty` branch, the remaining asserted requirements must also
filter. The implementation therefore uses precedence to choose the branch and
applies every other asserted requirement as a further filter. Under the literal
reading, Case 7 would return `HNN_LNN` and the adversarial case would not be
adversarial.

## Repository layout

```
taxonomy.yaml              complete schema: 11 nodes, 16 leaves
cases/cases.yaml           task descriptors for §9.4 and the §7.3 worked example
sciml_taxonomy/
  descriptor.py            Definitions 1, 2, 4; precedence (2); Definition 5
  schema.py                loader and structural validation
  selection.py             Algorithms 1 and 2
  structural.py            Table 5
  cases.py                 case loading
  cli.py                   command-line interface
tests/                     regression suite
```

## Schema conventions

Each leaf carries four fields, and the split between the last two is
substantive rather than cosmetic (Remark 1):

- `recovers` — the mathematical object the training objective returns. This is
  what governs macro-class membership under the allocation rule of §6, not the
  data the family consumes or the community that produced it.
- `discharges` — which of the four **ranked requirements** (`c_guar`, `c_unc`,
  `c_lat`, `c_interp`) the family satisfies. An empty list is a claim, not a
  gap: PINNs discharge nothing, which is the point of Case 5.
- `requires` — **task conditions** that must hold for the family to apply.
  These include the two environment facts (`c_diff`, `c_disc`) and the
  knowledge level. They act as filters and are never traded against anything.
- `known_failure_modes` — documented pathologies. The schema validator rejects
  any leaf without at least one, because a selector that recommends without
  warning is worse than no selector.

Adding an architecture family means adding a leaf. No decision node changes,
which is the extendibility ending condition of Nickerson et al.

## Contributing

Corrections to `known_failure_modes` and `requires` are the most valuable
contributions, because those fields are what make the taxonomy falsifiable by
evidence rather than revisable only by opinion. A new family is a pull request
adding one leaf plus a regression case.

## Citation

See `CITATION.cff`. Please cite both the extended article and the SIMULTECH
conference version.

## Licence

MIT. See `LICENSE`.

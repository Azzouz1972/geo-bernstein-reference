# On the Sensitivity of Shock-Capturing PINN Benchmarks to Reference Resolution

Reproducibility repository for the paper
*"On the Sensitivity of Shock-Capturing PINN Benchmarks to Reference Resolution."*

## TL;DR

A geometry-aligned Bernstein collocation sampler beats uniform and
residual-adaptive PINN baselines on inviscid Burgers (2D, torus) on **all
metrics, all 10 seeds, by up to 26%, p ≈ 1e-3** — *when scored against a
first-order finite-volume reference*. The **same trained models**, scored
against a second-order (MUSCL) reference **on the same grid**, **lose every
metric by 19–43%**. The advantage correlates **+0.79** with the reference
front width: it exists only insofar as the reference is under-resolved.

This repo lets you reproduce that reversal from saved checkpoints.

## Key point: no PyTorch needed

`src/load_pt.py` reads the `.pt` checkpoints with a pure-NumPy unpickler, so
every number is reproducible with only NumPy + Matplotlib. No GPU, no DL runtime.

```bash
pip install -r requirements.txt
```

## Reproduce each result

| Command | Reproduces |
|---|---|
| `python src/step0.py` | Front-width diagnostic; validation vs first-order 160² (Table 2); reversal vs MUSCL; 6-reference battery (Table 3); clean same-grid control |
| `python src/r1r2.py`  | R1 (gain vs reference front width, corr +0.79) and R2 (per-seed diffusion→error, +0.60) |
| `python src/r1_fig.py` | Figure: `figures/r1_gain_vs_reference_width.png` |
| `python src/heatmaps_burgers.py` | Figure: Burgers signed-error map (diffuse halo) |
| `python src/heatmaps.py` | Figure: Euler signed-error maps (amplitude bias) |

All scripts expect the data layout in `data/` (paths near the top of each script
may need adjusting to `data/burgers/`).

## Data

- `data/burgers/` — `config.json` and 30 checkpoints (Vanilla, RAR,
  Geo-Bernstein × seeds 0–9). **Fully reproducible**: re-evaluate against any
  reference with `step0.py`.
- `data/euler_config3/` — saved seed-0 profiles (`profile_*_seed0.npz`),
  `metrics_all.csv`, and the FV reference for the two-front Euler case.

## Honest scope / limitations

- The Euler case (Sec. 4.2) is an **optimized Bernstein variant** (width 72,
  5 seeds) on a related setup, **not** the scalar Geo-Bernstein model of Table 2.
  Its figures are reproduced **from saved profiles**; full re-evaluation
  checkpoints for the inviscid two-front Euler run were not retained.
- Analysis concerns the **strong-form** residual only; no claim is made about
  weak-form / variational PINNs.
- The viscous regime is excluded (null result: all methods tied within seed
  noise).
- We make **no prevalence claim**: high-fidelity references (WENO, exact
  Riemann) are common; this repo demonstrates a mechanism and a detection test,
  not a survey.

## The differential test (the reusable takeaway)

Re-score fixed checkpoints against references of increasing fidelity and check
that the ranking is stable. It needs no retraining and would have caught this
artifact before submission. See `src/step0.py`.

## Citation

```bibtex
@misc{azzouz2026refartifact,
  title  = {On the Sensitivity of Shock-Capturing PINN Benchmarks to
            Reference Resolution},
  author = {Azzouz, Abdelhalim},
  year   = {2026},
  note   = {Preprint}
}
```

## License
MIT — see `LICENSE`.

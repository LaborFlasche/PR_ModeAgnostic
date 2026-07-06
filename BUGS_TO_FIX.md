# Bugs & Issues Found in Benchmark Results (2026-07-06)

Findings from analyzing `results_config-tree.csv`, `data_final/results_config-accuracy.csv`,
`data_final/results_config-dimensionality.csv`, and `results_config-neural-networks-RQ3.csv`.
A row counts as *failed* when its `shapley_values` are all-NaN (the runner writes these on
timeout `[SKIP]` or crash `[BUG]`, see `Benchmarking/runner.py:84-89`).

Overall status per sweep:

| Results CSV | Grid coverage | Failed rows | Verdict |
|---|---|---|---|
| `results_config-accuracy.csv` | 2160/2160 | 0 | ✅ usable (n_bg=50 half; see Bug 4) |
| `results_config-dimensionality.csv` | 864/864 | 0 | ✅ usable |
| `results_config-tree.csv` | 6174/6174 | 1071 (all fasttreeshap) | ⚠️ usable except fasttreeshap (Bug 5) and woodelf classification cells (Bug 1) |
| `results_config-neural-networks-RQ3.csv` (rerun 2026-07-06) | 27/27 cells | 80/243 | ⚠️ mlp/cnn_1d usable; transformer unusable (Bug 9) |

---

## Bug 1 — woodelf returns sign-flipped Shapley values for random_forest classifiers 🔴

**Symptom:** 147/630 `woodelf_interventional` order-1 cells have `relative_mae ≈ 2.0` and
`mean_sample_rho = −1.0` vs the oracle. Affected cells are exactly
`random_forest × {adult_census, gisette}` (the two classification datasets), every seed,
depth, and feature count. Same subset poisons `woodelf_path_dependent` and
`woodelf_interaction`, and inflates woodelf's median relative additivity gap (0.49 vs 0.15
for the other path-dependent backends).

**Evidence:** element-wise comparison of a bad cell shows woodelf's output is exactly
`−1 ×` the oracle's values (ratio −1.0 on every feature). `relative_mae = 2.0` is the
mathematical signature of `candidate = −reference`.

**Root cause (hypothesis):** the woodelf backend explains the wrong class of the sklearn
`RandomForestClassifier` (class 0 probability where every other backend explains class 1).
Regression (ames_housing) and xgboost/lightgbm classifiers are unaffected.

**Fix:** correct the class-index selection in the woodelf backend
(`Benchmarking/backends/`, woodelf wiring). Existing data is recoverable by multiplying
the affected cells' values by −1 — no rerun strictly required once verified.

---

## Bug 2 — shapiq_interaction disagrees with the interaction oracle 🔴

**Symptom:** `woodelf_interaction` matches the `shap_interaction` oracle essentially
exactly (median rel MAE 0.0, rho 1.0, outside the Bug-1 cells). `shapiq_interaction`
deviates massively: median rel MAE **0.65**, rho 0.92, never better than 0.995 across all
441 order-2 cells.

**Root cause (hypothesis):** shapiq is likely computing a budget-capped *approximate*
interaction index (or a different index definition, e.g. k-SII) rather than exact
TreeSHAP interaction values.

**Fix:** check `ShapIQInteractionBackend`'s configuration. Until resolved, do not present
`shapiq_interaction` as a true-value/exact backend in results.

---

## Bug 3 — silent `except` swallows DeepExplainer errors in shap_nn backend 🟡

**Location:** `Benchmarking/backends/shap_nn_backend.py:83-84`

```python
except Exception:
    return nan_result(x)
```

**Impact:** the NN sweep's `shap_nn deep` failures left no trace in the SLURM logs;
diagnosing them required local reproduction. Reproduced cause: shap's `DeepExplainer`
fails on `TabularTransformer` with an additivity `AssertionError`
("unrecognized nn.Module: LayerNorm", max diff 0.077 vs tolerance 0.01) — DeepLIFT rules
don't support LayerNorm/attention. This is a fundamental limitation: expect NaN for
`shap_nn deep` on the transformer even after reruns (and treat `captum deep_lift_shap` on
the transformer as approximate — it only warns "unrecognized nn.Module").

**Fix:** log the exception like `runner.py` does (`[BUG] ... {e.__class__.__name__}: {e}`)
before returning the NaN frame.

---

## Bug 4 — accuracy sweep: systematic error floor at n_background=200 🟡

**Symptom:** at `n_background=50`, approximation error behaves like sampling noise
(shrinks sharply with budget, e.g. shapiq-kernel 0.073 → 0.000 going 64 → 512). At
`n_background=200`, **all seven library×approximator combinations** converge to a
~0.09–0.14 relative-MAE floor that an 8× budget increase barely moves.

**Root cause (hypothesis):** a budget-independent floor means a *value-function mismatch*,
not estimation noise — most likely one or more libraries silently subsample/cap the
background set internally (e.g. at 100 rows) while the oracle uses all 200.

**Fix:** audit each approx backend's background handling (shap KernelExplainer, lightshap
`bg_X`, dalex background sampling, shapiq imputer). Until verified, only interpret the
`n_background=50` half of the accuracy sweep as an approximation-quality comparison.

---

## Bug 5 — fasttreeshap produced zero usable data in the tree sweep 🟡

**Symptom:** all 1071 fasttreeshap rows (630 order-1 + 441 order-2) are all-NaN.
Runtimes of 0.1–0.6 ms show the backend never launched its subprocess.

**Causes:**
1. xgboost cells (357 rows): skipped **by design** — fasttreeshap 0.1.6 cannot parse
   XGBoost 3.x's model format (`fasttreeshap_backend.py:31-37`).
2. lightgbm + random_forest cells (714 rows): the dedicated numpy<2 venv
   (`~/.cache/pr-modeagnostic/.venv-fasttreeshap`, see
   `scripts/setup_fasttreeshap_env.sh`) **was never set up on the cluster**, so the
   `Path(venv_python).exists()` check failed → instant `[SKIP]`
   (`fasttreeshap_backend.py:40-43`). `slurm/bench_array.sh` also does not export
   `FASTTREESHAP_VENV_PYTHON`. Confirmed by cluster logs (2026-07-06): 1462×
   `[SKIP] ... no fasttreeshap venv found at /home/f/fuchsfe/.cache/...`.

**Fix:** run `scripts/setup_fasttreeshap_env.sh` on the cluster (home is NFS-mounted, so
once is enough), then rerun only the lightgbm/random_forest tree cells. All other
backends in `results_config-tree.csv` are complete and reusable.

---

## Bug 6 — NN results CSV is stale (wrong config generation) 🔴

**Symptom:** every row of `results_config-neural-networks-RQ3.csv` has
`seed=42, n_background=200` — the original config from commit `8cd8b3c`. The current
config (`seed: [0,1,2]`, `n_background: 100`, `backend_timeout_s: 3600`) defines a
27-cell grid of which **0 cells are present**.

Additional defects of that old run:
- `gisette × transformer` cell missing entirely — the run predated `backend_timeout_s`
  and slow cells ran for hours (lightshap-kernel on gisette-mlp: 17,109 s;
  shapiq reference on ames-transformer: 7,039 s) → almost certainly killed by the 12 h
  SLURM walltime before writing its CSV.
- 10/72 rows all-NaN (`shap_nn deep` in 7/8 cells, `captum deep_lift_shap` on both
  transformer cells + gisette-cnn_1d). Only the transformer failures reproduce with the
  current environment (see Bug 3); the rest were older library versions on the cluster.

**Fix:** full rerun via `uv run python slurm/submit_all.py --configs nn` (after clearing
`Benchmarking/slurm_results/config-neural-networks-RQ3/`). Requires the GRES fix (Bug 7).

**Status: RESOLVED (2026-07-06 rerun).** New CSV covers all 27 cells (seeds 0–2,
n_background 100, 9 backend rows each). Remaining problems are Bug 9 (transformer) and
the expected Bug-3 `shap_nn deep` failures.

---

## Bug 9 — rerun: the transformer model breaks nearly every backend, including the reference 🔴

**Symptom (2026-07-06 rerun, 80/243 rows all-NaN):** on transformer cells 69/81 rows
failed — crucially including **`shapiq_true_value` on all 9 transformer cells** (crash,
not timeout: runtimes 1–153 s, far below the 3600 s budget). Model-agnostic backends
(lightshap 9/9, dalex 6/9, shapiq_proxy 6/9) crashed too — these only ever call
`model.predict`, so the trained transformer's prediction path itself is the prime
suspect, not the explainers. Failure is dataset-dependent: on gisette-transformer,
dalex / shapiq_proxy / captum-gradient_shap / shap_nn-gradient succeeded (70–80 s),
while lightshap and the shapiq reference crashed everywhere.

**Consequence:** with the reference NaN on every transformer cell, even the 12 surviving
transformer rows have no accuracy metrics — **the transformer column of RQ3 is unusable
in this run.**

**Non-transformer failures in the rerun** (consistent with Bug 3):
`shap_nn deep` crashed on all 9 mlp cells and 2/9 cnn_1d cells (silent except, no log);
everything else on mlp/cnn_1d succeeded.

**Root causes — CONFIRMED from cluster logs (2026-07-06):**

1. **CUDA OOM: `predict()` runs the whole input in one forward pass.**
   `Models/trainers.py:66-70` (`predict`) / `:72-81` (`predict_proba`) convert the entire
   input to a single tensor and do one forward. The shapiq reference and lightshap hand
   it coalition batches of 10⁵–10⁶ imputed rows; the transformer's attention then needs
   **9.4–32 GiB on a 7.6 GiB GPU**:
   ```
   [BUG] shapiq_true_value crashed: OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 7.60 GiB ...
   [BUG] lightshap_approx ({'approximator': 'kernel', 'budget': 512}) crashed: OutOfMemoryError: ... Tried to allocate 9.34 GiB ...
   ```
   mlp/cnn_1d survive because their per-row activation memory is tiny.
   **Fix:** chunk inference in `predict`/`predict_proba` (loop over e.g. 1024–4096-row
   batches, concatenate on CPU). This alone likely also fixes most model-agnostic
   transformer failures.

2. **NaN predictions: no input standardization in the NN training path.**
   `PyTorchTrainer.fit` feeds raw feature values to the network
   (`Models/trainers.py:195-197`) and `TabularTransformer` projects raw scalars per
   token. ames/adult features reach 10⁴–10⁵ (LotArea, capital-gain) → activations blow
   up → the trained transformer emits NaN/inf. Direct evidence — ProxySHAP fits xgboost
   on the transformer's predictions and gets:
   ```
   [BUG] shapiq_proxy (... 'proxy_model': 'xgboost') crashed: XGBoostError: ... Label contains NaN, infinity or a value too large.
   ```
   (6×, exactly the adult/ames transformer cells; gisette's homogeneous features stayed
   finite, which is why its transformer cells partially worked.)
   **Fix:** standardize inputs inside the NN trainer (fit a scaler in `fit()`, apply it
   in `_to_tensor()` so explainers see original feature space), and/or add gradient
   clipping. Note the mlp/cnn accuracy numbers of this run were also trained unscaled —
   they didn't NaN, but retraining scaled will change (likely improve) them.

**Status: FIXED locally (2026-07-06), NN rerun required.** `Models/trainers.py`:
`predict`/`predict_proba` now run in `PREDICT_CHUNK`-row slices (bounded GPU memory),
and `fit` standardizes features + regression target, with the stats stored as buffers
inside `TorchPredictor` and applied in `forward` — so all explainers (gradient-based and
model-agnostic) still see the model in the original feature/output space and Shapley
values remain comparable across backends. Verified: 75/75 tests pass (incl. the two new
spec tests), transformer trained on 1e5-scale features gives finite predictions, 50k-row
chunked predict is bit-identical, shap GradientExplainer works through the scaled
forward with attributions in original feature space. **Do not mix the retrained models'
results with this run's mlp/cnn numbers — rerun the whole NN config after deploying.**

**Also suspicious (quality, not crash):** `captum` on mlp shows rel MAE ≈ 1.4–2.0 vs the
reference despite rho ≈ 0.72–0.94 — a systematic scale mismatch (possibly explaining
logits vs probabilities on classification cells). And `shapiq_proxy` tracks the reference
poorly everywhere (rho 0.51–0.56) — expected for a proxy method, but note it before
citing it.

---

## Bug 7 — GPU jobs rejected: cluster defines no GPU GRES ✅ fixed locally

**Symptom:** all 27 nn submissions failed with
`sbatch: error: Invalid generic resource (gres) specification`.

**Cause:** `sinfo -o "%P %G"` shows `GRES=(null)` on every partition — the CIP cluster
does not register GPUs with SLURM, so `--gres=gpu:1` / `--gpus=1` are invalid. The
NvidiaAll nodes expose their GPU directly; `--partition=NvidiaAll` alone is sufficient.

**Fixed** (2026-07-06, local working tree): `--gres=gpu:1` removed from
`slurm/submit_all.py` (`nn` + `tree-gpu` entries) and `slurm/bench_array_gpu.sh`; stale
advice corrected in `slurm/single_task.sh` and `SLURM.md`. Apply the same edit on the
cluster checkout (or pull once committed) before resubmitting.

---

## Bug 8 — submit_all.py submits merge jobs even when every task submission failed 🟢

**Symptom:** in the failed nn submission run (Bug 7), all 27 sbatch calls errored, yet
the script still submitted the merge job (171053), which can rebuild the merged CSV from
stale per-task files.

**Fix:** in `slurm/submit_all.py`, skip `submit_merge()` for configs whose tasks all
failed to submit (or exit non-zero after submission failures). Also: always clear the
per-task directory before a rerun, since `merge_results.py` globs every
`results_*.csv` in it and old rows (different seed) survive `drop_duplicates`.

---

# Non-bug findings (library comparison, for the write-up)

- **Tree-exact backends agree** (modulo Bugs 1–2) — speed is the differentiator:
  shap TreeSHAP 0.02 s ≪ shapiq path-dependent 0.14 s ≪ woodelf ~0.8 s ≪
  shapiq-interventional 6.8 s (median per cell). Runtime flattens above max_depth ≈ 20.
  Path-dependent vs interventional values differ by ~20% rel MAE (rho ≈ 0.95),
  consistently across libraries — the expected value-function difference.
- **For interactions,** woodelf is exact and 3–4× faster than shapiq.
- **Model-agnostic:** lightshap-permutation is the most accurate per model-eval and the
  only approximator that survives 256 features (rel MAE 0.032, rho 0.996 vs shap-perm
  0.24/0.91) — but it spends ~60× more model evals per "budget" unit (36M at d=256,
  102 s). KernelSHAP (shap & shapiq) collapses to noise at 256 features (rho ≈ 0.3).
  shapiq's approximators are weakest per eval; dalex is middling and permutation-only.
- **Cross-library "budget" is not comparable** — normalize on `n_model_evals`
  (e.g. at budget 64 lightshap-kernel already spends 419k evals vs shap-kernel's 32k).

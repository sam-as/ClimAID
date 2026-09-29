> **Note (release 0.4.0).** This is the testing log written during development, kept as a record. The
> development builds it mentions as 0.2.0 and 0.3.0 were released together as **0.4.0**; code described here as
> "0.3.0" is the 0.4.0 release. `V2_REVIEW_RESPONSE.md` is now `V2_FEEDBACK_RESPONSE.md`, and
> `CHANGELOG_v2.md` is merged into `CHANGELOG.md`. For the current state see `CHANGELOG.md` and the
> [Status & validation](https://sam-as.github.io/ClimAID/guide/status/) page.

# ClimAID v2 — Review Findings (this pass)

I read `README.md`, `MIGRATION_v2.md`, `CHANGELOG_v2.md` and `V2_REVIEW_RESPONSE.md`
first, then ran the test suite and read through the whole `forecasting_v2`
engine (`features.py`, `validation.py`, `stacking.py`, `renewal.py`, `ml.py`,
`ensemble.py`, `baselines.py`, `metrics.py`, `hindcast.py`, `forecaster.py`)
since that's the part carrying the reviewer-response claims about leakage,
out-of-fold residuals and calibration.

## Good news
The leakage-safety claims in `V2_REVIEW_RESPONSE.md` check out in the code:
- `ClimateFeatureBuilder.fit` only ever sees climate observations at/before
  the cutoff (month climatology is fit on history only, then applied
  unchanged going forward).
- `RollingOriginSplitter` and `TemporalResidualStack` never let a validation
  fold's index precede its training fold — genuinely leakage-safe temporal
  out-of-fold stacking.
- The recursive ML/renewal forecasts append the model's own point forecast
  (never true future observed cases) into the autoregressive history during
  multi-step prediction.
- `forecaster.predict()` now hard-fails on mismatched forecast calendars
  across models, which is exactly the kind of check a reviewer would want.

## Two real bugs fixed

### 1. `weighted_interval_score` (climaid/forecasting_v2/metrics.py)
The normalising constant was `1 + 2*sum(alpha_k/2)`, which depends on the
specific quantile levels chosen. The canonical WIS (Bracher et al. 2021)
normalises by `K + 1/2` (K = number of central intervals) regardless of the
alpha values, and weights the median term by 1/2, not 1. The bug inflated
WIS by a data-dependent, non-constant factor (verified numerically: ~2–3x
in a quick simulation, and the ratio wasn't even fixed row to row). Since
`V2_REVIEW_RESPONSE.md` leans on WIS specifically to answer the reviewer's
calibration concern, this was worth getting exactly right rather than
approximately right. Fixed and verified against the standard "WIS = 2x mean
pinball loss across the 2K+1 quantile levels" identity for several different
probability grids (see `tests/test_metrics_and_baselines.py`).

### 2. `SeasonalNaive.predict` (climaid/forecasting_v2/baselines.py)
The exact-lag lookup always used `pd.DateOffset(months=period)`, but
`ClimaidV2Forecaster._seasonal_period()` sets `period=52` for weekly-cadence
data (common for disease surveillance series). That looked up a date ~4.3
years in the past instead of ~1 year, missed the history index almost every
time, and silently fell back to a same-calendar-month multi-year median —
i.e. every weekly ClimAID v2 run was quietly using a coarser, less accurate
seasonal-naive baseline than intended. Fixed by inferring the cadence
(weekly vs. monthly) from the median spacing between observations at fit
time and using the matching offset unit at predict time.

Both fixes are covered by new regression tests in
`tests/test_metrics_and_baselines.py`. Full suite: `11 passed`.

## Smaller things
- Added a `test` extra (`pytest`, `httpx`) to `pyproject.toml` — the repo's
  own test suite wasn't declared as a dependency anywhere, and
  `test_browser_uploads.py` currently emits a deprecation warning under
  newer `starlette` recommending `httpx2` over `httpx`; worth revisiting
  next time `starlette`/`fastapi` are bumped.
- `HindcastEvaluator.evaluate()` silently swallows any exception per origin
  (`except Exception: continue`). Reasonable for robustness across many
  rolling origins, but consider at least collecting/logging which origins
  failed and why — a silently-shrinking hindcast set could mask a real
  problem (e.g. all origins failing) rather than genuine data-coverage
  gaps.

## What I didn't get to (first pass)
This is a large package (`climaid_model.py` alone is ~88K, plus
`reporting.py`, `reporting_v2.py`, `wizard.py`, the browser UI, and the CMIP6
projection code). I focused this pass on the v2 forecasting/scoring engine
since it's the part making the specific quantitative claims in your
reviewer-response document. Happy to do a similar pass over the reporting
layer, the browser/FastAPI interface, or the legacy `climaid_model.py` next
— let me know which matters most for what you're using this for.

---

# Second pass — reporting layer, browser UI separation, C-DSI v2

## Another real bug fixed

### `DiseaseReporter._llm_generate` (climaid/reporting.py)
When the LLM client raised an exception, the fallback returned a **tuple**
(`(str, self._deterministic_engine(prompt))`) instead of a string — while
every caller (`generate`, `policy_brief`, `chat`) is typed `-> str`, and
`open_report_in_browser()` immediately calls `html.escape(report_text)` on
the result. So the one scenario this fallback exists to handle safely — the
LLM being unavailable — was exactly the scenario that crashed report
rendering with a `TypeError`. Also, the buggy code called
`self._deterministic_engine(prompt)`, but `_deterministic_engine` expects a
`ReportArtifacts` object, not the prompt string, so it would have raised a
second, unrelated error even if the tuple bug weren't there. Fixed to return
a single, clear string. Covered by `tests/test_reporting.py`.

## Browser UI — v1 and v2 now clearly separated

Previously the "Optimization Preset" and full model-configuration card
(base/residual/correction models) sat at the top of the page, unlabeled,
above *both* the v1 and v2 sections — easy to mistake for a shared/required
step even though it only applies when the legacy pipeline runs. And only a
"run v2 only" shortcut existed; there was no "run v1 only".

Changes in `climaid/browser_ui/static/`:
- Moved the model-configuration card into the `① ClimAID v1 — Legacy
  Deterministic Pipeline` section, which now owns all v1-specific config
  end to end (report mode, CMIP6 toggle, preset, base/residual/correction
  models, trials) with its own "Run ClimAID v1 Only" button.
- The `② ClimAID v2 — Probabilistic Forecasting` section is unchanged in
  content but now has a matching "Run ClimAID v2 Only" button next to it
  (was previously a standalone button at the bottom of the whole page).
- Added `runClimAIDV1Only()` to `browser.js`, symmetric to the existing
  `runClimAIDV2Only()`.
- A shared-inputs hint at the top now explicitly says what's common to both
  pipelines (location, disease dataset) vs. pipeline-specific.
- Covered by `tests/test_browser_ui_separation.py`, and verified end-to-end
  by actually serving `index.html` through the real FastAPI app.

`climaid/browser_ui/api.py` (the backend) was not changed — it already
supported running v1/v2 independently via `run_legacy`/`run_v2` flags; this
pass only fixed the frontend's presentation of that.

## C-DSI v2 — a real v2-native deterministic interpreter, not just a v1 appendix

Previously, `reporting_v2.py`'s report embedded the **unchanged v1 C-DSI
narrative** as-is under the heading "Legacy deterministic C-DSI report" —
useful as a reference, but it says nothing about the v2 forecast itself
(v1's C-DSI describes a point-forecast ML pipeline and CMIP6 scenario
trends; it has no concept of quantile forecasts, WIS, or hindcast
calibration).

Added `generate_c_dsi_v2()` to `climaid/reporting_v2.py`: a new deterministic,
LLM-free interpreter built on the same design principle as the original
C-DSI (template-based narrative from precomputed numeric artifacts only —
zero risk of hallucinated content) but interpreting v2's *own* outputs:

1. **Forecast summary** — picks a primary model deterministically (ensemble
   if present → lowest mean hindcast WIS → lowest held-out WIS → the only
   fitted model), then states the trend and peak of its median forecast.
2. **Interval calibration** — compares empirical 50/80/95% coverage
   (preferring rolling hindcasts over the single held-out window, since
   hindcasts give more folds) against nominal levels using fixed thresholds
   (±15 points → "under-/over-covered", else "consistent with its nominal
   level").
3. **Model comparison** — ranks models by mean hindcast WIS and states
   the winner, or explicitly says no comparison is possible when hindcasts
   weren't run.
4. **Structural caveats** — restates the population/relative-incidence and
   climate-source (`observed` vs `projection`) status from the run's own
   metadata, plus the standing note that model agreement isn't a calibrated
   probability.
5. **Methodological warnings** — the run's own recorded warnings.

This is now its own card in the v2 report (`C-DSI v2 — Deterministic
Forecast Interpretation`), placed before the raw metric tables. The legacy
v1 report is still embeddable, but is now explicitly labelled "Appendix —
ClimAID v1 (legacy) C-DSI report" with a note that it describes a different
pipeline and isn't validation for the v2 forecast — addressing the same
v1/v2-conflation concern as the browser UI change above.

Every other v2 report card was also given an explicit "v2 ..." heading
(models fitted, probabilistic forecast, held-out evaluation, hindcast,
validation contract, warnings, model specification) so the whole report
reads unambiguously as v2, with the v1 material clearly boxed off at the
end. (One card's exact heading text, "Validation contract", was kept
byte-for-byte to avoid breaking an existing test that asserts on it — it's
now "v2 Validation contract".)

Covered by `tests/test_c_dsi_v2.py` (4 tests: section presence, graceful
handling when no metrics/hindcasts are available, deterministic
best-model selection, and correct embedding/ordering in the full report).

Full suite after this pass: **19 passed**.

## Still not done (as of second pass)
- `wizard.py` (the terminal interface) wasn't given the same v1/v2
  separation treatment — only the browser UI was addressed this pass.
- The legacy `climaid_model.py` (~88K) and the CMIP6 projection code
  (`climaid_projections.py`, `projection_plots.py`) haven't been reviewed.
- No changes were made to `climaid/browser_ui/api.py` (the backend) beyond
  what was already there.

---

# Third pass — climate-feature leakage in climaid_model.py (v1)

## The leakage bug

`DiseaseModel._merge_data()` (climaid/climaid_model.py) builds two
"average climate" features that feed directly into the **default** feature
grid used by both `optimize_lags()` and `train_final_model()`:
`YA_mean_temperature`/`YA_mean_Rain`/`YA_mean_SH` ("year average") and
`MA_mean_*` ("10-year moving average").

`YA_mean_*` was computed as:

```python
annual = climate.groupby("Year")[vars].mean().reset_index()
...
climate = climate.merge(annual, on="Year", how="left")
```

That's a **whole-calendar-year** mean, merged onto *every* row sharing that
year — including months earlier in the year than the value it was merged
into. A January 2016 row's `YA_mean_temperature` was the average of
January–December **2016**, i.e. it included up to eleven months of climate
that hadn't happened yet relative to that row. This is a textbook
target/feature leakage bug, and it's the same "climate leakage" category the
external reviewer raised (per `V2_REVIEW_RESPONSE.md`) — that response
describes how v2 avoids it, but v1's `_merge_data()` had this exact problem
untouched. It inflates historical validation performance (R²/RMSE) because
the model is effectively being handed a same-year climate summary it could
never have at real forecast time, and it would silently degrade at
deployment/forecast time once real future months genuinely aren't known yet.

A second, related issue: neither `YA_mean_*` nor the existing `MA_mean_*`
(10-year trailing average) had `climate` sorted by time before calling
`.rolling()` — trailing-window correctness assumes row order == chronological
order, which an uploaded climate file is not guaranteed to have.

**Fix** (in `_merge_data()`):
- Sort `climate` by `time` first, unconditionally, before computing either
  average.
- Replace the whole-calendar-year `groupby("Year").mean()` with a strictly
  backward-looking **trailing 12-month rolling average** ending at the
  current row — same trailing-window principle as `MA_mean_*`, just over 12
  months instead of 120. This preserves the "annual average climate"
  semantic while guaranteeing it can never see a later-dated value.

Verified with a numeric reproduction: on a synthetic strictly-increasing
climate series, the old code gave January 2016 a "trailing annual average"
of 29.5 when the actual January 2016 value was only 24.0 — i.e. it was
provably using future-dated data. After the fix, the feature never exceeds
the current row's own value on a monotonically increasing series (see
`tests/test_climaid_model_leakage.py`).

Also confirmed the fix doesn't break the pipeline that consumes these
features: ran `optimize_lags()` → `train_final_model()` end-to-end on
synthetic data through the real (unmocked) code path.

Covered by three tests in `tests/test_climaid_model_leakage.py`:
1. the trailing annual average never uses same-year future months (checked
   against the raw value at that exact row, on a monotonic series where any
   leak would provably inflate it),
2. it matches a hand-computed trailing-12-month mean exactly,
3. results are identical whether the input climate rows are already sorted
   or arbitrarily shuffled (i.e. the fix doesn't silently depend on the
   input file happening to already be sorted).

Full suite after this pass: **22 passed**.

---

# Fourth pass — test-set-reuse leakage, predict() typo, legacy report
# mode mislabelling, and wizard.py v1/v2 separation

## The big one: test-set reuse across model/feature selection

This is more consequential than the annual-average bug above, and it's
almost certainly the "leakage problem" being asked about specifically,
since it affects *every* run of `optimize_lags()`, not just certain
climate patterns.

`optimize_lags()` screened and ranked **every** candidate lag/feature/model
configuration (potentially hundreds to thousands: every lag combination x
base model x residual model x correction model) directly against
`self.test_df`:
- Stage 1 screening (`_evaluate_base_screen`) fit each candidate and scored
  it against `X_test_full`/`y_test`, both built from `self.test_df`.
- Stage 2 (`_evaluate_configuration`) did the same for the full
  base→residual→correction stack, and its `rmse`/`r2` (test-set-based) was
  the criterion used to rank *all* candidates and pick `self.best_config`.
- `train_final_model()` then reported that same test set's RMSE/R² as the
  "held-out" performance metric shown in every report.

Reusing a test set to choose among many candidates optimistically biases
the resulting score — a textbook "test-set reuse"/"double-dipping" leak,
distinct from the temporal (same-row) leaks above. I verified the general
mechanism numerically first: selecting the best of 300 *pure-noise* feature
sets by test RMSE beat a naive mean-prediction baseline (8.77 vs 9.35),
even though the noise features carry zero real signal — purely from
reusing the test set across many comparisons.

**Fix**: added `DiseaseModel._chronological_holdout()`, which carves a
genuine validation split out of `self.train_df` alone (`self.sel_train_df`
/ `self.sel_val_df`), used for:
- Stage 1 screening and Stage 2's config ranking (passed to
  `_evaluate_configuration` in place of `test_df` — its returned metrics
  are renamed `val_rmse`/`val_r2` to make the distinction explicit and
  avoid confusion with the final test-based report).
- The scaler fit (previously fit on the full `train_df`, which included
  what becomes the validation rows — now fit on `sel_train_df` only and
  applied everywhere else via `.transform()`).
- The correction-model accept/reject decision, which is now decided *once*,
  during search, on `sel_val_df` — `train_final_model()` was re-deciding
  this a second time against `test_df` (an additional, separate leak layer)
  via an "RMSE-based safety check"; that re-check is removed and
  `train_final_model()` now trusts the search-phase decision unconditionally.

`self.test_df` is now touched exactly once, at the very end of
`train_final_model()`, to report a genuinely honest generalisation metric.

## Base-model hyperparameter tuning was in-sample

Found while fixing the above: the base model's Optuna search scored
candidate hyperparameters by predicting on their own training data
(`_optuna_objective` fit and scored on the same `X_train`/`y_train`), while
the residual and correction stages *in the same function* already correctly
scored on a validation split. Fixed `_optuna_objective` to accept and use
`X_val`/`y_val` when supplied, and updated the base-model call site to pass
the same chronological split the other two stages use.

## predict() never applied the fitted correction model

`climaid_model.py`'s `predict()` checked `hasattr(self, "cor")` — a typo
for `"corr"`, the actual attribute `train_final_model()` sets. Since
`self.cor` never exists, this check is always `False`, so the fitted
correction/calibration model was silently **never** applied in real
predictions — even though the `test_rmse`/`test_r2` reported by
`train_final_model()` reflect the *corrected* model. Every real call to
`predict()` was quietly less accurate than the metrics implied.

Fixing the typo exposed a second, previously-unreachable bug: a non-isotonic
correction model was fit on a `(-1, 1)`-reshaped array in
`train_final_model()`, but `predict()` passed it a bare 1D array — would
have raised a shape error the first time anyone actually exercised this
path. Fixed to match shapes correctly per model type.

All three of the above are covered by 9 tests in
`tests/test_climaid_model_leakage.py` (full file, all passing): selection-
split disjointness from `test_df`, `val_rmse`/`val_r2` naming, a "corrupt
`test_df` with NaNs and confirm the selection is unaffected" proof,
`train_final_model()` trusting the search decision, and both isotonic and
non-isotonic correction models actually being applied by `predict()` with
the correct input shape.

## Browser UI: "Policy"/"Summary"/"Detailed" legacy report modes were silent no-ops

Found while auditing the rest of `climaid_model.py`: `DiseaseReporter.generate()`
only varies its output by `style` when an LLM client is supplied; with
`llm_client=None` it always returns the same deterministic C-DSI report
regardless of `style`. The browser UI's `_run_legacy()` hardcodes
`llm_client=None` (no LLM is wired up anywhere in the browser UI), so
selecting "Detailed", "Summary" or "Policy" in the "Legacy report" dropdown
silently produced **byte-identical** output to "C-DSI deterministic" —
verified directly (`out_policy == out_summary == out_deterministic`) — while
the UI labelled the result `legacy_policy`/`legacy_summary`/`legacy_detailed`
as if that style had actually been generated.

Fixed in `climaid/browser_ui/api.py`: when a narrative style is requested
without an LLM available, the deterministic engine is used explicitly and a
`note` is returned explaining why (surfaced in the frontend's status text),
and the report is always labelled `legacy_c_dsi` (what it actually is)
rather than the requested-but-unfulfilled style. The dropdown in
`index.html` now says so directly next to each non-functional option.
Covered by `tests/test_legacy_report_mode.py` (3 tests, using a lightweight
fake model rather than driving the real ML pipeline, since this is purely
an orchestration/labelling bug).

## wizard.py: v1/v2 separation, plus two more crash bugs it surfaced

Applied the same treatment as the browser UI last pass: `wizard.py`'s
`run_interactive_pipeline()` previously interleaved the v2 forecasting
prompt in the *middle* of the v1 flow (train/test split → outbreak
detection → **v2 block** → lag optimization → training → plots → CMIP6
projections → visualization → LLM/C-DSI report → runtime summary), with no
way to run one pipeline without walking through all of the other's
prompts too.

Now asks upfront ("1. ClimAID v1 only / 2. ClimAID v2 only / 3. Both") and
only asks each pipeline's own questions when that pipeline was selected.
The v1 flow (everything from train/test split through the final report) is
wrapped in one `if run_v1:` block; the v2 block's existing `if
_ask_yes_no(...)` became `if run_v2 and _ask_yes_no(...)`. Verified end to
end with a scripted-`input()` test (`tests/test_wizard_separation.py`) that
drives the real wizard function against a fake `DiseaseModel` and asserts,
for a "v2 only" run, that no v1-only method (`_train_test_split`,
`detect_historical_outbreaks`, `optimize_lags`, `train_final_model`) is ever
called and no v1 question is ever asked (an unscripted extra `input()` call
raises immediately) — and, for a "v1 only" run, that `forecast_v2` is never
called and its question is never asked.

This restructuring surfaced two more real, pre-existing crash bugs, found
because the new test is the first thing to actually exercise "decline this
step" paths in the wizard:

1. **`plt` referenced before import**: `import matplotlib.pyplot as plt`
   only happened inside the "Plot historical predictions?" and "Run CMIP6
   climate projections?" prompts. Declining both, then reaching the
   LLM-report section's unconditional `plt.close('all')`, raised
   `NameError`. Fixed by importing `matplotlib.pyplot` unconditionally at
   module level (matplotlib is already a hard dependency).
2. **`use_headless_backend` referenced before import**: same pattern —
   only imported inside "Run lag optimization?", but called unconditionally
   in the LLM-report section. Declining lag optimization raised
   `UnboundLocalError`. Fixed the same way.

Also fixed, as a side effect of properly scoping the v1 block:
`projection_summary` was referenced later in the function without being
assigned if lag optimization was declined (another latent `NameError`) —
now initialized to `None` at the top of the v1 block.

Full suite after this pass: **33 passed** (24 fast + 9 in the slower
`test_climaid_model_leakage.py`, which drives real `optimize_lags()` calls).

## Still not done
- The CMIP6 projection code (`climaid_projections.py`, `projection_plots.py`)
  hasn't been reviewed.
- The reporting `style="policy"` parameter passed to `DiseaseReporter.generate()`
  from `climaid_model.py`'s `generate_report()` docstring promises a
  "comprehensive policy report" but `generate()` doesn't dispatch to the
  separate `policy_brief()` method for it — flagged during the audit but not
  yet resolved; needs a product decision (should `generate_report(style="policy")`
  call `policy_brief()` instead, or should the docstring be corrected?) rather
  than a mechanical fix.

---

# Fifth pass — climaid_projections.py: CMIP6 projection features didn't match training

## The bug

`DiseaseProjection.prepare_features()` (`climaid_projections.py`) is the
feature builder used every time a CMIP6 projection is made — it's what
turns future climate-scenario data into the exact feature matrix the
trained model expects. It computed the `MA_*`/`YA_*` climate-average
features using **completely different formulas** than
`DiseaseModel._merge_data()` computes the identically-named features the
model was actually *trained* on:

- `YA_*` here was a whole-calendar-year pooled mean
  (`df.groupby("Year")[var].transform("mean")`) — the exact same bug already
  fixed in `_merge_data()` two passes ago, just re-appearing independently
  in the projection code path, which was never touched by that fix.
- `MA_*` here was a rolling mean across the last 10 *occurrences of the same
  calendar month* (e.g. the last 10 Januaries) — a different concept
  entirely from `_merge_data()`'s trailing 120-*row* (10 years, all months)
  window.

Same feature name, different formula, computed on two different sides of
train vs. predict. Verified numerically on a synthetic series: for the same
row, `YA_mean_temperature` came out as 17.5 at projection time vs. 11.5 at
training time; `MA_mean_temperature` came out as 11.0 vs. 8.5. Every CMIP6
projection was silently feeding the trained model values it had never seen
associated with those feature names during training — a train/inference
skew that degrades projection quality without raising any error, in exactly
the domain (climate feature construction) the external review specifically
scrutinised.

There was also a secondary issue: the `YA_*`/`MA_*` block ran *before* the
code that sorts by time and groups by `(model, ssp, member)` for the lag
features further down, so on a multi-GCM/multi-SSP input frame those
averages would (in principle) pool different climate scenarios sharing a
calendar year/month together. In practice `project()` always pre-filters to
one model+SSP before calling `prepare_features()`, so this wasn't
triggered via the normal path, but `prepare_features()` is also called
directly with unfiltered multi-scenario data in places, so it's a real
latent risk, not just a theoretical one.

**Fix**: moved the sorting/grouping logic before the average-computation
block, and rewrote `MA_*`/`YA_*` to use the exact same trailing-window
rolling means as `_merge_data()` (120-row and 12-row respectively),
computed per `(model, ssp, member)` group when the input spans more than
one scenario. Verified the two code paths now produce **identical** values
for the same synthetic series (`119.5 == 119.5`, `65.5 == 65.5`).

One consequence of matching training's `min_periods` exactly: the first
11/119 rows of any projection series now correctly come out as `NaN`
(previously they never did, because the old formulas didn't require a full
window — part of the same inconsistency). Extended the existing
lag-feature dropna step to also drop these rows, so `NaN` never silently
reaches `self.model.predict()`; the trained model never saw a partial-window
value for these features either, since `optimize_lags()` drops all-NaN rows
the same way before training.

Covered by 3 new tests in `tests/test_projection_feature_consistency.py`:
exact numeric match against `_merge_data()`'s output on shared synthetic
data, correct dropping of incomplete-window rows, and correct per-scenario
grouping (two synthetic GCM/SSP series with very different constant
temperatures must not blend into a shared average). Also verified the full
`DiseaseProjection.project()` path runs end-to-end through the fixed
`prepare_features()` without error.

Full suite after this pass: **27 passed** in the fast run (the separate,
slower `test_climaid_model_leakage.py` — 9 tests, unaffected by this
pass — still passes independently; not re-run in full here to save time).

## Scope of this pass
Also skimmed the rest of `climaid_projections.py` (`flag_outbreak_risk`,
`project_ensemble_mean`, `build_projection_summary`, `export_tidy_projections`)
and `projection_plots.py` (pure matplotlib/seaborn plotting, no data
transformation). Nothing else at the same severity as the bug above turned
up. Two things noted but not changed, since neither is clearly a bug rather
than a reasonable simplification:
- `flag_outbreak_risk`'s "dynamic" threshold is self-referential (flags the
  top `percentile` of each GCM/SSP/time-window group's *own* projected
  values), so it will flag roughly that fraction of each group by
  construction. This is already the exact pattern the review response
  reframes in v2 ("dual-baseline flag ≠ calibrated probability") — no v1
  code change made here, just noting it's consistent with that known
  limitation rather than a new issue.
- `build_projection_summary()`'s linear "trend" (increasing/decreasing/
  stable) is fit over `np.arange(len(df_sorted))` after sorting by time,
  across all GCM/SSP rows concatenated together. When multiple scenarios
  share the same calendar dates, the row-index x-axis doesn't correspond to
  true elapsed time per point, which could mildly distort the slope
  estimate. Not clearly wrong enough to be a confident "bug" fix without
  understanding intended behaviour better, so left as-is and flagged here.


---

# Sixth pass — robustness

Focus: places where the package failed silently, crashed on messy input,
or gave results that depended on hidden state.

## 1. Disease input validation (v1 loader)
`_load_disease_data()` coerced dates with `errors="coerce"` but kept the
resulting `NaT` rows, never converted `Count` to a number, accepted
negative counts, and passed weekly/daily or duplicated rows straight into a
Year/Month merge with *monthly* climate. That last case silently repeated
each month's climate across several rows, so the model trained on
duplicated months. New `_validate_disease_frame()` (called by the loader):
drops unparseable dates and non-numeric counts, converts numeric strings,
raises on negative counts rather than guessing a fix, drops exact duplicate
rows, and sums sub-monthly data to monthly totals. Every change is printed
in a "DISEASE DATA VALIDATION" block, so nothing is altered silently.

## 2. Hindcast failures are no longer hidden
`HindcastEvaluator.evaluate()` wrapped every fold in
`except Exception: continue` and skipped folds with too little future
climate without saying so. If every fold failed, callers got an empty table
and the report said "no hindcast scores were available". Now every skipped
fold is recorded in `evaluator.skipped_origins_` with its reason, partial
failures raise a `RuntimeWarning`, and if every fold fails a `RuntimeError`
listing the reasons is raised. `forecast_v2()` already catches that and
stores it as a `hindcast_error` row, so the real cause now reaches the
report. Unavailable models are also reported instead of dropped silently.

## 3. C-DSI v2 handles failed hindcasts correctly
Follow-on to #2: the `hindcast_error` placeholder row (NaN scores) could be
ranked as if it were a model, and an error-only hindcast table blocked the
calibration section from falling back to the held-out window. Ranking now
ignores non-finite scores, the model-comparison section states the
reported failure cause, and calibration falls back to held-out metrics.

## 4. `random_state` was never passed to any model
All 10 call sites in `climaid_model.py` checked
`"random_state" in str(model_cls)`. The string of a class is its repr
(e.g. `<class '...RandomForestRegressor'>`), which never contains that
text, so the check was always False. Replaced with
`_accepts_random_state()`, which checks the constructor signature and
falls back to `get_params()` for `**kwargs` wrappers like XGBoost.

To be precise about impact: I first expected this to make runs
non-reproducible, but testing showed `_evaluate_configuration()` and the
Stage 1 screen reseed NumPy's global RNG on every call, which masked the
bug during search (same seed gave the same result even with the old
check). So this was a latent bug rather than a visible one. The fix still
matters: fitted models now carry their own seed instead of depending on
global RNG state, which is not reliable across threads or repeated
`train_final_model()` calls in one process.

## 5. `project_multi_model_ssp()` KeyError
It clipped `lower_bound` unconditionally, but that column only exists when
the model has an RMSE. Now guarded.

Covered by `tests/test_robustness.py` (12 tests). Full suite: **48 passed**
(39 fast + 9 in the slower `test_climaid_model_leakage.py`).

## Known limitations not changed
- Sub-monthly data is summed to monthly for v1. That is the right default
  for count data with monthly climate, but a user who wants weekly v1
  modelling would need weekly climate inputs and a weekly-aware merge.
- The `style="policy"` / `policy_brief()` question from the fourth pass is
  still open (product decision).

---

# Seventh pass — Stage 1 pruning kept 90% of configurations

Found while running the synthetic benchmark. `optimize_lags()` with the
default `pruning_strategy="percentile", percentile=90` kept every
configuration whose Stage 1 RMSE was at or below the 90th percentile, i.e.
~90% of them, and ignored `top_k`. The docstring says it "keeps
top-performing configurations" and that `top_k` is the "number of top
configurations retained after pruning" (it also listed a non-existent
`"threshold"` option). On a 336-lag-combination grid this sent ~1,200
configurations to the Optuna stage, projecting to several hours on one CPU.

Fixed to match the documentation: keep the best `(100 - percentile)`%
(best 10% by default), capped at `top_k`. Docstring corrected. The synthetic
seasonal run then finished in 341 s and recovered the true lags exactly.
Side effect: `test_climaid_model_leakage.py` dropped from ~400 s to ~70 s.
Behaviour change to note: default searches are now much narrower in Stage 2,
which is the documented intent; users wanting the old breadth can set
`percentile=10` (keep best 90%) and a large `top_k`.

Covered by a new test in `tests/test_robustness.py`. Full suite: **49 passed**.

A synthetic benchmark (seasonal vs non-seasonal) is delivered separately as
`ClimAID_synthetic_benchmark.zip`.

---

# Eighth pass — fixes from interpreting a real Pune dengue report, and dashboard redesign

## Fixes prompted by the report
1. **Hindcasts used data after the forecast origin (leakage in the validation
   evidence).** `forecast_v2()` passed the full disease series to
   `HindcastEvaluator`, so a Dec-2020 forecast was "validated" with hindcasts
   running to Dec 2023. The forecasts themselves were unaffected, but the
   report's calibration and model-comparison sections were partly based on
   post-origin data. Hindcasts now receive only disease and climate data at or
   before the origin.
2. **C-DSI v2 calibration tolerance.** A fixed ±15-point band called 81%
   coverage of a 95% interval "consistent" (missing 19% of months instead of
   5%). Tolerance is now ±max(2·√(p(1−p)/n), 3 points), shown in the report.
3. **C-DSI v2 trend sentence** now describes start, low, peak and end, notes a
   dip-and-recovery, and states the 95% interval span, instead of comparing
   only the first and last month.
4. **New model-disagreement flag** when fitted models' horizon totals differ by
   more than 2x, noting the ensemble median is not a consensus.
5. **Renewal stability cap.** Expected cases are capped at 5x the training
   maximum (and at the population when known); if >1% of simulated months hit
   the cap, a warning is added to the report. (An early hindcast in the Pune
   report had RMSE ≈ 19,000 on monthly counts under 100.)

## New option: `drop_2020` (default True)
Shared by both pipelines (dashboard, `forecast_v2(drop_2020=...)`, wizard).
- v2: each 2020 month in the training history is replaced by that calendar
  month's median across the other training years (rows are replaced, not
  deleted, because seasonal-naive, renewal and lagged ML models need an
  unbroken monthly series). Post-origin 2020 observations are kept for
  scoring. The substitution is recorded in the report warnings.
- v1: already dropped 2020 by default, but this was not configurable, and the
  default split hard-coded `Year < 2020` so `drop_2020=False` still discarded
  2020. Now exposed and honoured (`Year <= 2020` for training).

## Wider default v2 model set
`DEFAULT_V2_MODELS = seasonal_naive, renewal, poisson, random_forest,
extra_trees, gradient_boosting`, used by `forecast_v2()`, the API, dashboard
and wizard. All 14 installed v2 models remain selectable; previously only
Random Forest was ticked, so the ensemble was a median of three models.

## Dashboard
- v2 and v1 are separate pages with a tab bar; **v2 is the default** (`#v1`
  in the URL opens v1). Each page has its own settings and Run button. The
  redundant "Run legacy pipeline", "Run v2 forecasting" and "Run both"
  controls were removed; "Test year" moved to the v1 page (v1-only).
  Trade-off: v2 can no longer embed the v1 C-DSI appendix from the dashboard
  in a single run (still available via the Python API).
- An ⓘ tooltip on every control, section heading and model checkbox,
  keyboard-focusable and screen-reader labelled.
- Verified in headless Chromium: v2 visible on load, v1 hidden, tab switch
  works, 14 models listed with the 6 defaults ticked. (This caught a class
  collision: the page wrapper already used `.page`, which the first version
  of the page switcher hid.)

Tests: `tests/test_v2_origin_and_2020.py` (new), updated
`test_browser_ui_separation.py` and `test_wizard_separation.py`.
Full suite: **62 passed**.

---

# Ninth pass — hybrid near-term + CMIP6 scenario outlook

## What was added
- `climaid/forecasting_v2/scenario.py`: `HybridScenarioProjector`,
  `ClimateResponseModel`, `bias_correct`, `select_lags_cv`,
  `near_best_structures`.
- `DiseaseModel.project_v2(...)` and `climaid/reporting_scenario.py`
  (scenario report with its own deterministic C-DSI v2 interpretation).
- Dashboard: "Climate scenario outlook (CMIP6)" card on the v2 page (run
  toggle, end year, SSPs, how climate effects are learned, lag structure),
  every control with an info tip; API fields `run_scenarios`,
  `scenario_end_year`, `scenario_ssps`, `scenario_response`,
  `scenario_lag_selection`.

## How it works
1. Each GCM x SSP series is canonicalised separately (the v2 loader would
   otherwise average rows sharing a date, blending scenarios) and
   bias-corrected by monthly mean shift against the observed baseline.
2. Months 1-12: v2 ensemble forecast (renewal excluded by default) driven by
   each GCM's corrected climate. Months 13-24: linear hand-over.
3. Long term: Poisson GLM with NB dispersion, year-block bootstrap, samples
   pooled across GCMs within each SSP, so uncertainty grows with horizon.
4. Changes are reported against the model's own simulation of the baseline.
5. The other response mode is run as a sensitivity check and reported.

## Findings while building it (synthetic benchmark with known truth)
- **Month-of-year terms hide the climate signal.** With seasonality
  absorbed, temperature effects are learned only from ±0.5 °C within-month
  deviations; the temperature coefficient was ~0 and SSP5-8.5 projected
  *fewer* cases than SSP2-4.5 by the 2050s. Hence `response="seasonal"`
  (default), with `"anomaly"` kept as the sensitivity check.
- **Shared seasonal cycles confound variables.** With lags 0-3 for every
  variable the model fitted history slightly better (deviance explained
  0.758 vs 0.740) yet put ~0 weight on temperature and over-credited
  rainfall, roughly halving the projected warming response. One lag per
  variable (v1's lags) recovered temperature 0.27 vs true 0.30 and projected
  +48% vs true +60% (SSP5-8.5, 2050s). In-sample fit cannot choose here.
- **A single cross-validated "best" structure is fragile**: it picked
  temperature lag 0 + humidity lag 0 (r = 0.85), projected +4% vs true +60%,
  truth inside the 10-90% range in only 2/4 cases. Six structures were
  within 2% on CV. Default is therefore `lag_selection="ensemble"`
  (average near-best structures): truth inside the range in 4/4 cases, but
  medians still low (+27% vs +60%) because some equally-fitting structures
  give warming little weight. The report states this explicitly and
  recommends supplying lags from v1 or prior studies.

Tests: `tests/test_scenario_outlook.py` (9), dashboard test extended.
Full suite: **71 passed**.

## Not verified here
Real CMIP6 data (downloaded from HuggingFace on the user's machine) could not
be reached from the sandbox; all testing used synthetic CMIP6-shaped data.

---

# Tenth pass — six improvements (release 0.3.0)

1. **Calibrated v2 intervals.** Split-conformal rescaling per lead-time band from the
   hindcasts (`forecasting_v2/calibration.py`, on by default). Band factors use the pooled
   value unless there are ≈2/(1−level) points, and are shrunk toward it. Checked on data the
   calibration never saw (synthetic benchmarks, origin Dec 2018, scored 2019–2020): ensemble
   95% coverage 0.88/0.92 → 1.00/1.00, 80% coverage moved toward nominal, WIS better or similar
   for 7 of 10 model–dataset pairs. An unshrunk first version produced noisy factors (95% factor
   4.1 in one band vs 2.0 in the others); fixed before release.
2. **Multi-district pooling** (`extra_districts`). Shared climate coefficients, own intercepts,
   pooled lag selection and bootstrap. Six synthetic districts with a shared temperature-driven
   response, 2050s truth +27%: target-only +13/+16/+30% (truth outside the range once);
   pooled +25/+24/+30% with narrower ranges, truth inside 3/3.
3. **Thermal-suitability curve** (`temperature_curve`). Unimodal shape with limits and optimum;
   the data estimate strength only. Where the truth has an optimum (hot district, warming past
   it), true change ≈ −25%: default −15/−18%, curve −19/−22%. Modest gain. Preset values for
   *Aedes aegypti* (17.8/29.1/34.6 °C) were entered from memory of Mordecai et al. (2017) and
   are flagged for verification.
4. **Long-term backtest** (`run_backtest=True`). Refits without the last N years and projects
   them from observed climate. On the synthetic benchmark: 3 of 4 years inside the 80% range,
   median error 11% vs 12% for the training-period mean; the report states this cannot
   confirm skill (verdict is three-way, not "better/worse").
5. **Population scaling** (`population_projection`). Results scaled by projected population /
   baseline (constant incidence per person), reported alongside climate-only results. No SSP
   population data is bundled; the user supplies it.
6. **Release tidy.** Version 0.3.0; `CHANGELOG.md`; README "what's new"; technical
   documentation sections 3.2, 4.1, 4.2, 6.2 and 8.1 corrected (8.1 previously claimed the old
   split "prevents data leakage") and section 13 added; `pytest.ini` with a `slow` marker;
   GitHub Actions workflow (fast suite on 3.10–3.12, full suite on 3.12); stale `build/` and
   `dist/` (old unfixed code and a 0.2.0 wheel) removed.

Full suite: **81 passed** (72 fast + 9 slow).
Dashboard: temperature-curve choice added to the scenario card; pooling and population
scaling are Python-API only (they need several districts' files / a population table).


---

# Eleventh pass — reports rewritten for non-specialists

Both v2 reports now open with a plain-language layer (`climaid/reporting_plain.py`), with the
full technical content in a collapsible "Technical details" section.

- Forecast report: short version (total, busiest month, compared with a typical year, which
  months are higher/lower than usual, trust rating, and how many already-known months fell inside
  the likely range), a chart and month-by-month table, "How much can you trust this?" with a
  rule-based rating, plain caveats, glossary. The method shown is named in plain words.
- Scenario report: per-pathway sentences ("about N cases a year, roughly P% more than in the
  baseline, likely range …"), plain pathway names with SSP codes, a confidence rating from the
  backtest, climate-model agreement, sensitivity and structural uncertainty, and what the
  projections do not include.
- Found while reviewing the draft: a sentence claiming "the difference between the pathways grows
  over time" was written unconditionally and was false for the demo data (≈6 points in both the
  2030s and 2050s). It is now computed from the numbers, using the same rounded percentages the
  reader sees. An "overall similar to a typical year" line next to "6 months higher than usual"
  was confusing; it now names the months that are higher and lower.

Tests: `tests/test_plain_reports.py` (7), including a check that no jargon (WIS, quantile,
hindcast, coverage) appears in the plain sections. Full suite: **87 passed**.


---

# Twelfth pass — compulsory tuning, more models, ENSO interactions

- Correction to an earlier statement: v2 already stacked each ML model with a residual random
  forest (temporal out-of-fold). What it lacked versus v1 was tuning and v1's correction stage.
- Tuning (`forecasting_v2/tuning.py`): compulsory, presets Fast/Balanced/Deep. On years never
  seen (4 origins x 2 synthetic datasets, 12-month forecasts), tuning improved WIS in 7 of 10
  model–dataset pairs: Poisson 4.05 -> 2.07 (non-seasonal) and 7.73 -> 6.77 (seasonal), gradient
  boosting 2.22 -> 1.98 and 7.13 -> 6.58; slightly worse for random forest and XGBoost (seasonal)
  and HistGB (non-seasonal). Tuning optimises one-month-ahead training-period error, which usually
  but not always carries over to 12-month forecasts.
- Seven added models; all 19 ML models fit, tune and forecast in a smoke test (CatBoost slowest,
  ~35 s on Fast).
- ENSO interactions: with a true temperature x ENSO interaction, gradient boosting WIS
  7.50 -> 6.08 (RMSE 19.0 -> 14.1), Poisson 7.12 -> 6.98; without one, no degradation.
- Bugs fixed: v2 default-parameter name mismatch (random forests had 100 trees, not 400);
  unscaled linear-type models; Poisson non-convergence.

Tests: `tests/test_v2_tuning_models.py` (7); conftest sets the Fast preset for the suite.
Full suite: **94 passed**.


---

# Thirteenth pass — v1 stack inside v2; tree comparison in scenarios

- `v1_stack` (`forecasting_v2/v1_stack.py`): on the synthetic seasonal benchmark (origin
  Dec 2020, Fast preset) it scored WIS 3.99 vs Poisson 4.22 and seasonal-naive 5.84, with 80%
  coverage 0.83 after calibration. About 70 s per fit on Fast; it is refitted at every hindcast
  origin, so it is off by default.
- Tree comparison: on the synthetic scenario (true SSP5-8.5 2050s change +60%), random forest
  +18% and gradient boosting +16% vs the main projection's +27%, with 29% of projected months
  outside the training climate. Consistent with tree models' inability to extrapolate; reported as
  a comparison only.

Full suite: **98 passed** (one test adjusted: `v1_stack` is a pipeline, not a single estimator).


---

# Fourteenth pass — benchmarks folder and documentation

- `benchmarks/generators.py` collects every synthetic dataset (fixed seeds) and adds
  `realistic_hard()`. `benchmarks/run_benchmarks.py`: 12-month forecasts from 4 origins per dataset
  through `forecast_v2()` (Fast tuning, hindcasts, calibration). WIS relative to seasonal naive:
  seasonal ensemble 0.63; non-seasonal Poisson 0.85 / ensemble 0.88 but random forest 1.23 and
  gradient boosting 1.09 (worse, with 80% coverage 0.58–0.69); realistic ensemble 0.63, driven by
  years where the baseline failed; the 2019 serotype-outbreak year was missed by every method
  (RMSE ≈ 82). 95% ranges covered ~100% of months: probably wider than necessary.
- Documentation (separate MkDocs site): "under testing" banner, status & validation page, v2 /
  scenario / report / benchmark guides, v2 API page, changelog; fixed wrong sample-data link,
  Windows image paths, dependency list, non-existent `climaid[full]` extra.

Full suite: **102 passed**.


---

# Fifteenth pass — COVID-19 period choice

`climaid/exclusion.py`: one rule for disrupted months, used by v1 (`_train_test_split`), v2
(`forecast_v2`, hindcast scoring exclusion, `v1_stack`) and `project_v2`. Options: "2020"
(default), custom "YYYY-MM:YYYY-MM" (whole months, inclusive), "none". No "2020-2021" preset, by
design for South Asia. Dashboard: "COVID-19 disruption period" with month pickers for a custom
period and validation; wizard: 1/2/3 choice. Reports name the excluded period. Also fixed:
`v1_stack` previously always excluded 2020 regardless of the user's choice.


---

# Sixteenth pass — v1 modes in v2; distributed lag effects

- v1 modes (`model_parameters.V1_MODES`): Fast xgb/rf/isotonic 50 trials; Balanced same, 200; Deep
  rf+xgb / xgb+rf+extra_trees / isotonic+poisson+elasticnet, 500; v1 default lag ranges. Dashboard,
  wizard, `v1_stack` and scenario `lag_selection="v1"` all use them. Wizard Deep previously used
  `elastic_net` (not registered: would fail) and wizard Fast differed from the dashboard; unified on the
  dashboard definitions (to be confirmed by the author).
- "Use v1 lags" now runs v1's search on data up to the origin; previously it silently fell back from
  the dashboard because each run starts a fresh model.
- Distributed lags. Forecasts (4 origins x 3 datasets, Fast tuning): WIS better in 7 of 12 pairs;
  Poisson better on all three; trees mixed -> on for regression models only. Scenarios: seasonal
  benchmark +58% vs true +60% (averaged +27%, v1 lags +48%) but ranges up to +200%; six-district world
  (truth +27%): single district +14/+41/+54%, pooled +41/+33/+34% (averaged pooled +25/+24/+30%).
  Kept as an option, not the default. Fitted curves recover total effect better than timing.


---

# Seventeenth pass — effect of the number of optimisation trials; docs bundled

- v2 (WIS vs seasonal naive, origins 2017 and 2021, 3 datasets x 3 models, 5/10/30/80 trials):
  5 -> 10 sometimes helped (seasonal GB 0.63 -> 0.52, Poisson 0.92 -> 0.78); beyond 10-30 flat or worse
  (realistic GB 0.62 -> 0.91 at 80). Runtime roughly proportional (GB 8/16/37/86 s per fit).
- v1 (reduced lag grid, origin 2019): RMSE seasonal 11.6 / 10.2 / 11.0 and realistic 34.9 / 25.3 / 27.5
  at 5 / 20 / 50 trials; ~6.5 s per trial. 200 and 500 trials extrapolated, not run.
- Dashboard tooltips now state what each preset buys; docs page and status summary added.
- Docs shipped in the package and served at /documentation (not /docs, which is FastAPI's API page);
  `climaid docs` CLI command.

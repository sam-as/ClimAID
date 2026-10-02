# ClimAID — Climate change impact using AI on Diseases

**Version 0.4.1 (beta, under active testing)** · [Documentation](https://sam-as.github.io/ClimAID/) ·
[PyPI](https://pypi.org/project/climaid/) · [Changelog](https://github.com/sam-as/ClimAID/blob/main/CHANGELOG.md) ·
[Paper](https://doi.org/10.21203/rs.3.rs-9394047/v1)

ClimAID is an integrated toolkit for modelling, forecasting and projecting climate-sensitive diseases such as
dengue and malaria, using machine learning, a climate-informed transmission model and CMIP6 climate-model
ensembles.

* Built-in climate data for South Asia: India, Nepal, Bhutan, Sri Lanka, Myanmar, Afghanistan, Pakistan and
  Bangladesh.
* Data from other countries are supported through the global mode of the browser interface.

> **Under active testing.** ClimAID 0.4.1 has been checked on synthetic data with a known answer, but **not yet
> validated on real surveillance data or real CMIP6 projections**. Treat outputs as research estimates and do not
> use them as the sole basis for public-health decisions. See
> [Status & validation](https://sam-as.github.io/ClimAID/guide/status/).

---

## What's new in 0.4.1

* **`climaid ai`: built-in assistant.** Describe what you want in plain language ("forecast dengue in Pune for
  the next 6 months"); it asks for anything missing, runs the forecast or climate outlook, and explains the
  results and the methods. Fully offline and with no AI model: every number comes from ClimAID's own results.
  In the terminal, or as a chat page (`climaid ai --browser`, or *Assistant* in the dashboard).
* **SARIMAX** as an optional v2 model (seasonal ARIMA with climate inputs), tuned, backtested and calibrated
  like every other model. Best of all models on the non-seasonal synthetic benchmark; weaker on the realistic
  one, so not a default.

**0.4.0** was the first public release of **ClimAID v2** (the previous public release was 0.1.2, v1 only):
probabilistic forecasts with calibrated likely ranges, a climate-informed renewal (transmission) model, a
seasonal-naive benchmark, models with compulsory leakage-safe tuning, rolling backtests ("hindcasts"), a hybrid
near-term + CMIP6 scenario outlook (`DiseaseModel.project_v2()`), plain-language reports with a trust rating,
and leakage fixes in v1. **v1 metrics from 0.1.x were optimistic; rerun before citing them.**

Full list, including every change that alters results:
[CHANGELOG.md](https://github.com/sam-as/ClimAID/blob/main/CHANGELOG.md).

---

## Installation

```bash
pip install climaid              # core
pip install "climaid[ml]"        # + XGBoost, LightGBM, CatBoost
```

Requires Python 3.10 or newer. For development: `pip install -e ".[test]"` from a clone of this repository.

Optional local-LLM reports: install [Ollama](https://ollama.com) and run `ollama serve`. No extra Python package is
needed; without it, ClimAID uses its built-in deterministic C-DSI reports.

---

## Two pipelines

| | **ClimAID v2** (recommended) | **ClimAID v1** (legacy) |
|---|---|---|
| Purpose | Forecasts for the coming months, and climate-scenario outlooks to 2050–2100 | Lag-optimised point predictions and CMIP6 projections |
| Output | Expected cases **with likely ranges** | Expected cases |
| Models | 21, including a climate transmission model and v1's stacked model | Stacked base → residual → correction models |
| Checked by | Held-out months, rolling backtests, calibrated ranges | One test year |
| Report | Plain-language summary + technical details | C-DSI deterministic report |

---

## Quick examples

```python
from climaid.climaid_model import DiseaseModel

dm = DiseaseModel(district="IND_Pune_MAHARASHTRA", disease_file="dengue.xlsx", disease_name="Dengue")

# v2: probabilistic forecast for the next 12 months
result = dm.forecast_v2(forecast_origin="2023-12-31", horizon=12, tuning="balanced", save_report=True)
print(result["report_path"])

# v2: climate-scenario outlook
outlook = dm.project_v2(end_year=2050, ssps=["ssp245", "ssp585"])
print(outlook["decades"])

# v1 (legacy)
dm.optimize_lags()
dm.train_final_model()
```

Lower-level v2 engine and migration notes: [MIGRATION_v2.md](https://github.com/sam-as/ClimAID/blob/main/MIGRATION_v2.md).

---

## Interfaces

```bash
climaid ai        # built-in assistant: describe what you want; it guides you and runs ClimAID (offline)
climaid ai -b     # the same assistant as a chat page in your browser (also in the dashboard menu)
climaid browse    # browser dashboard (South Asian and global data); v2 and v1 pages
climaid wizard    # terminal wizard (South Asian data); choose v1, v2 or both
climaid docs      # open the documentation bundled with this installation (offline)
climaid --version
```

---

## Documentation and project files

| Where | What |
|---|---|
| [sam-as.github.io/ClimAID](https://sam-as.github.io/ClimAID/) | User guide, API reference, status & validation, changelog |
| [CHANGELOG.md](https://github.com/sam-as/ClimAID/blob/main/CHANGELOG.md) | All changes, by version |
| [MIGRATION_v2.md](https://github.com/sam-as/ClimAID/blob/main/MIGRATION_v2.md) | Moving from v1 to v2 |
| [V2_FEEDBACK_RESPONSE.md](https://github.com/sam-as/ClimAID/blob/main/V2_FEEDBACK_RESPONSE.md) | Reviewer concerns and how v2 addresses them |
| [benchmarks/](https://github.com/sam-as/ClimAID/tree/main/benchmarks) | Synthetic datasets, benchmark runner and saved results |
| [README_DOCS.md](https://github.com/sam-as/ClimAID/blob/main/README_DOCS.md) | Building and releasing the documentation |

---

## Designed for

Epidemiologists, climate scientists, public-health analysts and data scientists.

## Citation

> Sam, A.K., Phuleria, H.C. (2026). ClimAID: An AI-integrated Global Hybrid Climate-Disease Modelling Framework. Preprint: [https://doi.org/10.21203/rs.3.rs-9394047/v1](https://doi.org/10.21203/rs.3.rs-9394047/v1)

## License

MIT. Designed by **Avik Kumar Sam** & **Harish C. Phuleria** as open-access software through the [National Disease Modelling Consortium](www.ndmcconsortium.com). 

- Full text:
[LICENSE](https://github.com/sam-as/ClimAID/blob/main/LICENSE). 

- For technical feedback:
[avik.sam@iitb.ac.in](mailto:avik.sam@iitb.ac.in) | [avik.sam@nus.edu.sg](mailto:avik.sam@nus.edu.sg).

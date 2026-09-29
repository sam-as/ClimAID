# ClimAID — Climate change impact using AI on Diseases

**Version 0.4.0 (beta, under active testing)** · [Documentation](https://sam-as.github.io/ClimAID/) ·
[PyPI](https://pypi.org/project/climaid/) · [Changelog](https://github.com/sam-as/ClimAID/blob/main/CHANGELOG.md) ·
[Paper](https://doi.org/10.21203/rs.3.rs-9394047/v1)

ClimAID is an integrated toolkit for modelling, forecasting and projecting climate-sensitive diseases such as
dengue and malaria, using machine learning, a climate-informed transmission model and CMIP6 climate-model
ensembles.

* Built-in climate data for South Asia: India, Nepal, Bhutan, Sri Lanka, Myanmar, Afghanistan, Pakistan and
  Bangladesh.
* Data from other countries are supported through the global mode of the browser interface.

> **Under active testing.** ClimAID 0.4.0 has been checked on synthetic data with a known answer, but **not yet
> validated on real surveillance data or real CMIP6 projections**. Treat outputs as research estimates and do not
> use them as the sole basis for public-health decisions. See
> [Status & validation](https://sam-as.github.io/ClimAID/guide/status/).

---

## What's new in 0.4.0

0.4.0 is the first public release of **ClimAID v2**; the previous public release was 0.1.2 (v1 only).

* **ClimAID v2 (additive):** probabilistic forecasts with calibrated likely ranges, a climate-informed renewal
  (transmission) model, a seasonal-naive benchmark, 21 models with compulsory leakage-safe tuning, rolling
  backtests ("hindcasts"), and a hybrid near-term + CMIP6 scenario outlook (`DiseaseModel.project_v2()`).
* **Plain-language reports** with a Good / Moderate / Low trust rating; technical details kept in a
  collapsible section.
* **Leakage fixes in v1** (annual-average climate feature, test-set reuse during lag optimisation, projection
  features that did not match training). **v1 metrics from 0.1.x were optimistic; rerun before citing them.**
* Dashboard with separate v2 (default) and v1 pages; documentation bundled offline (`climaid docs`).

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
| [Claude-Testing_FINDINGS_2026-09.md](https://github.com/sam-as/ClimAID/blob/main/Claude-Testing_FINDINGS_2026-09.md) | Testing log: the evidence behind each fix |
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

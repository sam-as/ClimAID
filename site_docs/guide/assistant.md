# ClimAID assistant (`climaid ai`)

!!! warning "Under testing"
    The assistant is new in 0.4.1 and under active testing. It runs the same ClimAID v2 methods as the
    dashboard and wizard, so everything on the [Status & validation](status.md) page applies to its results.

```text
climaid ai              # in the terminal
climaid ai --browser    # as a chat page in your web browser
```

The assistant is a conversational guide, in the terminal or as a chat page in the browser interface
(**Assistant** in the dashboard's menu, or `climaid ai --browser`). You describe what you want in plain words; it asks
for anything missing, shows a plan, runs ClimAID when you say yes, and explains the results.

**It is built in and works offline.** It does not use an AI language model, local or online: it recognises
common requests from keywords and phrases, and every number and sentence about your results comes from
ClimAID's own calculations and its rule-based reports. It cannot invent results. When it does not understand
something, it says so and suggests what you can type.

---

## An example

```text
you> I want a dengue forecast in Pune for the next 6 months
     I found several districts with that name. Which one?
       1. IND_Pune_MAHARASHTRA  ...
you> 1
     Where is your disease data file? ...
you> data/pune_dengue.xlsx
     Data file looks good: 156 monthly records from January 2012 to December 2024 ...
     Here's the plan: a Dengue forecast for IND_Pune_MAHARASHTRA, 6 months ahead ...
     Shall I run it? (yes / no, or tell me what to change)
you> quick tuning, use data up to December 2023
you> yes
     ... expected cases month by month, likely ranges, trust rating, link to the full report
you> how reliable is this?
you> what about 2050 under high emissions?
```

---

## In the browser

The chat page works the same way, with a few conveniences:

* **📎 Upload** your disease data (and, outside South Asia, your own climate data or climate projections)
  instead of typing a path.
* **Suggestion buttons** under the conversation: *Yes, run it*, district choices, follow-up questions.
* **Long runs** continue in the background; the page shows how long the run has taken so far. Keep the
  `climaid browse` (or `climaid ai --browser`) terminal open until it finishes.
* **Links** open the saved report and the bundled documentation directly.

Each browser keeps its own conversation; *Start a new conversation* clears it.

---

## What it can do

| Ask for | Example |
|---|---|
| A forecast of the coming months | `forecast dengue in Pune for the next 12 months` |
| A climate-change outlook | `what could happen by 2050 under SSP2-4.5 and SSP5-8.5?` |
| How much to trust a result | `how reliable is this?` |
| An explanation of results | `explain the results`, `open the report` |
| Answers from the documentation | `what is WIS?`, `why is 2020 excluded?`, `how many trials do I need?` |
| How the methods work | `methods`, `how does calibration work?`, then `more detail` |
| How a run was made | `how was this forecast made?` |
| Finding a district | `find district Kathmandu` |
| Your current settings | `settings` |

## Explaining the methods

Type **`methods`** for a numbered list of topics, or ask about any of them directly ("how does tuning work?",
"what is the renewal model?", "how are climate models bias-corrected?"). Each answer comes in plain language
first; say **`more detail`** for the technical version (equations, defaults, lag ranges), with a link to the
documentation page.

| Forecasting | Climate outlook | General |
|---|---|---|
| Data checks; the COVID-19 period; climate inputs and lags; distributed lags; the models; the baseline; the transmission (renewal) model; residual learning and the ensemble; tuning; backtests; calibrated likely ranges; scoring (WIS) | How the outlook is built; bias correction; seasonal vs anomaly response; lag structure; pooling districts; thermal-suitability curve | Overview; the trust rating; avoiding leakage; ClimAID v1; limitations |

After a run, **"how was this forecast made?"** describes that particular run from its own record: the data
period, the models compared, which models tuning improved and which kept their defaults, the backtest start
dates, the calibration, the COVID-19 handling and why the model in the summary was chosen.

Settings it understands anywhere in a sentence:

| Setting | Say |
|---|---|
| Data file | a path ending in `.csv`, `.xlsx` or `.xls` (in quotes if it has spaces) |
| Forecast start | `use data up to December 2023`, `from 2023-12` |
| Months ahead | `6 months`, `next year` (1–36; for longer, use the outlook) |
| Tuning | `quick` / `balanced` / `thorough`, or `50 trials` |
| COVID-19 period | `keep 2020`, `exclude 2020-03 to 2020-12` |
| Emissions pathways | `SSP2-4.5`, `high emissions`, `all scenarios`; `also include ...` adds to the list |
| Outlook end | `by 2050`, `to 2080` |
| Your own climate data | `my climate file is "weather.csv"` (places outside the built-in South Asia data); for outlooks also `my projection file is "cmip6.csv"` |

The assistant always shows the plan and waits for **yes** before running anything. Reports are saved in
`climaid_outputs/reports`; ClimAID's progress messages go to `climaid_outputs/assistant_log.txt`.

---

## Limits

* It understands common phrasings, not every possible sentence. Rephrase, or type `help` for examples.
* It runs one district at a time with ClimAID's default v2 models. Pooling several districts
  (`extra_districts`), choosing individual models and other advanced options are available from Python
  (see the [v2 API](../api/v2.md)) and the dashboard.
* Documentation answers are short summaries with a link; the linked page has the full explanation.
* The first run for a region downloads ClimAID's climate data, which needs an internet connection once.

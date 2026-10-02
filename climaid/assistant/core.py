"""The ClimAID assistant: a rule-based conversational guide (no language model).

It understands plain requests ("forecast dengue in Pune for the next 6 months"), asks for
whatever is missing, shows a plan and asks before running anything, runs ClimAID, and explains
the results and the documentation. All numbers come from ClimAID's own code.

``Assistant.respond(text)`` yields the reply as one or more messages, so a terminal (or a
dashboard) can show "Running ..." before a long run finishes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from climaid import __version__
from . import knowledge, methods
from .actions import Runner, format_months, inspect_disease_file, summarise_forecast, summarise_projection
from .understanding import district_mentions, match_districts, parse

SSP_NAMES = {"ssp119": "SSP1-1.9 (very low emissions)", "ssp126": "SSP1-2.6 (low emissions)",
             "ssp245": "SSP2-4.5 (middle of the road)", "ssp370": "SSP3-7.0 (high emissions)",
             "ssp585": "SSP5-8.5 (very high emissions)"}
TUNING_NAMES = {"fast": "Fast (10 trials per model)", "balanced": "Balanced (30 trials per model)",
                "deep": "Deep (80 trials per model)"}
CAVEAT = (f"Remember: ClimAID {__version__} is under active testing. Its methods have been checked on synthetic "
          "data only, so treat these results as research estimates, not as the sole basis for decisions.")
REQUEST = re.compile(r"^(can|could|would|will) you|^please|^let'?s|^i (want|need|would like|'d like)|"
                     r"^(run|do|make|give|create|start|show me a|forecast|predict|project)\b")

# Questions about how ClimAID works go to the documentation even when they mention a task.
DOCS_QUESTION = re.compile(r"^(how (does|do|is|are|can i|should i)|what (is|are|does|do)\b|why|explain|define|"
                           r"what's|whats|meaning of)")

INTRO = (f"Hello! I'm the ClimAID assistant (ClimAID {__version__}). I work fully offline and don't use an AI "
         "model: I understand common requests, run ClimAID for you, and explain the results using ClimAID's own "
         "reports and documentation.\n\n"
         "I can:\n"
         "  - forecast the coming months, e.g. \"forecast dengue in Pune for the next 12 months\"\n"
         "  - project how cases could change with climate, e.g. \"what could happen by 2050 under high emissions?\"\n"
         "  - explain results and how much to trust them, e.g. \"how reliable is this?\"\n"
         "  - explain the methods, e.g. \"how does tuning work?\", or type \"methods\" for all topics\n"
         "  - answer questions from the documentation, e.g. \"what is WIS?\" or \"why is 2020 excluded?\"\n\n"
         "To start, tell me what you'd like to do and where your disease data file is (CSV or Excel, with a date "
         "column and a case-count column). Type \"help\" at any time, or \"quit\" to leave.")

HELP = """Things you can say:
  forecast dengue in Pune for the next 6 months using "data/pune_dengue.xlsx"
  use data up to December 2023            (forecast start)
  quick / balanced / thorough tuning      (Fast / Balanced / Deep)
  keep 2020   or   exclude 2020-03 to 2020-12   (COVID-19 period)
  climate change to 2050 under SSP2-4.5 and SSP5-8.5   (or "all scenarios")
  my climate file is "weather.csv"        (for places outside the built-in South Asia data)
  how reliable is this?   explain the results   open the report
  settings   find district Kathmandu   start over   quit
  methods   (all topics)    explain calibration    more detail    how was this forecast made?
Or ask a question, e.g. "what does likely range mean?" """


class Assistant:
    def __init__(self, runner: Runner | None = None, districts: list[str] | None = None,
                 report_link=None, docs_base: str | None = None, file_prompt: str | None = None):
        """report_link: function turning a saved report path into a link (default: a file:// URI);
        docs_base: where the bundled documentation is served (default: file:// links)."""
        self.runner = runner or Runner()
        self._districts = districts
        self.report_link = report_link or (lambda p: Path(p).resolve().as_uri())
        self.docs_base = docs_base
        self.file_prompt = file_prompt
        self.settings: dict = {}
        self.task: str | None = None              # "forecast" | "project"
        self.awaiting: str | None = None          # what the last question asked for
        self.choices: list[str] = []              # numbered district choices
        self.data_info: dict | None = None
        self.results: dict = {}                   # kind -> summary
        self.last_kind: str | None = None
        self._dm = None
        self._dm_key = None
        self.finished = False
        self.last_topic = None                    # methods topic last explained (for "more detail")

    # ------------------------------------------------------------------ helpers
    @property
    def districts(self) -> list[str]:
        if self._districts is None:
            from climaid.districts import get_available_districts
            try:
                self._districts = get_available_districts()
            except Exception:
                self._districts = []
        return self._districts

    def ask(self, text: str) -> str:
        """Whole reply as one string (convenient for tests and simple frontends)."""
        return "\n\n".join(self.respond(text))

    def greet(self) -> str:
        return INTRO

    def reset(self):
        self.__init__(runner=self.runner, districts=self._districts, report_link=self.report_link,
                      docs_base=self.docs_base, file_prompt=self.file_prompt)

    # ------------------------------------------------------------------ main entry
    def respond(self, text: str) -> Iterator[str]:
        text = (text or "").strip()
        if not text:
            yield "Type a request or a question (or \"help\")."
            return
        p = parse(text)
        low = p.text.lower().strip()
        intent = p.intent

        if intent == "quit" and len(low.split()) <= 3:
            self.finished = True
            yield "Goodbye! Your reports are saved in climaid_outputs/reports."
            return
        if intent == "reset" and len(low.split()) <= 4:
            self.reset()
            yield "OK, starting over. What would you like to do?"
            return

        # answers to a question I asked
        if self.awaiting == "district_choice" and p.number is not None:
            if 1 <= p.number <= len(self.choices):
                self.settings["district"] = self.choices[p.number - 1]
                self.awaiting, self.choices = None, []
                yield f"District: {self.settings['district']}."
                yield from self._advance()
            else:
                yield f"Please pick a number from 1 to {len(self.choices)}."
            return
        # methods: "more detail", the topics menu, a topic number
        if methods.MORE_WORDS.match(low) and self.last_topic:
            yield methods.describe(self.last_topic, detailed=True, base=self._docs_root())
            return
        if self.awaiting == "topic_choice" and p.number is not None:
            if 1 <= p.number <= len(methods.TOPICS):
                self.awaiting = None
                self.last_topic = methods.TOPICS[p.number - 1]
                yield methods.describe(self.last_topic, base=self._docs_root())
            else:
                yield f"Please pick a number from 1 to {len(methods.TOPICS)}."
            return
        if methods.MENU_WORDS.match(low.strip(" ?.!")):
            if self.awaiting != "confirm":
                self.awaiting = "topic_choice"
            yield methods.menu()
            return
        if self.awaiting == "confirm":
            if p.yes:
                self.awaiting = None
                yield from self._run()
                return
            if p.no:
                self.awaiting = None
                yield "OK, not running it. Tell me what to change (for example \"6 months\", \"quick tuning\", " \
                      "\"keep 2020\"), or ask me anything."
                return

        messages = list(self._take_settings(p))
        yield from messages

        if self.awaiting == "disease_name" and "disease_name" not in p.slots and not p.question and not p.intents:
            name = re.sub(r"[^A-Za-z \-]", "", text).strip()
            self.settings["disease_name"] = name.title() if name and name.lower() not in ("skip", "none", "unknown") \
                else "Disease"
            self.awaiting = None
            yield f"Disease: {self.settings['disease_name']}."
            yield from self._advance()
            return
        if self.awaiting == "disease_file" and "disease_file" not in p.slots and Path(text.strip("'\"")).exists():
            yield from self._set_file(text.strip("'\""))
            yield from self._advance()
            return

        is_request = bool(REQUEST.search(low))
        asking = p.question or low.startswith(("tell me about", "describe", "what about the", "explain"))
        if asking and not is_request:
            if methods.RUN_WORDS.search(low):
                yield self._explain_run()
                return
            t = methods.find_topic(low)
            if t is not None and not (intent == "explain" and self.results and t.key == "overview"):
                self.last_topic = t
                yield methods.describe(t, base=self._docs_root())
                return
        about_how = bool(DOCS_QUESTION.search(low))
        task_words = intent in ("forecast", "project")
        if p.question and not is_request and (about_how or not task_words) \
                and intent not in ("settings", "districts", "report", "trust", "explain"):
            yield from self._docs(text)
            return

        if intent in ("forecast", "project"):
            self.task = intent
            yield from self._advance()
            return
        if intent == "trust":
            yield from self._trust()
            return
        if intent == "explain" and self.results:
            yield self._show_result(self.results[self.last_kind])
            return
        if intent == "settings":
            yield self._settings_text()
            return
        if intent == "districts":
            yield from self._find_districts(re.sub(r".*district[s]?\s*", "", low))
            return
        if intent == "report":
            yield self._report_text()
            return
        if p.greeting or intent == "help":
            yield INTRO if p.greeting else HELP
            return
        if self.task and (messages or p.slots):
            yield from self._advance()
            return
        if messages:
            if not self.task:
                yield "What would you like to do with it: a forecast for the coming months, or a climate-change outlook?"
            return
        if p.question:
            yield from self._docs(text)
            return
        yield ("I didn't understand that. You can ask for a forecast (\"forecast dengue for the next 6 months\"), "
               "a climate outlook (\"what could happen by 2050?\"), or ask a question (\"what is WIS?\"). "
               "Type \"help\" for more examples.")

    # ------------------------------------------------------------------ settings from a message
    def _take_settings(self, p) -> Iterator[str]:
        s = dict(p.slots)
        if "disease_file" in s:
            yield from self._set_file(s.pop("disease_file"))
        for key in ("weather_file", "projection_file"):
            if key in s:
                path = s.pop(key)
                if Path(path).expanduser().exists():
                    self.settings[key] = str(Path(path).expanduser())
                    yield f"{'Climate' if key == 'weather_file' else 'Climate-projection'} file: {path}."
                else:
                    yield f"I can't find {path}; check the path."
        if "horizon" in s:
            h = int(s.pop("horizon"))
            if not 1 <= h <= 36:
                yield "Forecasts can look 1 to 36 months ahead; for longer questions use the climate outlook."
            else:
                self.settings["horizon"] = h
                if h > 12:
                    yield (f"Note: {h} months is a long forecast. Forecasts rely on recent cases, which stop being "
                           "informative after about a year; for multi-year questions the climate outlook is better.")
        for key in ("disease_name", "forecast_origin", "tuning", "exclude_period", "end_year"):
            if key in s:
                self.settings[key] = s.pop(key)
        if "ssps" in s:
            new = s.pop("ssps")
            old = self.settings.get("ssps")
            if new and old and re.search(r"\b(also|add|include|plus|as well)\b", p.text.lower()):
                new = sorted(set(old) | set(new))
            self.settings["ssps"] = new
        # district: mentioned in passing ("in Pune"), or the answer to my question
        if p.intent != "districts" and ("district" not in self.settings or district_mentions(p.text)):
            candidates = district_mentions(p.text)
            if self.awaiting == "district" and not candidates:
                candidates = [p.text]
            for c in candidates:
                found = list(self._resolve_district(c))
                if found:
                    yield from found
                    break

    def _set_file(self, path) -> Iterator[str]:
        info = inspect_disease_file(path)
        if not info["ok"]:
            self.settings.pop("disease_file", None)
            self.awaiting = "disease_file" if self.task else None
            yield "There's a problem with that file:\n  - " + "\n  - ".join(info["problems"])
            return
        self.settings["disease_file"] = info["path"]
        self.data_info = info
        if self.awaiting == "disease_file":
            self.awaiting = None
        yield f"Data file looks good: {info['summary']}."

    def _resolve_district(self, text: str) -> Iterator[str]:
        if self.settings.get("weather_file"):
            # own climate data: the name must match its Dist_States column; accept as given
            self.settings["district"] = text.strip()
            yield f"District: {text.strip()} (from your climate file)."
            return
        matches = match_districts(text, self.districts)
        if len(matches) == 1:
            self.settings["district"] = matches[0]
            if self.awaiting == "district":
                self.awaiting = None
            yield f"District: {matches[0]}."
        elif len(matches) > 1:
            self.choices = matches
            self.awaiting = "district_choice"
            yield "I found several districts with that name. Which one?\n" + \
                  "\n".join(f"  {i}. {d}" for i, d in enumerate(matches, 1))
        elif self.awaiting == "district":
            yield ("I couldn't find that district in the built-in South Asia data. Try another spelling, or type "
                   "\"find district <name>\". For other countries, give me your own climate file "
                   "(\"my climate file is weather.csv\").")

    def _find_districts(self, query: str) -> Iterator[str]:
        query = query.strip(" ?.")
        if not query:
            yield (f"The built-in data cover {len(self.districts):,} districts in South Asia, named like "
                   "IND_Pune_MAHARASHTRA. Tell me a place name, e.g. \"find district Pune\".")
            return
        found = match_districts(query, self.districts, limit=15)
        if not found:
            yield f"No built-in district matches \"{query}\"."
            return
        self.choices, self.awaiting = found, "district_choice"
        yield "Matching districts (type a number to choose one):\n" + \
              "\n".join(f"  {i}. {d}" for i, d in enumerate(found, 1))

    # ------------------------------------------------------------------ the guided flow
    def _advance(self) -> Iterator[str]:
        if not self.task:
            return
        if self.awaiting == "district_choice":
            return
        if "disease_file" not in self.settings:
            self.awaiting = "disease_file"
            yield (self.file_prompt or "Where is your disease data file? Give me the path, e.g. "
                   "\"data/pune_dengue.xlsx\" (CSV or Excel, with a date column and a case-count column).")
            return
        if "district" not in self.settings:
            self.awaiting = "district"
            yield ("Which district is it for? Give me a place name, e.g. \"Pune\" (or the full code, "
                   "e.g. IND_Pune_MAHARASHTRA).")
            return
        if "disease_name" not in self.settings:
            self.awaiting = "disease_name"
            yield "Which disease is it (e.g. dengue, malaria)? Used for the report title; say \"skip\" to leave it out."
            return
        self.awaiting = "confirm"
        yield self._plan()

    def _plan(self) -> str:
        s = self.settings
        place = s["district"]
        tuning = s.get("tuning") or "balanced"
        tuning_txt = TUNING_NAMES.get(tuning, f"{tuning} trials per model")
        excl = s.get("exclude_period") or "2020"
        excl_txt = {"2020": "2020 left out of the training history (COVID-19 disruption)",
                    "none": "no period left out"}.get(excl, f"{excl} left out of the training history")
        start = s.get("forecast_origin")
        start_txt = (f"using data up to {pd.Timestamp(start):%B %Y}" if start else
                     f"using all your data (up to {self.data_info['end']:%B %Y})" if self.data_info else
                     "using all your data")
        if self.task == "forecast":
            h = int(s.get("horizon") or 12)
            lines = [f"Here's the plan: a {s['disease_name']} forecast for {place}, {h} months ahead, {start_txt}.",
                     f"  - Models: ClimAID's six default v2 models, compared with the 'same as recent years' baseline",
                     f"  - Tuning: {tuning_txt}",
                     f"  - COVID-19 period: {excl_txt}",
                     "  - Backtests from 4 earlier start dates, used to check the models and calibrate the likely ranges",
                     "The first run for a region downloads ClimAID's climate data (internet needed once).",
                     "This can take from a few minutes (Fast) to well over ten minutes (Balanced or Deep) on one CPU core, "
                     "because the backtests repeat the fitting; say \"quick tuning\" for a faster first look."]
        else:
            end = int(s.get("end_year") or 2050)
            ssps = s.get("ssps")
            ssp_txt = ", ".join(SSP_NAMES.get(x, x) for x in ssps) if ssps else "all available emissions pathways"
            lines = [f"Here's the plan: a climate-change outlook for {s['disease_name']} in {place} to {end}, {start_txt}.",
                     f"  - Emissions pathways: {ssp_txt}",
                     "  - Every available climate model, bias-corrected to your district's observed climate",
                     f"  - Tuning: {tuning_txt};  COVID-19 period: {excl_txt}",
                     "  - Checks: a backtest on held-out past years and a sensitivity run",
                     "The first outlook may need to download climate projections (internet needed once). "
                     "It takes longer than a forecast."]
        lines.append("Shall I run it? (yes / no, or tell me what to change)")
        return "\n".join(lines)

    def _model(self):
        key = tuple(self.settings.get(k) for k in ("disease_file", "district", "disease_name",
                                                   "weather_file", "projection_file"))
        if self._dm is None or key != self._dm_key:
            self._dm = self.runner.load_model(self.settings)
            self._dm_key = key
        return self._dm

    def _run(self) -> Iterator[str]:
        kind = self.task
        log = getattr(self.runner, "log_path", None)
        yield (("Running the forecast..." if kind == "forecast" else
                "Running the climate outlook... this can take a while.")
               + (f" (progress is written to {log})" if log else ""))
        try:
            dm = self._model()
            if kind == "forecast":
                raw = self.runner.forecast(dm, self.settings)
                res = summarise_forecast(raw, dm, self.settings["disease_name"], self.settings["district"])
            else:
                raw = self.runner.project(dm, self.settings)
                res = summarise_projection(raw, self.settings["disease_name"], self.settings["district"])
        except Exception as exc:
            msg = f"The run stopped with an error: {type(exc).__name__}: {exc}"
            if re.search(r"HTTP|Connection|Timeout|resolve|urlopen|Max retries", f"{type(exc).__name__} {exc}", re.I):
                msg += ("\nClimAID downloads its climate-projection data the first time it is used for a region, "
                        "which needs an internet connection once; afterwards it works offline. Check your connection "
                        "(or proxy) and try again.")
            elif "future climate" in str(exc).lower():
                end = f" Your data end in {self.data_info['end']:%B %Y}." if self.data_info else ""
                msg += ("\nA forecast needs climate data for the months ahead, and ClimAID's climate data may end "
                        f"when your case data do.{end} Try an earlier forecast start, e.g. \"use data up to "
                        "December 2023\", to test the forecast against months that are already known.")
            else:
                msg += ("\nCommon causes: the district has no climate data for your dates, too few months of data, or "
                        "(for outlooks) no climate projections for this district.")
            log = getattr(self.runner, "log_path", None)
            yield msg + (f"\nDetails are in {log}." if log else "")
            return
        self.results[kind] = res
        self.last_kind = kind
        yield self._show_result(res)
        yield ("You can now ask \"how reliable is this?\", \"open the report\", ask what a term means, or "
               + ("ask for a climate outlook (\"what about 2050?\")." if kind == "forecast"
                  else "ask for a forecast of the coming months."))

    # ------------------------------------------------------------------ results
    def _show_result(self, r: dict) -> str:
        out = []
        if r["summary"]:
            out.append(r["summary"])
        if r["kind"] == "forecast":
            from climaid.reporting_plain import MODEL_PLAIN
            name = MODEL_PLAIN.get(r["primary"], r["primary"])
            why = {"past tests": ", the method that did best in past tests",
                   None: ""}.get(r.get("primary_source"), f", the method that did best on {r.get('primary_source')}")
            out.append(f"Month by month (from the {name}{why}):\n" + format_months(r["months"]))
        out.append(f"Trust rating: {r['rating']}" + ("".join(f"\n  - {x}" for x in r["reasons"]) if r["reasons"] else ""))
        if r["kind"] == "projection" and r.get("not_included"):
            out.append("Not included in these projections:\n" + r["not_included"])
        if r.get("warnings"):
            out.append("Warnings from the run:\n" + "\n".join(f"  - {w}" for w in r["warnings"][:5]))
        if r.get("report_path"):
            out.append(f"Full report (with charts and technical details): {self.report_link(r['report_path'])}")
        out.append(CAVEAT)
        return "\n\n".join(out)

    def _trust(self) -> Iterator[str]:
        if not self.results:
            yield ("I haven't run anything yet, so there's no result to rate. Every ClimAID report includes a "
                   "Good / Moderate / Low trust rating.")
            ans = knowledge.answer("trust rating", self.docs_base)
            if ans:
                yield ans
            return
        r = self.results[self.last_kind]
        rule = ("one point each for: the likely range contained the real number about as often as it should; "
                "the method beat the 'same as recent years' baseline in past tests; the methods broadly agree; "
                "and the forecast looks no more than 12 months ahead" if r["kind"] == "forecast" else
                "one point each for: doing at least as well as the historical average on past years it had not seen; "
                "the climate models agreeing on the direction of change; results holding up under a different way "
                "of estimating climate effects; and the data pointing to a clear explanation")
        yield (f"Trust rating: {r['rating']}.\n" + "\n".join(f"  - {x}" for x in r["reasons"]) +
               f"\n\nHow it's worked out: {rule} (4 points = Good, 2-3 = Moderate, 0-1 = Low). A Good rating means "
               "the method did well in past tests on your data; it is not a guarantee.\n\n" + CAVEAT)

    def _settings_text(self) -> str:
        if not self.settings and not self.task:
            return "Nothing set yet. Tell me what you'd like to do."
        names = {"disease_file": "Disease data", "district": "District", "disease_name": "Disease",
                 "weather_file": "Climate file", "projection_file": "Projection file", "horizon": "Months ahead",
                 "forecast_origin": "Forecast start", "tuning": "Tuning", "exclude_period": "COVID-19 period left out",
                 "ssps": "Emissions pathways", "end_year": "Outlook to"}
        lines = [f"Task: {self.task or 'not chosen yet'}"]
        for k, label in names.items():
            if k in self.settings:
                v = self.settings[k]
                if k == "ssps":
                    v = ", ".join(v) if v else "all available"
                lines.append(f"  {label}: {v}")
        return "\n".join(lines)

    def _report_text(self) -> str:
        paths = [r["report_path"] for r in self.results.values() if r.get("report_path")]
        if not paths:
            return "There's no report yet. Run a forecast or an outlook first."
        return "Reports (open in your browser):\n" + "\n".join(f"  {self.report_link(p)}" for p in paths)

    def _docs_root(self) -> str:
        return self.docs_base if self.docs_base is not None else knowledge.SITE_URL

    def _explain_run(self) -> str:
        r = self.results.get("forecast")
        if r is None:
            if "projection" in self.results:
                return ("The climate outlook was made as described here:\n\n"
                        + methods.describe(methods.topic("scenarios"), base=self._docs_root()))
            return ("I haven't run anything yet. Type \"methods\" to see how ClimAID's methods work, or ask for a "
                    "forecast.")
        return methods.explain_run(r.get("metadata", {}), r.get("hindcast_metrics"), r.get("primary"),
                                   r.get("primary_source"))

    def _docs(self, question: str) -> Iterator[str]:
        ans = knowledge.answer(question, self.docs_base)
        if ans:
            yield ans
        else:
            yield ("I couldn't find that in the documentation. Try other words, or browse it with "
                   "\"climaid docs\" (offline) or https://sam-as.github.io/ClimAID/.")

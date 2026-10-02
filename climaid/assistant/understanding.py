"""Rule-based understanding of what the user typed.

No language model is used: intents are recognised from keywords and phrases, and
settings (file, district, disease, dates, scenarios ...) are extracted with regular
expressions. Everything here is deterministic and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

DATA_SUFFIXES = (".csv", ".xlsx", ".xls", ".parquet")

# Intent -> phrases that suggest it. Each matching phrase adds its weight; the best
# score wins. Multi-word phrases are weighted higher than single words.
INTENT_PHRASES = {
    "forecast": ["forecast", "predict", "prediction", "next season", "next year", "coming months",
                 "next few months", "next months", "months ahead", "outlook for next", "early warning"],
    "project": ["climate change", "scenario", "scenarios", "projection", "project", "ssp", "cmip6",
                "future climate", "warming", "2030", "2040", "2050", "2060", "2070", "2080", "2090", "2100",
                "long term", "long-term", "emissions"],
    "trust": ["trust", "reliable", "reliability", "confident", "confidence", "accurate", "accuracy",
              "how good", "can i use", "should i believe", "rating"],
    "explain": ["explain", "what does the result", "what do the results", "summarise", "summarize",
                "summary", "interpret", "what did you find", "results", "result"],
    "settings": ["settings", "what do you have", "what have i told", "current setup", "show setup", "status"],
    "districts": ["list districts", "which districts", "available districts", "what districts",
                  "find district", "search district"],
    "report": ["open report", "show report", "open the report", "where is the report", "report path"],
    "help": ["help", "what can you do", "how do i", "how does this work", "get started", "start", "guide me"],
    "reset": ["reset", "start over", "start again", "new analysis", "clear"],
    "quit": ["quit", "exit", "bye", "goodbye"],
}
YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "go", "go ahead", "run", "run it", "do it",
       "please do", "correct", "proceed", "start"}
NO = {"no", "n", "nope", "not now", "cancel", "stop", "don't", "do not", "wait"}
GREETINGS = {"hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"}
QUESTION_WORDS = ("what", "why", "how", "when", "which", "where", "who", "does", "do ", "is ", "are ",
                  "can ", "should", "explain")

DISEASES = ["dengue", "malaria", "chikungunya", "zika", "leptospirosis", "cholera", "diarrhoea",
            "diarrhea", "typhoid", "japanese encephalitis", "scrub typhus", "influenza", "covid"]

SSP_WORDS = {
    "ssp119": ["ssp119", "ssp1-1.9", "very low emission"],
    "ssp126": ["ssp126", "ssp1-2.6", "low emission", "strong climate action"],
    "ssp245": ["ssp245", "ssp2-4.5", "middle", "moderate emission", "medium emission", "intermediate"],
    "ssp370": ["ssp370", "ssp3-7.0", "high emission"],
    "ssp585": ["ssp585", "ssp5-8.5", "very high emission", "worst case", "worst-case", "business as usual"],
}
TUNING_WORDS = {"fast": ["fast", "quick", "quickly", "rough"],
                "balanced": ["balanced", "normal", "standard", "default"],
                "deep": ["deep", "thorough", "careful", "best possible"]}
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


@dataclass
class Parsed:
    """What was understood from one message."""
    text: str
    intents: dict = field(default_factory=dict)     # intent -> score
    slots: dict = field(default_factory=dict)       # setting -> value
    yes: bool = False
    no: bool = False
    greeting: bool = False
    question: bool = False
    number: int | None = None                      # a bare number (e.g. a menu choice)

    @property
    def intent(self) -> str | None:
        return max(self.intents, key=self.intents.get) if self.intents else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has(phrase: str, text: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def _month_end(year: int, month: int) -> str:
    import calendar
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def extract_file(raw: str) -> str | None:
    """A path to a data file: quoted, or a token ending in a data suffix."""
    for q in re.findall(r"[\"']([^\"']+)[\"']", raw):
        if q.lower().endswith(DATA_SUFFIXES):
            return q
    m = re.search(r"(\S+(?:" + "|".join(re.escape(s) for s in DATA_SUFFIXES) + r"))(?![\w])", raw, re.I)
    return m.group(1).strip(",;") if m else None


def extract_origin(text: str) -> str | None:
    """Forecast start, e.g. 'from 2023-12', 'up to December 2023', 'data until 2023'."""
    m = re.search(r"(?:from|after|starting|start|origin|up to|until|till|through|data to)\s+"
                  r"(?:the\s+)?(?:end of\s+)?(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return _month_end(y, mo)
    m = re.search(r"(?:from|after|starting|start|origin|up to|until|till|through|data to)\s+"
                  r"(?:the\s+)?(?:end of\s+)?([a-z]{3})[a-z]*\.?\s+(\d{4})", text)
    if m and m.group(1) in MONTHS:
        return _month_end(int(m.group(2)), MONTHS[m.group(1)])
    m = re.search(r"(?:up to|until|till|through|data to|end of)\s+(\d{4})(?![\d-])", text)
    if m:
        return f"{int(m.group(1)):04d}-12-31"
    return None


def extract_horizon(text: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*(?:-\s*)?months?", text)
    if m:
        return int(m.group(1))
    if re.search(r"\b(?:a|one|next)\s+year\b", text):
        return 12
    m = re.search(r"\b(\d)\s*years?\b", text)
    if m and not re.search(r"\b(19|20)\d\d\b", text[m.start():m.end()]):
        return 12 * int(m.group(1))
    return None


def extract_end_year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(20[3-9]\d|2100)\b", text)]
    return max(years) if years else None


def extract_ssps(text: str) -> list[str] | None:
    """SSP codes mentioned by code or in words. [] means 'all available'."""
    if re.search(r"\b(all|every)\s+(scenarios|pathways|ssps)\b", text):
        return []
    rest, found = text, []
    # longest phrases first, removed once matched, so "very high emission" is not also "high emission"
    for phrase, ssp in sorted(((w, s) for s, ws in SSP_WORDS.items() for w in ws), key=lambda t: -len(t[0])):
        if phrase in rest:
            rest = rest.replace(phrase, " ")
            if ssp not in found:
                found.append(ssp)
    return sorted(found) or None


def extract_tuning(text: str) -> str | int | None:
    m = re.search(r"(\d{1,4})\s*trials?", text)
    if m:
        return int(m.group(1))
    for preset, words in TUNING_WORDS.items():
        if any(_has(w, text) for w in words):
            return preset
    return None


def extract_exclude_period(text: str) -> str | None:
    if re.search(r"(keep|include|don'?t (?:exclude|drop|remove)|do not (?:exclude|drop|remove)|no exclusion)"
                 r".{0,20}(2020|covid)", text) or re.search(r"(2020|covid).{0,15}(keep|include)", text):
        return "none"
    m = re.search(r"(?:exclude|drop|remove|skip)\D{0,20}(\d{4})-(\d{1,2})\s*(?:to|-|until|:)\s*(\d{4})-(\d{1,2})",
                  text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}:{int(m.group(3)):04d}-{int(m.group(4)):02d}"
    if re.search(r"(exclude|drop|remove|skip).{0,15}(2020|covid)", text):
        return "2020"
    return None


def extract_disease(text: str) -> str | None:
    text = re.sub(r"\S+(?:" + "|".join(re.escape(x) for x in DATA_SUFFIXES) + r")\b", " ", text)   # not file names
    found = [(m.start(), d) for d in DISEASES for m in [re.search(r"(?<![a-z0-9])" + re.escape(d) + r"(?![a-z0-9])", text)] if m]
    if found:
        d = min(found)[1]
        return d.title() if d != "covid" else "COVID-19"
    m = re.search(r"\bdisease (?:is|name is|called)\s+([a-z][a-z \-]{2,30}?)(?:[,.;]|$| and | in )", text)
    return m.group(1).strip().title() if m else None


def parse(raw: str) -> Parsed:
    """Understand one message."""
    text = _norm(raw)
    p = Parsed(text=raw)
    bare = text.strip(" .!?")
    p.yes = bare in YES
    p.no = bare in NO
    p.greeting = bare in GREETINGS or any(text.startswith(g + " ") for g in GREETINGS)
    p.question = text.endswith("?") or text.startswith(QUESTION_WORDS)
    if re.fullmatch(r"\d{1,3}", bare):
        p.number = int(bare)

    for intent, phrases in INTENT_PHRASES.items():
        score = sum((2 if " " in ph else 1) for ph in phrases if _has(ph, text))
        if score:
            p.intents[intent] = score

    slots = {}
    f = extract_file(raw)
    if f:
        if re.search(r"projections?\s+(?:data|file)", text) or "cmip6 file" in text:
            slots["projection_file"] = f
        elif re.search(r"(climate|weather)\s+(?:data|file)", text):
            slots["weather_file"] = f
        else:
            slots["disease_file"] = f
    for key, fn in (("disease_name", extract_disease), ("forecast_origin", extract_origin),
                    ("ssps", extract_ssps), ("tuning", extract_tuning), ("exclude_period", extract_exclude_period)):
        v = fn(text)
        if v is not None:
            slots[key] = v
    if p.number is None:
        h = extract_horizon(text)
        if h:
            slots["horizon"] = h
        y = extract_end_year(text)
        if y and (p.intents.get("project") or re.search(r"\b(by|to|until|till|through)\s+" + str(y), text)):
            slots["end_year"] = y
    p.slots = slots
    return p


# --------------------------------------------------------------------------- districts

def match_districts(query: str, districts: list[str], limit: int = 8) -> list[str]:
    """Districts that match free text such as 'Pune', 'pune maharashtra' or 'IND_Pune_MAHARASHTRA'."""
    q = _norm(query)
    if not q:
        return []
    exact = [d for d in districts if d.lower() == q]
    if exact:
        return exact
    words = [w for w in re.split(r"[\s_,]+", q) if len(w) > 2]
    if not words:
        return []

    def parts(d):
        return [x.lower() for x in d.split("_")]

    scored = []
    for d in districts:
        ps = parts(d)
        name = ps[1] if len(ps) > 1 else ps[0]
        hits = sum(1 for w in words if any(w == x for x in ps))
        if hits:
            scored.append((hits + (1 if any(w == name for w in words) else 0), d))
    if not scored:   # partial names, e.g. "kath" -> Kathmandu
        for d in districts:
            ps = parts(d)
            name = ps[1] if len(ps) > 1 else ps[0]
            if any(name.startswith(w) for w in words if len(w) >= 3):
                scored.append((1, d))
    if scored:
        best = max(s for s, _ in scored)
        return [d for s, d in sorted(scored, key=lambda t: (-t[0], t[1])) if s == best][:limit]
    # fuzzy fallback (spelling mistakes): compare against the district-name part
    fuzzy = []
    for d in districts:
        ps = parts(d)
        name = ps[1] if len(ps) > 1 else ps[0]
        r = max(SequenceMatcher(None, w, name).ratio() for w in words)
        if r >= 0.8:
            fuzzy.append((r, d))
    return [d for _, d in sorted(fuzzy, key=lambda t: -t[0])][:limit]


NOT_PLACES = set("""the a an my our this that next coming last first all every each some data file
months month years year weeks future climate scenario scenarios dengue malaria cases case disease example
high low very middle moderate emissions emission ssp quick deep balanced fast tuning report results result
me you it them us details more trust use using with from to and or of on by""".split()) | set(DISEASES)


def district_mentions(raw: str) -> list[str]:
    """Candidate place words from a message, e.g. 'in Pune', 'for Kathmandu district'."""
    out = []
    for m in re.finditer(r"\b(?:in|for|at|district(?: is)?)\s+([A-Za-z][A-Za-z_\-]+(?:[ _][A-Za-z][A-Za-z\-]+)?)",
                         raw):
        words = [w for w in re.split(r"[ _]", m.group(1)) if w]
        while words and words[-1].lower() in NOT_PLACES:
            words.pop()
        if words and words[0].lower() not in NOT_PLACES:
            out.append(" ".join(words))
    out += re.findall(r"\b[A-Z]{3}_[A-Za-z\-]+_[A-Za-z_\-]+\b", raw)
    return out

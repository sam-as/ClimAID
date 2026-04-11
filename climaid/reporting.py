from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import warnings
warnings.filterwarnings("ignore")
from climaid.utils import pretty_country

# =====================================================
# DATA CONTAINER FOR REPORTING (MODEL + PROJECTIONS)
# =====================================================
@dataclass
class ReportArtifacts:
    """
    Structured container passed from:
    - disease_model (metrics, model info)
    - disease_projections (CMIP6 summaries)

    This keeps reporting fully decoupled from modelling logic.
    """
    district: str
    disease_name : str
    date_range: str
    metrics: Dict[str, Any]
    selected_lags: Dict[str, Any]
    interaction_lags: list[dict]
    features: list
    importance: Dict[str, float]
    projection_summary: Dict[str, Any]
    model_info: Optional[Dict[str, Any]] = None
    data_summary: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    download_data: Optional[Any] = None

# -------------------------------------------------
# ClimAID logo
# -------------------------------------------------
from importlib.resources import files
import base64

# For plotly figures
logo_path = files("climaid.assets") / "ClimAID logo.png"

with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

logo_uri = f"data:image/png;base64,{logo_base64}"

# For website header
logo_path = files("climaid.assets") / "ClimAID_logo_AS.png"

with open(logo_path, "rb") as f:
    logo_base64_web = base64.b64encode(f.read()).decode()

# logo_uri_web = f"data:image/png;base64,{logo_base64_web}"

# =====================================================
# MAIN REPORTER CLASS (LLM-OPTIONAL, LOCAL-LLM SAFE)
# =====================================================
class DiseaseReporter:
    """
    High-level reporting interface for ClimAID.

    The `DiseaseReporter` is responsible for transforming model outputs into
    structured, human-readable narratives. It acts strictly as a *post-processing*
    layer and does not perform any statistical modeling or inference.

    This separation ensures that reporting remains modular, reproducible,
    and independent of the underlying modeling pipeline.

    Features
    --------

    - Template-based reports
        Fully deterministic summaries generated without LLMs (offline-safe).
    - LLM-generated scientific reports
        Rich, narrative outputs using local or remote language models.
    - Policy briefs
        Concise, decision-oriented summaries for public health stakeholders.
    - Interactive Q&A
        Exploratory analysis interface for research workflows.

    LLM Compatibility
    -----------------

    The reporter is backend-agnostic and works with any LLM client that
    implements the following interface:

        generate(prompt: str) -> str

    - This includes:
        - Local LLMs (e.g., Ollama, LM Studio)
        - Remote APIs (e.g., OpenAI, other providers)

    Notes
    -----

    - This class never performs modeling or prediction.
    - All inputs are expected to be precomputed outputs from ClimAID models.
    - Designed for reproducibility and flexible deployment (offline/online).
    """

    def __init__(self, llm_client=None, max_json_chars: int = 8000):
        """
        Parameters
        ----------
        llm_client : optional
            Any LLM client with method: generate(prompt: str) -> str
            If None → fallback deterministic report (recommended for reproducibility)

        max_json_chars : int
            Maximum characters allowed for JSON blocks in prompt.
            Prevents prompt explosion for large CMIP6 projection summaries.
        """
        self.llm = llm_client
        self.max_json_chars = max_json_chars

    # =====================================================
    # INTERNAL: SAFE LLM GENERATION (NO CRASHES)
    # =====================================================
    def _llm_generate(self, prompt: str) -> str:
        """
        Safe wrapper around LLM call.
        Prevents crashes if:
        - Local API is down
        - Timeout occurs
        - Model not loaded
        """
        if self.llm is None:
            raise RuntimeError("LLM client not provided.")

        try:
            return self.llm.generate(prompt)
        except Exception as e:
            return (
                "Switching to ClimAID Deterministic Scientific Interpreter (C-DSI): "
                f"Reason: {str(e)}\n\n"
                "Local LLM Client unavailable:\n",
                self._deterministic_engine(prompt)
            )

    # =====================================================
    # INTERNAL: SAFE JSON TRUNCATION (CRITICAL FOR LOCAL LLM)
    # =====================================================
    def _safe_json(self, obj: Dict[str, Any]) -> str:
        """
        Safely serialize large dictionaries (e.g., CMIP6 projection summaries)
        without overwhelming local LLM context windows.
        Automatically rounds floats to 2 decimals.
        """
        from climaid.utils import _round_numeric
        try:
            # Round numeric values first
            cleaned_obj = _round_numeric(obj)

            text = json.dumps(cleaned_obj, indent=2, default=str)

        except Exception:
            text = str(obj)

        if len(text) > self.max_json_chars:
            return text[:self.max_json_chars] + "\n... (truncated for report clarity)"

        return text

    # =====================================================
    # MAIN REPORT (SCIENTIFIC / SUMMARY)
    # =====================================================
    def generate(self, artifacts: ReportArtifacts, style: str = "summary") -> str:
        """
        Generate a disease risk report.

        style options:
        - "summary"
        - "detailed"
        - "technical"
        """
        prompt = self._build_prompt(artifacts, style)

        # Fully offline mode (recommended for reproducibility)
        if self.llm is None:
            return self._deterministic_engine(artifacts)

        return self._llm_generate(prompt)

    # =====================================================
    # POLICY BRIEF (GOVERNMENT / WHO STYLE)
    # =====================================================
    def policy_brief(self, artifacts: ReportArtifacts) -> str:
        """
        Generate a policy-oriented disease climate risk brief.
        """
        if self.llm is None:
            return (
                "Policy brief requires an LLM client. "
                "Initialize diseaseReporter(llm_client=...) to enable."
            )
        
        # Formatting
        district_state = getattr(artifacts, "district", "Unknown Region")
        parts = district_state.split("_")

        if len(parts) >= 3:
            country = pretty_country(parts[0])
            district_name = parts[1].title()
            state = parts[2].title()
            district = f"{district_name}, {state}, {country}"
        elif len(parts) >= 2:
            district = f"{parts[0].title()}, {parts[1].title()}"
        else:
            district = district_state.title()

        proj = getattr(artifacts, "projection_summary", {}) or {}
        projection_period = proj.get("projection_period", "Future climate scenarios")
        if isinstance(projection_period, dict):
            projection_period = f"{projection_period.get('start')} → {projection_period.get('end')}"

        data_summary = getattr(artifacts, "data_summary", {}) or {}
        train_period = data_summary.get("train_period", "Predefined split")
        test_period = data_summary.get("test_period", "Post-training evaluation")

        prompt = f"""
            You are a public health policy advisor specialising in {artifacts.disease_name} and climate change risk.

            ========================
            REGION
            ========================
            District: {district}
            Study Period: f"{train_period} → {test_period}"
            Target Disease: {artifacts.disease_name}

            ========================
            MODEL PERFORMANCE
            ========================
            {self._safe_json(artifacts.metrics)}

            =====================
            CLIMATE VARIABLE DEFINITIONS
            =====================
            - mean_SH: Specific humidity (proxy for atmospheric moisture)
            - mean_temperature: Mean air temperature (°K)
            - mean_Rain: Monthly rainfall (kg m⁻² s⁻¹)
            - Nino_anomaly: ENSO climate variability index

            ========================
            KEY CLIMATE DRIVERS
            ========================
            {self._safe_json(artifacts.importance)}

            ========================
            SELECTED CLIMATE LAGS
            ========================
            {self._safe_json(artifacts.selected_lags)}

            ========================
            CMIP6 CLIMATE-DRIVEN PROJECTIONS
            ========================
            Projection Period : {projection_period}
            {self._safe_json(artifacts.projection_summary)}

            Instructions:
            - Write a policy-ready disease risk brief
            - Focus on climate-driven risk implications
            - Highlight uncertainty from multi-model projections
            - Provide actionable public health recommendations
            - Do NOT invent numerical values
            - Use formal, evidence-based language

            Sections required:
            1. Risk level assessment for {artifacts.district}
            2. Climate-sensitive transmission drivers of {artifacts.disease_name}
            3. Seasonal vulnerability insights for {artifacts.district}
            4. Future risk under climate change scenarios for {artifacts.district}
            5. Public health intervention recommendations for {artifacts.disease_name}
            """
        return self._llm_generate(prompt)

    # =====================================================
    # INTERACTIVE CHAT (RESEARCH MODE)
    # =====================================================
    def chat(self, artifacts: ReportArtifacts, question: str) -> str:
        """
        Ask research questions about model outputs interactively.
        """
        if self.llm is None:
            return (
                "Interactive mode requires an LLM client. "
                "Use a local LLM (e.g., Ollama) to enable chat."
            )

        context = self._build_prompt(artifacts, style="technical")

        prompt = f"""
                    You are a disease epidemiology and climate-health modelling expert.

                    MODEL CONTEXT:
                    {context}

                    USER QUESTION:
                    {question}

                    Rules:
                    - Use ONLY the provided model and projection outputs
                    - Do NOT hallucinate values
                    - Be scientifically cautious
                    - Acknowledge uncertainty when relevant
                    - Interpret climate-disease relationships mechanistically
                    """
        return self._llm_generate(prompt)

    # =====================================================
    # CORE PROMPT (CMIP6 + SCIENTIFIC INTERPRETATION)
    # =====================================================
    def _build_prompt(self, a: ReportArtifacts, style: str) -> str:
        """
        Construct the LLM prompt for scientific report generation.

        This method transforms structured report artifacts into a formatted
        prompt suitable for language model inference. It encodes ClimAID’s
        scientific context (e.g., CMIP6 projections, epidemiological signals)
        and ensures consistent, high-quality narrative generation.

        The prompt is carefully designed to:
            - Preserve scientific accuracy from input artifacts
            - Guide the LLM toward structured, domain-specific outputs
            - Minimize ambiguity and reduce hallucination risk

        Parameters
        ----------

        a : ReportArtifacts
            Structured outputs from the ClimAID pipeline, including projections,
            summary statistics, and derived indicators.

        style : str
            Output style for the generated report. 
            
            - Typical options include:
                - "scientific" : formal, publication-style narrative
                - "policy"     : concise, decision-oriented summary
                - "technical"  : detailed analytical interpretation

        Returns
        -------

        str :
            A fully constructed prompt ready to be passed to an LLM client.

        Notes
        -----

        - This method does not perform inference; it only prepares the prompt.
        - Prompt design is critical for ensuring reproducible and reliable outputs.
        - Compatible with both local and remote LLM backends.

        """
        from climaid.utils import _json_safe_numbers

        # Formatting
        district_state = getattr(a, "district", "Unknown Region")
        parts = district_state.split("_")

        if len(parts) >= 3:
            country = pretty_country(parts[0])
            district_name = parts[1].title()
            state = parts[2].title()
            district = f"{district_name}, {state}, {country}"
        elif len(parts) >= 2:
            district = f"{parts[0].title()}, {parts[1].title()}"
        else:
            district = district_state.title()

        return f"""
                    You are a disease epidemiology and climate-health modelling expert.

                    Study Region: {district}
                    Study Period: {a.date_range}
                    Target Disease: {a.disease_name}

                    ==================================================
                    SECTION 1: HISTORICAL MODEL VALIDATION
                    ==================================================
                    Model Performance Metrics:
                    {json.dumps(a.metrics, indent=2)}

                    Model Information:
                    {json.dumps(a.model_info, indent=2) if a.model_info else "Not provided"}

                    Training & Testing Data Summary:
                    {json.dumps(a.data_summary, indent=2) if a.data_summary else "Not provided"}

                    Selected Climate Lags (months):
                    {json.dumps(a.selected_lags, indent=2)}

                    Selected Interaction Lags (months)
                    {json.dumps(a.interaction_lags, indent=2)}

                    Important Climate Drivers:
                    {json.dumps(a.importance, indent=2, default=_json_safe_numbers)}

                    ==================================================
                    SECTION 2: FUTURE CLIMATE PROJECTIONS (CMIP6)
                    ==================================================
                    Projection Summary:
                    {json.dumps(a.projection_summary, indent=2)}

                    ==================================================
                    REPORTING INSTRUCTIONS
                    ==================================================
                    Write a {style} disease risk report with CLEARLY SEPARATED sections:

                    1. Historical Model Reliability and Predictive Performance for {a.district}
                    - Interpret R² and RMSE scientifically  
                    - Discuss strengths and limitations  
                    - DO NOT exaggerate performance  

                    2. Climate–disease Mechanistic Relationships for {a.disease_name} 
                    - Link selected lags to mosquito ecology and transmission dynamics  
                    - Interpret specific humidity (mean_SH) correctly  
                    - Avoid incorrect variable definitions  

                    3. Future Climate Projections (CMIP6-Based) for {district}
                    - Interpret ensemble mean projections  
                    - Discuss SSP scenario differences  
                    - Highlight trend direction (increasing/decreasing/stable)  

                    4. Uncertainty and Ensemble Interpretation for {district}
                    - Explain lower_bound and upper_bound meaning  
                    - Discuss multi-model variability  

                    5. Public Health and Policy Implications for {a.disease_name}
                    - Early warning insights  
                    - Seasonal preparedness relevance  
                    - Climate adaptation relevance  

                    CRITICAL RULES:
                    - DO NOT fabricate numbers
                    - Use ONLY provided metrics and summaries
                    - Maintain scientific tone (journal-quality)
                    - Clearly distinguish historical validation vs future projections
                    """

    # =====================================================
    # ClimAID Deterministic Scientific Interpreter (DSI)
    # =====================================================
    def _deterministic_engine(self, artifacts) -> str:
        """
        ClimAID Deterministic Scientific Interpreter (C-DSI).

        This method generates structured, human-readable reports directly from
        precomputed model artifacts, without relying on any language model.
        It is automatically used as a fallback when an LLM is unavailable.

        The engine operates purely on validated inputs and does not introduce
        any generative or probabilistic interpretation, ensuring zero risk of
        hallucinated content.

        Parameters
        ----------

        artifacts : dict or object
            Structured outputs from the modeling pipeline (e.g., projections,
            summaries, statistics).

        Returns
        -------

        str :
            A deterministic, scientifically grounded report.

        Notes
        -----

        - Uses only explicit, precomputed values from the pipeline.
        - No external dependencies or model calls.
        - Ensures reproducibility and auditability of results.
        - Intended for secure, offline, or high-integrity workflows.
        """

        import textwrap
        import markdown
        import calendar

        # -------------------------------------------------
        # BUG FIX: Initialize variables to prevent UnboundLocalError
        # -------------------------------------------------
        ensemble_trend = "Future climate projections indicate variable disease risk."
        projection_sentence = ""
        trend_sentence = ""
        seasonal_sentence = ""
        peak_month_text = "Peak transmission months not identified"

        # -------------------------------------------------
        # SAFE EXTRACTION
        # -------------------------------------------------
        # Formatting
        district_state = getattr(artifacts, "district", "Unknown Region")
        parts = district_state.split("_")

        if len(parts) >= 3:
            country = pretty_country(parts[0])
            district_name = parts[1].title()
            state = parts[2].title()
            district = f"{district_name}, {state}, {country}"
        elif len(parts) >= 2:
            district = f"{parts[0].title()}, {parts[1].title()}"
        else:
            district = district_state.title()

        disease = getattr(artifacts, "disease_name", "Climate-sensitive disease")
        date_range = getattr(artifacts, "date_range", "Unknown period")

        metrics = getattr(artifacts, "metrics", {}) or {}
        lags = getattr(artifacts, "selected_lags", {}) or {}
        interaction_lags = getattr(artifacts, "interaction_lags", {}) or {}
        importance = getattr(artifacts, "importance", {}) or {}
        proj = getattr(artifacts, "projection_summary", {}) or {}
        runtime = getattr(artifacts, "runtime", {}) or {}

        data_summary = getattr(artifacts, "data_summary", {}) or {}
        model_info = getattr(artifacts, "model_info", {}) or {}

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------
        r2 = metrics.get("test_r2", "Not available")
        rmse = metrics.get("test_rmse", "Not available")

        train_period = data_summary.get("train_period", "Unknown")
        test_period = data_summary.get("test_period", "Unknown")

        # -------------------------------------------------
        # LAG SUMMARY
        # -------------------------------------------------
        if isinstance(lags, dict) and lags:

            var_names = {
                "mean_SH": "Specific Humidity",
                "mean_temperature": "Temperature",
                "mean_Rain": "Rainfall",
                "Nino_anomaly": "ENSO"
            }

            lines = []

            for var, lag in lags.items():

                name = var_names.get(var, var)

                if lag == 0:
                    lines.append(f"- {name} (current month)")
                else:
                    lines.append(f"- {name} (lag {lag} months)")

            lag_text = "\n".join(lines)

        else:
            lag_text = "- Automatic lag selection applied"

        if isinstance(interaction_lags, list) and interaction_lags:

            var_names = {
                "mean_SH": "Specific Humidity (Mean)",
                "mean_temperature": "Temperature (Mean)",
                "mean_Rain": "Rainfall (Mean)",
                "Nino_anomaly": "ENSO"
            }

            lines = []

            for inter in interaction_lags:

                v1 = var_names.get(inter["var1"], inter["var1"])
                v2 = var_names.get(inter["var2"], inter["var2"])

                l1 = inter["lag1"]
                l2 = inter["lag2"]

                lines.append(f"- {v1} (lag {l1}) × {v2} (lag {l2})")

            interaction_lag_text = "\n".join(lines)

        else:
            interaction_lag_text = "- No interaction lags selected"

        # Function for interpreting interactions. 
        def interpret_interactions(interactions):

            if not isinstance(interactions, list) or not interactions:
                return "No significant climate–ENSO interaction terms were detected."

            var_names = {
                "mean_SH": "specific humidity",
                "mean_temperature": "temperature",
                "mean_Rain": "rainfall",
                "Nino_anomaly": "ENSO"
            }

            lines = ["Thus, the model identified the following climate–ENSO interactions:\n"]

            for inter in interactions:

                v1 = var_names.get(inter["var1"], inter["var1"])
                v2 = var_names.get(inter["var2"], inter["var2"])

                l1 = inter["lag1"]
                l2 = inter["lag2"]

                lines.append(f"- {v1.capitalize()} (lag {l1}) × {v2} (lag {l2})")

            lines.append(
                "\nThese interactions suggest that large-scale climate variability may "
                "modulate local environmental conditions influencing disease transmission."
            )

            return "\n".join(lines)

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        importance_text = "- Feature importance not available when base model is mlp, nn or lasso/ridge/elasticnet"

        if importance is not None:

            # Convert pandas Series → dict if needed
            if hasattr(importance, "to_dict"):
                importance = importance.to_dict()

            # Ensure we actually have values
            if isinstance(importance, dict) and len(importance) > 0:

                var_names = {
                    "mean_SH": "Specific Humidity (Mean)",
                    "mean_temperature": "Temperature (Mean)",
                    "mean_Rain": "Rainfall (Mean)",
                    "Nino_anomaly": "ENSO",
                    "MA_mean_temperature": "10-Year Mean Temperature",
                    "MA_mean_Rain": "10-Year Mean Rainfall",
                    "MA_mean_SH": "10-Year Mean Humidity",
                    "YA_mean_temperature": "Annual Mean Temperature",
                    "YA_mean_Rain": "Annual Mean Rainfall",
                    "YA_mean_SH": "Annual Mean Humidity",
                    "Year": "Long-term Trend (Year)"
                }

                def pretty_feature(name):

                    # Interaction terms
                    if "_x_" in name:

                        left, right = name.split("_x_")

                        if "_lag" in left:
                            v1, l1 = left.split("_lag")
                            v1 = var_names.get(v1, v1)
                        else:
                            v1, l1 = left, None

                        if "_lag" in right:
                            v2, l2 = right.split("_lag")
                            v2 = var_names.get(v2, v2)
                        else:
                            v2, l2 = right, None

                        return f"{v1} (lag {l1}) × {v2} (lag {l2})"

                    # Lag terms
                    if "_lag" in name:

                        var, lag = name.split("_lag")
                        var = var_names.get(var, var)

                        if lag == "0":
                            return f"{var} (current month)"
                        else:
                            return f"{var} (lag {lag} months)"

                    # Regular variable
                    return var_names.get(name, name)

                # Remove NaNs
                clean_importance = {
                    k: v for k, v in importance.items()
                    if v is not None
                }

                if len(clean_importance) > 0:

                    top_feats = sorted(
                        clean_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    lines = [
                        f"- {pretty_feature(k)} ({v:.3f})"
                        for k, v in top_feats
                    ]

                    importance_text = (
                        "The following variables contributed most strongly to the prediction model:\n\n"
                        + "\n".join(lines)
                    )

        # -------------------------------------------------
        # PROJECTION SUMMARY
        # -------------------------------------------------
        projection_period_raw = proj.get("projection_period", "Future climate scenarios")

        projection_period = projection_period_raw

        if isinstance(projection_period_raw, dict):

            start = projection_period_raw.get("start")
            end = projection_period_raw.get("end")
            steps = projection_period_raw.get("n_timesteps")

            from datetime import datetime

            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)

                start_fmt = start_dt.strftime("%B %Y")
                end_fmt = end_dt.strftime("%B %Y")

                start_year = start_dt.year
                end_year = end_dt.year

            except Exception:
                start_fmt = start
                end_fmt = end
                start_year = start
                end_year = end

            # Main report formatting
            if steps:
                projection_period = f"{start_fmt} – {end_fmt} ({steps:,} projection timesteps)"
            else:
                projection_period = f"{start_fmt} – {end_fmt}"

            # Scientific explanatory sentence
            projection_sentence = (
                f"Climate-driven disease projections were simulated from "
                f"{start_fmt} to {end_fmt} using CMIP6 climate model scenarios. "
                f"The projection horizon spans approximately {end_year - start_year} years "
                f"with {steps:,} simulated timesteps."
            )

        # -------------------------------------------------
        # LONG-TERM PROJECTION TREND (DETERMINISTIC)
        # -------------------------------------------------

        ensemble_ts = proj.get("ensemble_timeseries", [])

        if isinstance(ensemble_ts, list) and len(ensemble_ts) > 10:

            try:
                start_mean = ensemble_ts[0].get("mean")
                end_mean = ensemble_ts[-1].get("mean")

                if start_mean and end_mean:

                    pct_change = ((end_mean - start_mean) / start_mean) * 100

                    if pct_change > 10:
                        trend_sentence = (
                            f"Long-term projections indicate that {disease.lower()} incidence "
                            f"in {district} may increase by approximately "
                            f"{pct_change:.1f}% by the end of the century "
                            f"relative to early projection years."
                        )

                    elif pct_change < -10:
                        trend_sentence = (
                            f"Long-term projections indicate a potential decline of "
                            f"approximately {abs(pct_change):.1f}% in "
                            f"{disease.lower()} incidence in {district} "
                            f"by the end of the projection period."
                        )

                    else:
                        trend_sentence = (
                            f"Projected {disease.lower()} incidence in {district} "
                            f"remains relatively stable across the simulation horizon."
                        )

            except Exception:
                trend_sentence = ""

        # -------------------------------------------------
        # SEASONAL RISK INTERPRETATION
        # -------------------------------------------------

        risk_matrix = proj.get("risk_matrix", [])

        if isinstance(risk_matrix, list) and len(risk_matrix) > 0:

            try:
                import calendar
                from collections import defaultdict

                monthly_risk = defaultdict(list)

                for row in risk_matrix:

                    date_str = row.get("time")
                    risk_val = row.get("risk")

                    if date_str and risk_val is not None:

                        month = int(date_str.split("-")[1])
                        monthly_risk[month].append(risk_val)

                # compute average monthly risk
                avg_risk = {
                    m: sum(vals)/len(vals)
                    for m, vals in monthly_risk.items()
                    if len(vals) > 0
                }

                if avg_risk:

                    # sort months by risk
                    peak_months = sorted(avg_risk, key=avg_risk.get, reverse=True)[:3]

                    peak_month_names = [
                        calendar.month_name[m] for m in peak_months
                    ]

                    seasonal_sentence = (
                        f"Seasonal transmission risk is highest during "
                        f"{', '.join(peak_month_names)}, suggesting elevated "
                        f"{disease.lower()} transmission potential during these months."
                    )

            except Exception:
                seasonal_sentence = ""

            # Ensemble information
            ensemble_info = proj.get("ensemble_mean", {}) or {}

            ensemble_trend = ensemble_info.get(
                "trend",
                "Future climate projections indicate variable disease risk."
            )

            peak_months = ensemble_info.get("peak_transmission_months", [])

            if isinstance(peak_months, (list, tuple)) and peak_months:
                peak_month_text = ", ".join(
                    calendar.month_name[int(m)]
                    for m in peak_months
                    if isinstance(m, (int, float)) and 1 <= int(m) <= 12
                )
            else:
                peak_month_text = "Peak transmission months not identified"

        # -------------------------------------------------
        # SSP SCENARIO SUMMARIES
        # -------------------------------------------------
        ssp_summary = proj.get("ssp_ensemble", {}) or {}

        ssp_lines = []

        for ssp, stats in ssp_summary.items():

            mean_proj = stats.get("mean_projection")
            max_proj = stats.get("max_projection")
            min_proj = stats.get("min_projection")

            if mean_proj is not None:

                ssp_lines.append(
                    f"- **{ssp.upper()}**: mean ≈ {mean_proj:.1f} "
                    f"(range {min_proj:.1f}–{max_proj:.1f})"
                )

        ssp_text = "\n".join(ssp_lines) if ssp_lines else "- SSP-specific projections unavailable."

        # -------------------------------------------------
        # UNCERTAINTY SUMMARY
        # -------------------------------------------------
        uncertainty_info = proj.get("uncertainty", {}) or {}
        unc_val = uncertainty_info.get("mean_uncertainty_range")

        if unc_val is not None:

            uncertainty = (
                f"Average projection spread across climate models "
                f"is approximately **±{unc_val/2:.1f} cases** "
                f"(mean spread ≈ {unc_val:.1f})."
            )

        else:
            uncertainty = "Projection uncertainty could not be estimated."

        # -------------------------------------------------
        # RUNTIME SUMMARY
        # -------------------------------------------------
        runtime_text = ""

        if runtime:

            runtime_lines = [
                f"- {k.replace('_',' ').title()}: {round(v,2)} seconds"
                for k, v in runtime.items()
                if isinstance(v, (int, float))
            ]

            if runtime_lines:
                runtime_text = "\n\n## Computational Performance\n\n" + "\n".join(runtime_lines)

        # -------------------------------------------------
        # MODEL INFO
        # -------------------------------------------------
        if isinstance(model_info, dict) and model_info:

            model_info_text = (
                f"{model_info.get('stacking_pipeline','Pipeline')} "
                f"(Base: {model_info.get('base_model','N/A')}, "
                f"Residual: {model_info.get('residual_model','N/A')}, "
                f"Correction: {model_info.get('correction_model','N/A')}, "
                f"Features: {model_info.get('n_features','N/A')})"
            )

        else:
            model_info_text = "Model configuration unavailable"

        # -------------------------------------------------
        # FINAL REPORT (MARKDOWN)
        # -------------------------------------------------
        report = textwrap.dedent(f"""
            # Climate-Driven Disease Risk Assessment Report (C-DSI)

            **Region:** {district}  
            **Study Period:** {date_range}  
            **Target Disease:** {disease}  
            **Report Mode:** ClimAID Deterministic Scientific Interpreter (C-DSI)

            ---

            ## 1. Historical Model Validation

            A stacked climate-driven disease modelling pipeline  
            (Base → Residual → Correction) was trained using historical
            climate and disease observations.

            **Training Period:** {train_period}  
            **Testing Period:** {test_period}

            **Model Performance**

            - Model used: {model_info_text}
            - R²: {r2}
            - RMSE: {rmse}

            The modelling workflow included automated lag optimisation and
            feature selection to capture delayed climate–disease responses.

            ---

            ## 2. Climate–Disease Mechanistic Relationships

            **Selected Climate Lags**

            {lag_text}

            **Selected Interaction Lags**

            {interaction_lag_text}

            {interpret_interactions(interaction_lags)}

            **Top Contributing Features**

            {importance_text}

            ---

            ## 3. Future Climate-Driven Disease Projections (CMIP6)

            **Projection Period:** {projection_period}

            {projection_sentence}

            {trend_sentence}

            **Ensemble Projection Trend**

            {ensemble_trend}

            **Peak Transmission Months**

            {peak_month_text}

            **Scenario-Specific Trends**

            {ssp_text}

            ---

            ## 4. Projection Uncertainty

            {uncertainty}

            ---

            ## 5. Public Health Interpretation

            {trend_sentence}

            {seasonal_sentence}

            Few general comments:

            - Climate-sensitive disease risk may evolve under changing
            temperature and precipitation regimes.

            - Early warning systems should incorporate the identified
            climate lags to improve outbreak prediction.

            - Seasonal preparedness strategies may require adjustment
            if projected peak transmission periods shift.

            {runtime_text}

            ---

            *Note:* This report was generated using the ClimAID Deterministic Scientific Interpreter (C-DSI).
            """).strip()

        # Clean indentation
        report = "\n".join(line.lstrip() for line in report.splitlines())

        # -------------------------------------------------
        # MARKDOWN → HTML
        # -------------------------------------------------
        report_html = markdown.markdown(
            report,
            extensions=["extra", "tables", "sane_lists"]
        )

        # -------------------------------------------------
        # HTML DASHBOARD TEMPLATE
        # -------------------------------------------------
        css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                        Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f7f9fb;
            margin: 0;
            padding: 40px;
        }

        .report {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .report h1 {
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }

        .report h2 {
            margin-top: 30px;
        }

        .report ul {
            margin-left: 20px;
        }
        """

        html = f"""
            <html>
            <head>
            <meta charset="UTF-8">
            <style>
            {css}
            </style>
            </head>

            <body>

            <div class="report">
            {report_html}
            </div>

            </body>
            </html>
            """

        return html

# -------------------------------------------------------- #
#                      Plotters                            #
# -------------------------------------------------------- #
def build_projection_from_summary(projection_summary: dict):
    """
    Create an interactive mean projection plot from summary outputs.

    Parameters
    ----------

    projection_summary : dict
        Must contain "ensemble_timeseries".

    Returns
    -------

    plot : 
        plotly.graph_objects.Figure 
    """
    import plotly.graph_objects as go

    timeseries = projection_summary.get("ensemble_timeseries", [])

    if not timeseries:
        return "<p><b>No projection time-series data available.</b></p>"

    years = [item["time"] for item in timeseries] 
    mean_vals = [item["mean"] for item in timeseries]
    lower = [item["lower_bound"] for item in timeseries]
    upper = [item["upper_bound"] for item in timeseries]

    fig = go.Figure()

    # Upper Bound Line (Transparent, used for fill boundary)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=upper,
            mode="lines",
            line=dict(width=0), # No visible line
            showlegend=False,
            hoverinfo='skip'
        )
    )

    # Lower Bound + Fill (The Uncertainty Band)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(212, 163, 115, 0.2)", # Soft earthy tan with transparency
            line=dict(width=0),
            name="95% Uncertainty Interval",
            hoverinfo='skip'
        )
    )

    # Ensemble Mean (The Hero Line)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=mean_vals,
            mode="lines",
            name="Ensemble Mean",
            line=dict(width=3, color="#bc6c25"), # Stronger contrast for the mean
            hovertemplate="<b>Year: %{x}</b><br>Value: %{y:.2f}<extra></extra>"
        )
    )

    # Layout Refinements
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, serif", size=14, color="#2c3e50"),
        height=500,
        margin=dict(l=80, r=40, t=80, b=80),
        hovermode="x unified",
        title={
            'text': "<b>CMIP6 Ensemble Climate Projection</b>",
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Year",
            showgrid=False,
            linecolor="#bdc3c7"
        ),
        yaxis=dict(
            title="Projected Disease Burden",
            showgrid=True,
            gridcolor="#ecf0f1",
            linecolor="#bdc3c7"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.5
        )
    )

    fig.update_layout(
        images=[
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                sizex=0.15,
                sizey=0.15,
                xanchor="right",
                yanchor="bottom",
                opacity=0.9,
                layer="above"
            )
            ]
    )

    return fig.to_html(full_html=False, include_plotlyjs=True)

def build_dual_seasonal_heatmap(projection_summary: dict, years_ahead=5):
    """
    Create dual heatmaps of seasonal projections and uncertainty.

    Parameters
    ----------

    projection_summary : dict
        Must contain "ssp_timeseries".

    years_ahead : int, default=5
        Number of years to include.

    Returns
    -------

    plot : 
        plotly.graph_objects.Figure or str 
    """

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ssp_data = projection_summary.get("ssp_timeseries", {})

    if not ssp_data:
        return "<p><b>No SSP time-series available.</b></p>"

    month_labels = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    ssps = sorted(ssp_data.keys())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Mean Projection","Projection Uncertainty"]
    )

    traces_per_ssp = 2

    for ssp in ssps:

        df = pd.DataFrame(ssp_data[ssp])

        df["time"] = pd.to_datetime(df["time"])
        df["Year"] = df["time"].dt.year
        df["Month"] = df["time"].dt.month

        start_year = df["Year"].min()
        end_year = start_year + years_ahead
        df = df[df["Year"] <= end_year]

        df["maximum"] = df["upper_bound"].copy()

        mean_pivot = df.pivot_table(
            index="Month",
            columns="Year",
            values="mean"
        )

        unc_pivot = df.pivot_table(
            index="Month",
            columns="Year",
            values="maximum"
        )

        y_labels = [month_labels[m-1] for m in mean_pivot.index]

        fig.add_trace(
            go.Heatmap(
                z=mean_pivot.values,
                x=mean_pivot.columns.astype(str),
                y=y_labels,
                colorscale="YlOrRd",
                visible=False,
                colorbar=dict(
                    title="Mean Cases",
                    orientation='h',
                    # Positioning
                    x=0.22,     
                    y=-0.25,    
                    len=0.4,    
                    thickness=15
                ),
                hovertemplate="Month %{y}<br>Year %{x}<br>Mean %{z:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=unc_pivot.values,
                x=unc_pivot.columns.astype(str),
                y=y_labels,
                colorscale="Blues",
                visible=False,
                colorbar=dict(
                    title="Max Cases",
                    orientation='h',
                    x=0.78,     
                    y=-0.25,    
                    len=0.4,    
                    thickness=15
                ),
                hovertemplate="Month %{y}<br>Year %{x}<br>Maximum %{z:.2f}<extra></extra>"
            ),
            row=1, col=2
        )

        fig.update_layout(margin=dict(b=100))

    # Make first SSP visible
    for i in range(traces_per_ssp):
        fig.data[i].visible = True

    # Create dropdown

    dropdown = []

    for i, ssp in enumerate(ssps):

        visible = [False] * len(fig.data)

        visible[i*traces_per_ssp] = True
        visible[i*traces_per_ssp + 1] = True

        dropdown.append(
            dict(
                label=ssp.upper(),
                method="update",
                args=[{"visible": visible}]
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=80, t=100, b=40),

        updatemenus=[
            dict(
                buttons=dropdown,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
                bgcolor="#eef5fb",
                bordercolor="#1d6fa5",
                borderwidth=1
            )
        ],

        annotations=[
            dict(
                text="Select Climate Scenario (SSP)",
                x=0.5,
                y=1.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13)
            )
        ],

        font=dict(family="Georgia, serif", size=14, color="#2c3e50"),

        images=[
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=1,
                y=1.12,
                sizex=0.12,
                sizey=0.12,
                xanchor="right",   
                yanchor="top",     
                opacity=0.9,
                layer="above"
            )
        ]
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

def build_ssp_projection_grid(projection_summary: dict):
    """
    Create a grid of SSP-specific projection time series.

    Each subplot represents a different SSP scenario for comparison.

    Parameters
    ----------

    projection_summary : dict
        Must contain "ssp_timeseries".

    Returns
    -------

    plot : 
        plotly.graph_objects.Figure 

    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ssp_data = projection_summary.get("ssp_timeseries", {})

    if not ssp_data:
        return "<p><b>No SSP time-series available.</b></p>"

    ssps = list(ssp_data.keys())
    
    # Custom color palette for SSPs (Progressing from "low" to "high" impact)
    # Using a professional palette: Blue, Green, Orange, Red
    colors = ['#2E91E5', '#2CA02C', '#FF7F0E', '#D62728', '#9467BD']
    
    fig = make_subplots(
        rows=1,
        cols=len(ssps),
        shared_yaxes=True,
        subplot_titles=[f"<b>{ssp}</b>" for ssp in ssps],
        horizontal_spacing=0.05
    )

    for i, ssp in enumerate(ssps, start=1):
        data = ssp_data[ssp]
        color = colors[(i-1) % len(colors)]

        years = [d["time"] for d in data]
        mean_vals = [d["mean"] for d in data]
        lower = [d["lower_bound"] for d in data]
        upper = [d["upper_bound"] for d in data]

        # Upper Bound (Hidden line)
        fig.add_trace(
            go.Scatter(
                x=years, y=upper, 
                mode="lines",
                line=dict(width=0), 
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=i
        )

        # Confidence Interval (Shaded Area)
        fig.add_trace(
            go.Scatter(
                x=years, y=lower, 
                mode="lines",
                fill="tonexty", 
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}", # Dynamic RGBA with 0.2 alpha
                line=dict(width=0),
                showlegend=False,
                name="95% CI",
                hoverinfo='skip'
            ),
            row=1, col=i
        )

        # Mean Trend Line
        fig.add_trace(
            go.Scatter(
                x=years, y=mean_vals,
                mode="lines",
                line=dict(color=color, width=3),
                name=f"Mean {ssp}",
                hovertemplate="<b>Year: %{x}</b><br>Value: %{y:.2f}<extra></extra>",
                showlegend=False
            ),
            row=1, col=i
        )

    # Global Layout Refinements
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, serif", size=14, color="#2c3e50"),
        height=500,
        title={
            'text': "<b>SSP-Specific Climate–Disease Projections</b>",
            'y':1,
            'x':1,
            'xanchor': 'left',
            'yanchor': 'bottom',
            'font': dict(size=20)
        },
        margin=dict(l=40, r=120, t=100, b=60),
        hovermode="x unified"
    )

    # Style axes
    fig.update_xaxes(showgrid=False, zeroline=False, title_text="Year")
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)

    fig.update_layout(
        images=[
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                sizex=0.15,
                sizey=0.15,
                xanchor="left",
                yanchor="bottom",
                opacity=0.9,
                layer="above"
            )
            ]
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

def build_climate_sensitivity_panel(importance: dict):
    """
    Visualize feature importance for climate sensitivity 
    based on the base model selected inside the DiseaseModel Class.

    Parameters
    ----------

    importance : dict
        Mapping of variables to importance scores.

    Returns
    -------

    plot : 
        plotly.graph_objects.Figure or str 
    """
    import plotly.graph_objects as go

    if not importance:
        return "<p><b>No feature importance data available.</b></p>"

    # ----------------------------------
    # Human-readable variable names
    # ----------------------------------
    var_names = {
        "mean_SH": "Specific Humidity",
        "mean_temperature": "Temperature",
        "mean_Rain": "Rainfall",
        "Nino_anomaly": "ENSO",
        "MA_mean_temperature": "10-Year Mean Temperature",
        "MA_mean_Rain": "10-Year Mean Rainfall",
        "MA_mean_SH": "10-Year Mean Humidity",
        "YA_mean_temperature": "Annual Mean Temperature",
        "YA_mean_Rain": "Annual Mean Rainfall",
        "YA_mean_SH": "Annual Mean Humidity",
        "Year": "Long-term Trend (Year)"
    }

    # ----------------------------------
    # Feature name formatter
    # ----------------------------------
    def pretty_feature(name):

        # Interaction terms
        if "_x_" in name:

            left, right = name.split("_x_")

            if "_lag" in left:
                v1, l1 = left.split("_lag")
                v1 = var_names.get(v1, v1)
            else:
                v1, l1 = left, None

            if "_lag" in right:
                v2, l2 = right.split("_lag")
                v2 = var_names.get(v2, v2)
            else:
                v2, l2 = right, None

            return f"{v1} (lag {l1}) × {v2} (lag {l2})"

        # Lag terms
        if "_lag" in name:

            var, lag = name.split("_lag")
            var = var_names.get(var, var)

            if lag == "0":
                return f"{var} (current month)"
            else:
                return f"{var} (lag {lag} months)"

        # Regular variables
        return var_names.get(name, name)

    # ----------------------------------
    # Sort importance
    # ----------------------------------
    sorted_items = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Pretty labels
    variables = [pretty_feature(k) for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=variables,
            x=values,
            orientation="h",
            # Apply a color gradient based on the values
            marker=dict(
                color=values,
                colorscale='Viridis',
                reversescale=True,
                line=dict(color='white', width=1)
            ),
            # Add text labels inside/outside the bars for clarity
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, serif", size=14, color="#2c3e50"),
        height=min(500, 100 + len(variables) * 40), # Responsive height based on item count
        margin=dict(l=20, r=40, t=80, b=40),
        title={
            'text': "<b>Climate Driver Sensitivity Analysis Based on the Feature Importances Obtained from Base model</b>",
            'subtitle': {'text': 'Relative impact of climate variables on disease projections'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis=dict(
            title="Relative Importance Weight",
            showgrid=True,
            gridcolor="#f0f0f0",
            range=[0, max(values) * 1.15] # Add breathing room for text labels
        ),
        yaxis=dict(
            title="",
            autorange="reversed", # Ensure highest is at top
            tickfont=dict(size=12, color="#34495e")
        ),
        bargap=0.3 # Space between bars for a cleaner look
    )

    # Remove the 'Modebar' clutter for a cleaner UI integration
    config = {'displayModeBar': False}

    fig.update_layout(
        images=[
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=1,
                y=0,
                sizex=0.15,
                sizey=0.15,
                xanchor="right",
                yanchor="bottom",
                opacity=0.9,
                layer="above"
            )
            ]
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)

def build_risk_matrix(projection_summary: dict, years_per_period: int = 5):
    """
    Create a temporal risk matrix visualization.

    Parameters
    ----------

    projection_summary : dict
        Must contain "risk_matrix".

    years_per_period: int
        default = 5
        
    Returns
    -------

    plot : 
        plotly.graph_objects.Figure or str

    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    risk_data = projection_summary.get("risk_matrix", [])
    if not risk_data: return "<div>No data provided.</div>"

    df = pd.DataFrame(risk_data)
    
    # -----------------------------
    # Pre-processing & Type Casting
    # -----------------------------
    df['date'] = pd.to_datetime(df['time'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.strftime('%b')

    # Ensure probability exists
    if "probability" not in df.columns:
        raise ValueError("risk_matrix must include 'probability' field")

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    m_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    df['month'] = pd.Categorical(df['month'], categories=m_order, ordered=True)
    df = df.sort_values(['year', 'month'])

    ssps = sorted(df['SSP'].unique())
    all_years = sorted(df['year'].unique())
    
    # Chunking years into periods
    periods = [all_years[i:i + years_per_period] for i in range(0, len(all_years), years_per_period)]
    period_labels = [f"{p[0]} - {p[-1]}" for p in periods]

    fig = go.Figure()

    trace_map = [] 

    for ssp in ssps:
        for p_idx, period_years in enumerate(periods):

            sub_df = df[(df['SSP'] == ssp) & (df['year'].isin(period_years))]
            if sub_df.empty: continue

            # --------------------------------------------------
            # FIX: Use pivot_table (handles duplicates correctly)
            # --------------------------------------------------
            prob_p = sub_df.pivot_table(
                index='year',
                columns='month',
                values='probability',
                aggfunc='mean'
            )

            prob_p = prob_p.reindex(columns=m_order)

            # --------------------------------------------------
            # Fill missing ONLY for visualization (not logic)
            # --------------------------------------------------
            prob_plot = prob_p.fillna(0)

            # --------------------------------------------------
            # Heatmap (Probability-based)
            # --------------------------------------------------
            fig.add_trace(go.Heatmap(
                z=prob_plot.values * 100,
                x=prob_plot.columns,
                y=prob_plot.index,

                # Clean hover showing probability
                hovertemplate=(
                    "<b>%{y} %{x}</b><br>"
                    "Outbreak Probability: %{z}%"
                    "<extra></extra>"
                ),

                colorscale="Viridis",   
                zmin=0,
                zmax=100,

                xgap=3,
                ygap=3,
                visible=False,

                colorbar=dict(
                    title="Outbreak Probability",
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    len=0.5
                )
            ))

            trace_map.append({"ssp": ssp, "period": p_idx})

    # -----------------------------
    # Set Initial State
    # -----------------------------
    if len(fig.data) > 0:
        fig.data[0].visible = True

    # -----------------------------
    # Dynamic visibility logic
    # -----------------------------
    def create_vis_list(target_ssp, target_p_idx):
        vis = []
        for item in trace_map:
            is_match = (item['ssp'] == target_ssp and item['period'] == target_p_idx)
            vis.append(is_match)
        return vis

    ssp_buttons = [dict(
        label=f"Scenario: {s}",
        method="update",
        args=[{"visible": create_vis_list(s, 0)},
              {"title": f"Risk Matrix: {s} ({period_labels[0]})"}]
    ) for s in ssps]

    period_buttons = [dict(
        label=f"Period: {label}",
        method="update",
        args=[{"visible": create_vis_list(ssps[0], idx)},
              {"title": f"Risk Matrix: {ssps[0]} ({label})"}]
    ) for idx, label in enumerate(period_labels)]

    fig.update_layout(
        updatemenus=[
            {"buttons": ssp_buttons, "x": 0.0, "y": 1.25, "xanchor": "left", "active": 0},
            {"buttons": period_buttons, "x": 0.75, "y": 1.25, "xanchor": "left", "active": 0}
        ],
        template="plotly_white",
        font=dict(family="Georgia, serif", size=14, color="#2c3e50"),
        height=600,
        margin=dict(t=130, b=100, l=80, r=80),
        hoverlabel=dict(bgcolor="white", font_size=13),
        yaxis=dict(type='category', autorange="reversed", title="Forecast Years"),
        xaxis=dict(title="Month"), 
    )

    fig.update_layout(
        images=[
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=1.05,
                y=0,
                sizex=0.15,
                sizey=0.15,
                xanchor="left",
                yanchor="bottom",
                opacity=0.9,
                layer="above"
            )
            ]
    )


    # -----------------------------
    # Clean UI (remove modebar)
    # -----------------------------
    config = {'displayModeBar': False}

    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)

# =========================
# Browser Viewer (HTML)
# =========================

import webbrowser
import tempfile
from pathlib import Path
import html


def open_report_in_browser(
    report_text: str,
    title: str = "ClimAID disease Report",
    save_copy: bool = False,
    artifacts =  None, 
    output_dir: str = None,
) -> str:
    """
    Render and open a generated report in the user's default web browser.

    This function converts a report string into a formatted HTML document
    and launches it locally in the system's default browser. It is designed
    to provide a seamless, fully offline viewing experience for ClimAID
    reports.

    Parameters
    ----------

    report_text : str
        The generated report content (plain text or formatted text).

    title : str, default="ClimAID Disease Report"
        Title of the HTML report displayed in the browser tab.

    save_copy : bool, default=False
        If True, saves a persistent HTML copy of the report to disk.

    artifacts : optional
        Additional structured outputs (e.g., plots, summaries) that may be
        embedded into the report.

    output_dir : str, optional
        Directory where the report should be saved if `save_copy=True`.
        If not provided, a default temporary or working directory is used.

    Returns
    -------

    str
        Path to the generated HTML file.

    Notes
    -----

    - Runs fully offline; no external dependencies or network calls required.
    - Safe for use in packaged environments (e.g., pip installations).
    - Designed for quick visualization of results without requiring
      Jupyter notebooks or external viewers.
    - Can optionally embed visual artifacts such as Plotly figures.

    Behavior
    --------
    - Generates an HTML file from the report content.
    - Opens the file in the default system browser.
    - Optionally saves a persistent copy for later use.
    """

    # Escape HTML-sensitive characters (important for LLM + fallback text)
    from datetime import datetime
    safe_text = html.escape(report_text)
    timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    total_runtime = "N/A"
    if artifacts.runtime:
        valid_times = [
            v for v in artifacts.runtime.values()
            if isinstance(v, (int, float))
        ]
        if valid_times:
            total_runtime = round(sum(valid_times), 2)

    # -------------------------------------------------
    # AUTO FORMAT REPORT TEXT (HEADINGS + HIGHLIGHTING)
    # -------------------------------------------------
    import re
    import markdown

    # -------------------------------------------------
    # Detect if report is already HTML
    # -------------------------------------------------
    if "<html" in report_text.lower() or "<div" in report_text.lower():
        formatted_report = report_text
    else:
        # Assume Markdown or plain text (LLM output)
        formatted_report = markdown.markdown(
            report_text,
            extensions=["extra", "tables", "sane_lists"]
        )

    # # Convert markdown-style headings into styled HTML sections
    # formatted_report = re.sub(
    #     r"^##\s*(.+)$",
    #     r'<h2 class="section">📊 \1</h2>',
    #     formatted_report,
    #     flags=re.MULTILINE
    # )

    # formatted_report = re.sub(
    #     r"^\*\*(.+?)\*\*",
    #     r'<h3 class="subsection">🔹 \1</h3>',
    #     formatted_report,
    #     flags=re.MULTILINE
    # )

    try:
        district_state = getattr(artifacts, "district")
    except:
        from climaid import climaid_model
        district_state = climaid_model().district

    district_state = district_state or "Unknown Region"
    parts = district_state.split("_")

    if len(parts) >= 3:
        district = f"{parts[1].title()}, {parts[2].title()}, {parts[0].upper()}"
    elif len(parts) >= 2:
        district = f"{parts[0].title()}, {parts[1].title()}"
    else:
        district = district_state.title()

    projection_html = ""
    seasonal_heatmap_html = ""
    ssp_grid_html = ""
    sensitivity_html = ""
    risk_html = ""

    if artifacts and artifacts.projection_summary:
        projection_html = build_projection_from_summary(
            artifacts.projection_summary
        )

        seasonal_heatmap_html = build_dual_seasonal_heatmap(
            artifacts.projection_summary
        )
        

        ssp_grid_html = build_ssp_projection_grid(
            artifacts.projection_summary
        )

        sensitivity_html = build_climate_sensitivity_panel(
            artifacts.importance
        )

        risk_html = build_risk_matrix(
            artifacts.projection_summary
        )

    # Highlight key scientific metrics automatically
    for keyword in ["R²", "RMSE", "SSP", "Ensemble", "Uncertainty", "Climate", "Projection"]:
        formatted_report = formatted_report.replace(
            keyword,
            f'<span class="highlight">{keyword}</span>'
        )

    # Define the filename for saving the projections data
    csv_filename = f"climaid_{artifacts.disease_name}_data_{timestamp}.csv"

    # Button for downloading the csv file.

    download_section = f"""
    <div style="
        margin-top: 30px;
        padding: 24px;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    ">

        <div>
            <div style="font-weight: 600; color: #111827; font-size: 16px; margin-bottom: 4px;">
                Download Projected Cases
            </div>
            <div style="font-size: 14px; color: #6B7280;">
                Includes model projections and outbreak probabilities across scenarios
            </div>
        </div>

        <a href="{csv_filename}" download style="text-decoration: none;">
            <button style="
                background-color: #CBAEBA;
                color: #131a2a;
                padding: 10px 18px;
                border: none;
                border-radius: 8px;
                font-family: 'Inter', -apple-system, system-ui, sans-serif;
                font-weight: 500;
                cursor: pointer;
                font-size: 14px;
                transition: opacity 0.2s;
            " onmouseover="this.style.opacity='0.9'" onmouseout="this.style.opacity='1'">
                Download CSV
            </button>
        </a>
    </div>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{artifacts.disease_name} Risk Report</title>

        <style>
        :root {{
            --primary: #463f3a;
            --secondary: #8a817c;
            --accent: #cbaeba;
            --card-bg: #f4f3ee;
            --border: #e6ecf2;
            --text-main: #1f2933;
            --text-muted: #6b7280;
        }}

        body {{
            margin: 0;
            padding: 0;
            font-family: "Inter", -apple-system, system-ui, sans-serif;
            background: #efe5dc;
            color: var(--text-main);
        }}

        .container {{
            width: 95%;
            max-width: 1200px;
            margin: 40px auto;
            padding-bottom: 60px;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 45px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: flex;
            flex-direction:column;
            gap:8px;
        }}

        .header-bottom{{
            display:flex;
            justify-content:space-between;
            align-items:flex-end;
        }}

        .badges{{
            display:flex;
            gap:10px;
        }}

        .logo{{
            height:75px;
            width:auto;
        }}

        .header h1 {{
            margin: 0;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}

        .meta {{
            margin-top: 15px;
            font-size: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            opacity: 0.95;
        }}

        .badge {{
            background: rgba(255,255,255,0.15);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Metric Cards Styling */
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}

        .metric-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            text-align: center;
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: var(--secondary);
        }}

        .metric-title {{
            font-size: 15px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 28px;
            font-weight: 800;
            color: var(--primary);
        }}

        /* Nav Buttons Styling */
        .plot-nav {{
            display: flex;
            gap: 12px;
            margin-top: 40px;
            justify-content: center;
            flex-wrap: wrap;
        }}

        .nav-btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 30px;
            background-color: #fff;
            color: var(--text-main);
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid var(--border);
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}

        .nav-btn.active {{
            background-color: var(--bg); /* Set via inline style */
            color: white;
            border-color: transparent;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        /* Plot Cards Styling */
        .plot-card {{
            display: none;
            background: white;
            padding: 30px;
            margin-top: 20px;
            border-radius: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        }}

        .plot-card.active {{
            display: block;
            animation: fadeIn 0.5s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .plot-card h2 {{
            margin-top: 0;
            font-size: 22px;
            color: var(--primary);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        /* Original Collapsible Styling */
        .card {{
            background: white;
            padding: 30px;
            margin-top: 30px;
            border-radius: 20px;
            border: 1px solid var(--border);
        }}

        .collapsible {{
            cursor: pointer;
            width: 100%;
            text-align: left;
            padding: 20px;
            border: none;
            font-size: 18px;
            font-weight: 700;
            color: var(--primary);
            background: var(--accent);
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .content {{
            margin-top: 20px;
            display: none; /* Hidden by default now */
        }}

        .scroll-box {{
            max-height: 600px;
            overflow-y: auto;
            padding: 0 15px;
        }}

        .report {{
            font-family: "Georgia", serif;
            font-size: 18px;
            line-height: 1.75;
            color: #2d3748;
        }}

        .footer {{
            text-align: center;
            margin-top: 60px;
            font-size: 14px;
            color: var(--text-muted);
        }}
        </style>
    </head>

    <body>
        <div class="container">
            <div class="header">
                    <h1>ClimAID - Climate-Driven {artifacts.disease_name} Risk Intelligence</h1>

                    <div class="meta">
                        <span><strong>Region:</strong> {district}</span>
                        <span><strong>Study Period:</strong> {artifacts.date_range}</span>
                    </div>

                    <div class="header-bottom">
                        <div class="badges">
                            <span class="badge">Scientific Dashboard</span>
                            <span class="badge">CMIP6-Integrated</span>
                        </div>

                        <img src="data:image/png;base64,{logo_base64_web}" class="logo">
                    </div>
                </div>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Model R²</div>
                    <div class="metric-value">{artifacts.metrics.get("test_r2", "N/A")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">RMSE</div>
                    <div class="metric-value">{artifacts.metrics.get("test_rmse", "N/A")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Training Size</div>
                    <div class="metric-value">{artifacts.data_summary.get("train_size", "N/A") if artifacts.data_summary else "N/A"}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Runtime (s)</div>
                    <div class="metric-value">{total_runtime}</div>
                </div>
            </div>

            <div class="card">
                <button class="collapsible" id="toggleBtn">View Full Scientific Report</button>
                <div class="content" id="reportContent">
                    <div class="scroll-box">
                        <div class="report">
                            {formatted_report}
                        </div>
                    </div>
                </div>
            </div>

            <div class="plot-nav">
                <button class="nav-btn active" onclick="showPlot(event, 'projection')" style="--bg: #2E91E5;">CMIP6 Ensemble</button>
                <button class="nav-btn" onclick="showPlot(event, 'seasonal')" style="--bg: #8E44AD;">Seasonal Heatmap</button>
                <button class="nav-btn" onclick="showPlot(event, 'ssp_grid')" style="--bg: #2CA02C;">SSP Grid</button>
                <button class="nav-btn" onclick="showPlot(event, 'sensitivity')" style="--bg: #FF7F0E;">Sensitivity</button>
                <button class="nav-btn" onclick="showPlot(event, 'risk')" style="--bg: #D62728;">Risk Matrix</button>
            </div>

            <div id="plot-container">
                <div id="projection" class="plot-card active">
                    <h2>CMIP6 Climate–{artifacts.disease_name} Projections</h2>
                    {projection_html}
                </div>

                <div id = "seasonal" class="plot-card">
                    <h2> Seasonal Projection Heatmap</h2>
                    {seasonal_heatmap_html}
                </div>

                <div id="ssp_grid" class="plot-card">
                    <h2>SSP-wise Projection Grid</h2>
                    {ssp_grid_html}
                </div>

                <div id="sensitivity" class="plot-card">
                    <h2>Climate Sensitivity Analysis</h2>
                    {sensitivity_html}
                </div>

                <div id="risk" class="plot-card">
                    <h2>Dual-Baseline Outbreak Risk Matrix</h2>
                    {risk_html}
                </div>
            </div>

            {download_section}

            <div class="footer">
                <strong>ClimAID: Climate Change impact using AI on Diseases </strong><br>
                Interactive Scientific Dashboard | Designed for Epidemiologists and Policy Analysts
            </div>
        </div>

        <script>
        function showPlot(event, plotId) {{
            // Hide all plots
            document.querySelectorAll('.plot-card').forEach(card => {{
                card.classList.remove('active');
            }});
            
            // Deactivate all buttons
            document.querySelectorAll('.nav-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show target plot
            document.getElementById(plotId).classList.add('active');
            
            // Set button as active
            event.currentTarget.classList.add('active');

            // Force Plotly to recalculate width for the newly shown chart
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
            }}, 200);
        }}

        // Logic for Scientific Report Toggle
        const coll = document.getElementById("toggleBtn");
        const content = document.getElementById("reportContent");
        
        coll.addEventListener("click", function() {{
            if (content.style.display === "block") {{
                content.style.display = "none";
                this.style.borderRadius = "12px";
            }} else {{
                content.style.display = "block";
                this.style.borderRadius = "12px 12px 0 0";
            }}
        }});
        </script>
    </body>
    </html>
    """

    # -----------------------------
    # Default output directory
    # -----------------------------
    if save_copy and output_dir is None:
        output_dir = "climaid_outputs/reports"

    # -----------------------------
    # Save report + CSV together
    # -----------------------------
    if save_copy:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"climaid_{artifacts.disease_name}_report_{timestamp}.html"
        file_path = Path(output_dir) / filename

        # -----------------------------
        # Save downloadable dataset
        # -----------------------------
        csv_filename = f"climaid_{artifacts.disease_name}_data_{timestamp}.csv"
        csv_path = Path(output_dir) / csv_filename

        if artifacts.download_data is not None:
            artifacts.download_data.to_csv(csv_path, index=False)

        # Save HTML
        file_path.write_text(html_content, encoding="utf-8")

    # -----------------------------
    # Temporary file case
    # -----------------------------
    else:

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".html",
            mode="w",
            encoding="utf-8"
        )

        tmp.write(html_content)
        tmp.close()

        file_path = Path(tmp.name)

        # -----------------------------
        # Save CSV in SAME temp folder
        # -----------------------------
        csv_filename = f"climaid_{artifacts.disease_name}_data_{timestamp}.csv"
        csv_path = file_path.parent / csv_filename

        if artifacts.download_data is not None:
            artifacts.download_data.to_csv(csv_path, index=False)

    # -----------------------------
    # Open report
    # -----------------------------
    webbrowser.open(f"file://{file_path.resolve()}")
    return str(file_path)
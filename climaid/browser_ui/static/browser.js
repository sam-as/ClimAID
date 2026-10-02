/* ======================================================
ClimAID Browser Wizard
====================================================== */

let uploadedFile = null;
let districtCatalog = {};
let MODE = "southasia";

const SESSION_STORAGE_KEY = "climaid_session_id";
let climaidSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
if(!climaidSessionId){
    climaidSessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + "-" + Math.random().toString(16).slice(2));
    localStorage.setItem(SESSION_STORAGE_KEY, climaidSessionId);
}

let diseaseUploaded = false;
let weatherUploaded = false;
let projectionUploaded = false;

function apiHeaders(extra={}){
    return Object.assign({"X-ClimAID-Session": climaidSessionId}, extra);
}

async function apiFetch(url, options={}){
    const opts = Object.assign({}, options);
    opts.headers = apiHeaders(options.headers || {});
    return fetch(url, opts);
}

function setFileStatus(labelId, text, kind="ok"){
    const el=document.getElementById(labelId);
    if(!el) return;
    el.textContent=text;
    el.classList.remove("upload-ok","upload-error","upload-busy");
    el.classList.add(kind === "error" ? "upload-error" : kind === "busy" ? "upload-busy" : "upload-ok");
}

const SOUTH_ASIA_COUNTRIES = [
    "IND","AFG","BGD","BTN","LKA","MMR","NPL","PAK"
];

const COUNTRY_NAMES = {
    IND:"India",
    BGD:"Bangladesh",
    NPL:"Nepal",
    BTN:"Bhutan",
    LKA:"Sri Lanka",
    MMR:"Myanmar",
    PAK:"Pakistan",
    AFG:"Afghanistan"
};


/* ======================================================
INITIALIZE
====================================================== */

/* ======================================================
MODE SWITCH
====================================================== */

function setMode(mode){

    MODE = mode;

    const saBtn = document.getElementById("saModeBtn");
    const globalBtn = document.getElementById("globalModeBtn");

    const stateSelect = document.getElementById("state");
    const districtSelect = document.getElementById("district");

    const stateManual = document.getElementById("manual_state");
    const districtManual = document.getElementById("manual_district");

    const climateUploads = document.getElementById("externalClimateSection");

    if(mode === "southasia"){

        saBtn.classList.add("active");
        globalBtn.classList.remove("active");

        stateSelect.style.display = "block";
        districtSelect.style.display = "block";

        stateManual.style.display = "none";
        districtManual.style.display = "none";

        climateUploads.style.display = "none";

        loadDistrictCatalog();

    }else{

        globalBtn.classList.add("active");
        saBtn.classList.remove("active");

        stateSelect.style.display = "none";
        districtSelect.style.display = "none";

        stateManual.style.display = "block";
        districtManual.style.display = "block";

        climateUploads.style.display = "block";

        loadGlobalCountries();
    }
}


/* ======================================================
LOAD SOUTH ASIA
====================================================== */

async function loadDistrictCatalog(){

    try{

        const res = await apiFetch("/district_catalog");

        if(!res.ok)
            throw new Error("district catalog unavailable");

        districtCatalog = await res.json();

        const countrySelect = document.getElementById("country");
        countrySelect.innerHTML="";

        const countries = Object.keys(districtCatalog);

        const sortedCountries = [
            ...SOUTH_ASIA_COUNTRIES.filter(c=>countries.includes(c)),
            ...countries.filter(c=>!SOUTH_ASIA_COUNTRIES.includes(c)).sort()
        ];

        sortedCountries.forEach(country=>{

            const opt=document.createElement("option");
            opt.value=country;
            opt.text=COUNTRY_NAMES[country] || country;

            countrySelect.appendChild(opt);
        });

        if(countries.includes("IND"))
            countrySelect.value="IND";

        countrySelect.onchange=updateStates;

        updateStates();

    }catch(err){
        console.warn("District catalog not loaded:",err);
    }
}


/* ======================================================
GLOBAL COUNTRIES
====================================================== */

function loadGlobalCountries(){

    const countrySelect = document.getElementById("country");
    countrySelect.innerHTML="";

    const countries = {
        IND:"India",
        BGD:"Bangladesh",
        NPL:"Nepal",
        BTN:"Bhutan",
        LKA:"Sri Lanka",
        MMR:"Myanmar",
        PAK:"Pakistan",
        AFG:"Afghanistan",
        ALB:"Albania",
        DZA:"Algeria",
        AND:"Andorra",
        AGO:"Angola",
        ARG:"Argentina",
        ARM:"Armenia",
        AUS:"Australia",
        AUT:"Austria",
        AZE:"Azerbaijan",
        BHR:"Bahrain",
        BGD:"Bangladesh",
        BLR:"Belarus",
        BEL:"Belgium",
        BEN:"Benin",
        BTN:"Bhutan",
        BOL:"Bolivia",
        BIH:"Bosnia and Herzegovina",
        BWA:"Botswana",
        BRA:"Brazil",
        BRN:"Brunei",
        BGR:"Bulgaria",
        BFA:"Burkina Faso",
        BDI:"Burundi",
        KHM:"Cambodia",
        CMR:"Cameroon",
        CAN:"Canada",
        CAF:"Central African Republic",
        TCD:"Chad",
        CHL:"Chile",
        CHN:"China",
        COL:"Colombia",
        COM:"Comoros",
        COG:"Congo",
        COD:"DR Congo",
        CRI:"Costa Rica",
        CIV:"Côte d’Ivoire",
        HRV:"Croatia",
        CUB:"Cuba",
        CYP:"Cyprus",
        CZE:"Czech Republic",
        DNK:"Denmark",
        DJI:"Djibouti",
        DOM:"Dominican Republic",
        ECU:"Ecuador",
        EGY:"Egypt",
        SLV:"El Salvador",
        GNQ:"Equatorial Guinea",
        ERI:"Eritrea",
        EST:"Estonia",
        ETH:"Ethiopia",
        FJI:"Fiji",
        FIN:"Finland",
        FRA:"France",
        GAB:"Gabon",
        GMB:"Gambia",
        GEO:"Georgia",
        DEU:"Germany",
        GHA:"Ghana",
        GRC:"Greece",
        GTM:"Guatemala",
        GIN:"Guinea",
        GNB:"Guinea-Bissau",
        GUY:"Guyana",
        HTI:"Haiti",
        HND:"Honduras",
        HUN:"Hungary",
        ISL:"Iceland",
        IDN:"Indonesia",
        IRN:"Iran",
        IRQ:"Iraq",
        IRL:"Ireland",
        ISR:"Israel",
        ITA:"Italy",
        JAM:"Jamaica",
        JPN:"Japan",
        JOR:"Jordan",
        KAZ:"Kazakhstan",
        KEN:"Kenya",
        KWT:"Kuwait",
        KGZ:"Kyrgyzstan",
        LAO:"Laos",
        LVA:"Latvia",
        LBN:"Lebanon",
        LSO:"Lesotho",
        LBR:"Liberia",
        LBY:"Libya",
        LTU:"Lithuania",
        LUX:"Luxembourg",
        MDG:"Madagascar",
        MWI:"Malawi",
        MYS:"Malaysia",
        MDV:"Maldives",
        MLI:"Mali",
        MLT:"Malta",
        MRT:"Mauritania",
        MUS:"Mauritius",
        MEX:"Mexico",
        MDA:"Moldova",
        MNG:"Mongolia",
        MAR:"Morocco",
        MOZ:"Mozambique",
        NAM:"Namibia",
        NLD:"Netherlands",
        NZL:"New Zealand",
        NIC:"Nicaragua",
        NER:"Niger",
        NGA:"Nigeria",
        PRK:"North Korea",
        MKD:"North Macedonia",
        NOR:"Norway",
        OMN:"Oman",
        PAN:"Panama",
        PNG:"Papua New Guinea",
        PRY:"Paraguay",
        PER:"Peru",
        PHL:"Philippines",
        POL:"Poland",
        PRT:"Portugal",
        QAT:"Qatar",
        ROU:"Romania",
        RUS:"Russia",
        RWA:"Rwanda",
        SAU:"Saudi Arabia",
        SEN:"Senegal",
        SRB:"Serbia",
        SLE:"Sierra Leone",
        SGP:"Singapore",
        SVK:"Slovakia",
        SVN:"Slovenia",
        SOM:"Somalia",
        ZAF:"South Africa",
        KOR:"South Korea",
        ESP:"Spain",
        SDN:"Sudan",
        SUR:"Suriname",
        SWE:"Sweden",
        CHE:"Switzerland",
        SYR:"Syria",
        TWN:"Taiwan",
        TJK:"Tajikistan",
        TZA:"Tanzania",
        THA:"Thailand",
        TGO:"Togo",
        TUN:"Tunisia",
        TUR:"Turkey",
        UGA:"Uganda",
        UKR:"Ukraine",
        ARE:"United Arab Emirates",
        GBR:"United Kingdom",
        USA:"United States",
        URY:"Uruguay",
        UZB:"Uzbekistan",
        VEN:"Venezuela",
        VNM:"Vietnam",
        YEM:"Yemen",
        ZMB:"Zambia",
        ZWE:"Zimbabwe"
    };

    Object.entries(countries).forEach(([code,name])=>{
        const opt=document.createElement("option");
        opt.value=code;
        opt.text=name;
        countrySelect.appendChild(opt);
    });
}


/* ======================================================
STATE + DISTRICT
====================================================== */

function updateStates(){

    const country=document.getElementById("country").value;
    const stateSelect=document.getElementById("state");

    stateSelect.innerHTML="";

    if(!districtCatalog[country]) return;

    Object.keys(districtCatalog[country]).forEach(state=>{
        const opt=document.createElement("option");
        opt.value=state;
        opt.text=state;
        stateSelect.appendChild(opt);
    });

    stateSelect.onchange=updateDistricts;
    updateDistricts();
}

function updateDistricts(){

    const country=document.getElementById("country").value;
    const state=document.getElementById("state").value;

    const districtSelect=document.getElementById("district");
    districtSelect.innerHTML="";

    if(!districtCatalog[country] || !districtCatalog[country][state])
        return;

    districtCatalog[country][state].forEach(d=>{
        const opt=document.createElement("option");
        opt.value=d;
        opt.text=d;
        districtSelect.appendChild(opt);
    });
}


/* ======================================================
DROPZONE (FIXED)
====================================================== */

function setupDropzone(zoneId,inputId,labelId,onFile=null){
    const zone=document.getElementById(zoneId);
    const input=document.getElementById(inputId);
    const label=document.getElementById(labelId);
    if(!zone || !input) return;

    const acceptFile = async (file) => {
        if(!file) return;
        if(label) label.textContent = `${file.name} — uploading…`;
        setFileStatus(labelId, `${file.name} — uploading…`, "busy");
        try{
            if(onFile) await onFile(file);
        }catch(err){
            console.error(err);
            setFileStatus(labelId, `${file.name} — ${err.message || err}`, "error");
        }
    };

    zone.onclick = () => input.click();
    input.onchange = async (e)=>{
        const file=e.target.files[0];
        if(file) await acceptFile(file);
        // Allows selecting the same file again after an upload/error.
        input.value = "";
    };
    zone.ondragover = (e)=>{ e.preventDefault(); zone.classList.add("dragover"); };
    zone.ondragleave = ()=> zone.classList.remove("dragover");
    zone.ondrop = async (e)=>{
        e.preventDefault();
        zone.classList.remove("dragover");
        const file=e.dataTransfer.files[0];
        if(file) await acceptFile(file);
    };
}


/* ======================================================
UPLOAD SETUP
====================================================== */

function setupFileUpload(){
    setupDropzone("dropzone","fileInput","filename", async file => {
        uploadedFile = file;
        const data = await uploadDataset(file);
        diseaseUploaded = true;
        setFileStatus("filename", `${file.name} ✓ uploaded (${data.rows} rows)`, "ok");
    });
}

let weatherFile = null;
let projectionFile = null;

function setupClimateUploads(){
    setupDropzone("weatherDropzone","weather_file","weather_filename", async file => {
        weatherFile = file;
        const data = await uploadWeather(file);
        weatherUploaded = true;
        setFileStatus("weather_filename", `${file.name} ✓ uploaded`, "ok");
    });
    setupDropzone("projectionDropzone","projection_file","projection_filename", async file => {
        projectionFile = file;
        const data = await uploadProjection(file);
        projectionUploaded = true;
        setFileStatus("projection_filename", `${file.name} ✓ uploaded`, "ok");
    });
}

async function uploadDataset(file=uploadedFile){
    if(!file) throw new Error("Please upload a disease dataset");
    const formData=new FormData();
    formData.append("file",file);
    const res=await apiFetch("/upload_dataset",{method:"POST",body:formData});
    let data={};
    try{ data=await res.json(); }catch(_){}
    if(!res.ok || data.error || data.detail) throw new Error(data.detail || data.error || `Upload failed (${res.status})`);
    return data;
}

async function uploadWeather(file=weatherFile){
    if(!file) throw new Error("Please upload a weather file");
    const formData = new FormData();
    formData.append("file", file);
    const res = await apiFetch("/upload_weather", {method:"POST", body:formData});
    const data = await res.json();
    if(!res.ok || data.error || data.detail) throw new Error(data.detail || data.error || `Upload failed (${res.status})`);
    return data;
}

async function uploadProjection(file=projectionFile){
    if(!file) throw new Error("Please upload a projection file");
    const formData = new FormData();
    formData.append("file", file);
    const res = await apiFetch("/upload_projection", {method:"POST", body:formData});
    const data = await res.json();
    if(!res.ok || data.error || data.detail) throw new Error(data.detail || data.error || `Upload failed (${res.status})`);
    return data;
}

/* ======================================================
RUN PIPELINE
====================================================== */

async function runClimAID(options={}){

    const status=document.getElementById("status");

    try{
        status.textContent=diseaseUploaded ? "Disease dataset ready." : "Uploading disease dataset...";
        if(!diseaseUploaded) await uploadDataset();

        let state, district;
        if(MODE==="southasia"){
            state=document.getElementById("state").value;
            district=document.getElementById("district").value;
        }else{
            state=document.getElementById("manual_state").value;
            district=document.getElementById("manual_district").value;
        }

        if(!state || !district)
            throw new Error("Please select/enter a state and district.");

        if(MODE==="global"){
            status.textContent="Checking uploaded climate data...";
            if(!weatherUploaded) await uploadWeather();
            if(!projectionUploaded) await uploadProjection();
        }

        const originYearRaw=document.getElementById("forecast_origin_year")?.value;
        const runV2 = options.v1Only ? false : (options.v2Only ? true : document.getElementById("run_v2")?.value !== "no");
        const runLegacy = options.v2Only ? false : (options.v1Only ? true : document.getElementById("run_legacy")?.value === "yes");
        const horizon = parseInt(document.getElementById("forecast_horizon")?.value || "12");
        const sims = parseInt(document.getElementById("simulations")?.value || "2000");

        let cfg={
            mode:MODE,
            country:document.getElementById("country").value,
            state:state,
            district:district,
            disease_name:document.getElementById("disease").value,
            preset:document.getElementById("preset").value,
            test_year:parseInt(document.getElementById("test_year").value)||null,
            run_v2:runV2,
            run_legacy:runLegacy,
            forecast_origin_year:parseInt(originYearRaw)||null,
            forecast_horizon:horizon,
            simulations:sims,
            population_at_risk:parseFloat(document.getElementById("population_at_risk")?.value)||null,
            population_column:(document.getElementById("population_column")?.value||"population").trim()||"population",
            forecast_climate_source:document.getElementById("forecast_climate_source")?.value||"auto",
            run_hindcasts:document.getElementById("run_hindcasts")?.value !== "no",
            hindcast_origins:parseInt(document.getElementById("hindcast_origins")?.value||"4"),
            hindcast_horizon:parseInt(document.getElementById("hindcast_horizon")?.value||String(horizon)),
            save_html:document.getElementById("save_html")?.value !== "no",
            v2_models:getCheckedModels("v2_models_container"),
            run_cmip6:document.getElementById("run_cmip6")?.value !== "no",
            legacy_report_mode:document.getElementById("legacy_report_mode")?.value||"deterministic",
            exclude_period:excludePeriodValue(),
            v2_tuning:document.getElementById("v2_tuning")?.value||"balanced",
            run_scenarios:document.getElementById("run_scenarios")?.value !== "no",
            scenario_end_year:parseInt(document.getElementById("scenario_end_year")?.value||"2050"),
            scenario_ssps:(document.getElementById("scenario_ssps")?.value||"").split(",").map(x=>x.trim().toLowerCase()).filter(Boolean),
            scenario_response:document.getElementById("scenario_response")?.value||"seasonal",
            scenario_lag_selection:document.getElementById("scenario_lag_selection")?.value||"ensemble",
            scenario_compare_trees:document.getElementById("scenario_compare_trees")?.value !== "no",
            scenario_temperature_curve:document.getElementById("scenario_temperature_curve")?.value||null
        };

        if(runV2 && (!cfg.v2_models || cfg.v2_models.length===0))
            throw new Error("Select at least one v2 forecasting model.");

        if (cfg.preset === "custom" || runLegacy) {
            const trialsInput=document.getElementById("trials").value;
            cfg.n_trials=trialsInput==="" ? null : parseInt(trialsInput);
            cfg.base_models=getCheckedModels("base_models_container");
            cfg.residual_models=getCheckedModels("residual_models_container");
            cfg.correction_models=getCheckedModels("correction_models_container");
        }

        status.textContent=runLegacy && runV2 ? "Running legacy + ClimAID v2..." : (runV2 ? "Running ClimAID v2..." : "Running legacy ClimAID...");

        const res=await apiFetch("/run",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(cfg)
        });

        if(!res.ok) throw new Error("Server error: "+res.status);
        const data=await res.json();
        if(data.error) throw new Error(data.error);

        status.textContent="ClimAID pipeline completed successfully.";
        renderReportLinks(data.reports || []);

        if(data.scenario_error){
            status.textContent += `\nScenario outlook skipped: ${data.scenario_error}`;
        }
        if(data.legacy && data.legacy.note){
            status.textContent += `\n${data.legacy.note}`;
        }

        if(data.v2 && data.v2.metadata){
            const m=data.v2.metadata;
            status.textContent += `\nForecast: ${m.forecast_start || ""} → ${m.forecast_end || ""}; models: ${(m.models||[]).join(", ")}`;
        }

        return data;

    }catch(err){
        console.error(err);
        status.textContent="Error: "+err.message;
        throw err;
    }
}

async function runClimAIDV2Only(){
    return runClimAID({v2Only:true});
}

async function runClimAIDV1Only(){
    return runClimAID({v1Only:true});
}

function renderReportLinks(reports){
    const box=document.getElementById("report-links");
    if(!box) return;
    if(!reports.length){ box.innerHTML=""; return; }
    box.innerHTML = "<strong>Generated reports</strong>" + reports.map(r=>{
        const label = r.type || "report";
        const url = r.url || "#";
        return `<a class="report-link" href="${url}" target="_blank" rel="noopener">${label}</a>`;
    }).join(" ");
}


/* ======================================================
CUSTOM PRESET TOGGLE
====================================================== */

function setupPresetToggle() {

    const preset = document.getElementById("preset");
    const custom = document.getElementById("model-selection");

    preset.onchange = () => {

        if (preset.value === "custom")
            custom.style.display = "block";

        else
            custom.style.display = "none";
    };
}



/* ======================================================
LOAD AVAILABLE MODELS
====================================================== */

async function loadAvailableModels() {

    let res = await apiFetch("/available_models");
    let data = await res.json();

    const models = data.models;

    populateModelBox("base_models_container", models);
    populateModelBox("residual_models_container", models);
    populateModelBox("correction_models_container", models);
}



/* ======================================================
POPULATE CHECKBOX CONTAINERS
====================================================== */

function populateModelBox(containerId, models) {

    const container = document.getElementById(containerId);
    container.innerHTML = "";


    /* remove duplicates / aliases */

    const aliasMap = {

        rf: "random_forest",
        extratrees: "extra_trees",
        gbr: "gradient_boosting",

        xgb: "xgboost",
        lgbm: "lightgbm",

        nn: "mlp",
        neural_net: "mlp"
    };


    models = models.map(m => aliasMap[m] || m);
    models = [...new Set(models)];


    /* model categories */

    const groups = {

        "Linear Models": [
            "linear",
            "ridge",
            "lasso",
            "elasticnet",
            "poisson"
        ],

        "Tree-Based Models": [
            "random_forest",
            "extra_trees",
            "gradient_boosting"
        ],

        "Boosting Libraries": [
            "xgboost",
            "lightgbm",
            "catboost"
        ],

        "Neural Networks": [
            "mlp"
        ],

        "Calibration": [
            "isotonic"
        ]
    };


    for (let group in groups) {

        const wrapper = document.createElement("div");
        wrapper.className = "model-group";


        const header = document.createElement("div");
        header.className = "model-header";
        header.textContent = group;


        const body = document.createElement("div");
        body.className = "model-body";


        header.onclick = () => {

            body.style.display =
                body.style.display === "block"
                    ? "none"
                    : "block";
        };


        groups[group].forEach(model => {

            if (!models.includes(model)) return;

            const row = document.createElement("div");
            row.className = "model-option";


            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = model;


            const text = document.createElement("span");
            text.textContent = formatModelName(model);


            row.appendChild(checkbox);
            row.appendChild(text);
            const tip = modelTip(model);
            if (tip) row.appendChild(infoIcon(tip));

            body.appendChild(row);
        });


        wrapper.appendChild(header);
        wrapper.appendChild(body);

        container.appendChild(wrapper);
    }

}



/* ======================================================
HELPER: Format the model names
====================================================== */

function formatModelName(name) {

    const labels = {

        linear: "Linear Regression",
        ridge: "Ridge Regression",
        lasso: "Lasso Regression",
        elasticnet: "Elastic Net",
        poisson: "Poisson Regression",

        random_forest: "Random Forest",
        extra_trees: "Extra Trees",
        gradient_boosting: "Gradient Boosting",

        xgboost: "XGBoost",
        lightgbm: "LightGBM",
        catboost: "CatBoost",

        mlp: "Neural Network (MLP)",

        isotonic: "Isotonic Calibration"
    };

    return labels[name] || name;
}



/* ======================================================
HELPERS
====================================================== */

function selectAllModels(containerId) {

    const boxes =
        document
        .getElementById(containerId)
        .querySelectorAll("input[type=checkbox]");

    boxes.forEach(b => b.checked = true);
}



function clearModels(containerId) {

    const boxes =
        document
        .getElementById(containerId)
        .querySelectorAll("input[type=checkbox]");

    boxes.forEach(b => b.checked = false);
}



function selectRecommended(containerId) {

    const recommended = [
        "random_forest",
        "xgboost",
        "lightgbm",
        "poisson"
    ];

    const boxes =
        document
        .getElementById(containerId)
        .querySelectorAll("input[type=checkbox]");

    boxes.forEach(b => {

        b.checked = recommended.includes(b.value);

    });
}



/* ======================================================
GET SELECTED MODELS FROM A CONTAINER
====================================================== */

function getCheckedModels(containerId) {

    const container = document.getElementById(containerId);

    if (!container) return [];

    const boxes =
        container.querySelectorAll("input[type=checkbox]");

    let selected = [];

    boxes.forEach(box => {

        if (box.checked)
            selected.push(box.value);

    });

    return selected;
}



/* ======================================================
INITIALIZE WIZARD
====================================================== */

window.onload = function () {

    console.log("ClimAID browser.js loaded");

    loadDistrictCatalog();
    loadAvailableModels();
    loadAvailableV2Models();

    setupFileUpload();
    setupClimateUploads();
    setupPresetToggle();

};

/* ======================================================
CLIMAID V2 MODEL SELECTION (ADDITIVE)
====================================================== */
async function loadAvailableV2Models(){
    try{
        const res=await apiFetch("/available_v2_models");
        if(!res.ok) throw new Error("v2 model endpoint unavailable");
        const data=await res.json();
        const models=data.models || [];
        const container=document.getElementById("v2_models_container");
        if(!container) return;
        container.innerHTML="";
        const labels={
            seasonal_naive:"Seasonal Naïve",
            renewal:"Climate renewal + susceptibility depletion",
            linear:"Linear Regression",
            ridge:"Ridge Regression",
            lasso:"Lasso Regression",
            elasticnet:"Elastic Net",
            poisson:"Poisson Regression",
            random_forest:"Random Forest",
            extra_trees:"Extra Trees",
            gradient_boosting:"Gradient Boosting",
            xgboost:"XGBoost",
            lightgbm:"LightGBM",
            catboost:"CatBoost",
            mlp:"Neural Network (MLP)",
            tweedie:"Tweedie Regression",
            spline_poisson:"Smooth-curve Poisson (GAM-style)",
            bayesian_ridge:"Bayesian Ridge",
            huber:"Huber (spike-resistant)",
            hist_gradient_boosting:"Hist Gradient Boosting (Poisson)",
            svr:"Support Vector Regression",
            knn:"Nearest Neighbours (KNN)",
            v1_stack:"ClimAID v1 stacked model",
            sarimax:"SARIMAX (seasonal ARIMA)"
        };
        const defaults=data.defaults || ["seasonal_naive","renewal","poisson","random_forest","extra_trees","gradient_boosting"];
        models.forEach(model=>{
            const row=document.createElement("div"); row.className="model-option";
            const cb=document.createElement("input"); cb.type="checkbox"; cb.value=model; cb.checked=defaults.includes(model);
            const text=document.createElement("span"); text.textContent=labels[model]||model;
            row.appendChild(cb); row.appendChild(text);
            const tip=modelTip(model); if(tip) row.appendChild(infoIcon(tip));
            container.appendChild(row);
        });
    }catch(err){ console.warn("v2 model list unavailable:",err); }
}


/* ======================================================
PAGES (v2 default, v1 legacy) AND INFO TOOLTIPS
====================================================== */
function showPage(page){
    document.querySelectorAll(".pipeline-page").forEach(el=>{ el.hidden = el.dataset.page !== page; });
    document.querySelectorAll(".page-tab").forEach(btn=>{
        const on = btn.id === "tab-" + page;
        btn.classList.toggle("active", on);
        btn.setAttribute("aria-selected", on ? "true" : "false");
    });
    try{ history.replaceState(null, "", "#" + page); }catch(e){}
}
document.addEventListener("DOMContentLoaded", ()=>{
    showPage(location.hash === "#v1" ? "v1" : "v2");   // v2 is the default page
});

function infoIcon(tip){
    const i=document.createElement("span");
    i.className="info"; i.tabIndex=0; i.setAttribute("role","note");
    i.setAttribute("aria-label",tip); i.dataset.tip=tip; i.textContent="i";
    return i;
}

const MODEL_TIPS = {
    seasonal_naive:"Repeats last year's value for the same month. The simple benchmark every other model should beat.",
    renewal:"Mechanistic transmission model: each month's cases arise from recent cases, scaled by a climate-driven reproduction rate.",
    linear:"Straight-line relationship between the climate and lag features and cases.",
    ridge:"Linear model with shrinkage, which reduces overfitting when features are correlated.",
    lasso:"Linear model with shrinkage that can drop unhelpful features entirely.",
    elasticnet:"Linear model mixing ridge and lasso shrinkage.",
    poisson:"Generalised linear model for counts, where climate effects multiply the expected number of cases.",
    random_forest:"Average of many decision trees. Captures non-linear climate effects; a robust default.",
    extra_trees:"Like random forest but with more random splits; often smoother and less prone to overfitting.",
    gradient_boosting:"Trees built one after another, each correcting the previous ones' errors. Often the most accurate tree model.",
    xgboost:"Fast, optimised gradient boosting (optional package).",
    lightgbm:"Fast, optimised gradient boosting that handles many features well (optional package).",
    catboost:"Gradient boosting that is robust with default settings (optional package).",
    mlp:"Small neural network. Needs more data than the others and can be unstable on short series.",
    isotonic:"Monotonic calibration: adjusts predictions up or down without changing their order.",
    tweedie:"Regression for counts that vary more than a Poisson model allows (overdispersion), common in outbreak data.",
    spline_poisson:"Poisson regression on smooth curves of each input (GAM-style), so climate effects can bend rather than follow a straight line.",
    bayesian_ridge:"Linear model that estimates how much shrinkage to apply from the data itself.",
    huber:"Linear model that down-weights unusually large spikes, so a single outbreak does not distort the fit.",
    hist_gradient_boosting:"Fast gradient boosting with a Poisson loss, designed for count data.",
    svr:"Support vector regression: fits a smooth curve while ignoring small errors. Often weaker on short series.",
    knn:"Predicts from the most similar past months. Simple, but usually weaker on short series and cannot extrapolate.",
    sarimax:"Seasonal ARIMA with climate as external inputs: the classic time-series benchmark in climate-and-disease studies. Fitted on log(1 + cases); tuning picks its orders and the climate lag. Monthly data only; slower than the regression models.",
    v1_stack:"The full ClimAID v1 pipeline (lag search, then base, residual and correction models) run inside v2, so it is tested and given likely ranges exactly like the other models. Slow: about a minute per fit on Fast tuning, and it is refitted at every hindcast start date."
};
const MODEL_ALIASES = {rf:"random_forest", extratrees:"extra_trees", gbr:"gradient_boosting", xgb:"xgboost",
                       lgbm:"lightgbm", neural_net:"mlp", nn:"mlp"};
function modelTip(model){ return MODEL_TIPS[MODEL_ALIASES[model] || model] || ""; }


/* ======================================================
COVID-19 DISRUPTION PERIOD
====================================================== */
function toggleCustomPeriod(){
    const custom = document.getElementById("exclude_period_mode")?.value === "custom";
    document.querySelectorAll(".custom-period").forEach(el => el.hidden = !custom);
}
function excludePeriodValue(){
    const mode = document.getElementById("exclude_period_mode")?.value || "2020";
    if (mode !== "custom") return mode;
    const a = document.getElementById("exclude_start")?.value, b = document.getElementById("exclude_end")?.value;
    if (!a || !b) throw new Error("Please choose both months of the custom COVID-19 period.");
    if (b < a) throw new Error("The custom COVID-19 period ends before it starts.");
    return `${a}:${b}`;
}

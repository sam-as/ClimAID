/* ======================================================
ClimAID Browser Wizard
====================================================== */

let uploadedFile = null;
let districtCatalog = {};
let MODE = "southasia";

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

window.onload = function () {

    console.log("ClimAID initialized");

    loadDistrictCatalog();
    loadAvailableModels();

    setupFileUpload();
    setupClimateUploads();
    setupPresetToggle();
};


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

        const res = await fetch("/district_catalog");

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

    zone.onclick = () => input.click();

    input.onchange = (e)=>{
        const file=e.target.files[0];
        if(!file) return;

        label.textContent=file.name;

        if(onFile) onFile(file);
    };

    zone.ondragover = (e)=>{
        e.preventDefault();
        zone.classList.add("dragover");
    };

    zone.ondragleave = ()=>{
        zone.classList.remove("dragover");
    };

    zone.ondrop = (e)=>{
        e.preventDefault();
        zone.classList.remove("dragover");

        const file=e.dataTransfer.files[0];
        if(!file) return;

        label.textContent=file.name;

        if(onFile) onFile(file);
    };
}


/* ======================================================
UPLOAD SETUP
====================================================== */

function setupFileUpload(){

    setupDropzone(
        "dropzone",
        "fileInput",
        "filename",
        file => {
            uploadedFile = file;
            console.log("Disease file uploaded:", file.name);
        }
    );
}

let weatherFile = null;
let projectionFile = null;

function setupClimateUploads(){

    setupDropzone(
        "weatherDropzone",
        "weather_file",
        "weather_filename",
        file => {
            weatherFile = file;
            console.log("Weather selected:", file.name);
        }
    );

    setupDropzone(
        "projectionDropzone",
        "projection_file",
        "projection_filename",
        file => {
            projectionFile = file;
            console.log("Projection selected:", file.name);
        }
    );
}

/* ======================================================
UPLOAD DATASET
====================================================== */

async function uploadDataset(){

    if(!uploadedFile)
        throw new Error("Please upload a disease dataset");

    const formData=new FormData();
    formData.append("file",uploadedFile);

    await fetch("/upload_dataset",{   // FIXED ENDPOINT
        method:"POST",
        body:formData
    });

    console.log("Uploading:", uploadedFile.name);
}

async function uploadWeather(){

    if(!weatherFile)
        throw new Error("Please upload a weather file");

    const formData = new FormData();
    formData.append("file", weatherFile);

    const res = await fetch("/upload_weather", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if(data.error){
        throw new Error(data.error);
    }

    console.log("Weather uploaded:", data);
}

async function uploadProjection(){

    if(!projectionFile)
        throw new Error("Please upload a projection file");

    const formData = new FormData();
    formData.append("file", projectionFile);

    const res = await fetch("/upload_projection", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if(data.error){
        throw new Error(data.error);
    }

    console.log("Projection uploaded:", data);
}

/* ======================================================
RUN PIPELINE
====================================================== */

async function runClimAID(){

    const status=document.getElementById("status");

    try{

        status.textContent="Uploading dataset...";
        await uploadDataset();

        let state, district;

        if(MODE==="southasia"){
            state=document.getElementById("state").value;
            district=document.getElementById("district").value;
        }else{
            state=document.getElementById("manual_state").value;
            district=document.getElementById("manual_district").value;
        }

        if(MODE==="global"){
            status.textContent="Uploading climate data...";
            await uploadWeather();
            await uploadProjection();
        }

        let cfg={
            mode:MODE,
            country:document.getElementById("country").value,
            state:state,
            district:district,
            disease_name:document.getElementById("disease").value,
            preset:document.getElementById("preset").value,
            test_year:parseInt(document.getElementById("test_year").value)||null
        };

        if (cfg.preset === "custom") {

            const trialsInput =
                document.getElementById("trials").value;

            cfg.n_trials =
                trialsInput === ""
                    ? null
                    : parseInt(trialsInput);

            cfg.base_models =
                getCheckedModels("base_models_container");

            cfg.residual_models =
                getCheckedModels("residual_models_container");

            cfg.correction_models =
                getCheckedModels("correction_models_container");
        }

        status.textContent="Running ClimAID pipeline...";

        const res=await fetch("/run",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(cfg)
        });

        if(!res.ok)
            throw new Error("Server error: "+res.status);

        const data = await res.json();

        if(data.error){
            throw new Error(data.error);
        }

        status.textContent="ClimAID pipeline completed successfully.";

    }catch(err){

        console.error(err);
        status.textContent="Error: "+err.message;
    }
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

    let res = await fetch("/available_models");
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

    setupFileUpload();
    setupClimateUploads();
    setupPresetToggle();

};
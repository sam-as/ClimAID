'''
Created Date: Wednesday, February 18th 2026, 3:52:33 pm
Author: Avik Sam

'''

import sys
from pathlib import Path
from climaid.utils import build_district_tree
from climaid.districts import get_available_districts, print_districts
from climaid.climaid_model import DiseaseModel
from climaid.climaid_projections import DiseaseProjection
from climaid.projection_plots import DiseaseVisualizer
from climaid.llm_client import LocalOllamaLLM


def _ask_yes_no(question: str, default: str = "y") -> bool:
    """Simple robust Yes/No prompt (pip-safe, cross-platform)."""

    prompt = f"{question} (Y/N) [default: {default.upper()}]: "
    ans = input(prompt).strip().lower()

    if ans == "":
        ans = default.lower()

    if ans in ("y", "yes"):
        return True
    elif ans in ("n", "no"):
        return False
    else:
        print("Invalid input detected. Exiting program.")
        sys.exit(1)


def run_interactive_pipeline():
    import difflib
    print("=" * 70)
    print("ClimAID Interactive Climate-Disease Terminal Interface")
    print("=" * 70)
    print("This terminal-based wizard is only meant for South Asian countries.")
    print("The browser interface allows both South Asian and global analysis.")
    print('')
    print("If you wish to use browser interface, exit the terminal by typing exit.")
    print("After exiting, you may use the below code in the terminal.")
    print("climaid browse")
    print("=" * 70)
    print("\nTip: You may enter a number OR type the name directly.\n")

    valid_districts = get_available_districts()
    tree = build_district_tree(valid_districts)

    # -------------------------------------------------
    # COUNTRY SELECTION
    # -------------------------------------------------

    countries = sorted(tree.keys())

    print("Available Countries")
    print("-" * 30)

    for i, c in enumerate(countries, 1):
        print(f"{i:2d}. {c}")

    country_input = input("\nSelect country: ").strip()

    if country_input.lower() in {"list", "show"}:
        print_districts()
        raise SystemExit(0)

    try:
        if country_input.isdigit():
            country = countries[int(country_input) - 1]
        else:
            matches = difflib.get_close_matches(
                country_input.upper(),
                countries,
                n=1,
                cutoff=0.4
            )
            if not matches:
                raise ValueError
            country = matches[0]

    except Exception:

        print("\nInvalid country selection - Restart")

        suggestions = difflib.get_close_matches(
            country_input.upper(),
            countries,
            n=5,
            cutoff=0.4
        )

        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"  - {s}")

        raise SystemExit(1)

    # -------------------------------------------------
    # STATE SELECTION
    # -------------------------------------------------

    states = sorted(tree[country].keys())

    print(f"\nStates in {country}")
    print("-" * 30)

    for i, s in enumerate(states, 1):
        print(f"{i:2d}. {s}")

    state_input = input("\nSelect state: ").strip()

    try:
        if state_input.isdigit():
            state = states[int(state_input) - 1]
        else:
            matches = difflib.get_close_matches(
                state_input.title(),
                states,
                n=1,
                cutoff=0.4
            )
            if not matches:
                raise ValueError
            state = matches[0]

    except Exception:

        print("\nInvalid state selection.")

        suggestions = difflib.get_close_matches(
            state_input.title(),
            states,
            n=5,
            cutoff=0.4
        )

        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"  - {s}")

        raise SystemExit(1)

    # -------------------------------------------------
    # DISTRICT SELECTION
    # -------------------------------------------------

    districts = sorted(tree[country][state])

    district_names = [d[0] for d in districts]

    print(f"\nDistricts in {state}")
    print("-" * 30)

    for i, (dname, _) in enumerate(districts, 1):
        print(f"{i:2d}. {dname}")

    district_input = input("\nSelect district: ").strip()

    try:
        if district_input.isdigit():
            district = districts[int(district_input) - 1][1]
        else:
            matches = difflib.get_close_matches(
                district_input.title(),
                district_names,
                n=1,
                cutoff=0.4
            )

            if not matches:
                raise ValueError

            idx = district_names.index(matches[0])
            district = districts[idx][1]

    except Exception:

        print("\nInvalid district selection.")

        suggestions = difflib.get_close_matches(
            district_input.title(),
            district_names,
            n=5,
            cutoff=0.4
        )

        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"  - {s}")

        raise SystemExit(1)

    print(f"\nSelected district: {district}")
    # Continue remaining inputs AFTER district is validated
    disease_name = input(
        "Enter disease name (e.g., Dengue, Malaria): "
    ).strip()

    disease_file = input(
        "Enter path to disease Excel/CSV file: "
    ).strip()

    disease_path = Path(disease_file)
    if not disease_path.exists():
        print(f"\nDisease file not found: {disease_file}")
        print("Please provide a valid file path without inverted commas.")
        raise SystemExit(1)

    print("\nInitializing DiseaseModel...")
    
    dm = DiseaseModel(
        district=district,
        disease_file=str(disease_path),
        random_state=42,
        disease_name=disease_name,
    )

    print("\nPreview of merged dataset:")
    print(dm.df_merged.head())

    # -------------------------------------------------
    # TRAIN / TEST SPLIT CONFIGURATION
    # -------------------------------------------------
    if _ask_yes_no("Configure train/test split?", default="y"):

        print('By default, 2020 has been removed from the data.')

        print("\nTrain/Test Split Options:")
        print("1. Default split")
        print("2. Custom test size")

        choice = input("Select option [1/2, default=1]: ").strip()

        if choice == "" or choice == "1":
            print("\nUsing default train/test split from DiseaseModel.")
            dm._train_test_split()

        elif choice == "2":
            test_year_input = input(
                "Enter the year from which you want the test data to be incorporated: "
            ).strip()
            test_year_input = int(test_year_input)
            dm._train_test_split(test_year=test_year_input)

        else:
            print("Invalid option. Using default split.")
            dm._train_test_split()

        print("Train/Test split completed.")

    else:
        print('Using default train/test split from DiseaseModel.')

    # -------------------------------------------------
    # Historical outbreak detection 
    # -------------------------------------------------
    if _ask_yes_no("Detect historical outbreak signals?", default="y"):
        hist_out = dm.detect_historical_outbreaks()
        print(f"Detected {len(hist_out)} potential historical outbreak periods.")
        print('--------------------------------------')
        print(hist_out)

    # -------------------------------------------------
    # LAG OPTIMIZATION 
    # -------------------------------------------------
    if _ask_yes_no("Run lag optimization (Optuna search) using AutoML?", default="y"):

        # Full supported model registry 
        SUPPORTED_MODELS = [
            'catboost', 'elasticnet', 'extra_trees', 'extratrees',
            'gbr', 'gradient_boosting', 'isotonic', 'lasso',
            'lgbm', 'lightgbm', 'linear', 'mlp', 'neural_net', 'nn',
            'poisson', 'random_forest', 'rf', 'ridge',
            'xgb', 'xgboost'
        ]

        # Clean display list 
        DISPLAY_MODELS = sorted(list(set([
            "rf", "xgb", "lgbm", "catboost",
            "gradient_boosting", "extra_trees",
            "linear", "ridge", "lasso", "elasticnet",
            "poisson", "mlp", "isotonic"
        ])))

        print("\nSupported Models (ClimAID Registry):")
        print(", ".join(DISPLAY_MODELS))
        print("\nWARNING! : For the correction model, we recommend isotonic, ridge, poisson, lasso and elasticnet")

        # -------------------------------------------------
        # PRESET SELECTION 
        # -------------------------------------------------
        print("\nSelect optimization preset:")
        print("1. Fast (50 trials, RF + XGB + isotonic) – Quick policy runs")
        print("2. Balanced (200 trials, XGB + RF + Isotonic) – Recommended")
        print("3. Deep (500 trials, Multi-model search) – Research grade")
        print("4. Custom (manual model selection)")

        preset = input("Enter choice [1/2/3/4, default=2]: ").strip()

        if preset == "" or preset == "2":
            # CURRENT SCIENTIFIC DEFAULT
            base_models = ("xgb",)
            residual_models = ("rf",)
            correction_models = ("isotonic",)
            n_trials = 200
            print("\nUsing BALANCED preset (recommended).")

        elif preset == "1":
            base_models = ("rf",)
            residual_models = ("xgb",)
            correction_models = ("isotonic",)
            n_trials = 50
            print("\nUsing FAST preset (policy mode).")

        elif preset == "3":
            base_models = ("rf", "xgb")
            residual_models = ("xgb", "rf", "extra_trees")
            correction_models = ("isotonic", "poisson", "elastic_net")
            n_trials = 500
            print("\nUsing DEEP preset (research-grade optimization).")

        elif preset == "4":
            # -------------------------
            # BASE MODELS (CUSTOM)
            # -------------------------
            base_input = input(
                "Enter base models (comma-separated) [default: rf]: "
            ).strip()

            base_models = (
                tuple(m.strip().lower() for m in base_input.split(","))
                if base_input else ("rf",)
            )

            # Validate base models
            base_models = tuple(m for m in base_models if m in SUPPORTED_MODELS)
            if not base_models:
                print("No valid base models entered. Falling back to ('rf',).")
                base_models = ("rf",)

            # -------------------------
            # RESIDUAL MODELS (CUSTOM)
            # -------------------------
            residual_input = input(
                "Enter residual models (comma-separated) [default: xgb]: "
            ).strip()

            residual_models = (
                tuple(m.strip().lower() for m in residual_input.split(","))
                if residual_input else ("xgb",)
            )

            residual_models = tuple(m for m in residual_models if m in SUPPORTED_MODELS)
            if not residual_models:
                print("No valid residual models entered. Falling back to ('xgb',).")
                residual_models = ("xgb",)

            # -------------------------
            # CORRECTION MODELS (CUSTOM)
            # -------------------------
            correction_input = input(
                "Enter correction models (comma-separated) [default: isotonic]: "
            ).strip()

            correction_models = (
                tuple(m.strip().lower() for m in correction_input.split(","))
                if correction_input else ("isotonic",)
            )

            correction_models = tuple(
                m for m in correction_models if m in SUPPORTED_MODELS
            )
            if not correction_models:
                print("No valid correction models entered. Using ('isotonic',).")
                correction_models = ("isotonic",)

            # -------------------------
            # TRIALS (CUSTOM)
            # -------------------------
            trials_input = input(
                "Enter number of Optuna trials [default: 200 | recommended: 100–500]: "
            ).strip()

            try:
                n_trials = int(trials_input) if trials_input else 200

                if n_trials <= 0:
                    raise ValueError("The number of trials should be greater than 0")

            except ValueError:
                print("Invalid trial input. Using default = 200.")
                n_trials = 200

            print("\nCustom Configuration Selected:")

        else:
            print("Invalid choice. Using default BALANCED preset.")
            base_models = ("rf",)
            residual_models = ("xgb",)
            correction_models = ("isotonic",)
            n_trials = 200

        # -------------------------------------------------
        # FINAL CONFIG SUMMARY (VERY IMPORTANT FOR LOGGING)
        # -------------------------------------------------
        print("\nLag Optimization Configuration:")
        print("Base Models:", base_models)
        print("Residual Models:", residual_models)
        print("Correction Models:", correction_models)
        print("Optuna Trials:", n_trials)

        print("\nRunning lag optimization (this may take time)...")

        feature_metadata, lag_search_result, best_config = dm.optimize_lags(
            base_models=base_models,
            residual_models=residual_models,
            correction_models=correction_models,
            debug=False,
            n_jobs=-1,
            n_trials=n_trials,
        )

        print("\nBest Lag Configuration Found:")
        print(best_config)


        # -------------------------
        # FINAL TRAINING
        # -------------------------
        if _ask_yes_no("Train final stacked model?", default="y"):
            print("\nTraining final model...")
            final_out = dm.train_final_model()
            print("Final Test R2:", final_out["test_r2"])
            print("Final Test RMSE:", final_out["test_rmse"])

        # -------------------------
        # HISTORICAL PLOTS
        # -------------------------

        from climaid.utils import use_gui_backend, use_headless_backend
    
        if _ask_yes_no("Plot historical predictions?", default="y"):
            use_gui_backend()
            import matplotlib.pyplot as plt
            dm.plot_historical_predictions()

        # -------------------------
        # PROJECTIONS
        # -------------------------
        if _ask_yes_no("Run CMIP6 climate projections?", default="y"):
            print("\nPreparing projections...")
            import matplotlib.pyplot as plt
            plt.close('all')
            use_headless_backend()
            dp = DiseaseProjection(dm)

            fclim = dm.df_climate_proj
            fclim_lagged = dp.prepare_features(fclim)

            dpro = dp.project_multi_model_ssp(
                model_list=['ACCESS-ESM1-5', 'CESM2', 'CNRM-CM6-1', 'GFDL-ESM4',
                            'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0',
                            'NorESM2-LM', 'UKESM1-0-LL'],
                ssp_list=["ssp126", "ssp245", "ssp370", "ssp585"],
            )

            projection_summary = dp.build_projection_summary(dpro)

            if _ask_yes_no("Flag projected outbreak risk using historical and dynamic baselines?", default="y"):
                dpro3 = dp.flag_outbreak_risk(
                    dpro,
                    method="both",
                    percentile=0.9
                )

                hist_risk = dpro3["risk_flag_historical_baseline"].sum()
                dyn_risk = dpro3["risk_flag_dynamic_baseline"].sum()

                print(f"\nRisk periods (historical baseline): {hist_risk}")
                print(f"Risk periods (dynamic scenario baseline): {dyn_risk}")

        else:
            print("Skipping projections...")
            projection_summary = None
            raise SystemExit(0)

        # -------------------------------------------------
        # VISUALIZATION (INTERACTIVE SCIENTIFIC PLOTS)
        # -------------------------------------------------
        if dpro is not None and _ask_yes_no("Generate projection visualizations?", "y"):
            print("\n[6/8] Creating scientific visualizations...")

            viz = DiseaseVisualizer(dpro)

            # Extract available options dynamically  
            available_gcms = sorted(dpro["GCM"].unique().tolist())
            available_ssps = sorted(dpro["SSP"].unique().tolist())

            print("\nAvailable GCM Models:", available_gcms)
            print("Available SSP Scenarios:", available_ssps)

            use_gui_backend()

            # -------------------------
            # HEATMAP QUESTION
            # -------------------------
            if _ask_yes_no("Generate heatmap visualization?", "y"):
                gcm_default = available_gcms[0]
                ssp_default = available_ssps[-1]  # usually highest SSP (e.g., ssp585)

                gcm_choice = input(
                    f"Select GCM for heatmap [default: {gcm_default}]: "
                ).strip()
                if gcm_choice == "":
                    gcm_choice = gcm_default

                if gcm_choice not in available_gcms:
                    print("Invalid GCM selected. Using default.")
                    gcm_choice = gcm_default

                ssp_choice = input(
                    f"Select SSP for heatmap [default: {ssp_default}]: "
                ).strip()
                if ssp_choice == "":
                    ssp_choice = ssp_default

                if ssp_choice not in available_ssps:
                    print("Invalid SSP selected. Using default.")
                    ssp_choice = ssp_default

                print(f"Generating heatmap for {gcm_choice} - {ssp_choice}...")
                viz.plot_heatmap(gcm_choice, ssp_choice)

            # -------------------------
            # GRID PLOT QUESTION
            # -------------------------
            if _ask_yes_no("Generate multi-model projection grid plot?", "y"):
                print("\nUsing all available GCMs and SSPs for grid visualization.")
                viz.plot_projection_grid(
                    available_gcms,
                    available_ssps,
                )

            # -------------------------
            # DISTRIBUTION PLOT 
            # -------------------------
            if _ask_yes_no("Generate distribution + KDE plot for projections?", "n"):
                try:
                    # Use first GCM + SSP as default scientific diagnostic
                    subset = dpro[
                        (dpro["SSP"] == available_ssps[0]) &
                        (dpro["GCM"] == available_gcms[0])
                    ]
                    viz.plot_distribution_with_kde("disease_projection")
                except Exception:
                    print("Distribution plot skipped (data format mismatch).")


    # -------------------------
    # LLM REPORT (OFFLINE)
    # -------------------------
    plt.close('all')
    use_headless_backend()
    print('\n')
    print('Now, generating report through LLM ..... ')
    print('Offline LLM client will take time depending on your system specifications.')
    print('However, ClimAID supports deterministic report generation without LLM client')
    
    if projection_summary:

        # -----------------------------------------
        # Ask if user wants to save a copy
        # -----------------------------------------
        save_copy = False
        output_dir = None

        if _ask_yes_no("Save a copy of the HTML report to a directory?", default="y"):
            print("\nEnter directory path where the report should be saved.")
            print("Do not use inverted commas around the path.")

            output_dir = input("Directory path: ").strip()
            save_copy = True

        # -----------------------------------------
        # Ask if user wants LLM interpretation
        # -----------------------------------------
        if _ask_yes_no("Add AI-assisted interpretation using local LLM (Ollama)?", default="y"):

            try:
                print("\nInitializing local LLM (Ollama)...")

                llm = LocalOllamaLLM(model="phi3")

                report = dm.generate_report(
                    projection_summary=projection_summary,
                    llm_client=llm,
                    style="policy_brief",
                    open_browser=True,
                    save_copy=save_copy,
                    output_dir=output_dir,
                )

                print("\nReport generated successfully.")

            except Exception:
                print(
                    "Switching to ClimAID Deterministic Scientific Interpreter (C-DSI) "
                    "as local LLM is unavailable."
                )

                report = dm.generate_report(
                    projection_summary=projection_summary,
                    llm_client=None,
                    style="_deterministic_engine",
                    open_browser=True,
                    save_copy=save_copy,
                    output_dir=output_dir,
                )

        else:

            print("\nGenerating report using ClimAID Deterministic Scientific Interpreter (C-DSI)...")

            report = dm.generate_report(
                projection_summary=projection_summary,
                llm_client=None,
                style="_deterministic_engine",
                open_browser=True,
                save_copy=save_copy,
                output_dir=output_dir,
            )

            print("\nReport generated successfully.")

    # -------------------------
    # RUNTIME SUMMARY
    # -------------------------
    if _ask_yes_no("Print runtime summary?", default="y"):
        print('Warning! The total pipeline shown will be inaccurate if the terminal is kept idle.')
        dm.print_runtime_summary()

    print("\nClimAID pipeline completed successfully.")
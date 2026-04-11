# ClimAID/climate_data.py
'''
In this code, we will use the downloaded weather data and the projections for all Indian Districts. 
Through this code, the user will also be able to get the weather data for any district. 
'''

import os
import pandas as pd
from climaid.utils import _map_columns

class ClimateData:
    """
    Climate data loader for ClimAID.

    Built-in climate data is available for South Asia.
    Other countries require user-supplied weather and projection data.
    """

    SUPPORTED_COUNTRIES = {
        "IND", "AFG", "BGD", "BTN", "LKA", "MMR", "NPL", "PAK"
    }

    def __init__(self, weather_file=None, projection_file=None):

        base_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(base_dir, "data")

        self.weather_file = weather_file
        self.projection_file = projection_file

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_country(self, district):
        try:
            return district.split("_")[0].upper()
        except Exception:
            raise ValueError(
                f"Invalid district format: {district}\n"
                "Expected format: COUNTRY_district_state"
            )
        
    def _standardize_dataframe(self, df, district, data_type="projection"):
        """
        Standardize input dataframe for ClimAID pipeline.
        Handles:
        - column renaming
        - time parsing
        - Dist_States construction
        """

        df = _map_columns(df)

        # -----------------------------
        # Standardize column names
        # -----------------------------
        df.columns = [c.strip() for c in df.columns]

        # ---- Standardize to 'ssp' ----
        if "scenario" in df.columns and "ssp" not in df.columns:
            df = df.rename(columns={"scenario": "ssp"})

        if "Scenario" in df.columns and "ssp" not in df.columns:
            df = df.rename(columns={"scenario": "ssp"})

        if "SSP" in df.columns and "ssp" not in df.columns:
            df = df.rename(columns={"SSP": "ssp"})

        if "Ssp" in df.columns and "ssp" not in df.columns:
            df = df.rename(columns={"Ssp": "ssp"})

        # ---- Model ----
        if "Model" in df.columns and "model" not in df.columns:
            df = df.rename({"Model": "model"}, axis = 1)

        # ---- Time ----
        for col in ["time", "date", "Date", "TIME"]:
            if col in df.columns:
                df = df.rename({col: "time"}, axis = 1)
                break

        # -----------------------------
        # Parse time safely
        # -----------------------------
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # -----------------------------
        # Construct Dist_States if missing
        # -----------------------------
        if "Dist_States" not in df.columns:

            if district:
                df["Dist_States"] = district
                print("Filled missing Dist_States using API district...")
            else:
                raise ValueError("Missing Dist_States and no district provided")
            
        try:
            df['Dist_States'] = df['Dist_States'].apply(lambda x : x.split('_')[0].upper() + '_' + x.split('_')[1].title() + '_' + x.split('_')[2].upper())
        except: 
            raise ValueError("Missing Dist_States and no district provided")

        # -----------------------------
        # Final required checks
        # -----------------------------
        required_cols = ["Dist_States", "time"]

        if data_type == "projection":
            required_cols += ["model", "ssp", "mean_temperature", "mean_Rain", "mean_SH"]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df
        
    def _read_data(self, path, district, data_type="projection"):

        ext = os.path.splitext(path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        elif ext == ".parquet":
            df = pd.read_parquet(path)

        else:
            raise ValueError("Unsupported format")

        # Standardize here
        df = self._standardize_dataframe(df, district, data_type=data_type)

        return df
    
    def _load_local(self, filename, district, data_type="climate"):

        path = os.path.join(self.data_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing local data file: {path}")

        return self._read_data(path, district, data_type=data_type)
    
    def _load_file(self, file_path, district, data_type="projection"):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return self._read_data(file_path, district, data_type=data_type)

    # --------------------------------------------------
    # Historical weather
    # --------------------------------------------------

    def get_historical(self, district):
        """
        Retrieve historical climate data for a given district.

        This method loads historical weather data either from:
            - Built-in ClimAID datasets (for supported countries), or
            - A user-provided dataset (CSV, Excel, or Parquet)

        It then filters the data for the specified district.

        Parameters
        ----------

        district : str
            Name of the district (must match 'Dist_States' column in dataset).

        Returns
        -------

        df : pandas.DataFrame :
            Historical climate data for the specified district.

        Raises
        ------

        ValueError :
            - If the country is not supported and no custom weather file is provided.
            - If no data is found for the given district.
            - If file format is unsupported.

        Notes
        -----

        - Built-in data currently includes South Asia weather datasets.
        - User datasets must contain:
            - 'Dist_States' column (district name)
            - 'time' column (date/time)
        - Time column is automatically parsed for supported formats.
        """

        country = self._get_country(district)

        # -----------------------------
        # Built-in climate data
        # -----------------------------
        if self.weather_file:
            df = self._load_file(self.weather_file, district, data_type = 'climate')
            print("Using USER weather file...")

        elif country in self.SUPPORTED_COUNTRIES:
            df = self._load_local("SouthAsia_weather_data.csv", district, data_type="climate")
            print("Using BUILT-IN weather data...")

        else:
            raise ValueError(
                    f"ClimAID does not include climate data for country '{country}'.\n\n"
                    "Please provide a weather dataset through the global mode.\n"
                    "ClimateData(weather_file='your_weather_data.csv')"
                )

        # -----------------------------
        # Filter district
        # -----------------------------
        df = df[df["Dist_States"] == district].copy().reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No data found for '{district}'.\n\n"
                "Please check spelling and capitalization.\n\n"
                "To see available districts:\n"
                "from climaid.districts import print_districts\n"
                "print_districts()"
            )

        return df

    # --------------------------------------------------
    # Climate projections
    # --------------------------------------------------

    def get_projection(self, district, model=None, ssp=None):
        """
        Retrieve climate projection data for a specific district.

        This method:
            - Automatically selects the appropriate dataset (India or South Asia)
            - Downloads and caches data if not already available
            - Filters projections based on district, model, and scenario

        Parameters
        ----------

        district : str
            District name (must match 'Dist_States' column in dataset).

        model : str, optional
            Climate model name (e.g., "MIROC6").
            If provided, filters dataset to that model.

        ssp : str, optional
            Emission scenario (e.g., "ssp245", "ssp585").
            Filters dataset based on scenario column.

        Returns
        -------

        df : pandas.DataFrame :
            Filtered projection data for the specified district.

        Raises
        ------
        
        ValueError:
            If no matching data is found for the given filters.

        Notes
        -----
        
        - Uses DatasetManager to fetch datasets lazily.
        - Data is cached in memory after first load for performance.
        - Supports both built-in datasets and user-provided files.
        """

        import pandas as pd
        from climaid.datasets.manager import DatasetManager

        country = self._get_country(district)

        manager = DatasetManager()

        # ==================================================
        # BUILT-IN DATASETS (REMOTE: HuggingFace / Zenodo)
        # ==================================================
        if self.projection_file:
            dataset_name = None

        elif country == "IND":
            dataset_name = "cmip6_india"

        elif country in self.SUPPORTED_COUNTRIES:
            dataset_name = "cmip6_south_asia"

        else:
            dataset_name = None

        # ==================================================
        # LOAD FROM DATASET MANAGER (REMOTE + CACHED)
        # ==================================================
        if dataset_name:

            file_path = manager.fetch(dataset_name)

            # CACHE (VERY IMPORTANT FOR PERFORMANCE)
            cache_name = f"_{dataset_name}_cache"

            if not hasattr(self, cache_name):
                setattr(self, cache_name, pd.read_parquet(file_path))

            df = getattr(self, cache_name)

            # -------------------------
            # FILTER
            # -------------------------
            df = df[df["Dist_States"] == district]

            if model:
                df = df[df["model"] == model]

            if ssp:
                df = df[df["scenario"] == ssp]

            if df.empty:
                raise ValueError(
                    f"No projection data found for district='{district}', "
                    f"model='{model}', ssp='{ssp}'."
                )

            return df.copy()

        # ==================================================
        # CUSTOM USER FILE (CSV / EXCEL / PARQUET)
        # ==================================================
        else:

            import os

            if not self.projection_file:
                raise ValueError(
                    f"ClimAID does not include projections for country '{country}'.\n\n"
                    "Please supply projection data:\n"
                    "ClimateData(projection_file='your_projection_data.csv/.xlsx/.parquet')"
                )

            file_path = self.projection_file

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Projection file not found: {file_path}")

            # -------------------------
            # Read
            # -------------------------
            df = self._load_file(file_path, district, data_type="projection")

            # -------------------------
            # FILTER
            # -------------------------
            df = df[df["Dist_States"] == district]

            if model:
                df = df[df["model"] == model]

            if ssp:
                if "scenario" in df.columns:
                    df = df[df["scenario"] == ssp]
                elif "ssp" in df.columns:
                    df = df[df["ssp"] == ssp]

            if df.empty:
                raise ValueError(
                    f"No projection data found for district='{district}', "
                    f"model='{model}', ssp='{ssp}'."
                )

            return df
        

    def load_sample_dataset(self, name):
        """
        Load bundled sample datasets included with climaid.

        These datasets are intended for demonstration and testing purposes,
        particularly for users exploring the Global Mode via the browser interface.

        The datasets do not represent real-world observations; they are synthetic
        or simplified samples designed to illustrate expected data structure,
        variable formats, and workflows within climaid. 

        Warning: 
            - The data is incomplete, so may not work with ClimAID. 
            - Users must use the dataset structure as a reference and then prepare their own. 
            - After verifying the content, they may upload the data through the ClimAID Browser Interface. 

        The datsets can also be viewed here: https://github.com/sam-as//climaid/docs/dataset_samples
        
        Parameters
        ----------

        name : str
            Name of the dataset to load. 
            
            - Available options:
                - "climate"     : Sample historical climate data
                - "projection"  : Sample climate projection data
                - "disease"     : Sample disease/incidence data

        Returns
        -------

        df : pandas.DataFrame :
            A DataFrame containing the requested sample dataset.

        Raises
        ------

        ValueError :
            If an invalid dataset name is provided.

        Notes
        -----

        These datasets are packaged with the library and accessed using
        `importlib.resources`, ensuring compatibility after installation.

            Examples
            --------

            >>> from climaid.climate_data import ClimateData
            >>> cl = ClimateData()
            >>> df = cl.load_sample_dataset("climate")
            >>> df.head()
        
        """
        from importlib import resources
        import pandas as pd

        files = {
            "climate": "sample_climate_data.csv",
            "projection": "sample_climateprojection_data.csv",
            "disease": "sample_disease_data.csv",
        }

        path = resources.files("climaid").joinpath(f"docs/dataset_samples/{files[name]}")
        return pd.read_csv(path)
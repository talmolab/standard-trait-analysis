import marimo

__generated_with = "0.11.17"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import re
    import argparse
    import sys

    from typing import Optional, List
    from pathlib import Path
    from omegaconf import OmegaConf, DictConfig

    # Ensure the project root is on sys.path
    project_root = Path(".").resolve()
    sys.path.append(project_root.as_posix())
    from pipeline.pipeline_logger import setup_step_logger

    return (
        List,
        Optional,
        Path,
        np,
        os,
        pd,
        plt,
        sns,
        re,
        argparse,
        sys,
        OmegaConf,
        DictConfig,
        setup_step_logger,
    )


@app.cell
def load_config(OmegaConf, Path, argparse, setup_step_logger, sys, re):
    try:
        # Parse CLI args from Marimo launch
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="config.yaml")
        args = parser.parse_args(sys.argv[1:])
        print(f"✅ Parsed args: {args}")
    except Exception as e:
        print(f"❌ Error parsing args: {e}")
        raise

    # Load the resolved config (DO NOT re-resolve interpolations)
    try:
        cfg = OmegaConf.load(args.config_path)
        print(f"✅ Loaded config from {args.config_path}")
    except Exception as e:
        print(f"❌ Failed to load config from {args.config_path}: {e}")
        raise

    LOGGING_LEVEL = cfg.logging.level.upper()

    # Infer step name from script filename safely
    try:
        STEP_NAME = Path(sys.argv[0]).stem
        print(f"✅ Inferred STEP_NAME: {STEP_NAME}")
    except Exception:
        STEP_NAME = "unknown_step"
        print(
            "⚠️ Could not infer STEP_NAME from sys.argv[0]; defaulting to 'unknown_step'"
        )

    run_root = Path(cfg.output_dir)
    log_dir = Path(cfg.logging.log_dir)

    logger = setup_step_logger(
        log_dir=log_dir,
        step_name=STEP_NAME,
        level=LOGGING_LEVEL,
    )

    output_dir = run_root / STEP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step '{STEP_NAME}' starting with output dir {output_dir}")

    # Get csvs from the previous step
    csvs_dir = run_root / "make_csvs"  # Hardcoded for now
    csvs = list(csvs_dir.glob("*.csv"))
    logger.info(f"Found {len(csvs)} CSV files in {csvs_dir}")
    logger.info(f"CSV files: {csvs}")
    if not csvs:
        raise ValueError(f"No CSV files found in {csvs_dir}")
    csvs_dict = {}
    for csv in csvs:
        # pipeline_runs\run_2025-04-04_09-48-31\make_csvs\traits_5DAG.csv
        filename = csv.name
        # Use regex to extract the number before 'DAG'
        match = re.search(r"(\d+)DAG", filename)
        if match:
            _age = int(match.group(1))
            print(_age)
            csvs_dict[_age] = csv
        else:
            print("No age found.")
    logger.info(f"CSV files: {csvs_dict}")

    # Step specific parameters
    COL_START = cfg.make_csvs_params.col_start
    GENO_COL_NAME = cfg.make_csvs_params.genotype_col_name
    NAN_THRESHOLD = cfg.make_csvs_params.thresholds.nan
    OUTLIER_THRESHOLD = cfg.make_csvs_params.thresholds.outlier
    ZERO_THRESHOLD = cfg.make_csvs_params.thresholds.zero

    return (
        LOGGING_LEVEL,
        STEP_NAME,
        COL_START,
        NAN_THRESHOLD,
        OUTLIER_THRESHOLD,
        ZERO_THRESHOLD,
        GENO_COL_NAME,
        args,
        cfg,
        log_dir,
        logger,
        output_dir,
        parser,
        run_root,
        csvs_dict,
    )


@app.cell
def define_utilities(np, pd, plt, sns, Optional, Path):
    def count_outliers_per_trait(col: pd.Series) -> int:
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((col < lower_bound) | (col > upper_bound)).sum()

    def eda_computation(col: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "Num_NaNs": col.isna().sum(),
                "Num_Zeroes": (col == 0).sum(),
                "Variance": col.var(),
            }
        )

    def get_eda_metrics(
        df: pd.DataFrame, col_start: int, output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        trait_columns = df.columns[col_start:]
        eda_results = df[trait_columns].apply(eda_computation).T
        eda_results["Num_Outliers"] = df[trait_columns].apply(count_outliers_per_trait)
        eda_results["Fraction_NaNs"] = eda_results["Num_NaNs"] / df.shape[0]
        eda_results["Fraction_Zeroes"] = eda_results["Num_Zeroes"] / df.shape[0]
        eda_results["Fraction_Outliers"] = eda_results["Num_Outliers"] / df.shape[0]
        eda_results.reset_index(inplace=True)
        eda_results.rename(columns={"index": "Trait"}, inplace=True)
        if output_csv_path:
            eda_results.to_csv(output_csv_path, index=False)
        return eda_results

    def process_csv(
        csv_path: Path,
        col_start: int,
        geno_col_name: str,
        output_dir: Path,
        thresholds: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv_path)
        traits = df.columns[col_start:]
        df = df[["plant_qr_code", geno_col_name] + traits.tolist()]

        Y = df.iloc[:, col_start:].to_numpy()
        df = df.iloc[~np.isnan(Y).all(axis=1)]

        eda_path = output_dir / f"{csv_path.stem}_eda.csv"
        eda = get_eda_metrics(df, col_start, str(eda_path))

        remove_traits = eda[
            (eda["Fraction_NaNs"] >= thresholds["nan"])
            | (eda["Fraction_Zeroes"] >= thresholds["zero"])
            | (eda["Fraction_Outliers"] >= thresholds["outlier"])
        ]["Trait"].tolist()
        df_filtered = df.drop(columns=remove_traits)

        Y_clean = df_filtered.iloc[:, col_start:].to_numpy()
        df_no_nans = df_filtered.iloc[~np.isnan(Y_clean).any(axis=1)]

        eda_clean_path = output_dir / f"{csv_path.stem}_eda_clean.csv"
        eda_clean = get_eda_metrics(df_no_nans, col_start, str(eda_clean_path))

        final_csv = output_dir / f"{csv_path.stem}_cleaned.csv"
        df_no_nans.to_csv(final_csv, index=False)

        return df_no_nans, eda_clean

    def plot_eda_summary(eda_df: pd.DataFrame, output_dir: Path, label: str):
        _res = eda_df.copy()
        _res["Prefix"] = _res["Trait"].apply(
            lambda x: x.split("_")[0] if "_" in x else "NoPrefix"
        )

        _f, _axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
        sns.barplot(x="Trait", y="Fraction_NaNs", hue="Prefix", data=_res, ax=_axes[0])
        _axes[0].set_title(f"{label}: Fraction of NaNs")
        _axes[0].tick_params(labelbottom=False)

        sns.barplot(
            x="Trait", y="Fraction_Zeroes", hue="Prefix", data=_res, ax=_axes[1]
        )
        _axes[1].set_title(f"{label}: Fraction of Zeroes")
        _axes[1].tick_params(labelbottom=False)

        sns.barplot(
            x="Trait", y="Fraction_Outliers", hue="Prefix", data=_res, ax=_axes[2]
        )
        _axes[2].set_title(f"{label}: Fraction of Outliers")
        _axes[2].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        out_path = output_dir / f"{label}_eda_trait_overview.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor="white")
        print(f"Saved plot to {out_path}")
        plt.close()

    return get_eda_metrics, process_csv, plot_eda_summary


@app.cell
def main(
    csvs_dict,
    COL_START,
    NAN_THRESHOLD,
    OUTLIER_THRESHOLD,
    ZERO_THRESHOLD,
    GENO_COL_NAME,
    output_dir,
    get_eda_metrics,
    process_csv,
    plot_eda_summary,
):
    results_by_age = {}
    thresholds = {
        "nan": NAN_THRESHOLD,
        "zero": ZERO_THRESHOLD,
        "outlier": OUTLIER_THRESHOLD,
    }

    for age, csv_path in csvs_dict.items():
        print(f"\nProcessing {age}DAG -> {csv_path.name}")
        age_dir = output_dir / f"{age}DAG"
        age_dir.mkdir(parents=True, exist_ok=True)

        df_cleaned, eda_clean = process_csv(
            csv_path=csv_path,
            col_start=COL_START,
            geno_col_name=GENO_COL_NAME,
            output_dir=age_dir,
            thresholds=thresholds,
        )

        plot_eda_summary(eda_clean, age_dir, f"{age}DAG")
        results_by_age[age] = {"df": df_cleaned, "eda": eda_clean}

    return results_by_age


# @app.cell
# def _(Optional, pd):
#     def filter_rows_by_values(df: pd.DataFrame, col: str, values: list):
#         """Filters a DataFrame to remove rows where a specified column's value is in a given list.

#         Args:
#             df (pd.DataFrame): The DataFrame to filter.
#             col (str): The name of the column to consider for filtering.
#             values (list): The list of values that should be filtered out from the DataFrame.

#         Returns:
#             pd.DataFrame: A DataFrame with the rows removed where the specified column's value is in the given list.
#         """
#         # Use the `isin` method to identify rows where column value is in the list 'values'
#         # The `~` operator negates the boolean series, filtering out those rows
#         return df[~df[col].isin(values)]

#     def count_outliers_per_trait(col: pd.Series) -> pd.Series:
#         """Count the number of outliers in a Pandas Series using the IQR method.

#         Args:
#             col (pd.Series): The pandas Series (column) for which to count the number of outliers using the IQR method.

#         Returns:
#             The count (scalar) of number of outliers for the input column.
#         """
#         Q1 = col.quantile(0.25)
#         Q3 = col.quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         outliers = col[(col < lower_bound) | (col > upper_bound)]
#         return len(outliers)

#     def eda_computation(col: pd.Series) -> pd.Series:
#         """Compute the number of NaNs, zeroes, and variance for a given pandas Series (column).

#         Args:
#             col (pd.Series): The pandas Series (column) for which to compute exploratory data analysis (EDA) metrics.

#         Returns:
#             pd.Series: A Series containing the computed EDA metrics ('Num_NaNs', 'Num_Zeroes', 'Variance').
#         """
#         num_nans = col.isna().sum()
#         num_zeroes = (col == 0).sum()
#         variance = col.var()
#         return pd.Series(
#             {"Num_NaNs": num_nans, "Num_Zeroes": num_zeroes, "Variance": variance}
#         )

#     def get_eda_metrics(
#         input_dataframe: pd.DataFrame,
#         col_start: int,
#         output_csv_path: Optional[str] = None,
#         write_csv: bool = True,
#     ) -> pd.DataFrame:
#         """Perform exploratory data analysis (EDA) on a dataframe with root trait data to assess the number of NaNs,
#         zeroes, outliers, and variance for each root trait.

#         Args:
#             input_dataframe (pd.Dataframe): Pandas dataframe object with root traits.
#             col_start (int): The index of the column where root traits start relative to beginning of dataframe.
#             output_csv_path (Optional[str]): The path to save the EDA results as a CSV file. If None, results will not be saved.
#             write_csv (bool): Flag indicating whether to save the EDA results to a CSV file. Default is True.

#         Returns:
#             pd.DataFrame: A DataFrame summarizing the EDA results.
#         """
#         df = input_dataframe.copy()
#         trait_columns = df.columns[col_start:]
#         eda_results = df[trait_columns].apply(eda_computation)
#         eda_results = eda_results.T
#         outlier_counts = df[trait_columns].apply(count_outliers_per_trait)
#         eda_results["Num_Outliers"] = outlier_counts
#         eda_results["Fraction_NaNs"] = eda_results["Num_NaNs"] / df.shape[0]
#         eda_results["Fraction_Zeroes"] = eda_results["Num_Zeroes"] / df.shape[0]
#         eda_results["Fraction_Outliers"] = eda_results["Num_Outliers"] / df.shape[0]
#         eda_results.reset_index(inplace=True)
#         eda_results.rename(columns={"index": "Trait"}, inplace=True)
#         if write_csv and output_csv_path:
#             eda_results.to_csv(output_csv_path, index=False)
#         return eda_results

#     return (
#         count_outliers_per_trait,
#         eda_computation,
#         filter_rows_by_values,
#         get_eda_metrics,
#     )


# @app.cell
# def _(SUB_DIR, TOP_DIR, os):
#     # Create the top-level directory
#     top_dir = TOP_DIR
#     os.makedirs(top_dir, exist_ok=True)

#     # Create a subdirectory inside the top-level directory
#     sub_dir = os.path.join(top_dir, SUB_DIR)
#     os.makedirs(sub_dir, exist_ok=True)

#     sub_dir
#     return sub_dir, top_dir


# @app.cell
# def _(INPUT_CSV_PATH, pd):
#     # Data from experiment (Lines, Barcodes etc.) and root traits
#     master_data = pd.read_csv(INPUT_CSV_PATH)
#     master_data
#     return (master_data,)


# @app.cell
# def _(COL_START, master_data):
#     # Columns from sleap-roots. Check the column names and adjust COL_START accordingly!
#     pipeline_columns = master_data.columns[COL_START:]
#     pipeline_columns
#     return (pipeline_columns,)


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""7 Day-Old Data Cleanup""")
#     return


# @app.cell
# def _(
#     ROOT_BIOMASS_COL_NAME,
#     SHOOT_BIOMASS_COL_NAME,
#     master_data,
#     pipeline_columns,
# ):
#     # List of columns to keep
#     columns = [
#         "plant_qr_code",
#         "Line",
#         ROOT_BIOMASS_COL_NAME,
#         SHOOT_BIOMASS_COL_NAME,
#     ] + pipeline_columns.tolist()

#     # Filter the merged data to keep only the columns of interest
#     new_df = master_data[columns]
#     new_df
#     return columns, new_df


# @app.cell
# def _(new_df):
#     _col_start = 4
#     # Check the new DataFrame starting at the root traits
#     new_df.iloc[:, _col_start:]
#     return


# @app.cell
# def _(new_df):
#     # Make a numpy array of the root traits
#     _col_start = 4
#     Y = new_df.iloc[:, _col_start:].to_numpy()
#     Y.shape
#     return (Y,)


# @app.cell
# def _(Y, new_df, np):
#     # Find rows with only NaNs
#     is_missing_all = np.isnan(Y).all(axis=1)
#     missing_barcodes_all = new_df["plant_qr_code"].iloc[is_missing_all]
#     miss_all_rows = new_df.iloc[is_missing_all]
#     print(f"{len(missing_barcodes_all)}/{len(new_df)} missing all features")
#     print(missing_barcodes_all)
#     return is_missing_all, miss_all_rows, missing_barcodes_all


# @app.cell
# def _(miss_all_rows, sub_dir):
#     # Save this info to a csv
#     miss_all_rows.to_csv(f"{sub_dir}/df_miss_all_rows.csv", index=False)
#     print(f"Saved to {sub_dir}/df_miss_all_rows.csv")
#     miss_all_rows
#     return


# @app.cell
# def _(is_missing_all, new_df, sub_dir):
#     df = new_df.iloc[~is_missing_all]
#     df.to_csv(f"{sub_dir}/df_all_nan_rows_removed.csv", index=False)
#     print(f"Saved to {sub_dir}/df_all_nan_rows_removed.csv")
#     df.shape
#     return (df,)


# @app.cell
# def _(df, get_eda_metrics, sub_dir):
#     _output_csv = f"{sub_dir}/eda.csv"
#     eda_results = get_eda_metrics(df, 4, _output_csv)
#     eda_results
#     return (eda_results,)


# @app.cell
# def _(eda_results, plt, sns, sub_dir):
#     _res = eda_results.copy()
#     _res["Prefix"] = _res["Trait"].apply(
#         lambda x: x.split("_")[0] if "_" in x else "NoPrefix"
#     )
#     _f, _axes = plt.subplots(3, 1, figsize=(50, 20), sharex=True)
#     sns.barplot(x="Trait", y="Fraction_NaNs", hue="Prefix", data=_res, ax=_axes[0])
#     _axes[0].set_title("Fraction of NaNs for Each Trait")
#     _axes[0].set_xticklabels([])
#     sns.barplot(x="Trait", y="Fraction_Zeroes", hue="Prefix", data=_res, ax=_axes[1])
#     _axes[1].set_title("Fraction of Zeroes for Each Trait")
#     _axes[1].set_xticklabels([])
#     sns.barplot(x="Trait", y="Fraction_Outliers", hue="Prefix", data=_res, ax=_axes[2])
#     _axes[2].set_title("Fraction of Outliers for Each Trait")
#     _axes[2].set_xticklabels([])
#     plt.tight_layout()
#     plt.savefig(
#         f"{sub_dir}/eda_trait_overview.png", bbox_inches="tight", facecolor="white"
#     )
#     plt.show()
#     print(f"Saved to {sub_dir}/eda_trait_overview.png")
#     return


# @app.cell
# def _(eda_results, plt, sns, sub_dir):
#     _f, _axes = plt.subplots(3, 1, figsize=(14, 20))
#     sns.histplot(eda_results["Fraction_NaNs"], bins=100, ax=_axes[0], kde=True)
#     _axes[0].set_title("Distribution of Number of NaNs")
#     _axes[0].set_xlabel("Number of NaNs per trait / total number of samples")
#     _axes[0].set_ylabel("Frequency")
#     sns.histplot(eda_results["Fraction_Zeroes"], bins=100, ax=_axes[1], kde=True)
#     _axes[1].set_title("Distribution of Number of Zeroes")
#     _axes[1].set_xlabel("Number of Zeroes per trait / total number of samples")
#     _axes[1].set_ylabel("Frequency")
#     sns.histplot(eda_results["Fraction_Outliers"], bins=100, ax=_axes[2], kde=True)
#     _axes[2].set_title("Distribution of Number of Outliers")
#     _axes[2].set_xlabel("Number of Outliers per trait / total number of samples")
#     _axes[2].set_ylabel("Frequency")
#     plt.savefig(f"{sub_dir}/histograms_eda.png", bbox_inches="tight", facecolor="white")
#     plt.tight_layout()
#     plt.show()
#     print(f"Saved to {sub_dir}/histograms_eda.png")
#     return


# @app.cell
# def _(eda_results):
#     # Removal of traits based on above analysis
#     # Set thresholds to remove traits based on number of NaNs, number of zeroes, or number of outliers

#     # Initialize list of traits to remove from analysis
#     remove_traits = []

#     # Look at traits where number of NaNs / total number of plants is greater than `nan_threshold`
#     nan_threshold = 0.10
#     traits_top_nan = eda_results[eda_results["Fraction_NaNs"] >= nan_threshold]["Trait"]
#     print(
#         f"Number of traits with Fraction of NaNs greater than or equal to {nan_threshold} is {traits_top_nan.shape[0]}"
#     )
#     remove_traits.extend(traits_top_nan.tolist())

#     # Look at traits where number of zeroes / total number of plants is greater than `zero_threshold`
#     zero_threshold = 0.40
#     traits_top_zero = eda_results[eda_results["Fraction_Zeroes"] >= zero_threshold][
#         "Trait"
#     ]
#     print(
#         f"Number of traits with Fraction of Zeroes greater than or equal to {zero_threshold} is {traits_top_zero.shape[0]}"
#     )
#     remove_traits.extend(traits_top_zero.tolist())

#     # Look at traits where Fraction of Outliers is greater than `outlier_threshold`
#     outlier_threshold = 0.60
#     traits_top_outlier = eda_results[
#         eda_results["Fraction_Outliers"] >= outlier_threshold
#     ]["Trait"]
#     print(
#         f"Number of traits with Fraction of Outliers greater than or equal to {outlier_threshold} is {traits_top_outlier.shape[0]}"
#     )
#     remove_traits.extend(traits_top_outlier.tolist())
#     remove_traits
#     return (
#         nan_threshold,
#         outlier_threshold,
#         remove_traits,
#         traits_top_nan,
#         traits_top_outlier,
#         traits_top_zero,
#         zero_threshold,
#     )


# @app.cell
# def _(pd, remove_traits, sub_dir):
#     # Create a DataFrame from the list
#     df_remove_traits = pd.DataFrame(remove_traits, columns=["Traits_Removed"])

#     # Save the DataFrame of removed traits to a CSV file
#     csv_path = f"{sub_dir}/removed_traits.csv"
#     df_remove_traits.to_csv(csv_path, index=False)
#     print(f"Saved to {csv_path}")
#     df_remove_traits
#     return csv_path, df_remove_traits


# @app.cell
# def _(df, remove_traits):
#     # Create a new DataFrame without the unwanted traits
#     df_filtered = df.drop(columns=remove_traits)
#     df_filtered
#     return (df_filtered,)


# @app.cell
# def _(df_filtered, np):
#     _col_start = 4
#     Y_1 = df_filtered.iloc[:, _col_start:].to_numpy()
#     is_missing_any = np.isnan(Y_1).any(axis=1)
#     _missing_barcodes_any = df_filtered["plant_qr_code"].iloc[is_missing_any]
#     miss_any_rows = df_filtered.iloc[is_missing_any]
#     print(f"{len(_missing_barcodes_any)}/{len(df_filtered)} missing any features")
#     print(_missing_barcodes_any)
#     return Y_1, is_missing_any, miss_any_rows


# @app.cell
# def _(miss_any_rows, sub_dir):
#     # Save this info to a csv
#     miss_any_rows.to_csv(f"{sub_dir}/df_filtered_miss_any_rows.csv", index=False)
#     return


# @app.cell
# def _(OUTPUT_CSV, df_filtered, is_missing_any, sub_dir):
#     # Save Dataframe without any NANs to a csv
#     df_no_nans = df_filtered.iloc[~is_missing_any]
#     df_no_nans.to_csv(f"{sub_dir}/{OUTPUT_CSV}", index=False)
#     df_no_nans.shape
#     return (df_no_nans,)


# @app.cell
# def _(df_no_nans, np):
#     _col_start = 4
#     Z = df_no_nans.iloc[:, _col_start:].to_numpy()
#     is_missing_any_1 = np.isnan(Z).any(axis=1)
#     _missing_barcodes_any = df_no_nans["plant_qr_code"].iloc[is_missing_any_1]
#     print(f"{len(_missing_barcodes_any)}/{len(df_no_nans)} missing any features")
#     print(_missing_barcodes_any)
#     return Z, is_missing_any_1


# @app.cell
# def _(Z, np):
#     # Make sure there are no NaNs!
#     assert ~(np.isnan(Z).any())
#     return


# @app.cell
# def _(Z):
#     Z.shape
#     return


# @app.cell
# def _(df_no_nans, get_eda_metrics, sub_dir):
#     _col_start = 4
#     _output_csv = f"{sub_dir}/eda_clean.csv"
#     eda_results_1 = get_eda_metrics(df_no_nans, _col_start, _output_csv)
#     print(f"Saved to {_output_csv}")
#     eda_results_1
#     return (eda_results_1,)


# @app.cell
# def _(eda_results_1, plt, sns, sub_dir):
#     _res = eda_results_1.copy()
#     _res["Prefix"] = _res["Trait"].apply(
#         lambda x: x.split("_")[0] if "_" in x else "NoPrefix"
#     )
#     _f, _axes = plt.subplots(3, 1, figsize=(50, 20), sharex=True)
#     sns.barplot(x="Trait", y="Fraction_NaNs", hue="Prefix", data=_res, ax=_axes[0])
#     _axes[0].set_title("Fraction of NaNs for Each Trait")
#     _axes[0].set_xticklabels([])
#     sns.barplot(x="Trait", y="Fraction_Zeroes", hue="Prefix", data=_res, ax=_axes[1])
#     _axes[1].set_title("Fraction of Zeroes for Each Trait")
#     _axes[1].set_xticklabels([])
#     sns.barplot(x="Trait", y="Fraction_Outliers", hue="Prefix", data=_res, ax=_axes[2])
#     _axes[2].set_title("Fraction of Outliers for Each Trait")
#     _axes[2].set_xticklabels([])
#     plt.tight_layout()
#     plt.savefig(
#         f"{sub_dir}/eda_trait_overview_final.png",
#         bbox_inches="tight",
#         facecolor="white",
#     )
#     plt.show()
#     print(f"Saved to {sub_dir}/eda_trait_overview_final.png")
#     return


# @app.cell
# def _(eda_results_1, plt, sns, sub_dir):
#     _f, _axes = plt.subplots(3, 1, figsize=(14, 20))
#     sns.histplot(eda_results_1["Fraction_NaNs"], bins=100, ax=_axes[0], kde=True)
#     _axes[0].set_title("Distribution of Number of NaNs")
#     _axes[0].set_xlabel("Number of NaNs per trait / total number of samples")
#     _axes[0].set_ylabel("Frequency")
#     sns.histplot(eda_results_1["Fraction_Zeroes"], bins=100, ax=_axes[1], kde=True)
#     _axes[1].set_title("Distribution of Number of Zeroes")
#     _axes[1].set_xlabel("Number of Zeroes per trait / total number of samples")
#     _axes[1].set_ylabel("Frequency")
#     sns.histplot(eda_results_1["Fraction_Outliers"], bins=100, ax=_axes[2], kde=True)
#     _axes[2].set_title("Distribution of Number of Outliers")
#     _axes[2].set_xlabel("Number of Outliers per trait / total number of samples")
#     _axes[2].set_ylabel("Frequency")
#     plt.savefig(
#         f"{sub_dir}/histograms_eda_final.png", bbox_inches="tight", facecolor="white"
#     )
#     plt.tight_layout()
#     plt.show()
#     print(f"Saved to {sub_dir}/histograms_eda_final.png")
#     return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

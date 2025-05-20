import marimo

__generated_with = "0.12.5"
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
        DictConfig,
        List,
        OmegaConf,
        Optional,
        Path,
        argparse,
        np,
        os,
        pd,
        plt,
        project_root,
        re,
        setup_step_logger,
        sns,
        sys,
    )


@app.cell
def load_config(OmegaConf, Path, argparse, re, setup_step_logger, sys):
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
    COL_START = cfg.parameters.col_start
    GENO_COL_NAME = cfg.parameters.genotype_col_name
    NAN_THRESHOLD = cfg.parameters.thresholds.nan
    OUTLIER_THRESHOLD = cfg.parameters.thresholds.outlier
    ZERO_THRESHOLD = cfg.parameters.thresholds.zero
    return (
        COL_START,
        GENO_COL_NAME,
        LOGGING_LEVEL,
        NAN_THRESHOLD,
        OUTLIER_THRESHOLD,
        STEP_NAME,
        ZERO_THRESHOLD,
        args,
        cfg,
        csv,
        csvs,
        csvs_dict,
        csvs_dir,
        filename,
        log_dir,
        logger,
        match,
        output_dir,
        parser,
        run_root,
    )


@app.cell
def define_utilities(Optional, Path, np, pd, plt, sns):
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
        print(f"Traits start with {traits[:5]}")

        # Attempt to coerce all trait columns to numeric
        trait_df = df.iloc[:, col_start:].apply(pd.to_numeric, errors="coerce")
        df.iloc[:, col_start:] = trait_df  # Replace with coerced values

        # Raise an error if entire trait columns are still non-numeric
        non_numeric_cols = trait_df.columns[trait_df.isna().all()]
        if not non_numeric_cols.empty:
            raise ValueError(
                f"The following columns could not be converted to numeric values and contain only NaNs: {list(non_numeric_cols)}"
            )

        # Detect rows missing all trait values
        Y = trait_df.to_numpy()
        is_missing_all = np.isnan(Y).all(axis=1)
        missing_barcodes_all = df["plant_qr_code"].iloc[is_missing_all]
        miss_all_rows = df.iloc[is_missing_all]
        print(f"{len(missing_barcodes_all)}/{len(df)} rows missing all features")
        print(f"Missing barcodes: {missing_barcodes_all.tolist()}")

        # Save diagnostics
        output_dir.mkdir(parents=True, exist_ok=True)
        miss_all_rows.to_csv(output_dir / "df_miss_all_rows.csv", index=False)

        # Run EDA and filtering
        eda_path = output_dir / f"{csv_path.stem}_eda.csv"
        eda = get_eda_metrics(df, col_start, eda_path.as_posix())

        remove_traits = eda[
            (eda["Fraction_NaNs"] >= thresholds["nan"])
            | (eda["Fraction_Zeroes"] >= thresholds["zero"])
            | (eda["Fraction_Outliers"] >= thresholds["outlier"])
        ]["Trait"].tolist()
        print(f"Removing traits: {remove_traits}")
        # Save removed traits to a CSV
        removed_traits_path = output_dir / "removed_traits.csv"
        removed_traits_df = trait_df[remove_traits]
        removed_traits_df.to_csv(removed_traits_path, index=False)
        print(f"Saved removed traits to {removed_traits_path}")
        df_filtered = df.drop(columns=remove_traits)
        print(f"Removed {len(remove_traits)} traits")

        # Remove rows with NaNs in the remaining traits
        Y_clean = df_filtered.iloc[:, col_start:].to_numpy()
        is_missing_any = np.isnan(Y_clean).any(axis=1)
        missing_barcodes_any = df_filtered["plant_qr_code"].iloc[is_missing_any]
        miss_any_rows = df_filtered.iloc[is_missing_any]
        print(
            f"{len(missing_barcodes_any)}/{len(df_filtered)} rows missing any features"
        )
        print(f"Missing barcodes: {missing_barcodes_any.tolist()}")
        # Save diagnostics
        miss_any_rows.to_csv(output_dir / "df_miss_any_rows.csv", index=False)
        df_no_nans = df_filtered.iloc[~np.isnan(Y_clean).any(axis=1)]
        print(f"Removed {len(miss_any_rows)} rows with NaNs")
        print(f"Remaining rows: {len(df_no_nans)}")

        eda_clean_path = output_dir / f"{csv_path.stem}_eda_clean.csv"
        eda_clean = get_eda_metrics(df_no_nans, col_start, eda_clean_path.as_posix())

        final_csv = output_dir / f"{csv_path.stem}_cleaned.csv"
        df_no_nans.to_csv(final_csv, index=False)
        print(f"Saved cleaned CSV to {final_csv}")
        print(f"EDA results saved to {eda_path}")

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

    return plot_eda_summary, process_csv


@app.cell
def _(COL_START, csvs_dict, pd):
    dag_df = pd.read_csv(csvs_dict[11].as_posix())

    array = dag_df.iloc[:, COL_START:].to_numpy()
    # np.isnan(array).all(axis=1)
    array
    return


@app.cell
def main(
    COL_START,
    GENO_COL_NAME,
    NAN_THRESHOLD,
    OUTLIER_THRESHOLD,
    ZERO_THRESHOLD,
    csvs_dict,
    mo,
    output_dir,
    plot_eda_summary,
    process_csv,
):
    results_by_age = {}
    tables = []
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

        # Append a heading and the table itself to the display list
        tables.append(mo.ui.text(f"### {age} DAG"))
        tables.append(mo.ui.table(df_cleaned, max_rows=5))
        plot_eda_summary(eda_clean, age_dir, f"{age}DAG")
        results_by_age[age] = {"df": df_cleaned, "eda": eda_clean}
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

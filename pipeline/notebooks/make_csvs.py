import marimo

__generated_with = "0.12.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import datetime
    import openpyxl
    import logging
    import argparse
    import sys
    import marimo as mo

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
        datetime,
        logging,
        mo,
        np,
        openpyxl,
        os,
        pd,
        plt,
        project_root,
        setup_step_logger,
        sns,
        sys,
    )


@app.cell
def load_config(OmegaConf, Path, argparse, setup_step_logger, sys):
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

    # Get the remaining config parameters for this step
    TRAITS_CSV_PATH = Path(cfg.input.traits_csv)
    EXPERIMENT_EXCEL_PATH = Path(cfg.input.experimental_design_excel)
    SHEET_NAME = cfg.input.experimental_design_sheet
    logger.info(f"TRAITS_CSV_PATH: {TRAITS_CSV_PATH}")
    logger.info(f"EXPERIMENT_EXCEL_PATH: {EXPERIMENT_EXCEL_PATH}")
    logger.info(f"SHEET_NAME: {SHEET_NAME}")
    return (
        EXPERIMENT_EXCEL_PATH,
        LOGGING_LEVEL,
        SHEET_NAME,
        STEP_NAME,
        TRAITS_CSV_PATH,
        args,
        cfg,
        log_dir,
        logger,
        output_dir,
        parser,
        run_root,
    )


@app.cell
def read_master(EXPERIMENT_EXCEL_PATH, SHEET_NAME, mo, pd):
    # Data from experiment (Lines, Barcodes etc.)
    master_data_df = pd.read_excel(
        EXPERIMENT_EXCEL_PATH,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
    )
    print(
        f"📊 Read master data from {EXPERIMENT_EXCEL_PATH} (sheet: {SHEET_NAME}) with shape: {master_data_df.shape}"
    )
    # Remove leading and trailing whitespace from column names
    master_data_df.columns = master_data_df.columns.str.strip()
    print(f"🔍 Columns in Excel sheet: {master_data_df.columns.tolist()}")
    mo.ui.table(master_data_df)
    return (master_data_df,)


@app.cell
def read_traits(TRAITS_CSV_PATH, mo, pd):
    traits_df = pd.read_csv(TRAITS_CSV_PATH)
    print(f"📊 Read traits data from {TRAITS_CSV_PATH} with shape: {traits_df.shape}")
    mo.ui.table(traits_df)
    return (traits_df,)


@app.cell
def merge_dfs(logger, master_data_df, pd, traits_df):
    # `master_data_df` needs "Barcode" column renamed "plant_qr_code" to match `traits_df`
    master_data_df.rename(columns={"Barcode": "plant_qr_code"}, inplace=True)
    # Merge the two dataframes
    merged_data = pd.merge(master_data_df, traits_df, on="plant_qr_code", how="left")
    print(f"📊 Merged data shape: {merged_data.shape}")
    logger.info(f"Merged data shape: {merged_data.shape}")

    # Remove rows with missing values in the "plant_age_days" column
    merged_data = merged_data[merged_data["plant_age_days"].notna()]
    logger.info(f"Shape after dropping missing values: {merged_data.shape}")

    # Make "plant_age_days" column numeric
    merged_data["plant_age_days"] = pd.to_numeric(
        merged_data["plant_age_days"], errors="coerce"
    )
    # Get unique values in the "plant_age_days" column
    unique_plant_ages = merged_data["plant_age_days"].unique()
    print(f"Unique plant ages: {unique_plant_ages}")
    return merged_data, unique_plant_ages


@app.cell
def make_csvs_by_age(Path, logger, merged_data, output_dir, unique_plant_ages):
    for age in unique_plant_ages:
        # Filter the merged DataFrame for the current age
        filtered_data = merged_data[merged_data["plant_age_days"] == age]

        # Save the filtered DataFrame to a CSV file
        filtered_data.to_csv(
            Path(output_dir) / f"traits_{int(age)}DAG.csv",
            index=False,
        )
        logger.info(
            f"Saved {age} Day-Old Plants data to CSV file: {Path(output_dir) / f'traits_{int(age)}DAG.csv'} with shape: {filtered_data.shape}"
        )
    return age, filtered_data


if __name__ == "__main__":
    app.run()

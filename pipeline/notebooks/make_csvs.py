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
    import datetime
    import openpyxl
    import logging
    import OmegaConf

    from typing import Optional, List
    from pathlib import Path
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
        datetime,
        openpyxl,
        logging,
        OmegaConf,
        setup_step_logger,
    )


@app.cell
def _():
    # Define step name
    STEP_NAME = "make_csvs"
    return STEP_NAME


@app.cell
def _(OmegaConf, Path, setup_step_logger, STEP_NAME):
    # Load and resolve the config
    cfg = OmegaConf.load("config.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Pull logging level from config
    LOGGING_LEVEL = cfg.logging.level.upper()  # ensures e.g., "info" becomes "INFO"

    run_root = Path(cfg.output_dir)
    log_dir = run_root / "logs"

    logger = setup_step_logger(
        log_dir=log_dir,
        step_name=STEP_NAME,
        level=LOGGING_LEVEL,
    )
    logger.info(f"Logger initialized for step '{STEP_NAME}'")

    step_cfg = cfg[STEP_NAME]
    output_dir = run_root / STEP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step '{STEP_NAME}' starting with output dir {output_dir}")

    return cfg, step_cfg, output_dir, logger


@app.cell
def _(os, datetime, Path, setup_step_logger, LOGGING_LEVEL):
    logger, run_dir = setup_step_logger(
        base_dir="runs",
        run_prefix="run",
        log_name="log.txt",
        level=LOGGING_LEVEL,
    )

    logger.info("This log goes to both the notebook and a file.")

    # Get current date and time
    now = datetime.now()

    # Format it to a string (e.g., 2025-04-02_15-30-00)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a folder name with the timestamp
    folder_name = f"output_{timestamp}"

    # Create the folder in the current directory
    folder_path = Path(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Created folder: {folder_path}")
    return (folder_path,)


@app.cell
def _(pd, EXPERIMENT_EXCEL_PATH):
    # Data from experiment (Lines, Barcodes etc.)
    master_data_df = pd.read_csv(EXPERIMENT_EXCEL_PATH)
    master_data_df.shape
    return (master_data_df,)


@app.cell
def _(pd, TRAITS_CSV_PATH):
    traits_df = pd.read_csv(TRAITS_CSV_PATH)
    traits_df.shape
    return (traits_df,)


@app.cell
def _(master_data_df, pd, traits_df):
    # Merge the two dataframes
    merged_data = pd.merge(master_data_df, traits_df, on="plant_qr_code", how="left")
    merged_data.shape
    return (merged_data,)


@app.cell
def _(pd, merged_data):
    # Make "plant_age_days" column numeric
    merged_data["plant_age_days"] = pd.to_numeric(
        merged_data["plant_age_days"], errors="coerce"
    )
    # Get unique values in the "plant_age_days" column
    unique_plant_ages = merged_data["plant_age_days"].unique()
    print(f"Unique plant ages: {unique_plant_ages}")

    return


@app.cell
def _(Path, merged_data, top_dir):
    # 7 Day-Old Plants only
    sleap_root_traits_7d = merged_data[merged_data["plant_age_days"] == 7]

    # Save the filtered DataFrame to a CSV file
    sleap_root_traits_7d.to_csv(
        Path(top_dir) / "biomass_and_sleap_root_traits_7d_after_qc.csv", index=False
    )
    print(
        f"Saved 7 Day-Old Plants data to CSV file: {Path(top_dir) / 'sleap_root_traits_7d.csv'} with shape: {sleap_root_traits_7d.shape}"
    )

    # 14 Day-Old Plants only
    sleap_root_traits_14d = merged_data[merged_data["plant_age_days"] == 14]

    # Save the filtered DataFrame to a CSV file
    sleap_root_traits_14d.to_csv(
        Path(top_dir) / "biomass_and_sleap_root_traits_14d_after_qc.csv", index=False
    )
    print(
        f"Saved 14 Day-Old Plants data to CSV file: {Path(top_dir) / 'sleap_root_traits_14d.csv'} with shape: {sleap_root_traits_14d.shape}"
    )
    return sleap_root_traits_14d, sleap_root_traits_7d


if __name__ == "__main__":
    app.run()

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
        datetime,
        openpyxl,
        logging,
        OmegaConf,
        setup_step_logger,
        argparse,
        sys,
        DictConfig,
    )


@app.cell
def load_config(Path, sys, argparse, OmegaConf, setup_step_logger):
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
    logger.info(f"Logger initialized for step '{STEP_NAME}' at {log_dir}")

    output_dir = run_root / STEP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step '{STEP_NAME}' starting with output dir {output_dir}")

    # Get the remaining config parameters for this step
    TRAITS_CSV_PATH = Path(cfg.input.traits_csv)
    EXPERIMENT_EXCEL_PATH = Path(cfg.input.experimental_design_excel)
    logger.info(f"TRAITS_CSV_PATH: {TRAITS_CSV_PATH}")
    logger.info(f"EXPERIMENT_EXCEL_PATH: {EXPERIMENT_EXCEL_PATH}")

    return cfg, output_dir, logger, TRAITS_CSV_PATH, EXPERIMENT_EXCEL_PATH


@app.cell
def read_master(pd, EXPERIMENT_EXCEL_PATH):
    # Data from experiment (Lines, Barcodes etc.)
    master_data_df = pd.read_excel(
        EXPERIMENT_EXCEL_PATH,
        sheet_name="Redo_Master_Data",  # Make configurable
        engine="openpyxl",
    )
    master_data_df.shape
    return (master_data_df,)


# @app.cell
# def _(pd, TRAITS_CSV_PATH):
#     traits_df = pd.read_csv(TRAITS_CSV_PATH)
#     traits_df.shape
#     return (traits_df,)


# @app.cell
# def _(master_data_df, pd, traits_df):
#     # Merge the two dataframes
#     merged_data = pd.merge(master_data_df, traits_df, on="plant_qr_code", how="left")
#     merged_data.shape
#     return (merged_data,)


# @app.cell
# def _(pd, merged_data):
#     # Make "plant_age_days" column numeric
#     merged_data["plant_age_days"] = pd.to_numeric(
#         merged_data["plant_age_days"], errors="coerce"
#     )
#     # Get unique values in the "plant_age_days" column
#     unique_plant_ages = merged_data["plant_age_days"].unique()
#     print(f"Unique plant ages: {unique_plant_ages}")

#     return


# @app.cell
# def _(Path, merged_data, top_dir):
#     # 7 Day-Old Plants only
#     sleap_root_traits_7d = merged_data[merged_data["plant_age_days"] == 7]

#     # Save the filtered DataFrame to a CSV file
#     sleap_root_traits_7d.to_csv(
#         Path(top_dir) / "biomass_and_sleap_root_traits_7d_after_qc.csv", index=False
#     )
#     print(
#         f"Saved 7 Day-Old Plants data to CSV file: {Path(top_dir) / 'sleap_root_traits_7d.csv'} with shape: {sleap_root_traits_7d.shape}"
#     )

#     # 14 Day-Old Plants only
#     sleap_root_traits_14d = merged_data[merged_data["plant_age_days"] == 14]

#     # Save the filtered DataFrame to a CSV file
#     sleap_root_traits_14d.to_csv(
#         Path(top_dir) / "biomass_and_sleap_root_traits_14d_after_qc.csv", index=False
#     )
#     print(
#         f"Saved 14 Day-Old Plants data to CSV file: {Path(top_dir) / 'sleap_root_traits_14d.csv'} with shape: {sleap_root_traits_14d.shape}"
#     )
#     return sleap_root_traits_14d, sleap_root_traits_7d


if __name__ == "__main__":
    app.run()

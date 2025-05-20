import marimo

__generated_with = "0.13.10"
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

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.covariance import MinCovDet
    from sklearn.mixture import GaussianMixture

    # Ensure the project root is on sys.path
    project_root = Path(".").resolve()
    sys.path.append(project_root.as_posix())
    from pipeline.pipeline_logger import setup_step_logger

    return (
        GaussianMixture,
        MinCovDet,
        OmegaConf,
        PCA,
        Path,
        StandardScaler,
        argparse,
        np,
        pd,
        plt,
        re,
        setup_step_logger,
        sns,
        sys,
    )


@app.cell
def load_config(OmegaConf, Path, argparse, re, setup_step_logger, sys):
    """Parse CLI arguments and load configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = OmegaConf.load(args.config_path)
    LOGGING_LEVEL = cfg.logging.level.upper()
    STEP_NAME = Path(sys.argv[0]).stem if sys.argv[0] else "unknown_step"

    run_root = Path(cfg.output_dir)
    log_dir = Path(cfg.logging.log_dir)
    logger = setup_step_logger(
        log_dir=log_dir, step_name=STEP_NAME, level=LOGGING_LEVEL
    )

    output_dir = run_root / STEP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Step '{STEP_NAME}' starting with output dir {output_dir}")

    csvs_dir = run_root / "data_cleanup"
    csvs = list(csvs_dir.glob("**/traits_*_cleaned.csv"))
    if not csvs:
        raise ValueError(f"No CSV files found in {csvs_dir}")

    csvs_dict = {
        int(match.group(1)): csv
        for csv in csvs
        if (match := re.search(r"(\d+)DAG", csv.name))
    }

    MAHALANOBIS_THRESHOLD = cfg.parameters.thresholds.mahalanobis
    COL_START = cfg.parameters.col_start
    return COL_START, MAHALANOBIS_THRESHOLD, csvs_dict, logger, output_dir


@app.cell
def define_utilities(np, plt, sns):
    def save_diagnostic_plots(pca, Z_pca, df_pca, sub_dir, gmm=None):
        """Generate and save plots for PCA and outlier scoring."""
        plt.figure(figsize=(10, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Explained variance")
        plt.savefig(
            sub_dir / "pca_95_elbow_plot_per_plant.png",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()

        plt.figure(figsize=(30, 7))
        plt.plot(Z_pca)
        plt.title("Transformed data by PCA")
        plt.savefig(
            sub_dir / "transformed_data_95percent.png",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()

        if "gmm_score" in df_pca.columns:
            # PCA scatterplot with GMM score
            plt.figure(figsize=(15, 15))
            sns.scatterplot(x=df_pca.PC1, y=df_pca.PC2, hue=df_pca.gmm_score)
            plt.savefig(
                sub_dir / "2d_scatterplot_pc1_pc2_gmm_score.png",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.show()

            # PCA scatterplot with GMM score and centers
            if gmm is not None:
                plt.figure(figsize=(15, 15))
                sns.scatterplot(
                    x=df_pca.PC1, y=df_pca.PC2, hue=df_pca.gmm_score, palette="viridis"
                )
                centers = gmm.means_
                plt.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    c="red",
                    s=200,
                    marker="X",
                    label="GMM Centers",
                )
                plt.legend()
                plt.savefig(
                    sub_dir / "2d_scatterplot_pc1_pc2_gmm_score_centers.png",
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.show()

            # Histogram of negative log likelihood
            plt.figure(figsize=(15, 15))
            sns.histplot(df_pca.gmm_score, bins=50, log_scale=True)
            plt.xlabel("GMM negative log likelihood")
            plt.savefig(
                sub_dir / "gmm_score_hist.png", bbox_inches="tight", facecolor="white"
            )
            plt.show()

        if "m_dist" in df_pca.columns:
            plt.figure(figsize=(15, 15))
            sns.scatterplot(x=df_pca.PC1, y=df_pca.PC2, hue=df_pca.m_dist)
            plt.savefig(
                sub_dir / "2d_scatterplot_pc1_pc2_mdist.png",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.show()

            sns.histplot(df_pca.m_dist, bins=50, log_scale=True)
            plt.xlabel("Mahalanobis distance")
            plt.savefig(
                sub_dir / "m_dist_hist.png", bbox_inches="tight", facecolor="white"
            )
            plt.show()

    return (save_diagnostic_plots,)


@app.cell
def run_pca_outlier_pipeline(
    COL_START,
    GaussianMixture,
    MAHALANOBIS_THRESHOLD,
    MinCovDet,
    PCA,
    StandardScaler,
    csvs_dict,
    logger,
    mo,
    np,
    output_dir,
    pd,
    plt,
    save_diagnostic_plots,
):
    """Run PCA and Mahalanobis outlier detection for each age group."""

    tables = []
    for age, csv_path in csvs_dict.items():
        logger.info(f"Processing {age}DAG -> {csv_path.name}")
        sub_dir = output_dir / f"{age}DAG"
        sub_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)

        # data cleanup should have already done this
        trait_df = df.iloc[:, COL_START:]
        # Append a heading and the table itself to the display list
        tables.append(mo.ui.text(f"### {age} DAG"))
        # Add trait_df to tables for later use
        tables.append(mo.ui.table(trait_df))
        # Check for NaN values
        if trait_df.isnull().values.any():
            raise ValueError(
                f"NaN values found in the dataframe: {trait_df.isnull().sum()}"
            )
        # Check for non-numeric values
        if not np.issubdtype(trait_df.values.dtype, np.number):
            raise ValueError(
                f"Non-numeric values found in the dataframe: {trait_df.dtypes}"
            )

        # Standardize traits
        Y = trait_df.to_numpy()
        scaler = StandardScaler().fit(Y)
        Z_scaled = scaler.transform(Y)

        # PCA to retain 95% variance
        pca = PCA(n_components=0.95, random_state=2020).fit(Z_scaled)
        Z_pca = pca.transform(Z_scaled)

        # Build dataframe with PCs and metadata
        pc_cols = [f"PC{i+1}" for i in range(Z_pca.shape[1])]
        df_pca = pd.DataFrame(Z_pca, columns=pc_cols)
        df_pca[df.columns[:COL_START]] = df[df.columns[:COL_START]].reset_index(
            drop=True
        )

        # Get the number of components that explain 75% of variance
        var_idx = np.argmax(np.cumsum(pca.explained_variance_ratio_ * 100) > 75) + 1
        Z_reduced = Z_pca[:, :var_idx]

        if MAHALANOBIS_THRESHOLD is not None:
            # Mahalanobis-based outlier detection
            mcd = MinCovDet().fit(Z_reduced)
            m_dist = mcd.mahalanobis(Z_reduced)
            df_pca["m_dist"] = m_dist
            outliers = df_pca[df_pca.m_dist > MAHALANOBIS_THRESHOLD][
                "plant_qr_code"
            ].tolist()
            df_pca.to_csv(sub_dir / "pca95_mdist.csv", index=False)
            logger.info(
                f"Outliers detected using Mahalanobis distance: {len(outliers)} out of {len(df_pca)}"
            )
            # Generate and save diagnostic plots
            save_diagnostic_plots(pca, Z_pca, df_pca, sub_dir, gmm=None)
        else:
            # Automatically select best GMM using BIC
            lowest_bic = np.inf
            best_n = None
            bic_scores = []
            for n in range(1, 6):
                gmm_candidate = GaussianMixture(
                    n_components=n, covariance_type="full", random_state=42
                )
                gmm_candidate.fit(Z_reduced)
                bic = gmm_candidate.bic(Z_reduced)
                bic_scores.append(bic)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_n = n

            logger.info(f"Number of clusters with lowest BIC is {best_n}")

            gmm = GaussianMixture(
                n_components=best_n, covariance_type="full", random_state=42
            )
            gmm.fit(Z_reduced)
            scores = -gmm.score_samples(Z_reduced)
            df_pca["gmm_score"] = scores
            df_pca["gmm_cluster"] = gmm.predict(Z_reduced)

            plt.figure()
            plt.plot(range(1, 6), bic_scores, marker="o")
            plt.xlabel("Number of GMM components")
            plt.ylabel("BIC")
            plt.title("GMM Model Selection via BIC")
            plt.savefig(
                sub_dir / "gmm_bic_model_selection.png",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.show()

            adaptive_threshold = np.percentile(scores, 99)
            logger.info(
                f"Adaptive threshold for GMM outlier detection: {adaptive_threshold}"
            )
            outliers = df_pca[df_pca.gmm_score > adaptive_threshold][
                "plant_qr_code"
            ].tolist()
            df_pca.to_csv(sub_dir / "pca95_gmm_scores.csv", index=False)
            logger.info(
                f"Outliers detected using GMM: {len(outliers)} out of {len(df_pca)}"
            )
            # Generate and save diagnostic plots
            save_diagnostic_plots(pca, Z_pca, df_pca, sub_dir, gmm=gmm)

        df_outliers = df[df["plant_qr_code"].isin(outliers)]
        df_outliers.to_csv(sub_dir / "df_outliers.csv", index=False)
        df_no_outliers = df[~df["plant_qr_code"].isin(outliers)]
        df_no_outliers.to_csv(sub_dir / "df_no_outliers.csv", index=False)
        logger.info(f"Outliers saved to {sub_dir / 'df_outliers.csv'}")
        logger.info(f"Final CSV with outliers removed saved to {sub_dir / 'df_no_outliers.csv'}")

        logger.info(f"Finished processing {age}DAG")

    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

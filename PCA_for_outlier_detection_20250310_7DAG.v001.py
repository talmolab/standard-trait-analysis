import marimo

__generated_with = "0.11.17"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os

    from sklearn.covariance import EmpiricalCovariance, MinCovDet
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from mpl_toolkits import mplot3d
    from string import ascii_letters
    return (
        EmpiricalCovariance,
        MinCovDet,
        PCA,
        StandardScaler,
        ascii_letters,
        make_axes_locatable,
        mpl,
        mplot3d,
        np,
        os,
        pd,
        plt,
        sns,
    )


@app.cell
def _():
    TOP_DIR = "2025-03-10" # top-level directory
    SUB_DIR = "outlier_detection" # subdirectory for notebook outputs
    INPUT_CSV = "2025-03-10/data_cleanup_7DAG/biomass_sleap_roots_7DAG_nonans.csv" # input csv file without any NaNs
    COL_START = 2 # column where traits starts to be used for PCA
    ROOT_BIOMASS_COL_NAME = "Root_Biomass_mg" # Name of the root biomass column
    SHOOT_BIOMASS_COL_NAME = "Shoot_Biomass_mg" # Name of the shoot biomass column
    THRESHOLD = 100 # threshold for Mahalanobis distance
    return (
        COL_START,
        INPUT_CSV,
        ROOT_BIOMASS_COL_NAME,
        SHOOT_BIOMASS_COL_NAME,
        SUB_DIR,
        THRESHOLD,
        TOP_DIR,
    )


@app.cell
def _(SUB_DIR, TOP_DIR, os):
    # Create the top-level directory
    top_dir = TOP_DIR
    os.makedirs(top_dir, exist_ok=True)

    # Create a subdirectory inside the top-level directory
    sub_dir = os.path.join(top_dir, SUB_DIR)
    os.makedirs(sub_dir, exist_ok=True)

    sub_dir
    return sub_dir, top_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Use clean data:""")
    return


@app.cell
def _(INPUT_CSV, pd):
    df = pd.read_csv(INPUT_CSV)
    df = df.copy()
    df.shape
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(COL_START, df):
    # Check the columns and adjust COL_START if needed
    df.iloc[:, COL_START:]
    return


@app.cell
def _(COL_START, df):
    # convert data from df to a single numpy array
    _col_start = COL_START # column where traits start
    Y = df.iloc[:, _col_start:].to_numpy()
    Y.shape
    return (Y,)


@app.cell
def _(Y, df, np):
    # Check for missing values (all NaNs)
    _is_missing_all = np.isnan(Y).all(axis=1)
    _missing_barcodes_all = df['plant_qr_code'].iloc[_is_missing_all]
    _miss_all_rows = df.iloc[_is_missing_all]
    print(f'{len(_missing_barcodes_all)}/{len(df)} missing all features')
    print(_missing_barcodes_all)
    return


@app.cell
def _(Y, df, np):
    # Check for missing values in the data (any NaNs)
    _is_missing_any = np.isnan(Y).any(axis=1)
    _missing_barcodes_any = df['plant_qr_code'].iloc[_is_missing_any]
    _miss_any_rows = df.iloc[_is_missing_any]
    print(f'{len(_missing_barcodes_any)}/{len(df)} missing any features')
    print(_missing_barcodes_any)
    return


@app.cell
def _(df, plt, sns, sub_dir):
    sns.set_context('talk')
    _f, _ax = plt.subplots(figsize=(5, 5))
    plt.xticks(rotation=90)
    _ax = sns.countplot(y='Line', data=df, order=df['Line'].value_counts().index)
    plt.title('')
    _ax.xaxis.grid(True)
    _ax.set(xlabel='Line')
    sns.despine(trim=True, left=True)
    plt.savefig(f'{sub_dir}/line_counts_no_nans.png', bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {sub_dir}/line_counts_no_nans.png")
    return


@app.cell
def _(COL_START, df, np, plt, sns, sub_dir):
    sns.set_theme(style='dark')
    sns.set_style('darkgrid')
    _col_start = COL_START
    _res = df.iloc[:, _col_start:]
    _corr = _res.corr()
    mask = np.triu(np.ones_like(_corr, dtype=bool))
    _f, _ax = plt.subplots(figsize=(50, 50))
    sns.heatmap(_corr, mask=mask, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.savefig(f'{sub_dir}/corr_table.png', bbox_inches='tight', facecolor='white', dpi=600)
    plt.show()
    print(f"Saved to {sub_dir}/corr_table.png")
    return (mask,)


@app.cell
def _(
    COL_START,
    ROOT_BIOMASS_COL_NAME,
    SHOOT_BIOMASS_COL_NAME,
    df,
    plt,
    sns,
    sub_dir,
):
    _col_start = COL_START
    _res = df.iloc[:, _col_start:]
    target_columns = [ROOT_BIOMASS_COL_NAME, SHOOT_BIOMASS_COL_NAME]
    _corr = _res.corr()
    target_corr = _corr.loc[:, target_columns]
    sns.set_theme(style='dark')
    sns.set_style('darkgrid')
    _f, _ax = plt.subplots(figsize=(10, 200))
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=_ax)
    _ax.set_yticklabels(_ax.get_yticklabels(), rotation=0, fontsize=12)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=45, fontsize=12)
    plt.savefig(f'{sub_dir}/corr_with_root_shoot.png', bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {sub_dir}/corr_with_root_shoot.png")
    return target_columns, target_corr


@app.cell
def _(target_corr):
    target_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Start PCA!:""")
    return


@app.cell
def _(COL_START, df):
    _col_start = COL_START
    Z = df.iloc[:, _col_start:].to_numpy()
    return (Z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$ 
        Z = \frac{x - \mu}{\sigma}
        $$
        """
    )
    return


@app.cell
def _(StandardScaler, Z):
    #Create the object
    scaler = StandardScaler()
    # Calculate the mean and standard deviation
    scaler.fit(Z)
    # Transform the values using z-score standardization
    Z_scaled = scaler.transform(Z)
    return Z_scaled, scaler


@app.cell
def _(Z_scaled):
    # plants X traits
    Z_scaled.shape
    return


@app.cell
def _(COL_START, df):
    _col_start = COL_START
    # Check the columns and adjust COL_START if needed
    df.iloc[:, _col_start:].columns
    return


@app.cell
def _(COL_START, Z_scaled, df, pd):
    _col_start = COL_START
    df_scaled = pd.DataFrame(Z_scaled, columns=df.iloc[:, _col_start:].columns)
    return (df_scaled,)


@app.cell
def _(PCA, Z_scaled):
    # PCA tranformation capturing 95% of variability in dataset 
    pca_95 = PCA(n_components=0.95, random_state=2020)
    pca_95.fit(Z_scaled)
    Z_pca_95 = pca_95.transform(Z_scaled)
    return Z_pca_95, pca_95


@app.cell
def _(Z_pca_95):
    # plants X number of principal components 
    Z_pca_95.shape
    return


@app.cell
def _(Z_pca_95):
    # number of pricipal components for 95% explained variance
    pca_n = Z_pca_95.shape[1]
    pca_n
    return (pca_n,)


@app.cell
def _(pca_95, pca_n):
    print(f"Variance explained by all {pca_n} principal components = ", sum(pca_95.explained_variance_ratio_*100))
    return


@app.cell
def _(pca_95):
    pca_95_explained = pca_95.explained_variance_ratio_*100
    pca_95_explained
    return (pca_95_explained,)


@app.cell
def _(np, pca_95):
    pca_95_cumsum = np.cumsum(pca_95.explained_variance_ratio_ * 100)
    pca_95_cumsum
    return (pca_95_cumsum,)


@app.cell
def _(np, pca_95, plt, sub_dir):
    plt.figure(figsize=(10, 4), dpi=100)
    plt.plot(np.cumsum(pca_95.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.savefig(f'{sub_dir}/pca_95_elbow_plot_per_plant.png', bbox_inches = "tight", facecolor="white")
    plt.show()
    print(f"Saved to {sub_dir}/pca_95_elbow_plot_per_plant.png")
    return


@app.cell
def _(Z_pca_95, plt, sub_dir):
    plt.figure(figsize=(30,7))
    plt.plot(Z_pca_95)
    plt.xlabel('Observation')
    plt.ylabel('Transformed Data')
    plt.title('Transformed data by the principle components (95% variability)', pad=15)
    plt.savefig(f'{sub_dir}/transformed_data_95percent.png', bbox_inches = "tight", facecolor="white")
    plt.show()
    print(f"Saved to {sub_dir}/transformed_data_95percent.png")
    return


@app.cell
def _(pca_n):
    pc_list = ['PC' + str(_i) for _i in range(1, pca_n + 1)]
    pc_list
    return (pc_list,)


@app.cell
def _(Z_pca_95, pc_list, pd):
    # Create a dataframe with the transformed data
    df_new = pd.DataFrame(Z_pca_95, columns=pc_list)
    df_new = df_new.copy()
    df_new
    return (df_new,)


@app.cell
def _(df):
    # Choose the columns you want with your PC data
    df
    return


@app.cell
def _(COL_START, df):
    # get df column names using iloc
    # choose the columns you want with your PC data
    col_end = COL_START
    list(df.iloc[:, :col_end].columns)
    return (col_end,)


@app.cell
def _(col_end, df, df_new):
    # Add the columns you want to the new dataframe
    df_new[list(df.iloc[:, :col_end].columns)] = df[list(df.iloc[:, :col_end].columns)]
    df_new
    return


@app.cell
def _(df_new, sub_dir):
    df_new.to_csv(f"{sub_dir}/df_pca_95.csv", index=False)
    print(f"Saved to {sub_dir}/df_pca_95.csv")
    return


@app.cell
def _(df_new, mpl, pca_95_cumsum, pca_95_explained, plt, sub_dir):
    plt.figure(figsize=(15, 15), dpi=120)
    cmap = mpl.cm.cool
    plt.scatter(x=df_new.PC1, y=df_new.PC2, c=df_new.PC3, cmap=cmap, linewidths=0.5, edgecolors='k', alpha=0.8)
    plt.colorbar(label=f'PC3: {pca_95_explained[2]:.4f}% variability')
    for _i, (_x, _y) in enumerate(df_new.iloc[:, 0:2].to_numpy()):
        plt.text(_x, _y, s=df_new.plant_qr_code[_i], fontsize=8)
    plt.xlabel(f'PC1: {pca_95_explained[0]:.4f}% variability')
    plt.ylabel(f'PC2: {pca_95_explained[1]:.4f}% variability')
    plt.title(f'Transformed data by the principle components ({pca_95_cumsum[2]:.4f}% variability)', pad=15)
    plt.savefig(f'{sub_dir}/2d_scatterplot_pc1_pc2.png', bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {sub_dir}/2d_scatterplot_pc1_pc2.png")
    return (cmap,)


@app.cell
def _(COL_START, df, np, pca_95, plt, sub_dir):
    feature_names = df.columns[COL_START:]
    plt.figure(figsize=(10, 100))
    plt.imshow(pca_95.components_[:5].T, interpolation='nearest', aspect='auto')
    plt.clim(-np.abs(pca_95.components_[:5]).max(), np.abs(pca_95.components_[:5]).max())
    plt.colorbar(label='Coefficient')
    plt.xticks(np.arange(5), labels=np.arange(5) + 1)
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.savefig(f'{sub_dir}/features_pca5_colorbar.png', bbox_inches='tight', facecolor='white')
    return (feature_names,)


@app.cell
def _(pca_95):
    pca_95.components_.shape
    return


@app.cell
def _(feature_names, pca_95, pd):
    df_pc_first_component = pd.DataFrame(pca_95.components_[0], columns=["PC1"])
    df_pc_first_component['Features'] = feature_names
    df_pc_first_component = df_pc_first_component.sort_values(by=['PC1'])
    df_pc_first_component
    return (df_pc_first_component,)


@app.cell
def _(feature_names, pca_95, pd):
    df_pc_second_component = pd.DataFrame(pca_95.components_[1], columns=["PC2"])
    df_pc_second_component = df_pc_second_component.copy()
    df_pc_second_component['Features'] = feature_names
    df_pc_second_component = df_pc_second_component.sort_values(by=['PC2'])
    df_pc_second_component
    return (df_pc_second_component,)


@app.cell
def _(feature_names, pca_95, pd):
    df_pc_third_component = pd.DataFrame(pca_95.components_[2], columns=["PC3"])
    df_pc_third_component = df_pc_third_component.copy()
    df_pc_third_component['Features'] = feature_names
    df_pc_third_component = df_pc_third_component.sort_values(by=['PC3'])
    df_pc_third_component
    return (df_pc_third_component,)


@app.cell
def _(feature_names, pca_95, pd):
    df_pc_fourth_component = pd.DataFrame(pca_95.components_[3], columns=["PC4"])
    df_pc_fourth_component = df_pc_fourth_component.copy()
    df_pc_fourth_component['Features'] = feature_names
    df_pc_fourth_component = df_pc_fourth_component.sort_values(by=['PC4'])
    df_pc_fourth_component
    return (df_pc_fourth_component,)


@app.cell
def _(mo):
    mo.md(r"""PCA for Outlier Detection:""")
    return


@app.cell
def _(df_new, pca_95_cumsum, pca_95_explained, plt, sub_dir):
    plt.figure(figsize=(15, 15), dpi=120)
    plt.scatter(x=df_new.PC1, y=df_new.PC2, c=df_new.PC3, linewidths=0.5, edgecolors='k', alpha=0.8)
    plt.colorbar(label=f'PC3: {pca_95_explained[2]:.4f}% variability')
    for _i, (_x, _y) in enumerate(df_new.iloc[:, 0:2].to_numpy()):
        plt.text(_x, _y, s=df_new.plant_qr_code[_i], fontsize=8)
    plt.xlim((-80, 80))
    plt.ylim((-80, 80))
    plt.xlabel(f'PC1: {pca_95_explained[0]:.4f}% variability')
    plt.ylabel(f'PC2: {pca_95_explained[1]:.4f}% variability')
    plt.title(f'Transformed data by the principle components ({pca_95_cumsum[2]:.4f}% variability)', pad=15)
    plt.savefig(f'{sub_dir}/2d_scatterplot_pc1_pc2_aspect_ratio_equal.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {sub_dir}/2d_scatterplot_pc1_pc2_aspect_ratio_equal.png")
    return


@app.cell
def _(df_new, pca_95_cumsum, pca_95_explained, plt, sns, sub_dir):
    sns.set_theme(style="dark")
    data = df_new.copy()
    # Draw a combo histogram and scatterplot with density contours
    plt.figure(figsize=(5, 30), dpi=120)
    joint = sns.jointplot(data=data, x="PC1", y="PC2", kind="hist", cmap="mako", cbar="True")
    sns.kdeplot(data=data, x="PC1", y="PC2", levels=5, color="w", linewidths=1)
    # JointGrid has a convenience function
    joint.set_axis_labels(f'PC1: {pca_95_explained[0]:.4f}% variability', f'PC2: {pca_95_explained[1]:.4f}% variability', fontsize=16)
    joint.fig.suptitle(f'Transformed data by the principle components ({pca_95_cumsum[1]:.4f}% variability)')
    joint.fig.tight_layout()
    plt.savefig(f'{sub_dir}/2d_scatterplot_pc1_pc2_heatmap.png', dpi=120, bbox_inches = "tight", facecolor="white")
    plt.show()
    print(f"Saved to {sub_dir}/2d_scatterplot_pc1_pc2_heatmap.png")
    return data, joint


@app.cell
def _(df_new, sns):
    data_pca = df_new.copy()
    sns.kdeplot(
        data=data_pca, x=df_new.PC1, y=df_new.PC2,
        fill=True, thresh=0, levels=100, cmap="mako", cbar=True,
    )
    return (data_pca,)


@app.cell
def _(df_new, pca_95_cumsum, pca_95_explained, plt, sns, sub_dir):
    sns.set_theme(style="dark")
    # Draw a combo histogram and scatterplot with density contours
    plt.figure(figsize=(10, 30))
    sns.lmplot(data=df_new, x="PC1", y="PC2", fit_reg=False, hue="Line")
    sns.kdeplot(x=df_new.PC1, y=df_new.PC2, levels=5, color="w", linewidths=1)
    plt.xlabel(f'PC1: {pca_95_explained[0]:.4f}% variability')
    plt.ylabel(f'PC2: {pca_95_explained[1]:.4f}% variability')
    plt.title(f'Transformed data by the principle components ({pca_95_cumsum[1]:.4f}% variability)', pad=15)
    plt.savefig(f'{sub_dir}/2d_scatterplot_pc1_pc2_lines.png', bbox_inches = "tight", facecolor="white")
    plt.show()
    print(f"Saved to {sub_dir}/2d_scatterplot_pc1_pc2_lines.png")
    return


@app.cell
def _(pca_95_cumsum):
    var = 75 # percentage of variability
    print('The original list is : ' + str(list(pca_95_cumsum)))
    res_var = list(filter(lambda i: i > var, list(pca_95_cumsum)))[0]
    var_idx = list(pca_95_cumsum).index(res_var)
    print(f' {var_idx + 1} principal components account for {var} % of variability  {var}')
    return res_var, var, var_idx


@app.cell
def _(Z_pca_95, var_idx):
    # Check the shape of the data accounting for 75% of variability
    Z_pca_95[:,:var_idx + 1].shape
    return


@app.cell
def _(MinCovDet, Z_pca_95, var_idx):
    # Use Mahalanobis distance
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data accounting for 75% of variability 
    robust_cov = MinCovDet().fit(Z_pca_95[:,:var_idx+1])
    # Get the Mahalanobis distance
    m_dist = robust_cov.mahalanobis(Z_pca_95[:,:var_idx+1])
    return m_dist, robust_cov


@app.cell
def _(df_new, m_dist, sub_dir):
    # Add the Mahalanobis distance to the dataframe with the PCA data 
    df_new['m_dist'] = m_dist.tolist()
    # Save the m_dist with PCA data to a csv
    df_new.to_csv(f"{sub_dir}/pca95_mdist.csv", index=False)
    return


@app.cell
def _(df_new, pca_95_cumsum, pca_95_explained, plt, sns, sub_dir):
    sns.set_theme(style='dark')
    _f, _ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=df_new.PC1, y=df_new.PC2, s=5, hue=df_new.m_dist)
    sns.kdeplot(x=df_new.PC1, y=df_new.PC2, levels=5, color='w', linewidths=1)
    plt.xlabel(f'PC1: {pca_95_explained[0]:.4f}% variability')
    plt.ylabel(f'PC2: {pca_95_explained[1]:.4f}% variability')
    plt.title(f'Transformed data by the principle components ({pca_95_cumsum[1]:.4f}% variability)', pad=15)
    plt.savefig(f'{sub_dir}/2d_scatterplot_pc1_pc2_mdist_contour.png', bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {sub_dir}/2d_scatterplot_pc1_pc2_mdist_contour.png")
    return


@app.cell
def _(df_new, plt, sns, sub_dir):
    # Set the number of bins to a high value
    num_bins = 50
    sns.set_theme(style="ticks")
    mdist_data = df_new.sort_values(by=['m_dist'])[['plant_qr_code', 'm_dist']]
    sns.histplot(data=mdist_data, x="m_dist", bins=num_bins, log_scale=True)
    plt.xlabel(f'Mahalanobis distance')
    plt.savefig(f'{sub_dir}/mdist_hist.png', bbox_inches = "tight", facecolor="white")
    plt.show()
    print(f"Saved to {sub_dir}/mdist_hist.png")
    return mdist_data, num_bins


@app.cell
def _(THRESHOLD, df_new):
    _threshold = THRESHOLD
    outliers = df_new[df_new.m_dist > _threshold]['plant_qr_code'].to_list()
    outliers
    return (outliers,)


@app.cell
def _(df, outliers):
    # remove outliers from original data
    df_no_outliers = df[~df["plant_qr_code"].isin(outliers)]
    df_no_outliers
    return (df_no_outliers,)


@app.cell
def _(df_no_outliers, sub_dir):
    # save data without outliers
    df_no_outliers.to_csv(f"{sub_dir}/df_no_outliers.csv", index=False)
    print(f"Saved to {sub_dir}/df_no_outliers.csv")
    return


@app.cell
def _(
    ROOT_BIOMASS_COL_NAME,
    SHOOT_BIOMASS_COL_NAME,
    df_no_outliers,
    plt,
    sns,
    sub_dir,
):
    _res = df_no_outliers.copy()
    _x = _res[SHOOT_BIOMASS_COL_NAME]
    _y = _res[ROOT_BIOMASS_COL_NAME]
    sns.jointplot(x=_x, y=_y, kind='hex', color='#4CB391')
    plt.xlabel('Shoot Biomass (mg)')
    plt.ylabel('Root Biomass (mg)')
    plt.show()
    plt.savefig(f'{sub_dir}/biomass_jointplot_no_outliers.png', bbox_inches='tight', facecolor='white')
    print(f"Saved to {sub_dir}/biomass_jointplot_no_outliers.png")
    return


@app.cell
def _(SHOOT_BIOMASS_COL_NAME, df_no_outliers, plt, sns, sub_dir):
    _res = df_no_outliers.copy()
    plt.figure(figsize=(8, 6))
    sns.histplot(data=_res, x=SHOOT_BIOMASS_COL_NAME, bins=50, kde=True, color='skyblue')
    plt.xlabel('Shoot Biomass (mg)')
    plt.savefig(f'{sub_dir}/shoot_biomass_hist_no_outliers.png', bbox_inches='tight', facecolor='white')
    plt.show
    print(f"Saved to {sub_dir}/shoot_biomass_hist_no_outliers.png")
    return


@app.cell
def _(ROOT_BIOMASS_COL_NAME, df_no_outliers, plt, sns, sub_dir):
    _res = df_no_outliers.copy()
    plt.figure(figsize=(8, 6))
    sns.histplot(data=_res, x=ROOT_BIOMASS_COL_NAME, bins=50, kde=True, color='skyblue')
    plt.xlabel('Root Biomass (mg)')
    plt.savefig(f'{sub_dir}/root_biomass_hist_no_outliers.png', bbox_inches='tight', facecolor='white')
    plt.show
    print(f"Saved to {sub_dir}/root_biomass_hist_no_outliers.png")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

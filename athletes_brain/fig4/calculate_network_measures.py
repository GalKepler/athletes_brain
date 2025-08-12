import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.formula.api as smf
import pandas as pd
import bct
from tqdm import tqdm


def degree(mat, edge_threshold="auto"):
    """Calculate the degree of each node in the network."""
    G = mat.copy()
    # Threshold the matrix to remove weak connections
    if edge_threshold == "auto":
        edge_threshold = np.mean(mat)  # Use the mean as a threshold
    G[mat < edge_threshold] = 0  # Set weak connections to zero
    return bct.degrees_und(G)


def local_efficiency(mat):
    """Calculate the local efficiency of the network."""
    return bct.efficiency_wei(mat, local=True)


def small_worldness_metrics(mat, n_randomizations=100):
    """
    Calculate the small-worldness metrics (gamma, lambda, sigma) of the network.

    A network is considered small-world if:
    gamma ($ \\gamma $): Clustering coefficient ratio (Actual Clustering / Random Clustering) > 1.
    lambda ($ \\lambda $): Path length ratio (Actual Path Length / Random Path Length) $ \\approx $ 1.
                            (Calculated as E_random / E_actual, where E is global efficiency,
                            as characteristic path length is inversely related to global efficiency).
    sigma ($ \\sigma $): Small-world coefficient ($ \\gamma / \\lambda $) > 1.

    Parameters:
        mat (np.ndarray): Weighted adjacency matrix of the network.
        n_randomizations (int): Number of random networks to generate for comparison.

    Returns:
        tuple: (gamma, lambda_val, sigma)
            gamma (float): Small-world gamma.
            lambda_val (float): Small-world lambda.
            sigma (float): Small-world sigma.
            Returns (np.nan, np.nan, np.nan) if calculations fail (e.g., due to disconnected graphs).
    """
    # Ensure the matrix is undirected and remove self-loops for consistency with BCT functions
    G = (mat + mat.T) / 2
    np.fill_diagonal(G, 0)

    # Calculate actual network metrics (average clustering and global efficiency)
    C_actual = np.mean(
        bct.clustering_coef_wu(G)
    )  # Average clustering coefficient for the whole network
    E_actual = bct.efficiency_wei(G)

    # Handle cases where actual network might be entirely disconnected or degenerate
    if E_actual == 0 or np.isnan(C_actual) or np.isnan(E_actual):
        print(
            "Warning: Actual network has zero global efficiency or NaN metrics. Small-worldness metrics might be undefined."
        )
        return np.nan, np.nan, np.nan

    # Calculate metrics for random networks
    C_random_list = []
    E_random_list = []

    for _ in range(n_randomizations):

        # Generate random graph preserving degree sequence (or strength for weighted) using edge swaps
        # bct.randmio_und_ei returns (randomized_matrix, num_swaps)
        Gr, _ = bct.randmio_und_signed(G, itr=10)

        # Check for empty or highly sparse random graphs that might cause issues
        if np.sum(Gr) == 0 or np.all(np.isclose(Gr, 0)):
            continue  # Skip this randomization if it resulted in an empty graph

        Cr = np.mean(bct.clustering_coef_wu(Gr))
        Er = bct.efficiency_wei(Gr)

        # Only add if the values are valid and non-zero
        if not np.isnan(Cr) and not np.isnan(Er) and Er != 0:
            C_random_list.append(Cr)
            E_random_list.append(Er)

    if not C_random_list or not E_random_list:
        print(
            f"Warning: Could not generate sufficient valid random networks ({len(C_random_list)}/{n_randomizations}) for small-worldness calculation."
        )
        return np.nan, np.nan, np.nan

    C_random_mean = np.mean(C_random_list)
    E_random_mean = np.mean(E_random_list)

    # Calculate gamma: Actual Clustering / Random Clustering
    gamma = C_actual / C_random_mean if C_random_mean != 0 else np.inf

    # Calculate lambda: Random Efficiency / Actual Efficiency (equivalent to L_actual / L_random)
    lambda_val = E_random_mean / E_actual if E_actual != 0 else np.inf

    # Calculate sigma: gamma / lambda
    sigma = gamma / lambda_val if lambda_val != 0 else np.inf

    return gamma, lambda_val, sigma


def local_efficiency(mat):
    """Calculate the local efficiency of the network."""
    return bct.efficiency_wei(mat, local=True)


def modularity(mat):
    """Calculate the modularity of the network."""
    _, Q = bct.modularity_und(mat)
    return Q


NETWORK_MEASURES = [
    {"func": degree, "outputs": ["degree"]},
    {"func": bct.strengths_und, "outputs": ["strength"]},
    # {"func": bct.clustering_coef_wu, "outputs": ["clustering"]},
    {"func": bct.betweenness_wei, "outputs": ["betweenness"]},
    {"func": bct.efficiency_wei, "outputs": ["global_efficiency"]},
    # {"func": bct.assortativity_wei, "outputs": ["assortativity"]},
    # {"func": modularity, "outputs": ["modularity"]},
    # {"func": local_efficiency, "outputs": ["local_efficiency"]},
    # {"func": small_worldness_metrics, "outputs": ["gamma", "lambda", "sigma"]},
]


def estimate_scn(
    wide_df: dict,
    metric: str,
    parcels: pd.DataFrame,
    region_col: str = "index",
    subject_col:str = "subject_code",
    group_by: str = "column"
):
    """
    Estimate Structural Covariance Networks (SCNs) from the provided data.
    Parameters:
    - data: DataFrame containing the data with columns 'subject_code', 'value', and 'region_col'
    - metric: Metric to be used for SCN estimation (e.g., 'gm_vol', 'thickness')
    - region_col: Column name for the brain region
    - age_col: Column name for age
    - n_bins: Number of age bins to create for grouping participants by age
    Returns:
    - scns: Dictionary containing SCNs for each age group with correlation matrices and metadata
    """
    scns = {}
    # long_df = data.copy().drop_duplicates(subset=[subject_col, region_col], keep="last")
    covariates = ["sex"] if metric != "gm_vol" else ["sex", "tiv"]

    # correct for sex
    # wide_df = long_df.pivot_table(index="subject_code", columns=[region_col], values="value")
    # residualise for sex, and if needed, for TIV
    for i, row in parcels.iterrows():
        col = row[region_col]
        df_tmp = pd.DataFrame(index=wide_df.index)
        df_tmp[subject_col] = wide_df.index
        df_tmp[covariates] = wide_df[covariates].copy()
        df_tmp["value"] = wide_df[col]
        # do mixed effects regression
        model = smf.ols("value ~ " + " + ".join(covariates), data=df_tmp).fit()
        wide_df[col] = model.resid + model.params["Intercept"]
        # break
    # break

    # Step 2: Add age group info to wide matrix
    for group_name, group_df in wide_df.groupby(group_by):
        # Create a new column for each age group
        region_matrix = wide_df.loc[group_df["subject_code"]]
        corr_matrix = region_matrix.corr(method="spearman")
        scns[group_name] = {"corr_matrix": corr_matrix, "metadata": group_df}
    # Step 3: Return the dictionary of SCNs
    return scns


def calculate_network_measures(
    scns: dict,
    parcels: pd.DataFrame,
    metrics_directory: Path,
    network_measures: dict = NETWORK_MEASURES,
    force: bool = False,
):
    """
    Calculate network measures for each SCN in the provided dictionary.

    Parameters:
    - scns: Dictionary containing SCNs with correlation matrices and metadata
    - metric: Metric to be used for network measure calculation
    - edge_threshold: Threshold for edge weights (default is 0.1)
    - n_randomizations: Number of randomizations for small-worldness metrics

    Returns:
    - DataFrame with calculated network measures for each age group
    """
    network_matrices_dict = {}
    for group_name, group_data in tqdm(scns.items(), desc="Calculating network matrices"):
        target_file = metrics_directory / f"{group_name}.csv"
        if not force and target_file.exists():
            print(f"Loading existing file: {target_file}")
            p = pd.read_csv(target_file, index_col=0)
            network_matrices_dict[group_name] = p
            continue
        mat = group_data["corr_matrix"].values.copy()
        metadata = group_data["metadata"].copy()
        np.fill_diagonal(mat, 0)  # Set diagonal to zero
        # Calculate the network matrices
        p = parcels.copy()
        for measure in network_measures:
            func = measure["func"]
            outputs = measure["outputs"]
            results = func(mat)
            if len(outputs) == 1:
                results = [results]
            for col, res in zip(outputs, results):
                p[col] = res
        p["group"] = group_name
        # Add metadata
        for col in ["median_age", "mean_age", "sd_age", "n_participants"]:
            if col in metadata.columns:
                p[col] = metadata[col].values[0]
            else:
                p[col] = np.nan
        p.to_csv(target_file, index=False)
        # network_matrices_dict[group_name] = p
        # break

    # network_matrices = pd.concat(network_matrices_dict.values(), ignore_index=True)
    # return network_matrices


if __name__ == "__main__":
    FORCE = True
    # metric = "adc"
    ATLAS = "schaefer2018tian2020_400_7"
    region_col = "index"
    DATA_DIR = Path("/home/galkepler/Projects/athletes_brain/data")
    metrics = ["adc", "gm_vol"]
    # Load important files

    # Load the data
    parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
    nifti = DATA_DIR / "external" / "atlases" / ATLAS / "atlas.nii.gz"
    nifti_matlab = DATA_DIR / "external" / "atlases" / ATLAS / "atlas_matlab.nii"
    distribution_metric = "qfmean"

    # Output directory for figures
    OUTPUT_DIR = Path("/home/galkepler/Projects/athletes_brain/figures/fig5") / ATLAS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bad_subjects = ["IN120120"]

    # Load the data
    data = {}
    for metric in metrics:
        data[metric] = pd.read_csv(
            DATA_DIR / "processed" / f"{metric}.csv", index_col=0
        ).reset_index(drop=True)
        # drop problematic subjects
        data[metric] = data[metric][~data[metric]["subject_code"].isin(bad_subjects)]
        data[metric]["sex"] = data[metric]["sex"].map({"M": 0, "F": 1})
    metric_cols = {
        metric: "volume" if metric == "gm_vol" else distribution_metric for metric in metrics
    }

    for m, df in data.items():
        df = df.rename(columns={metric_cols[m]: "value"})
        data[m] = df

    # Estimate SCNs
    scns = {}
    for m, df in data.items():
        scns[m] = estimate_scn(
            df,
            metric=m,
            region_col=region_col,
            age_col="age_at_scan",
            n_bins=50,
        )
        print(f"Estimated SCNs for {m}: {len(scns[m])} age groups")
    # Calculate network measures
    for m, scn in scns.items():
        print(f"Calculating network measures for {m}...")
        metrics_directory = OUTPUT_DIR / m / "network_matrices"
        metrics_directory.mkdir(parents=True, exist_ok=True)
        calculate_network_measures(
            scn,
            parcels,
            metrics_directory=metrics_directory,
            network_measures=NETWORK_MEASURES,
            force=FORCE,
        )

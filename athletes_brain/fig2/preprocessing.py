import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from athletes_brain.fig2.config import Fig2Config

REGION_COL = Fig2Config.REGION_COL
DEMOGRAPHIC_COLS = Fig2Config.DEMOGRAPHIC_COLS


def long_to_wide(
    long_df: pd.DataFrame,
    index_cols: list = None,
    columns_to_pivot: str = REGION_COL,
    values_col: str = "value",
    demographic_cols: list = None,
) -> pd.DataFrame:
    """
    Pivots a long metric table to a wide DataFrame (sessions x features),
    while preserving demographic and target columns.

    Parameters
    ----------
    long_df : pd.DataFrame
        Input DataFrame in long format with columns like index_cols, columns_to_pivot,
        values_col, and demographic_cols.
    index_cols : list of str, optional
        Columns to use as the index for pivoting (e.g., ['Participant_ID', 'Session_ID']).
        Defaults to ["subject_code", "session_id"].
    columns_to_pivot : str
        Column containing the region names/IDs to become new columns.
    values_col : str
        Column containing the metric values.
    demographic_cols : list of str, optional
        Other columns to preserve (demographics, target variables).
        Defaults to DEMOGRAPHIC_COLS from config.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with (Participant_ID, Session_ID) as a multi-index,
        region values as columns (prefixed with metric type), and
        demographic/target columns.
    """
    if index_cols is None:
        index_cols = ["subject_code", "session_id"]
    if demographic_cols is None:
        demographic_cols = DEMOGRAPHIC_COLS

    if not all(col in long_df.columns for col in index_cols + [columns_to_pivot, values_col]):
        raise ValueError(
            f"Missing essential columns for pivoting. Required: {index_cols + [columns_to_pivot, values_col]}"
        )

    present_demographic_cols = [col for col in demographic_cols if col in long_df.columns]

    if "session_id" not in long_df.columns:
        long_df["session_id"] = long_df.groupby("subject_code").cumcount() + 1
        print(
            "  Warning: 'Session_ID' not found in long_df. Generated dummy Session_ID for pivoting."
        )

    meta_df = long_df[index_cols + present_demographic_cols].drop_duplicates(subset=index_cols)
    meta_df = meta_df.set_index(index_cols)

    wide_metric = long_df.pivot_table(
        index=index_cols, columns=columns_to_pivot, values=values_col
    )

    full_wide_df = meta_df.join(wide_metric, how="inner")

    return full_wide_df.reset_index()


def init_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Initializes a ColumnTransformer for numerical and categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame from which to determine numerical and categorical columns.

    Returns
    -------
    ColumnTransformer
        A preprocessor that scales numerical features and one-hot encodes categorical features.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), X.select_dtypes(include="number").columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                X.select_dtypes(include="object").columns,
            ),
        ],
        remainder="passthrough",  # Keep other columns not specified
    )

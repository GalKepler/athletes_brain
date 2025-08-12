"""Statistical analysis module for athletes brain study Figure 1."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import wls
from statsmodels.stats.multitest import multipletests
from loguru import logger

from .config import Fig1Config


class GroupComparison:
    """Class for performing group comparisons with ANCOVA."""

    def __init__(self, config: Optional[Fig1Config] = None):
        """Initialize the group comparison analyzer.

        Args:
            config: Configuration object. If None, uses default Fig1Config.
        """
        self.config = config or Fig1Config()

    def _get_covariates(self, metric_type: str, df: pd.DataFrame) -> List[str]:
        """Get appropriate covariates for a given metric type.

        Args:
            metric_type: Type of metric (e.g., 'gm_vol', 'adc')
            df: DataFrame containing the data

        Returns:
            List of covariate column names
        """
        covariates = ["age_at_scan", "+ I(age_at_scan**2)", "sex"]
        if "vol" in metric_type and "tiv" in df.columns:
            covariates.append("tiv")
        return covariates

    def _ipw_weights(
        self,
        s: pd.Series,
        target: Optional[Dict[str, float]] = None,
        cap_pct: Optional[float] = 0.99,
    ) -> pd.Series:
        """
        Calculate inverse probability weights for a given series.

        Parameters
        ----------
        s : pd.Series
            Series for which to calculate weights
        target : Optional[Dict[str, float]], optional
            Target distribution for weighting, by default None
        cap_pct : Optional[float], optional
            Percentile cap for weights, by default 0.99

        Returns
        -------
        pd.Series
            Series of calculated weights
        """
        p_hat = s.value_counts(normalize=True)
        if target is None:
            target = {k: 1.0 / len(p_hat) for k in p_hat.index}  # equalize groups
        w = s.map(lambda g: target[g] / p_hat[g])
        if cap_pct is not None:
            cap = w.quantile(cap_pct)
            w = np.minimum(w, cap)
        return w

    def _eff_n(self, w: pd.Series) -> float:
        """
        Calculate effective sample size.

        Parameters
        ----------
        w : pd.Series
            Series of weights

        Returns
        -------
        float
            Effective sample size
        """
        w = np.asarray(w, float)
        return (w.sum() ** 2) / (w @ w)

    def _fit_weighted_model(
        self, formula: str, data: pd.DataFrame, group_col: str
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Fit a weighted least squares model to correct for group imbalance.

        Args:
            formula: Statistical formula for the model
            data: DataFrame containing the data
            group_col: Column name for grouping variable

        Returns:
            Fitted WLS model
        """
        # Calculate inverse weights based on group sizes
        w = self._ipw_weights(data[group_col])
        # Robust covariance because weights are for balancing, not variance
        return wls(formula, data=data, weights=w).fit()

    def _analyze_single_region(
        self,
        df: pd.DataFrame,
        region_value: int,
        formula: str,
        group_col: str,
        group_labels: Dict[str, str],
    ) -> Dict[str, float]:
        """Analyze a single brain region.

        Args:
            df: DataFrame containing the data
            region_value: Value of the region to analyze
            formula: Statistical formula
            group_col: Column name for grouping variable
            group_labels: Dictionary mapping group keys to labels

        Returns:
            Dictionary of statistical results
        """
        cur_df = df[df[self.config.REGION_COL] == region_value]

        try:
            model = self._fit_weighted_model(formula, cur_df, group_col)
            aov_table = sm.stats.anova_lm(model, typ=2)

            # Extract main effect statistics
            main_effect_key = f"C({group_col})"
            if main_effect_key not in aov_table.index:
                # Try alternative key formats
                possible_keys = [k for k in aov_table.index if group_col in k]
                if possible_keys:
                    main_effect_key = possible_keys[0]
                else:
                    raise KeyError(f"Could not find main effect for {group_col}")

            group_p_value = aov_table.loc[main_effect_key, "PR(>F)"]
            group_f_statistic = aov_table.loc[main_effect_key, "F"]

            # Calculate group means
            group_means = cur_df.groupby(group_col)["value"].mean().to_dict()

            # Extract model coefficients
            coef_keys = [k for k in model.params.index if group_col in k and "[T." in k]
            if coef_keys:
                coef_key = coef_keys[0]
                coefficient = model.params[coef_key]
                std_err = model.bse[coef_key]
                t_statistic = model.tvalues[coef_key]
            else:
                coefficient = std_err = t_statistic = np.nan

            result = {
                "F_statistic": group_f_statistic,
                "p_value": group_p_value,
                "coefficient": coefficient,
                "std_err": std_err,
                "t_statistic": t_statistic,
            }

            # Add group means
            for group_key, group_label in group_labels.items():
                result[f"{group_key}_Mean"] = group_means.get(group_label, np.nan)

            return result

        except Exception as e:
            logger.warning(f"Error analyzing region {region_value}: {e}")
            return {
                key: np.nan
                for key in (
                    ["F_statistic", "p_value", "coefficient", "std_err", "t_statistic"]
                    + [f"{k}_Mean" for k in group_labels.keys()]
                )
            }

    def athletes_vs_controls(
        self, data: Dict[str, pd.DataFrame], parcels: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Compare athletes vs controls across all metrics.

        Args:
            data: Dictionary mapping metric names to DataFrames
            parcels: DataFrame containing parcel information

        Returns:
            Dictionary mapping metric names to results DataFrames
        """
        logger.info("Starting athletes vs controls comparison")
        results = {}

        for metric_type, df_metric in data.items():
            logger.info(f"Analyzing metric: {metric_type}")

            covariates = self._get_covariates(metric_type, df_metric)
            temp_df = df_metric.copy()
            temp_df["target"] = temp_df["target"].astype(str)

            if temp_df.empty or len(temp_df["target"].unique()) < 2:
                logger.warning(f"Skipping {metric_type}: insufficient data")
                continue

            formula_parts = ["value ~ C(target)"] + covariates
            formula = " + ".join(formula_parts)

            metric_stats = parcels.copy()
            group_labels = {"True": "True", "False": "False"}

            for i, row in metric_stats.iterrows():
                region_result = self._analyze_single_region(
                    temp_df, row[self.config.REGION_COL], formula, "target", group_labels
                )
                for key, value in region_result.items():
                    metric_stats.loc[i, key] = value

            # Multiple comparisons correction
            metric_stats["adjusted_p_value"] = multipletests(
                metric_stats["p_value"], method="fdr_bh"
            )[1]

            results[metric_type] = metric_stats

        logger.success(f"Completed athletes vs controls comparison for {len(results)} metrics")
        return results

    def sport_vs_controls(
        self, data: Dict[str, pd.DataFrame], parcels: pd.DataFrame, sport: str
    ) -> Dict[str, pd.DataFrame]:
        """Compare specific sport vs controls across all metrics.

        Args:
            data: Dictionary mapping metric names to DataFrames
            parcels: DataFrame containing parcel information
            sport: Sport to compare (e.g., 'Climbing', 'Bjj')

        Returns:
            Dictionary mapping metric names to results DataFrames
        """
        logger.info(f"Starting {sport} vs controls comparison")
        results = {}

        for metric_type, df_metric in data.items():
            logger.info(f"Analyzing {sport} vs controls for {metric_type}")

            covariates = self._get_covariates(metric_type, df_metric)
            temp_df = df_metric.copy()

            # Filter to include only the specific sport and controls
            temp_df = temp_df.drop(
                temp_df[(temp_df["target"]) & (temp_df["group"] != sport)].index
            )
            temp_df["target"] = temp_df["target"].astype(str)

            if temp_df.empty or len(temp_df["target"].unique()) < 2:
                logger.warning(
                    f"Skipping {sport} vs controls for {metric_type}: insufficient data"
                )
                continue

            formula_parts = ["value ~ C(target)"] + covariates
            formula = " + ".join(formula_parts)

            metric_stats = parcels.copy()
            group_labels = {
                "climber": self.config.CLIMBER_GROUP_LABEL,
                "bjj": self.config.BJJ_GROUP_LABEL,
            }

            for i, row in metric_stats.iterrows():
                region_result = self._analyze_single_region(
                    temp_df, row[self.config.REGION_COL], formula, "target", group_labels
                )
                for key, value in region_result.items():
                    metric_stats.loc[i, key] = value

            # Multiple comparisons correction
            metric_stats["adjusted_p_value"] = multipletests(
                metric_stats["p_value"], method="fdr_bh"
            )[1]

            results[metric_type] = metric_stats

        logger.success(f"Completed {sport} vs controls comparison for {len(results)} metrics")
        return results

    def climbers_vs_bjj(
        self, data: Dict[str, pd.DataFrame], parcels: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Compare climbers vs BJJ athletes across all metrics.

        Args:
            data: Dictionary mapping metric names to DataFrames
            parcels: DataFrame containing parcel information

        Returns:
            Dictionary mapping metric names to results DataFrames
        """
        logger.info("Starting climbers vs BJJ comparison")
        results = {}

        for metric_type, df_metric in data.items():
            logger.info(f"Analyzing climbers vs BJJ for {metric_type}")

            covariates = self._get_covariates(metric_type, df_metric)

            # Filter to include only climbers and BJJ athletes
            temp_df = df_metric[
                (
                    df_metric["target"]
                    & df_metric["group"].isin(
                        [self.config.CLIMBER_GROUP_LABEL, self.config.BJJ_GROUP_LABEL]
                    )
                )
            ].copy()
            temp_df["group"] = temp_df["group"].astype(str)

            if temp_df.empty or len(temp_df["group"].unique()) < 2:
                logger.warning(f"Skipping climbers vs BJJ for {metric_type}: insufficient data")
                continue

            formula_parts = [
                f"value ~ C(group, Treatment(reference='{self.config.CLIMBER_GROUP_LABEL}'))"
            ] + covariates
            formula = " + ".join(formula_parts)

            metric_stats = parcels.copy()
            group_labels = {
                "climber": self.config.CLIMBER_GROUP_LABEL,
                "bjj": self.config.BJJ_GROUP_LABEL,
            }

            for i, row in metric_stats.iterrows():
                region_result = self._analyze_single_region(
                    temp_df, row[self.config.REGION_COL], formula, "group", group_labels
                )
                for key, value in region_result.items():
                    metric_stats.loc[i, key] = value

            # Multiple comparisons correction
            metric_stats["adjusted_p_value"] = multipletests(
                metric_stats["p_value"], method="fdr_bh"
            )[1]

            results[metric_type] = metric_stats

        logger.success(f"Completed climbers vs BJJ comparison for {len(results)} metrics")
        return results

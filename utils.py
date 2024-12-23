# ==========================
# Standard Library Imports
# ==========================
import json
import logging
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================
# Data Analysis & Processing
# ==========================
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ==========================
# Machine Learning Libraries
# ==========================
import optuna
import shap
import phik
import category_encoders as ce

# sklearn - Preprocessing, Models, Metrics, Feature Selection, Validation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import indexable
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import compute_class_weight, parallel_backend
from sklearn import clone

# ==========================
# Imbalanced Learning
# ==========================
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================
# Gradient Boosting Libraries
# ==========================
import lightgbm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==========================
# Plotting Libraries
# ==========================
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# GUI Library
# ==========================
import tkinter as tk

# ==========================
# Joblib for Serialization
# ==========================
import joblib


warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
os.environ["LIGHTGBM_VERBOSE"] = "-1"


RANDOM_STATE = 98
COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
EXPORT_FOLDER = "exports"
DATA_FOLDER = "data"
DATATYPE_COLOR_MAP = {
    "numeric": "#1f77b4",
    "categorical": "#ff7f0e",
}
HIGH_RISK_CATEGORIES = {
    "59": "retail_high_risk",  # Direct Marketing, Internet/Phone
    "55": "transport_high_risk",  # Airlines, Car Rentals
    "58": "bars_nightclubs",  # Drinking Places, Restaurants
    "73": "services_high_risk",  # Business/Technology Services
    "75": "gambling_gaming",  # Gambling, Gaming, Casino
    "78": "entertainment_high_risk",  # Entertainment Services
    "98": "other_high_risk",  # Other Services
    "60": "money_transfer",  # Money Transfer Services
    "54": "fuel_stores",  # Gas Stations, Fuel Dealers
    "79": "recreation_services",  # Recreation Services
}

np.random.seed(RANDOM_STATE)


class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        Initialize the MeanEncoder with columns to encode.

        Parameters
        ----------
        columns : list of str, optional
            The columns to encode. If None, all categorical columns
            are used. Defaults to None.
        """

        self.columns = columns
        self.mean_encodings = {}

    def fit(self, X, y):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns

        for col in self.columns:
            mean_encoding = y.groupby(X[col]).mean()
            self.mean_encodings[col] = mean_encoding
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, encoding in self.mean_encodings.items():
            X_encoded[col] = X_encoded[col].map(encoding)
            X_encoded[col] = X_encoded[col].fillna(encoding.mean())
        return X_encoded


class CustomTimeSeriesSplitter(BaseCrossValidator):
    def __init__(
        self, n_splits=5, test_period="30D", gap_period="30D", min_train_period="180D"
    ):
        self.n_splits = n_splits
        self.test_period = pd.Timedelta(test_period)
        self.gap_period = pd.Timedelta(gap_period)
        self.min_train_period = pd.Timedelta(min_train_period)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)  # Simply using len() instead of _num_samples
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")
        
        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            test_end = X.index[-1] - i * (self.test_period + self.gap_period)
            test_start = test_end - self.test_period
            train_end = test_start - self.gap_period
            train_start = max(
                X.index[0],  # Don't go before the first date
                train_end
                - max(self.min_train_period, test_end - X.index[0]),  # Expanding window
            )
            train_mask = (X.index > train_start) & (X.index <= train_end)
            test_mask = (X.index > test_start) & (X.index <= test_end)
            train_indices = indices[train_mask]
            test_indices = indices[test_mask]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


def get_screen_width() -> int:
    """Retrieves the screen width using a tkinter root window and returns the screen width value."""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()

    return screen_width


def set_font_size() -> dict:
    """Sets the font sizes for visualization elements based on the screen width."""
    base_font_size = round(get_screen_width() / 100, 0)
    font_sizes = {
        "font.size": base_font_size * 0.6,
        "axes.titlesize": base_font_size * 0.4,
        "axes.labelsize": base_font_size * 0.6,
        "xtick.labelsize": base_font_size * 0.4,
        "ytick.labelsize": base_font_size * 0.4,
        "legend.fontsize": base_font_size * 0.6,
        "figure.titlesize": base_font_size * 0.6,
    }

    return font_sizes


def custom_format(x: float) -> str:
    """
    Formats a given number to a string with a specific decimal precision.

    Args:
        x (float): The number to be formatted.

    Returns:
        str: The formatted number as a string. If the number is an integer, it is formatted as an integer with no decimal places.
        Otherwise, it is formatted with two decimal places.
    """
    if x == int(x):
        return "{:.0f}".format(x)
    else:
        return "{:.2f}".format(x)


def check_duplicates(df: pd.DataFrame, df_name: str) -> None:
    """
    Check for duplicate rows in a pandas DataFrame and print the results.

    Args:
        df (pandas.DataFrame): The DataFrame to check for duplicates.
        df_name (str): The name of the DataFrame for printing purposes.

    Returns:
        None
    """
    duplicate_count = df.duplicated().sum()
    print(f"DataFrame: {df_name}")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {duplicate_count}\n")
    duplicates = df[df.duplicated(keep=False)]
    sorted_duplicates = duplicates.sort_values(by=list(df.columns))
    sorted_duplicates[:10] if len(duplicates) > 0 else None


def boolean_analysis(df: pd.DataFrame, boolean_columns: List[str]) -> pd.DataFrame:
    """
    Analyze a boolean column in a DataFrame and return a DataFrame with the count, null count, true count, false count, true percentage, and false percentage.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the column to be analyzed.
    boolean_column : str
        The name of the boolean column to be analyzed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the analysis of the boolean column.
    """
    if isinstance(boolean_columns, str):
        boolean_columns = [boolean_columns]

    results = []
    for column in boolean_columns:
        analysis = {
            "column": column,
            "count": df[column].count(),
            "null_count": df[column].isnull().sum(),
            "true_count": df[column].sum(),
            "false_count": (~df[column]).sum(),
            "true_percentage": df[column].mean() * 100,
            "false_percentage": (1 - df[column].mean()) * 100,
        }
        results.append(analysis)

    return pd.DataFrame(results).set_index("column")


def datetime_analysis(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """
    Analyze datetime columns in a DataFrame and return analysis including transaction counts per month.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be analyzed.
    datetime_columns : List[str] or str
        The name(s) of the datetime column(s) to be analyzed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the analysis of the datetime columns.
    """
    if isinstance(datetime_columns, str):
        datetime_columns = [datetime_columns]

    results = []
    for column in datetime_columns:
        # Basic datetime statistics
        basic_analysis = {
            "column": column,
            "count": df[column].count(),
            "min": df[column].min(),
            "max": df[column].max(),
            "range": df[column].max() - df[column].min(),
            "mode": df[column].mode().iloc[0] if not df[column].mode().empty else None,
            "null_count": df[column].isnull().sum(),
            "unique_count": df[column].nunique(),
        }

        # Monthly transaction analysis
        monthly_counts = df.groupby(df[column].dt.to_period("M")).size()

        # Calculate average transactions per month
        avg_transactions = monthly_counts.mean()

        # Add monthly statistics
        monthly_analysis = {
            "avg_transactions_per_month": round(avg_transactions, 0),
            "max_transactions_in_month": monthly_counts.max(),
            "min_transactions_in_month": monthly_counts.min(),
            "total_months": len(monthly_counts),
            "busiest_month": monthly_counts.idxmax().strftime("%Y-%m"),
            "slowest_month": monthly_counts.idxmin().strftime("%Y-%m"),
        }

        # Combine all analyses
        combined_analysis = {**basic_analysis, **monthly_analysis}
        results.append(combined_analysis)

    # Create DataFrame and format dates
    result_df = pd.DataFrame(results).set_index("column")

    # Format datetime columns
    for col in ["min", "max"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return result_df


def missing_values(df: pd.DataFrame, missing_only=False) -> pd.DataFrame:
    """Returns a DataFrame that summarizes missing values in the train and test datasets, including column data types, sorted by column data type."""
    missing_values = round(df.isnull().sum(), 0)
    missing_values_perc = round((missing_values / len(df)) * 100, 1)
    column_data_types = df.dtypes

    missing_values = pd.DataFrame(
        {
            "Data Type": column_data_types,
            "Count #": missing_values,
            "Perc %": missing_values_perc,
        }
    )

    missing_values = missing_values.sort_values(by="Perc %", ascending=False)

    # Filter features with missing values
    if missing_only:
        missing_values = missing_values[(missing_values["Count #"] > 0)]

    return missing_values


def clean_feature_name(name: str) -> str:
    """
    Replace non-alphanumeric characters with underscores in a given string.

    Args:
        name (str): The string to clean.

    Returns:
        str: The cleaned string.
    """
    return re.sub(r"[^\w]+", "_", name)


def calculate_class_weights(y: pd.Series) -> Tuple[Dict[int, float], np.ndarray, float]:
    """
    Calculate class weights based on the distribution in y.

    :param y: Array-like, target variable
    :return: Dictionary of class weights, array of class weights, and the weight ratio
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    # Calculate the ratio of the majority class to the minority class
    weight_ratio = max(weights) / min(weights)

    return class_weights, weights, weight_ratio


def calculate_mutual_information_smote(
    df: pd.DataFrame, target_column: str, categorical_columns=None
) -> pd.DataFrame:
    """
    Calculate mutual information for features in a DataFrame and apply SMOTE to handle class imbalance.
    Categorical features are encoded using OneHotEncoder.
    Infinity values are replaced and large values are clipped.

    :param df: pandas DataFrame containing the features and target
    :param target_column: name of the target column
    :param categorical_columns: list of categorical column names (optional)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # Create a copy of X to avoid modifying the original dataframe
    X_encoded = X.copy()

    # Encode categorical columns using mean encoding
    if categorical_columns:
        mean_encoder = ce.TargetEncoder(cols=categorical_columns)
        X_encoded = mean_encoder.fit_transform(X_encoded, y)
    else:
        X_encoded = X_encoded.copy()

    # Handle infinity and large values
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    X_encoded = X_encoded.clip(
        lower=np.finfo(np.float64).min, upper=np.finfo(np.float64).max
    )

    # Calculate mutual information
    mi_scores = mutual_info_classif(X_encoded, y)

    # Create a DataFrame with mutual information scores
    mi_df = pd.DataFrame(
        {"feature": X_encoded.columns, "score": mi_scores}
    ).sort_values("score", ascending=False)

    # Identify difficult-to-classify samples (low mutual information)
    difficult_samples = mi_df[mi_df["score"] < mi_df["score"].quantile(0.25)].index

    # Ensure we have difficult samples from each class
    difficult_samples_per_class = []
    for cls in y.unique():
        class_samples = X_encoded[y == cls].iloc[difficult_samples]
        difficult_samples_per_class.append(class_samples)

    difficult_samples_balanced = pd.concat(difficult_samples_per_class)
    y_difficult_balanced = y.loc[difficult_samples_balanced.index]

    # Apply SMOTE to synthesize new samples
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        difficult_samples_balanced, y_difficult_balanced
    )

    # Combine original and synthetic samples
    X_combined = pd.concat(
        [X_encoded, pd.DataFrame(X_resampled, columns=X_encoded.columns)],
        ignore_index=True,
    )
    y_combined = pd.concat([y, pd.Series(y_resampled)], ignore_index=True)

    # Recalculate mutual information with the combined dataset
    mi_scores_combined = mutual_info_classif(X_combined, y_combined)

    # Create a DataFrame with the new mutual information scores
    mi_df_combined = pd.DataFrame(
        {"feature": X_combined.columns, "score": mi_scores_combined}
    ).sort_values("score", ascending=False)

    # Assign colors based on data type
    mi_df_combined["color"] = [
        (
            DATATYPE_COLOR_MAP["categorical"]
            if col in categorical_columns
            else DATATYPE_COLOR_MAP["numeric"]
        )
        for col in mi_df_combined["feature"]
    ]

    return mi_df_combined


def calculate_ensemble_feature_importance(
    X: pd.DataFrame, y: pd.Series, n_iterations: int = 5
) -> pd.DataFrame:
    """
    Calculate the ensemble feature importance by training multiple models and averaging their feature importances.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The target variable.
        n_iterations (int): The number of times to train each model. Defaults to 5.

    Returns:
        pd.Series: The ensemble feature importance, with the feature names as the index and the importance values as the values.
    """
    class_weights, _, weight_ratio = calculate_class_weights(y)
    models = {
        "xgboost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="auc",
            n_jobs=-1,
            scale_pos_weight=weight_ratio,
            enable_categorical=True,
        ),
        "lightgbm": lightgbm.LGBMClassifier(
            n_jobs=-1, class_weight=class_weights, verbose=-1
        ),
    }

    feature_importance_sum = {model: pd.Series(0, index=X.columns) for model in models}

    # Identify categorical columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    # Convert categorical columns to 'category' dtype
    for col in categorical_columns:
        X[col] = X[col].astype("category")

    # Cleanup column names
    X.columns = [clean_feature_name(name) for name in X.columns]

    for i in range(n_iterations):
        X_split, _, y_split, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE + i
        )

        for model_name, model in models.items():
            model.random_state = RANDOM_STATE + i

            if model_name == "xgboost":
                model.fit(X_split, y_split)
            elif model_name == "lightgbm":
                model.fit(
                    X_split, y_split, categorical_feature=categorical_columns.tolist()
                )

            importance = pd.Series(model.feature_importances_, index=X.columns)

            feature_importance_sum[model_name] += importance

    # Calculate average importance for each model
    average_importance = {
        model: importance_sum / n_iterations
        for model, importance_sum in feature_importance_sum.items()
    }

    # Calculate ensemble average importance
    ensemble_df = pd.DataFrame(average_importance).mean(axis=1).reset_index()
    ensemble_df.columns = ["feature", "score"]
    ensemble_df = ensemble_df.sort_values("score", ascending=False)
    ensemble_df["color"] = [
        (
            DATATYPE_COLOR_MAP["categorical"]
            if col in categorical_columns
            else DATATYPE_COLOR_MAP["numeric"]
        )
        for col in ensemble_df["feature"]
    ]

    return ensemble_df


def plot_feature_importances(
    df: pd.DataFrame, target_column: str, importance_method: str
) -> None:
    """
    Plot feature importances.

    Parameters:
        df (pd.DataFrame): A DataFrame with columns 'feature', 'score', and 'color'.
        target_column (str): The name of the target column.
        importance_method (str): The name of the importance method (e.g. 'Mutual Information', 'Permutation Importance').

    Returns:
        None
    """
    # Create the plot
    plt.figure(figsize=(12, 15))
    _ = sns.barplot(x="score", y="feature", data=df, palette=df["color"].tolist())
    plt.title(f"{importance_method.title()} Scores (target: {target_column})")
    plt.xlabel(f"{importance_method.title()}")
    plt.ylabel("Features")

    # Add a legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in DATATYPE_COLOR_MAP.values()
    ]
    labels = list(DATATYPE_COLOR_MAP.keys())
    plt.legend(handles, labels, title="Data Type", loc="lower right")

    plt.tight_layout()
    plt.show()


def numerical_predictor_significance_test(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    missing_strategy="drop",
    min_sample_size=30,
) -> dict:
    """
    Perform either Mann-Whitney U test and return the results.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the numerical predictor
    target (str): The name of the column containing the binary target
    missing_strategy (str): How to handle missing values. Options: 'drop', 'median_impute'
    min_sample_size (int): Minimum sample size required for each group

    Returns:
    dict: A dictionary containing the test results
    """
    # Handle missing values
    if missing_strategy == "drop":
        df = df.dropna(subset=[predictor, target])
    elif missing_strategy == "median_impute":
        df[predictor] = df[predictor].fillna(df[predictor].median())
    else:
        raise ValueError("Invalid missing_strategy. Choose 'drop' or 'median_impute'")

    # Separate the data into two groups based on the binary target
    group1 = df[df[target] == 0][predictor]
    group2 = df[df[target] == 1][predictor]

    # Check if there's enough data
    if len(group1) < min_sample_size or len(group2) < min_sample_size:
        return {
            "error": f"Insufficient data. Group sizes: {
                len(group1)}, {
                len(group2)}"
        }

    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    test_name = "Mann-Whitney U test"
    effect_size = 2 * statistic / (len(group1) * len(group2)) - 1

    results = {
        "test_name": test_name,
        "p_value": p_value,
        "statistic": statistic,
        "effect_size": effect_size,
        "group1_median": np.median(group1),
        "group2_median": np.median(group2),
        "group1_size": len(group1),
        "group2_size": len(group2),
    }

    return results


def interpret_results_numerical(
    df: pd.DataFrame, results: dict, col_name: str
) -> pd.DataFrame:
    """Interpret the results of the non-parametric test and store them in a DataFrame"""
    data = {
        "Column": col_name,
        "Test Name": [results["test_name"]],
        "P-value": [round(results["p_value"], 6)],
        "Test Statistic": [round(results["statistic"], 2)],
        "Effect Size": [round(results["effect_size"], 4)],
        "Median Group 0": [results["group1_median"]],
        "Median Group 1": [results["group2_median"]],
        "Significance": [
            (
                "Statistically significant"
                if results["p_value"] < 0.05
                else "Not statistically significant"
            )
        ],
        "Effect Magnitude": [],
    }

    if abs(results["effect_size"]) < 0.2:
        effect_magnitude = "negligible"
    elif abs(results["effect_size"]) < 0.5:
        effect_magnitude = "small"
    elif abs(results["effect_size"]) < 0.8:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"

    data["Effect Magnitude"].append(effect_magnitude)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def draw_predictor_numerical_plots(
    df: pd.DataFrame, predictor: str, target: str, hist_type="histogram"
) -> None:
    """
    Draws two plots to visualize the frequency counts and box plot of the distribution of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.
        hist_type (str): The type of plot to draw. Can be 'histogram' or 'kde'. Defaults to 'histogram'.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_width / 5))

    # Chart 1: Box Plot
    sns.boxplot(
        data=df,
        x=target,
        y=predictor,
        hue=target,
        palette=COLOR_PALETTE,
        saturation=0.75,
        legend=False,
        ax=ax1,
    )
    ax1.set_title(f"Distribution of {predictor} by {target}")
    ax1.set_xlabel(f"{target}")
    ax1.set_ylabel(f"{predictor}")

    # Chart 2: Histogram
    if hist_type == "kde":
        sns.kdeplot(
            data=df,
            x=predictor,
            hue=target,
            multiple="stack",
            palette=COLOR_PALETTE,
            ax=ax2,
        )
    else:
        sns.histplot(
            data=df,
            x=predictor,
            hue=target,
            multiple="stack",
            palette=COLOR_PALETTE,
            ax=ax2,
        )
    ax2.set_title(f"Frequency Distribution of {predictor.title()} by {target.title()}")
    ax2.set_xlabel(f"{predictor.title()}")
    ax2.set_ylabel("Count")

    plt.show()
    plt.close(fig)


def categorical_predictor_significance_test(
    df: pd.DataFrame, predictor: str, target: str, missing_strategy="drop"
) -> dict:
    """
    Performs chi-squared test for independence between a categorical predictor and binary target.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the categorical predictor
    target (str): The name of the column containing the binary target
    missing_strategy (str): How to handle missing values. Options: 'drop', 'most_frequent'

    Returns:
    dict: A dictionary containing the test results
    """
    # Handle missing values
    if missing_strategy == "drop":
        df = df.dropna(subset=[predictor, target])
    elif missing_strategy == "most_frequent":
        df[predictor] = df[predictor].fillna(df[predictor].mode()[0])
    else:
        raise ValueError("Invalid missing_strategy. Choose 'drop' or 'most_frequent'")

    # Create a contingency table
    contingency_table = pd.crosstab(df[predictor], df[target])

    # Perform chi-squared test
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)

    # Calculate Cramer's V for effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    results = {
        "test_name": "Chi-squared test",
        "p_value": p_value,
        "chi2_statistic": chi2,
        "degrees_of_freedom": dof,
        "effect_size": cramer_v,
        "contingency_table": contingency_table,
    }

    return results


def interpret_results_categorical(
    df: pd.DataFrame, results: dict, col_name: str
) -> pd.DataFrame:
    """Interpret the results of the chi-squared test. Store the summary in a DataFrame"""
    data = {
        "Column": col_name,
        "Test Name": [results["test_name"]],
        "P-value": [round(results["p_value"], 6)],
        "Chi-squared statistic": [round(results["chi2_statistic"], 2)],
        "Degrees of freedom": [round(results["degrees_of_freedom"], 4)],
        "Effect size (Cramer's V)": [round(results["effect_size"], 4)],
        "Significance": [
            (
                "Statistically significant"
                if results["p_value"] < 0.05
                else "Not statistically significant"
            )
        ],
        "Effect Magnitude": [],
    }

    # Interpret effect size (Cramer's V)
    if results["effect_size"] < 0.1:
        effect_magnitude = "negligible"
    elif results["effect_size"] < 0.3:
        effect_magnitude = "small"
    elif results["effect_size"] < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"
    data["Effect Magnitude"] = effect_magnitude

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def draw_predictor_categorical_plots(
    df: pd.DataFrame, predictor: str, target: str
) -> None:
    """
    Draws two plots to visualize the frequency counts and proportions of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_width / 5))

    # Chart 1: Frequencies
    sns.histplot(
        data=df,
        x=predictor,
        hue=target,
        multiple="stack",
        palette=COLOR_PALETTE,
        ax=ax1,
    )
    ax1.set_title(f"Frequency Counts of {predictor.title()} by {target.title()}")
    ax1.set_xlabel(f"{predictor.title()}")
    ax1.set_ylabel("Count")

    # Chart 2: Proportions
    sns.histplot(
        data=df,
        x=predictor,
        hue=target,
        multiple="fill",
        discrete=True,
        palette=COLOR_PALETTE,
        shrink=1,
        ax=ax2,
    )
    ax2.set_title(f"Proportion of {target.title()} by {predictor.title()}")
    ax2.set_xlabel(f"{predictor.title()}")
    ax2.set_ylabel("Proportion")
    plt.show()


def find_and_analyze_infinite_values(df: pd.DataFrame) -> None:
    """
    Check if a pandas Series contains infinite values.

    Args:
        x (pd.Series): The input Series

    Returns:
        pd.Series: A Series of the same shape as the input, with True values indicating infinite values and False otherwise
    """

    def _is_infinite(x):
        if pd.api.types.is_numeric_dtype(x):
            return np.isinf(x)
        else:
            return pd.Series(False, index=x.index)

    # Find rows with infinite values
    infinite_mask = df.apply(_is_infinite)
    infinite_rows = df[infinite_mask.any(axis=1)]

    # Count infinite values per feature
    infinite_counts = infinite_mask.sum()
    features_with_infinites = infinite_counts[infinite_counts > 0]

    # Collect information about infinite values
    infinite_info = {
        "rows": infinite_rows,
        "features": features_with_infinites.to_dict(),
        "total_infinites": infinite_mask.sum().sum(),
    }

    # Print results
    if infinite_rows.empty:
        print("No rows with infinite values found.")
    else:
        print(f"Found {len(infinite_rows)} row(s) with infinite values.")
        print("\nFeatures with infinite values:")
        for feature, count in features_with_infinites.items():
            print(f"  {feature}: {count} infinite value(s)")
        print(f"\nTotal number of infinite values: {infinite_info['total_infinites']}")


def phik_matrix(
    df: pd.DataFrame,
    numerical_columns: list,
    target_column: str,
    feature_importances: pd.DataFrame,
) -> tuple:
    """
    Calculates the Phi_k correlation coefficient matrix for the given DataFrame and columns,
    and returns the top 10 largest phik coefficients between the target feature and other features,
    as well as the top 10 interactions between any features.

    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_columns (list): List of numerical columns.
        target_column (str): Name of the target column. Defaults to 'TARGET'.
        feature_importances (pd.DataFrame): DataFrame containing feature importances

    Returns:
        tuple: (DataFrame of top 10 target correlations, DataFrame of top 10 overall interactions)
    """
    # Calculate Phi_k correlation matrix
    corr_matrix = df.phik_matrix(interval_cols=numerical_columns)

    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(get_screen_width() / 100, 14))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Phi_k Correlation Heatmap")
    plt.show()

    # Extract correlations with the target feature
    target_correlations = corr_matrix[target_column].sort_values(ascending=False)

    # Remove self-correlation (correlation of TARGET with itself)
    target_correlations = target_correlations[
        target_correlations.index != target_column
    ]

    # Get top 10 correlations with target
    top_10_target = target_correlations.head(10)

    # Create a DataFrame with feature names and their correlations to target
    target_df = pd.DataFrame(
        {"Feature": top_10_target.index, "Phik Coefficient": top_10_target.values}
    )

    # Retrieve top interactions between features and their FI scores
    corr_df = corr_matrix.unstack().reset_index()
    corr_df.columns = ["Feature1", "Feature2", "Phik Coefficient"]
    corr_df = corr_df[corr_df["Feature1"] < corr_df["Feature2"]]
    top_interactions = corr_df[corr_df["Phik Coefficient"] > 0.8].sort_values(
        "Phik Coefficient", ascending=False
    )

    def _get_importance(feature):
        return feature_importances[feature_importances["feature"] == feature][
            "score"
        ].values[0]

    top_interactions["Feature1 Score"] = top_interactions["Feature1"].apply(
        _get_importance
    )
    top_interactions["Feature2 Score"] = top_interactions["Feature2"].apply(
        _get_importance
    )

    return target_df, top_interactions


def check_zip_country_match(zip_code: str, country_code: int) -> int:
    """Check if zipcode format matches the expected format for the country."""
    if pd.isna(zip_code) or zip_code in ["Unknown", "0", "...."]:
        return 0

    # UK (826) postcodes typically have 2-4 characters, then a number, then 2 characters
    if country_code == 826:
        return int(bool(re.match(r"^[A-Z]{1,2}[0-9R][0-9A-Z]?", str(zip_code))))

    # Default to 1 if we don't have specific validation for the country
    return 1


def create_non_leaking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features that don't depend on time series and won't cause leakage."""

    # Ensure the dataframe is sorted by time
    df = df.sort_index()

    # Add MCC risk features
    def _create_mcc_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        """Creates MCC risk-related features based on historical patterns."""

        # MCC category grouping (based on first two digits)
        df["mccCategory"] = df["mcc"].astype(str).str[:2]

        df["isHighRiskCategory"] = (
            df["mccCategory"].map(lambda x: x in HIGH_RISK_CATEGORIES).astype(bool)
        )

        return df

    df = _create_mcc_risk_features(df)

    # Basic POS entry mode flags (fixed padding)
    df["isCardPresent"] = (
        df["posEntryMode"]
        .astype(str)
        .str.zfill(2)
        .isin(["05", "07", "90", "91"])
        .astype(bool)
    )
    df["isFallback"] = (
        df["posEntryMode"].astype(str).str.zfill(2).isin(["80"]).astype(bool)
    )
    df["isEcommerce"] = (
        df["posEntryMode"].astype(str).str.zfill(2).isin(["81"]).astype(bool)
    )
    df["isManualEntry"] = (
        df["posEntryMode"].astype(str).str.zfill(2).isin(["01"]).astype(bool)
    )

    # Amount Pattern Features
    df["isRoundAmount"] = (df["transactionAmount"] % 1 == 0).astype(bool)
    df["cashUtilizationRate"] = df["transactionAmount"] / df["availableCash"]

    # Time-based features (keep only what's needed)
    df["hourOfDay"] = df.index.hour
    df["isWeekend"] = df.index.dayofweek.isin([5, 6]).astype(bool)
    df["isLateNight"] = df["hourOfDay"].isin([23, 0, 1, 2, 3, 4]).astype(bool)

    # Cyclic encoding for time features
    df["hourOfDay_sin"] = np.sin(df["hourOfDay"] * (2 * np.pi / 24))
    df["hourOfDay_cos"] = np.cos(df["hourOfDay"] * (2 * np.pi / 24))

    # Geographical features
    df["isInternational"] = (
        df["merchantCountry"] != df["merchantCountry"].mode().iloc[0]
    ).astype(bool)

    # Zipcode Features
    df["zipMatchesCountry"] = df.apply(
        lambda row: check_zip_country_match(row["merchantZip"], row["merchantCountry"]),
        axis=1,
    ).astype(bool)

    return df


def create_time_dependent_features(
    df: pd.DataFrame, window: str = "30D"
) -> pd.DataFrame:
    """Creates time-dependent features, respecting the train/test split to prevent leakage."""

    # Ensure the dataframe is sorted by time
    df = df.sort_index()

    # Transaction pattern features
    df["timeSinceLastTransaction"] = (
        df.groupby("accountNumber")
        .apply(lambda x: x.index.to_series().diff().dt.total_seconds() / 3600)
        .reset_index(level=0, drop=True)
    )
    df["timeSinceLastTransaction"] = df["timeSinceLastTransaction"].fillna(
        8760
    )  # 1 year in hours

    # POS Entry Mode Pattern Features
    df["unusualEntryMode"] = df.groupby("accountNumber")["posEntryMode"].transform(
        lambda x: (x != x.mode().iloc[0]).astype(bool)
    )
    df["entryModeChanged"] = df.groupby("accountNumber")["posEntryMode"].transform(
        lambda x: (x.astype(str) != x.astype(str).shift(1)).astype(bool)
    )

    # Amount-based features
    df["runningAvgTransactionAmount"] = (
        df.groupby("accountNumber")["transactionAmount"]
        .apply(lambda x: x.cumsum() / (np.arange(len(x)) + 1))
        .reset_index(level=0, drop=True)
    )

    # Merchant-based features
    df["merchantAvgTransactionAmount"] = (
        df.groupby("merchantId")["transactionAmount"]
        .apply(lambda x: x.cumsum() / (np.arange(len(x)) + 1))
        .reset_index(level=0, drop=True)
    )

    df["isNewMerchant"] = df.groupby("accountNumber")["merchantId"].transform(
        lambda x: (~x.duplicated()).astype(int)
    )
    df["uniqueMerchantCount"] = df.groupby("accountNumber").cumcount().astype("int32")

    # MCC-based features
    df["mcc"] = df["mcc"].astype(str)
    mcc_spending = df.groupby("mcc")["transactionAmount"].cumsum()
    total_spending = df.groupby("accountNumber")["transactionAmount"].cumsum()
    df["spendingPercentageByMCC"] = np.where(
        total_spending > 0, (mcc_spending / total_spending) * 100, 0
    )
    df["mccDiversity"] = df.groupby("accountNumber")["mcc"].transform(
        "nunique"
    ) / df.groupby("accountNumber")["eventId"].transform("count")

    # Pattern change detection
    df["avgAmountChange"] = df.groupby("accountNumber")[
        "transactionAmount"
    ].pct_change()
    df["avgAmountChange"] = (
        df["avgAmountChange"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    df["freqChange"] = (
        df.groupby("accountNumber")["timeSinceLastTransaction"].pct_change().fillna(0)
    )

    # Quick succession transactions (new)
    time_diff = df.groupby("accountNumber")["transactionTime"].diff().dt.total_seconds()
    df["isQuickSuccession"] = (time_diff < 300).fillna(False).astype(bool)
    df["quickSuccessionCount_1H"] = (
        df.groupby("accountNumber")
        .apply(
            lambda x: pd.Series(df["isQuickSuccession"][x.index])
            .rolling("1H", closed="left")
            .sum()
        )
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Cash Flow Analysis
    def calculate_depletion_rate(group):
        """
        Calculate the cash depletion rate over time.

        Parameters
        ----------
        group : pandas.DataFrame
            A group of transactions for a single account.

        Returns
        -------
        pandas.Series
            A series of cash depletion rates, where the rate is the cumulative sum of
            transaction amounts divided by the initial cash balance. If the initial
            cash balance is 0, the rate is set to 1.0. The rate is clipped to a maximum
            value of 1.0 to prevent over-depletion.

        Notes
        -----
        This function assumes that the group is sorted by transaction time.
        """
        cumsum = group["transactionAmount"].cumsum()
        initial_cash = group["availableCash"].iloc[0]
        rate = (
            cumsum / initial_cash
            if initial_cash != 0
            else pd.Series(1, index=group.index)
        )
        return rate.clip(upper=1)

    df["cashDepletionRate"] = (
        df.groupby("accountNumber")
        .apply(calculate_depletion_rate)
        .reset_index(level=0, drop=True)
    )

    # Time window aggregate features
    for col in ["transactionAmount", "availableCash"]:
        df[f"{col}_mean_7D"] = (
            df.groupby("accountNumber")[col]
            .rolling("7D", closed="left")
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"{col}_std_7D"] = (
            df.groupby("accountNumber")[col]
            .rolling("7D", closed="left")
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # Fraud Pressure
    df["fraudPressure_1H"] = (
        df.groupby("accountNumber")
        .apply(
            lambda x: x["eventId"].rolling("1H", closed="left").count()
            / x["eventId"].expanding().count()
        )
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df["fraudPressure_1D"] = (
        df.groupby("accountNumber")
        .apply(
            lambda x: x["eventId"].rolling("1D", closed="left").count()
            / x["eventId"].expanding().count()
        )
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Additional time windows
    for window in ["1H", "1D", "7D"]:
        df[f"transactionCount_{window}"] = (
            df.groupby("accountNumber")["eventId"]
            .rolling(window, closed="left")
            .count()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"transactionAmount_{window}"] = (
            df.groupby("accountNumber")["transactionAmount"]
            .rolling(window, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"unique_entry_modes_{window}"] = (
            df.groupby("accountNumber")["posEntryMode"]
            .rolling(window, closed="left")
            .agg(lambda x: x.nunique())
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # Risk combination features
    high_amount_threshold = df["transactionAmount"].quantile(0.95)
    df["highRiskCombo"] = (
        ((df["isManualEntry"] == 1) & (df["transactionAmount"] > high_amount_threshold))
        | ((df["isFallback"] == 1) & (df["isInternational"] == 1))
        | ((df["isEcommerce"] == 1) & (df["unusualEntryMode"] == 1))
    ).astype(bool)

    # Drop unneeded features
    df = df.drop(columns=["availableCash"], axis=1)

    return df


def create_features(df: pd.DataFrame, window: str = "30D") -> pd.DataFrame:
    """Main function to create all features, both non-leaking and time-dependent."""
    df = create_non_leaking_features(df)
    df = create_time_dependent_features(df, window)
    return df


def visualize_performance(
    y: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series, model_name: str
) -> None:
    """
    Visualizes the model performance
    Args:
        y (ndarray): True labels
        y_pred (ndarray): Predicted labels
        y_pred_proba (ndarray): Predicted probabilities
        model_name (str): Name of the model
    Returns:
        None
    """
    # 3 subplots
    _, ax = plt.subplots(1, 3, figsize=(20, 6))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm / cm.sum()
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", ax=ax[0])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title(f"Confusion Matrix - {model_name}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    ax[1].plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax[1].plot([0, 1], [0, 1], "k--")  # Diagonal line
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("Receiver Operating Characteristic (ROC) Curve")
    ax[1].legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    auprc = auc(recall, precision)

    # Fill the area under the Precision-Recall curve
    ax[2].fill_between(recall, precision, alpha=0.2, color="b")
    ax[2].plot(recall, precision, color="b")

    # Add AUPRC score to the plot
    ax[2].text(
        0.05,
        0.95,
        f"AUPRC = {auprc:.2f}",
        transform=ax[2].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax[2].set_xlabel("Recall")
    ax[2].set_ylabel("Precision")
    ax[2].set_title("Precision-Recall Curve")
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.show()


def prepare_dataset(df: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset by merging the labels DataFrame and adding a 'fraud' column, and by setting the transactionTime column as the index.

    Args:
        df (pd.DataFrame): Main DataFrame
        df_labels (pd.DataFrame): Labels DataFrame

    Returns:
        pd.DataFrame: Prepared DataFrame
    """
    df.merge(df_labels, how="left", on="eventId")
    df["fraud"] = df["eventId"].isin(df_labels["eventId"])

    # Transaction time indexing
    df["transactionTime"] = (
        pd.to_datetime(df["transactionTime"]).dt.tz_convert("UTC").dt.tz_localize(None)
    )
    df = df.sort_values("transactionTime")
    cc = df.groupby("transactionTime").cumcount()
    df.index = df["transactionTime"] + pd.to_timedelta(cc, unit="ms")

    return df


def get_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    col_dtypes = {}
    for col in df.columns:
        col_dtypes[col] = df[col].dtype
    col_dtypes = {
        k: v for k, v in sorted(col_dtypes.items(), key=lambda item: str(item[1]))
    }
    df_dtypes = pd.DataFrame(col_dtypes, index=["dtype"]).T

    return df_dtypes


class RandomWalkOversampler(BaseEstimator):
    """
    Random Walk Oversampling (RWO) for imbalanced datasets.

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of nearest neighbors to consider for the random walk
    sampling_strategy : float or str, default='auto'
        If float, specifies the ratio of minority to majority class
        If 'auto', will make classes balanced
    random_state : int or None, default=None
        Controls the randomization
    step_size : float, default=0.5
        Controls how far along the random walk path to generate samples
        Values closer to 0 stay closer to original samples
        Values closer to 1 allow more exploration
    """

    def __init__(
        self, n_neighbors=5, sampling_strategy="auto", random_state=None, step_size=0.5
    ):
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.step_size = step_size

    def _calculate_num_samples(self, y):
        """Calculate number of samples to generate."""
        target_stats = Counter(y)
        if self.sampling_strategy == "auto":
            # Make the dataset balanced
            max_samples = max(target_stats.values())
            return {label: max_samples - count for label, count in target_stats.items()}
        else:
            # Use provided ratio
            majority_class = max(target_stats.items(), key=lambda x: x[1])[0]
            majority_count = target_stats[majority_class]
            return {
                label: int(majority_count * self.sampling_strategy) - count
                for label, count in target_stats.items()
                if count < majority_count
            }

    def _generate_random_walk_samples(self, X, num_samples):
        """Generate synthetic samples using random walk."""
        if num_samples <= 0:
            return np.array([])

        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)

        synthetic_samples = []
        for _ in range(num_samples):
            # Randomly select a seed point
            idx = rng.randint(0, X.shape[0])

            # Randomly select one of its neighbors
            neighbor_idx = indices[idx, rng.randint(1, self.n_neighbors + 1)]

            # Perform random walk
            direction = X[neighbor_idx] - X[idx]
            step = self.step_size * rng.uniform(0, 1)
            synthetic_sample = X[idx] + step * direction

            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)

    def fit_resample(self, X, y):
        """
        Resample the dataset using Random Walk Oversampling.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        X_resampled : array-like of shape (n_samples_new, n_features)
            Resampled training data
        y_resampled : array-like of shape (n_samples_new,)
            Resampled target values
        """
        X = np.array(X)
        y = np.array(y)

        # Calculate number of samples needed for each class
        sampling_strategy = self._calculate_num_samples(y)

        # Generate synthetic samples for each minority class
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_label, num_samples in sampling_strategy.items():
            if num_samples > 0:
                class_indices = np.where(y == class_label)[0]
                X_class = X[class_indices]

                synthetic_samples = self._generate_random_walk_samples(
                    X_class, num_samples
                )

                X_resampled = np.vstack((X_resampled, synthetic_samples))
                y_resampled = np.hstack(
                    (y_resampled, np.full(num_samples, class_label))
                )

        return X_resampled, y_resampled


def apply_class_weights(model: object, class_weight_dict: dict) -> object:
    """
    Set class weights for the given model, if supported by the model.

    Parameters:
    model : object
        Model instance to set class weights for
    class_weight_dict : dict
        Dictionary with class labels as keys and their corresponding weights as values

    Returns:
    object
        Model instance with class weights set
    """
    try:
        if isinstance(model, IsolationForest):
            # Isolation Forest doesn't use class weights - skip it
            return model

        elif isinstance(model, lightgbm.LGBMClassifier):
            # Create new instance with same parameters
            params = model.get_params()
            params["class_weight"] = class_weight_dict
            return lightgbm.LGBMClassifier(**params)

        elif isinstance(model, CatBoostClassifier):
            # Create new instance with same parameters
            params = model.get_params()
            params["class_weights"] = class_weight_dict
            return CatBoostClassifier(**params)

        elif isinstance(model, XGBClassifier):
            # Create new instance with same parameters
            params = model.get_params()
            params["scale_pos_weight"] = class_weight_dict[1] / class_weight_dict[0]
            return XGBClassifier(**params)

        elif isinstance(
            model, (LogisticRegression, MLPClassifier, BalancedRandomForestClassifier)
        ):
            model.set_params(class_weight=class_weight_dict)
            return model

        else:
            warnings.warn(
                f"Model {type(model).__name__} might not support class weights properly"
            )
            return model

    except Exception as e:
        warnings.warn(
            f"Failed to set class weights for {type(model).__name__}: {str(e)}"
        )
        return model


def visualize_folds(
    X: pd.DataFrame, y: pd.Series, splitter: CustomTimeSeriesSplitter
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Create a compact visualization of fold statistics and timeline using Plotly.

    Parameters:
    -----------
    X : pd.DataFrame
        The feature dataset with DatetimeIndex
    y : pd.Series
        The target variable (1 for fraud, 0 for non-fraud)
    splitter : CustomTimeSeriesSplitter
        The initialized splitter object

    Returns:
    --------
    pd.DataFrame
        Summary statistics for each fold
    plotly.graph_objects.Figure
        Interactive timeline visualization
    """
    # Create summary DataFrame
    summary_data = []
    timeline_data = []

    colors = {
        "Train": "rgb(46, 137, 205)",
        "Gap": "rgb(180, 180, 180)",
        "Test": "rgb(231, 76, 60)",
    }

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Summary statistics
        fold_data = {
            "Fold": f"Fold {fold_idx}",
            "Train Period": f'{X_train.index.min().strftime("%Y-%m-%d")} to {X_train.index.max().strftime("%Y-%m-%d")}',
            "Test Period": f'{X_test.index.min().strftime("%Y-%m-%d")} to {X_test.index.max().strftime("%Y-%m-%d")}',
            "Train Samples": f"{len(y_train):,}",
            "Test Samples": f"{len(y_test):,}",
            "Train Fraud %": f"{(y_train.sum() / len(y_train) * 100):.2f}%",
            "Test Fraud %": f"{(y_test.sum() / len(y_test) * 100):.2f}%",
        }
        summary_data.append(fold_data)

        # Timeline data
        train_end = X_train.index.max()
        test_start = X_test.index.min()

        # Add training period
        timeline_data.append(
            {
                "Task": f"Fold {fold_idx}",
                "Start": X_train.index.min(),
                "Finish": train_end,
                "Type": "Train",
            }
        )

        # Add gap period
        timeline_data.append(
            {
                "Task": f"Fold {fold_idx}",
                "Start": train_end,
                "Finish": test_start,
                "Type": "Gap",
            }
        )

        # Add test period
        timeline_data.append(
            {
                "Task": f"Fold {fold_idx}",
                "Start": test_start,
                "Finish": X_test.index.max(),
                "Type": "Test",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    timeline_df = pd.DataFrame(timeline_data)

    # Create Plotly timeline
    fig = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Type",
        color_discrete_map=colors,
        title="Time Series Fold Splits",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="",
        height=50 + (50 * splitter.n_splits),  # Adjust height based on number of folds
        showlegend=True,
        legend_title_text="Period Type",
    )

    return summary_df, fig


def summarize_results(
    model_name: str,
    df_performance_auprc: pd.DataFrame,
    results_df: pd.DataFrame,
    column_name: str = "Score",
) -> pd.DataFrame:
    """
    Update and return a DataFrame with the aggregated AUPRC score for a specific model.

    Args:
        model_name (str): The name of the model whose performance is being summarized.
        df_performance_auprc (pd.DataFrame): DataFrame containing performance metrics for various models.
        results_df (pd.DataFrame): DataFrame with true labels and predicted probabilities.
        column_name (str, optional): The column name in df_performance_auprc where the AUPRC score will be updated. Defaults to "Score".

    Returns:
        pd.DataFrame: Updated DataFrame with the aggregated AUPRC score for the specified model.
    """
    precision, recall, _ = precision_recall_curve(
        results_df["true"], results_df["pred_proba"]
    )

    aggregated_auprc = auc(recall, precision)

    df_performance_auprc.loc[
        df_performance_auprc["Model"] == model_name, column_name
    ] = aggregated_auprc

    return df_performance_auprc


def get_scorer(metric: str):
    """
    Returns the appropriate scorer function for the given metric.
    """
    if metric == "f1":
        return make_scorer(f1_score)
    elif metric == "auprc":
        return make_scorer(average_precision_score)


def preprocess_fold_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    fold_idx: int,
    cols_remove: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper function to preprocess each fold's data consistently."""
    try:
        # Create custom features
        X_train = create_features(X_train)
        X_test = create_features(X_test)

        # Drop specified columns
        X_train = X_train.drop(columns=cols_remove)
        X_test = X_test.drop(columns=cols_remove)

        # Initialize encoded DataFrames with proper indices
        X_train_encoded = pd.DataFrame(index=X_train.index)
        X_test_encoded = pd.DataFrame(index=X_test.index)

        # Process numerical features first
        numerical_columns = X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            X_train_encoded = pd.concat(
                [X_train_encoded, X_train[numerical_columns]], axis=1
            )
            X_test_encoded = pd.concat(
                [X_test_encoded, X_test[numerical_columns]], axis=1
            )

        # Process categorical features
        categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns
        low_cardinality = [
            col for col in categorical_columns if X_train[col].nunique() < 10
        ]
        high_cardinality = [
            col for col in categorical_columns if X_train[col].nunique() >= 10
        ]

        # Process low cardinality features
        if low_cardinality:
            onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            onehot_encoded_train = onehot.fit_transform(X_train[low_cardinality])
            onehot_encoded_test = onehot.transform(X_test[low_cardinality])
            onehot_columns = onehot.get_feature_names_out(low_cardinality)

            X_train_encoded = pd.concat(
                [
                    X_train_encoded,
                    pd.DataFrame(
                        onehot_encoded_train,
                        columns=onehot_columns,
                        index=X_train.index,
                    ),
                ],
                axis=1,
            )
            X_test_encoded = pd.concat(
                [
                    X_test_encoded,
                    pd.DataFrame(
                        onehot_encoded_test, columns=onehot_columns, index=X_test.index
                    ),
                ],
                axis=1,
            )

        # Process high cardinality features
        if high_cardinality:
            mean_encoder = MeanEncoder(columns=high_cardinality)
            mean_encoded_train = mean_encoder.fit_transform(X_train, y_train)
            mean_encoded_test = mean_encoder.transform(X_test)

            X_train_encoded = pd.concat(
                [X_train_encoded, mean_encoded_train[high_cardinality]], axis=1
            )
            X_test_encoded = pd.concat(
                [X_test_encoded, mean_encoded_test[high_cardinality]], axis=1
            )

        return X_train_encoded, X_test_encoded

    except Exception as e:
        print(f"Preprocessing error in fold {fold_idx}: {str(e)}")
        raise


def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    metric: str,
    cv_splitter: CustomTimeSeriesSplitter,
    cols_remove=None,
):
    best_score = float("-inf")
    best_model = None

    def _create_model_params(X, trial, model_class):
        n_samples, n_features = X.shape
        n_jobs = os.cpu_count()
        match model_class:
            case "LightGBM":
                # Calculate data-dependent ranges
                max_leaves = int(0.5 * n_samples)  # Limit leaves to half the dataset size
                min_data_in_leaf = max(1, int(n_samples * 0.01))  # 1% of data minimum per leaf
                
                return {
                    # Tree Structure Parameters
                    "num_leaves": trial.suggest_int("num_leaves", 20, min(128, max_leaves)),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 
                        min_data_in_leaf,
                        max(20, int(n_samples * 0.05))  # 5% of data maximum
                    ),
                    
                    # Sampling Parameters
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    
                    # Learning Parameters
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "n_estimators": 1000,  # Fixed value as recommended
                    
                    # Regularization Parameters
                    "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1.0, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    
                    # Class Imbalance
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0),
                    
                    # Model Type
                    "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                    "n_jobs": n_jobs,
                }
                
            case "Balanced Random Forest":
                # Calculate limits based on dataset size
                max_samples_split = min(100, int(0.1 * n_samples))  # Either 100 or 10% of data, whichever is smaller
                max_samples_leaf = min(50, int(0.05 * n_samples))   # Either 50 or 5% of data, whichever is smaller
                
                return {
                    # Tree Structure Parameters
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 
                        2,  # Always at least 2
                        max_samples_split
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 
                        1,  # Always at least 1
                        max_samples_leaf
                    ),
                    
                    # Feature Sampling
                    "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                    
                    # Bootstrapping
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    
                    # Fixed Parameters
                    "n_estimators": 500,
                    "class_weight": "balanced",
                    "n_jobs": n_jobs,
                }
                
            case "Neural Network":
                # Calculate dynamic layer sizes based on feature count
                n_units_first = min(512, max(64, 2 * n_features))  # Between 64 and 512, based on features
                layer_sizes = [
                    (n_units_first,),
                    (n_units_first, n_units_first // 2),
                    (n_units_first, n_units_first // 2, n_units_first // 4),
                ]
                
                return {
                    "hidden_layer_sizes": trial.suggest_categorical(
                        "hidden_layer_sizes", layer_sizes
                    ),
                    "activation": trial.suggest_categorical(
                        "activation", ["relu", "tanh"]  # Removed logistic for better performance
                    ),
                    "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),  # Wider range
                    "learning_rate_init": trial.suggest_float(
                        "learning_rate_init", 1e-4, 1e-1, log=True  # Extended upper bound
                    ),
                    "batch_size": trial.suggest_int(
                        "batch_size", 
                        32, 
                        min(512, n_samples // 10),  # Dynamic based on dataset size
                        log=True
                    ),
                    "learning_rate": "adaptive",  # Fixed to adaptive for better convergence
                    "max_iter": 2000,
                    "early_stopping": True,
                    "validation_fraction": 0.15,  # Reduced from 0.2 to save more data for training
                    "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 30),  # Adjusted range
                    "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
                }
                
            case "Logistic Regression":
                # Define penalty and solver combinations that are guaranteed to work
                penalty_solver_pairs = [
                    ("l2", "lbfgs"),     # Fast for l2
                    ("l1", "saga"),      # Required for l1
                    ("elasticnet", "saga")  # Required for elasticnet
                ]
                
                # Randomly select a valid penalty-solver pair
                penalty, solver = penalty_solver_pairs[trial.suggest_int("penalty_solver_idx", 0, len(penalty_solver_pairs)-1)]
                
                params = {
                    "C": trial.suggest_float("C", 1e-4, 100.0, log=True),  # Extended range
                    "penalty": penalty,
                    "solver": solver,
                    "max_iter": 5000,    # Increased for better convergence
                    "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
                    "class_weight": "balanced",
                    "n_jobs": n_jobs if solver in ["saga"] else None,  # Only saga supports parallel processing
                }
                
                # Only add l1_ratio if penalty is elasticnet
                if penalty == "elasticnet":
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
                    
                return params

            case _:
                raise ValueError(f"Unknown model class: {model_class}")

    def _evaluate_predictions(y_test_folds, y_pred_proba_folds, metric):
        """
        Helper function to evaluate predictions consistently by concatenating all folds first.

        Args:
            y_test_folds: List of arrays containing true labels for each fold
            y_pred_proba_folds: List of arrays containing predicted probabilities for each fold
            metric: Metric to evaluate ('AUPRC' or 'F1')
        """
        # Ensure all inputs are numpy arrays and concatenate
        y_test_combined = np.concatenate([np.array(y) for y in y_test_folds])
        y_pred_proba_combined = np.concatenate(
            [np.array(y) for y in y_pred_proba_folds]
        )

        if metric == "AUPRC":
            precision, recall, _ = precision_recall_curve(
                y_test_combined, y_pred_proba_combined
            )
            return auc(recall, precision)
        elif metric == "F1":
            y_pred = (y_pred_proba_combined > 0.5).astype(int)
            return f1_score(y_test_combined, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def objective(trial):
        nonlocal best_score, best_model

        try:
            # Get parameters and create model
            params = _create_model_params(X, trial, model_name)
            match model_name:
                case "Neural Network":
                    model = ImbPipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("mlpc", MLPClassifier(random_state=RANDOM_STATE, **params)),
                        ]
                    )

                case "Logistic Regression":
                    model = ImbPipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("lr",LogisticRegression(random_state=RANDOM_STATE, **params)),
                        ]
                    )

                case "LightGBM":
                    model = ImbPipeline(
                        [
                            (
                                "lgbm",
                                lightgbm.LGBMClassifier(
                                    random_state=RANDOM_STATE,
                                    **params,
                                    callbacks=[
                                        lightgbm.early_stopping(
                                            stopping_rounds=50,
                                            first_metric_only=True,
                                            verbose=False,
                                        )
                                    ],
                                ),
                            ),
                        ]
                    )

                case "Balanced Random Forest":
                    model = BalancedRandomForestClassifier(
                        random_state=RANDOM_STATE, **params
                    )

                case _:
                    model = ValueError(f"Unknown model class: {model_name}")

            y_test_all = []
            y_pred_proba_all = []
            current_fold_models = []

            # Process each fold
            for fold_idx, (train_index, test_index) in enumerate(
                cv_splitter.split(X, y)
            ):
                try:
                    # Split data
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Preprocess data
                    X_train_encoded, X_test_encoded = preprocess_fold_data(
                        X_train, X_test, y_train, fold_idx, cols_remove
                    )

                    # Clone model for this fold
                    fold_model = clone(model)

                    # Train model
                    fold_model.fit(X_train_encoded, y_train)
                    current_fold_models.append(fold_model)

                    # Get predictions
                    if hasattr(fold_model, "predict_proba"):
                        y_pred_proba = fold_model.predict_proba(X_test_encoded)[:, 1]
                    else:
                        y_pred = fold_model.predict(X_test_encoded)
                        y_pred_proba = y_pred

                    # Store predictions and true values (convert to numpy arrays)
                    y_test_all.append(
                        y_test.values
                    )  # Convert pandas Series to numpy array
                    y_pred_proba_all.append(
                        np.array(y_pred_proba)
                    )  # Ensure numpy array

                except Exception as e:
                    print(f"Error in fold {fold_idx}: {str(e)}")
                    raise optuna.exceptions.TrialPruned()

            if not y_test_all:
                raise optuna.exceptions.TrialPruned()

            # Pass the lists directly to _evaluate_predictions
            score = _evaluate_predictions(y_test_all, y_pred_proba_all, metric)

            # Update best model if we have a new best score
            if score > best_score:
                best_score = score
                # Refit on all data for the best model
                best_model = clone(model)
                # Use a small slice of X as dummy test set to avoid empty DataFrame issues
                X_encoded, _ = preprocess_fold_data(X, X.iloc[0:1], y, -1, cols_remove)
                best_model.fit(X_encoded, y)

            return score

        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    # Add helper method to get best model
    def get_best_model():
        return best_model

    # Add the helper method to the objective function
    objective.get_best_model = get_best_model

    return objective


def optimize_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: str,
    metric: str,
    cv_splitter: CustomTimeSeriesSplitter,
    model_trials: int,
    cols_remove: list = [],
) -> Tuple[dict, float, object]:
    """
    Optimize a model using Bayesian optimization and cross-validation.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The target variable.
        model_class (str): The name of the model class to optimize.
        metric (str): The metric to optimize.
        cv_splitter (CustomTimeSeriesSplitter): The custom time series cross-validation splitter.
        model_trials (int): The number of trials to run for the optimization.
        cols_remove (list, optional): The columns to remove from the data before optimization. Defaults to an empty list.

    Returns:
        tuple: A tuple containing the best parameters, the best score, and the best model.
    """
    n_jobs = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(n_jobs)

    study_name = f"{model_class}_{metric}"
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1,
        ),
    )

    objective_function = create_objective(
        X, y, model_class, metric, cv_splitter, cols_remove
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        with parallel_backend("threading", n_jobs=n_jobs):
            study.optimize(
                objective_function,
                n_trials=model_trials,
                n_jobs=n_jobs,
                gc_after_trial=True,
                show_progress_bar=True,
                catch=(Exception,),
            )

        if len(study.trials) == 0 or study.best_trial is None:
            print(f"No successful trials for {model_class}")
            return {}, float("-inf"), None

        # Get the best model from the objective function
        best_model = objective_function.get_best_model()

        return study.best_params, study.best_value, best_model
    except Exception as e:
        print(f"Optimization failed for {model_class}: {str(e)}")
        return {}, float("-inf"), None


def extract_params(prefix: str, params: dict) -> dict:
    """
    Extract a subset of parameters from the given dictionary that start with the given prefix.

    Parameters
    ----------
    prefix : str
        The prefix to extract parameters from.
    params : dict
        The parameters to extract from.

    Returns
    -------
    dict
        The extracted parameters.
    """
    return {k[len(prefix) + 1 :]: v for k, v in params.items() if k.startswith(prefix)}


def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cols_remove: list = [],
    metric: str = "F1",
    model_trials: int = 50,
) -> None:

    # Create exports directory if it doesn't exist
    Path("exports").mkdir(exist_ok=True)

    cv_splitter = CustomTimeSeriesSplitter(
        n_splits=5, test_period="30D", gap_period="30D"
    )

    print(f"\nOptimizing {model_name} for {metric}...")
    best_params, best_score, best_model = optimize_model(
        X,
        y,
        model_name,
        metric,
        cv_splitter,
        model_trials,
        cols_remove=cols_remove,
    )

    # Export results for this model
    filename = f"exports/results_{metric}_{model_name}.json"
    model_filename = f"exports/model_{metric}_{model_name}.joblib"

    with open(filename, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "metric": metric,
                "params": best_params,
                "score": best_score,
            },
            f,
            indent=4,
        )

    # Save the actual model
    if best_model is not None:
        joblib.dump(best_model, model_filename)
        print(f"Model saved to {model_filename}")

    print(f"Results exported to {filename}")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splitter: CustomTimeSeriesSplitter,
    model: object,
    model_name: str,
    production_model: bool = True,
    cols_remove: list = [],
    imbalance_handling: str = "none",
    export_model: bool = False,
) -> pd.DataFrame:
    y_test_all = []
    y_pred_proba_all = []
    predictions = {}

    start_time = time.time()

    for fold_idx, (train_index, test_index) in enumerate(cv_splitter.split(X, y)):
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Use the same preprocessing function as Optuna for consistency
        if production_model:
            X_train, X_test = preprocess_fold_data(
                X_train, X_test, y_train, fold_idx, cols_remove
            )

        # Handle class imbalance
        class_weight_dict = None
        match imbalance_handling:
            case "none":
                class_weight_dict = None
            case "undersampler":
                undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
                X_train, y_train = undersampler.fit_resample(X_train, y_train)
                class_weight_dict = None
            case "oversampler":
                oversampler = RandomOverSampler(random_state=RANDOM_STATE)
                X_train, y_train = oversampler.fit_resample(X_train, y_train)
                class_weight_dict = None
            case "cost":
                weights = compute_class_weight(
                    "balanced", classes=np.array([0, 1]), y=y_train
                )
                class_weight_dict = dict(zip([0, 1], weights))

        if class_weight_dict is not None:
            model = apply_class_weights(model, class_weight_dict)

        # Train the model
        model.fit(X_train, y_train)

        # Get predictions
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred

        # Store predictions and true values
        y_test_all.append(y_test.values)
        y_pred_proba_all.append(np.array(y_pred_proba))

        # Store individual predictions for later aggregation
        for idx, pred, pred_proba in zip(test_index, y_pred, y_pred_proba):
            if idx not in predictions:
                predictions[idx] = {"true": y.iloc[idx], "preds": [], "pred_probas": []}
            predictions[idx]["preds"].append(pred)
            predictions[idx]["pred_probas"].append(pred_proba)

    # Calculate metrics on combined predictions
    y_test_combined = np.concatenate([np.array(y) for y in y_test_all])
    y_pred_proba_combined = np.concatenate([np.array(y) for y in y_pred_proba_all])

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(
        y_test_combined, y_pred_proba_combined
    )
    final_auprc = auc(recall, precision)
    end_time = time.time()
    print(f"{model_name} trained in {round(end_time - start_time, 2)} seconds.")
    print(f"AUPRC: {round(final_auprc, 4)}\n")

    # Aggregate predictions
    results = []
    for idx, data in predictions.items():
        results.append(
            {
                "index": idx,
                "true": data["true"],
                "pred": np.mean(data["preds"]) > 0.5,  # Majority vote
                "pred_proba": np.mean(data["pred_probas"]),  # Average probability
            }
        )

    # Export model
    if export_model:
        model_filename = f"exports/model_{model_name}.joblib"
        joblib.dump(model, model_filename)

    results_df = pd.DataFrame(results).set_index("index").sort_index()

    return results_df


def evaluate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: CustomTimeSeriesSplitter,
    cols_remove: list,
    get_feature_importances: bool = False,
):
    """
    Evaluates a model using a CustomTimeSeriesSplitter and returns the model, results DataFrame, and feature importance DataFrame (if requested).

    Parameters:
    model (object): The model to evaluate.
    X (pd.DataFrame): The feature dataset with DatetimeIndex.
    y (pd.Series): The target variable (1 for fraud, 0 for non-fraud).
    splitter (CustomTimeSeriesSplitter): The initialized splitter object.
    cols_remove (list): List of columns to remove from the dataset.
    get_feature_importances (bool): If True, calculate feature importances using SHAP values.

    Returns:
    model (object): The trained model.
    results_df (pd.DataFrame): DataFrame containing true labels, predicted probabilities, and indices.
    importance_df (pd.DataFrame, optional): DataFrame containing feature names and importance values, sorted in descending order of importance.
    """
    y_test_all = []
    y_pred_proba_all = []
    predictions = {}
    feature_names = None  # Store feature names for importance calculation

    def get_feature_importance(model, X_encoded):
        """Extract feature importance using SHAP values for all model types"""

        if hasattr(model, "named_estimators_"):
            importances = []
            weights = []
            n_features = X_encoded.shape[1]  # Expected number of features

            if hasattr(model, "weights") and model.weights is not None:
                estimator_weights = dict(
                    zip(model.named_estimators_.keys(), model.weights)
                )
            else:
                estimator_weights = {name: 1 for name in model.named_estimators_.keys()}

            print("Processing models with weights:", estimator_weights)

            for name, estimator in model.named_estimators_.items():
                try:
                    # Create an explainer appropriate for the model type
                    if isinstance(
                        estimator,
                        (lightgbm.LGBMClassifier, BalancedRandomForestClassifier),
                    ):
                        explainer = shap.TreeExplainer(estimator)
                    else:
                        # Use the encoded data for background
                        background_data = shap.sample(X_encoded, 100)
                        explainer = shap.KernelExplainer(
                            lambda x: estimator.predict_proba(x)[:, 1], background_data
                        )

                    # Calculate SHAP values using encoded data
                    shap_values = explainer.shap_values(X_encoded)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[
                            1
                        ]  # For binary classification, take positive class

                    # Convert to numpy array and ensure 2D shape
                    shap_values = np.asarray(shap_values)
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values.reshape(shap_values.shape[0], -1)

                    # Get feature importance as mean absolute SHAP values
                    feature_importance = np.abs(shap_values).mean(axis=0)

                    # Ensure feature_importance is 1D
                    feature_importance = feature_importance.ravel()

                    # Check if the shape matches expected number of features
                    if len(feature_importance) != n_features:
                        if (
                            len(feature_importance) == 2 * n_features
                        ):  # Handle double shape case
                            feature_importance = feature_importance[:n_features]
                        else:
                            continue

                    print(
                        f"Feature importance shape for {name}: {feature_importance.shape}"
                    )
                    importances.append(feature_importance)
                    weights.append(estimator_weights[name])

                except Exception as e:
                    continue

            if importances:
                # Stack importances along first axis
                importances = np.stack(importances)

                # Normalize weights
                weights = np.array(weights) / sum(weights)

                # Calculate weighted average
                try:
                    weighted_avg = np.average(importances, axis=0, weights=weights)
                    return weighted_avg
                except Exception as e:
                    return None

        return None

    start_time = time.time()
    for fold_idx, (train_index, test_index) in enumerate(splitter.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_encoded, X_test_encoded = preprocess_fold_data(
            X_train, X_test, y_train, fold_idx, cols_remove
        )

        # Store feature names from the first fold's encoded data
        if feature_names is None and hasattr(X_train_encoded, "columns"):
            feature_names = X_train_encoded.columns

        # Train the model
        model.fit(X_train_encoded, y_train)

        # Get predictions
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test_encoded)
            y_pred_proba = y_pred

        # Store predictions and true values
        y_test_all.append(y_test.values)
        y_pred_proba_all.append(np.array(y_pred_proba))

        # Store individual predictions for later aggregation
        for idx, pred, pred_proba in zip(test_index, y_pred, y_pred_proba):
            if idx not in predictions:
                predictions[idx] = {"true": y.iloc[idx], "preds": [], "pred_probas": []}
            predictions[idx]["preds"].append(pred)
            predictions[idx]["pred_probas"].append(pred_proba)

    # Calculate metrics on combined predictions
    y_test_combined = np.concatenate([np.array(y) for y in y_test_all])
    y_pred_proba_combined = np.concatenate([np.array(y) for y in y_pred_proba_all])

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(
        y_test_combined, y_pred_proba_combined
    )
    final_auprc = auc(recall, precision)
    print(f"AUPRC: {round(final_auprc, 4)}")
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds.")

    # Aggregate predictions
    results = []
    for idx, data in predictions.items():
        results.append(
            {
                "index": idx,
                "true": data["true"],
                "pred": np.mean(data["preds"]) > 0.5,  # Majority vote
                "pred_proba": np.mean(data["pred_probas"]),  # Average probability
            }
        )

    results_df = pd.DataFrame(results).set_index("index").sort_index()

    # Calculate feature importances if requested
    if get_feature_importances:
        # Use the last fold's encoded training data for feature importance
        sample_size = min(250, len(X_train_encoded))
        importance_X = X_train_encoded.sample(sample_size, random_state=RANDOM_STATE)
        importance_values = get_feature_importance(model, importance_X)

        if importance_values is not None and feature_names is not None:
            # Create importance DataFrame with encoded feature names
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importance_values}
            )
            importance_df = importance_df.sort_values("importance", ascending=False)

            return model, results_df, importance_df

    return model, results_df


def optimize_ensemble_weights(
    X: pd.DataFrame,
    y: pd.Series,
    models: list,
    splitter: CustomTimeSeriesSplitter,
    cols_remove: list,
    metric: str = "average_precision",
    recall_threshold: float = 0.8,
    threshold: float = 0.5,
    beta: float = 1.0,
):
    """
    Optimize weights for ensemble models using cross-validation predictions.

    Parameters:
    -----------
    X : pandas DataFrame
        Features
    y : pandas Series
        Target variable
    models : list of tuples
        List of (name, model) tuples for ensemble
    splitter : cross-validation splitter
        CV splitter object
    cols_remove : list
        Columns to remove during preprocessing
    metric : str
        Metric to optimize:
        - 'average_precision': Area under Precision-Recall curve
        - 'precision_at_recall': Precision at a specific recall threshold
        - 'precision': Standard precision (requires threshold)
        - 'f1': F1 score
        - 'fbeta': F-beta score (requires beta parameter)
    recall_threshold : float
        Target recall threshold when using 'precision_at_recall' metric
    threshold : float
        Classification threshold for metrics requiring binary predictions
    beta : float
        Beta value for F-beta score (beta=1 is F1 score)
    """

    # Store predictions for each model
    cv_predictions = {name: {"y_true": [], "y_pred_proba": []} for name, _ in models}

    # Get predictions from each model using cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Preprocess fold data
        X_train_encoded, X_test_encoded = preprocess_fold_data(
            X_train, X_test, y_train, fold_idx, cols_remove
        )

        # Get predictions from each model
        for name, model in models:
            # Train model
            model.fit(X_train_encoded, y_train)

            # Get predictions
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
            else:
                y_pred_proba = model.predict(X_test_encoded)

            # Store predictions
            cv_predictions[name]["y_true"].extend(y_test)
            cv_predictions[name]["y_pred_proba"].extend(y_pred_proba)

    # Convert lists to numpy arrays
    for name in cv_predictions:
        cv_predictions[name]["y_true"] = np.array(cv_predictions[name]["y_true"])
        cv_predictions[name]["y_pred_proba"] = np.array(
            cv_predictions[name]["y_pred_proba"]
        )

    def get_precision_at_recall(y_true, y_pred_proba, target_recall):
        """Get precision at a specific recall threshold"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        valid_recalls = recalls >= target_recall
        if not any(valid_recalls):
            return 0.0
        idx = valid_recalls.nonzero()[0][-1]
        return precisions[idx]

    # Pre-compute thresholds for F1/F-beta optimization
    if metric in ["f1", "fbeta"]:
        # Get a sample of predictions to determine potential thresholds
        y_true = cv_predictions[models[0][0]]["y_true"]
        y_pred_sample = cv_predictions[models[0][0]]["y_pred_proba"]
        _, _, thresholds = precision_recall_curve(y_true, y_pred_sample)
        # Reduce number of thresholds to test by taking a subset
        # More thresholds = more precise but slower
        n_thresholds = 20  # Adjust this number for speed/precision trade-off
        threshold_indices = np.linspace(0, len(thresholds) - 1, n_thresholds).astype(
            int
        )
        test_thresholds = thresholds[threshold_indices]

    def objective(weights):
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)

        # Compute weighted average of predictions
        weighted_pred_proba = np.zeros_like(
            cv_predictions[models[0][0]]["y_pred_proba"]
        )
        for weight, (name, _) in zip(weights, models):
            weighted_pred_proba += weight * cv_predictions[name]["y_pred_proba"]

        y_true = cv_predictions[models[0][0]]["y_true"]

        # Calculate metric based on choice
        if metric == "average_precision":
            score = average_precision_score(y_true, weighted_pred_proba)
        elif metric == "precision_at_recall":
            score = get_precision_at_recall(
                y_true, weighted_pred_proba, recall_threshold
            )
        elif metric in ["f1", "fbeta"]:
            # Test pre-computed thresholds
            best_score = -np.inf
            for t in test_thresholds:
                weighted_pred = (weighted_pred_proba >= t).astype(int)
                if metric == "f1":
                    current_score = f1_score(y_true, weighted_pred)
                else:
                    current_score = fbeta_score(y_true, weighted_pred, beta=beta)
                best_score = max(best_score, current_score)
            score = best_score
        else:  # standard precision
            weighted_pred = (weighted_pred_proba > threshold).astype(int)
            score = precision_score(y_true, weighted_pred)

        # Add small regularization term to prevent exactly equal weights
        weight_variation = np.std(weights)
        regularization = 0.001 * (1.0 / (weight_variation + 1e-6))

        return -(score - regularization)  # Negative because we minimize

    # Try multiple random initializations
    best_score = float("inf")
    best_weights = None
    n_tries = 5

    for try_idx in range(n_tries):
        # Random initial weights
        initial_weights = np.random.dirichlet(np.ones(len(models)))

        # Constraints: weights sum to 1 and are non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(models))]

        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Optimize weights
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-8, "maxiter": 2000},
            )

        if result.fun < best_score:
            best_score = result.fun
            best_weights = result.x

    # Normalize final weights to sum to 1
    optimal_weights = best_weights / np.sum(best_weights)

    # Print results
    print(f"\nOptimal weights for {metric}:")
    for (name, _), weight in zip(models, optimal_weights):
        print(f"{name}: {weight:.4f}")

    return optimal_weights


def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cols_remove: list = [],
    metric: str = "F1",
    model_trials: int = 50,
):
    """
    Run Bayesian optimization to find the best hyperparameters for a given model.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The target variable.
        model_name (str): The name of the model to optimize.
        cols_remove (list, optional): The columns to remove from the data before optimization. Defaults to an empty list.
        metric (str, optional): The metric to optimize. Defaults to "F1".
        model_trials (int, optional): The number of trials to run for the optimization. Defaults to 50.

    Returns:
        None
    """
    Path("exports").mkdir(exist_ok=True)

    cv_splitter = CustomTimeSeriesSplitter(
        n_splits=5, test_period="30D", gap_period="30D"
    )

    print(f"\nOptimizing {model_name} for {metric}...")
    best_params, best_score, best_model = optimize_model(
        X,
        y,
        model_name,
        metric,
        cv_splitter,
        model_trials,
        cols_remove=cols_remove,
    )

    # Export results for this model
    filename = f"exports/results_{metric}_{model_name}.json"
    model_filename = f"exports/model_{metric}_{model_name}.joblib"

    with open(filename, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "metric": metric,
                "params": best_params,
                "score": best_score,
            },
            f,
            indent=4,
        )

    # Save the actual model
    if best_model is not None:
        joblib.dump(best_model, model_filename)
        print(f"Model saved to {model_filename}")

    print(f"Results exported to {filename}")


def analyze_monthly_fraud(df: pd.DataFrame) -> dict:
    """
    Analyze monthly fraudulent transactions.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data

    Returns:
        dict: A dictionary containing the statistics of the analysis, including
            - start_period (str): The start period of the data
            - end_period (str): The end period of the data
            - total_months (int): The total number of months in the data
            - total_fraud_cases (int): The total number of fraudulent transactions
    """
    fraud_df = df[df["fraud"] == True]
    monthly_fraud = fraud_df.resample("M").size()

    # Calculate statistics
    avg_monthly_fraud = monthly_fraud.mean()
    start_period = monthly_fraud.index.min().strftime("%Y-%m")
    end_period = monthly_fraud.index.max().strftime("%Y-%m")

    # Create bar plot
    plt.figure(figsize=(get_screen_width() / 100, 10))
    bars = plt.bar(
        monthly_fraud.index.strftime("%Y-%m"),
        monthly_fraud.values,
        color=COLOR_PALETTE[0],
    )

    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Customize the plot
    plt.title("Monthly Fraudulent Transactions")
    plt.xlabel("Month")
    plt.ylabel("Number of Fraudulent Transactions")
    plt.xticks(rotation=45)

    # Add average line
    plt.axhline(
        y=avg_monthly_fraud,
        color="r",
        linestyle="--",
        label=f"Average: {avg_monthly_fraud:.1f}",
    )
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Return statistics
    stats = {
        "start_period": start_period,
        "end_period": end_period,
        "total_months": len(monthly_fraud),
        "total_fraud_cases": monthly_fraud.sum(),
    }

    return stats


def plot_top_n_features(
    selected_models: dict, feature_names: list, top_n: int = 20
) -> None:
    """
    Plot the top N features by importance for each model in the given dictionary.

    Parameters
    ----------
    selected_models : dict
        A dictionary containing the models to plot, with the model name as the key and the model
        object as the value.
    feature_names : list
        The list of feature names for the models.
    top_n : int, optional
        The number of top features to show in each plot. Defaults to 20.

    Returns
    -------
    None
    """

    def _get_feature_importance(
        model: object, feature_names: List[str]
    ) -> pd.DataFrame:
        # Initialize a dictionary to store combined feature importances
        """
        Combine feature importances from multiple models (e.g. Random Forest, XGBoost, LightGBM) into a single DataFrame.

        Parameters:
        model (object): The model object (either a single model or an ensemble)
        feature_names (List[str]): The names of the features

        Returns:
        pd.DataFrame: A DataFrame containing the combined feature importances, sorted in descending order
        """

        combined_importances = {feature: 0 for feature in feature_names}

        # Check if the model is an ensemble
        if hasattr(model, "estimators_"):
            # Get the constituent models
            models = [estimator for _, estimator in model.named_estimators_.items()]
        else:
            # Treat the model as a single model
            models = [model]

        for model in models:
            if hasattr(model, "feature_importances_"):
                # For tree-based models (Random Forest, XGBoost, LightGBM)
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                # For linear models
                importances = np.abs(model.coef_)
            else:
                print(
                    f"Warning: {
                        type(model).__name__} doesn't have a standard feature importance attribute."
                )
                continue

            # Add importance to the combined importances
            for feature, importance in zip(feature_names, importances):
                combined_importances[feature] += importance

        # Average the importances
        for feature in combined_importances:
            combined_importances[feature] /= len(models)

        # Convert to DataFrame and sort
        importance_df = pd.DataFrame.from_dict(
            combined_importances, orient="index", columns=["Importance"]
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)

        return importance_df

    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    for i, (model_name, model) in enumerate(selected_models.items()):
        feature_importances = _get_feature_importance(model, feature_names)
        top_features = feature_importances.head(top_n)
        top_features.plot.barh(ax=ax[i])
        ax[i].set_title(f"{model_name} top {top_n} features by importance")
        ax[i].invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_model_importances(
    df_importances: pd.DataFrame, top_n: int = 20, model_name: str = ""
) -> None:
    """
    Plot the top feature importances for a given model.

    Parameters:
    df_importances (pd.DataFrame): DataFrame containing feature importances
    top_n (int): The number of top features to display. Defaults to 20.
    model_name (str): Name of the model used to generate the feature importances. Defaults to ""
    """

    df_importances = df_importances.sort_values("importance", ascending=False).head(
        top_n
    )

    plt.figure(figsize=(get_screen_width() / 100, 10))
    plt.barh(df_importances["feature"], df_importances["importance"])
    plt.xlabel("Importance Mean")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances for {model_name}")
    plt.gca().invert_yaxis()
    plt.show()


def plot_probability_distribution(y_true: pd.Series, y_pred_proba: pd.Series) -> None:
    """
    Plots the probability distribution of the predicted probabilities for each class.

    Args:
        y_true (pd.Series): The true labels of the data.
        y_pred_proba (pd.Series): The predicted probabilities for the positive class.

    Returns:
        None
    """
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({"Probability": y_pred_proba, "True_Class": y_true})

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot KDE for each class
    sns.kdeplot(
        data=df[df["True_Class"] == 0],
        x="Probability",
        shade=True,
        color="skyblue",
        label="Class 0 (Negative)",
    )
    sns.kdeplot(
        data=df[df["True_Class"] == 1],
        x="Probability",
        shade=True,
        color="orange",
        label="Class 1 (Positive)",
    )

    # Add vertical lines for potential thresholds
    plt.axvline(x=0.5, color="red", linestyle="--", label="Default Threshold (0.5)")

    # Customize the plot
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Predicted Probability of Positive Class")
    plt.ylabel("Density")
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def analyze_thresholds(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    thresholds: pd.Series = None,
    total_transactions: int = 300,
) -> pd.DataFrame:
    """
    Analyze model performance at different thresholds for fraud detection.

    Parameters:
    -----------
    y_true : array-like
        True labels (0 for legitimate, 1 for fraud)
    y_pred_proba : array-like
        Predicted probabilities of fraud
    thresholds : array-like, optional
        Specific thresholds to evaluate
    total_transactions : int, optional
        Total number of daily transactions (default: 300)
    """
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 25)

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        _, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        total_alerts = tp + fp  # Total cases to investigate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_true, y_pred)

        # Calculate daily workload
        dataset_size = len(y_true)
        alerts_per_transaction = total_alerts / dataset_size
        daily_alerts = int(total_transactions * alerts_per_transaction)

        # Calculate efficiency metrics
        fraud_catch_rate = recall * 100  # percentage of fraud caught
        investigation_efficiency = (
            tp / total_alerts if total_alerts > 0 else 0
        )  # fraud cases per alert
        alerts_per_fraud = (
            total_alerts / tp if tp > 0 else float("inf")
        )  # alerts needed to find one fraud

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "daily_alerts": daily_alerts,
                "fraud_catch_rate": fraud_catch_rate,
                "investigation_efficiency": investigation_efficiency
                * 100,  # as percentage
                "alerts_per_fraud": alerts_per_fraud,
            }
        )

    return pd.DataFrame(results)


def find_optimal_threshold(
    results_df: pd.DataFrame, max_daily_alerts: int, min_fraud_catch_rate: float = None
) -> pd.Series:
    """
    Identify the optimal threshold for fraud detection based on constraints.

    This function evaluates different thresholds to find the one that maximizes
    investigation efficiency while adhering to specified constraints on daily
    alerts and fraud catch rate.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing threshold evaluation metrics including 'daily_alerts',
        'fraud_catch_rate', and 'investigation_efficiency'.
    max_daily_alerts : int
        Maximum number of daily alerts allowed.
    min_fraud_catch_rate : float, optional
        Minimum acceptable fraud catch rate (as a percentage). Defaults to None.

    Returns:
    --------
    pd.Series or None
        Series containing the metrics for the optimal threshold if it exists,
        otherwise None if no thresholds meet the constraints.
    """
    # Filter for thresholds that meet the daily alert constraint
    valid_df = results_df[results_df["daily_alerts"] <= max_daily_alerts]

    # Apply minimum fraud catch rate if specified
    if min_fraud_catch_rate is not None:
        valid_df = valid_df[valid_df["fraud_catch_rate"] >= min_fraud_catch_rate]

    if valid_df.empty:
        return None

    # Find threshold that maximizes investigation efficiency within constraints
    optimal_threshold = valid_df.loc[valid_df["investigation_efficiency"].idxmax()]

    return optimal_threshold


def print_threshold_summary(threshold_info: dict) -> None:
    """Print detailed summary of threshold performance."""
    if threshold_info is None:
        print("No threshold meets all constraints.")
        return

    print("\nThreshold Performance Summary:")
    print(f"Recommended threshold: {threshold_info['threshold']:.3f}")
    print(f"Daily alerts to investigate: {threshold_info['daily_alerts']}")
    print(f"Fraud catch rate: {threshold_info['fraud_catch_rate']:.1f}%")
    print(
        f"Investigation efficiency: {threshold_info['investigation_efficiency']:.1f}%"
    )
    print(f"Alerts needed per fraud: {threshold_info['alerts_per_fraud']:.1f}")

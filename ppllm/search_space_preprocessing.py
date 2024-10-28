# search_spaces.py

import optuna

# Imputers and missing value handling
from sklearn.impute import SimpleImputer, KNNImputer

# Scalers and standardization
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    MaxAbsScaler, 
    QuantileTransformer, 
    PowerTransformer
)

# Dimensionality reduction techniques
from sklearn.decomposition import (
    PCA, 
    KernelPCA, 
    TruncatedSVD, 
    FastICA
)

# Feature selection
from sklearn.feature_selection import (
    SelectKBest, 
    SelectPercentile, 
    RFE, 
    RFECV
)

# Polynomial feature expansion
from sklearn.preprocessing import PolynomialFeatures

# Encoding for categorical variables (if needed for regression models with categorical inputs)
from sklearn.preprocessing import OneHotEncoder


# Imputation search spaces
def simple_imputer_search_space(trial):
    return {
        "strategy": trial.suggest_categorical("strategy", ["mean", "median", "most_frequent", "constant"]),
        "fill_value": trial.suggest_float("fill_value", -10, 10) if trial.params.get("strategy") == "constant" else None
    }

def knn_imputer_search_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["nan_euclidean"])
    }

# Scaling search spaces
def standard_scaler_search_space(trial):
    return {}  # No hyperparameters for StandardScaler

def min_max_scaler_search_space(trial):
    return {
        "feature_range": (
            trial.suggest_float("min", 0.0, 0.5),
            trial.suggest_float("max", 0.5, 1.0)
        )
    }

def robust_scaler_search_space(trial):
    return {
        "quantile_range": (
            trial.suggest_float("quantile_min", 0.0, 25.0),
            trial.suggest_float("quantile_max", 75.0, 100.0)
        )
    }

def max_abs_scaler_search_space(trial):
    return {}  # No hyperparameters for MaxAbsScaler

def quantile_transformer_search_space(trial):
    return {
        "output_distribution": trial.suggest_categorical("output_distribution", ["uniform", "normal"]),
        "n_quantiles": trial.suggest_int("n_quantiles", 10, 1000)
    }

def power_transformer_search_space(trial):
    return {
        "method": trial.suggest_categorical("method", ["yeo-johnson", "box-cox"])
    }

# Dimensionality reduction search spaces
def pca_search_space(trial):
    return {
        "n_components": trial.suggest_int("n_components", 1, 100),
        "svd_solver": trial.suggest_categorical("svd_solver", ["auto", "full", "arpack", "randomized"])
    }

def kernel_pca_search_space(trial):
    return {
        "n_components": trial.suggest_int("n_components", 1, 100),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"])
    }

def truncated_svd_search_space(trial):
    return {
        "n_components": trial.suggest_int("n_components", 1, 100)
    }

def fast_ica_search_space(trial):
    return {
        "n_components": trial.suggest_int("n_components", 1, 100),
        "algorithm": trial.suggest_categorical("algorithm", ["parallel", "deflation"])
    }

# Feature selection search spaces
def select_k_best_search_space(trial):
    return {
        "k": trial.suggest_int("k", 1, 100)
    }

def select_percentile_search_space(trial):
    return {
        "percentile": trial.suggest_int("percentile", 1, 100)
    }

def rfe_search_space(trial):
    return {
        "n_features_to_select": trial.suggest_int("n_features_to_select", 1, 100)
    }

def rfecv_search_space(trial):
    return {
        "min_features_to_select": trial.suggest_int("min_features_to_select", 1, 10)
    }

# Polynomial feature expansion
def polynomial_features_search_space(trial):
    return {
        "degree": trial.suggest_int("degree", 2, 5),
        "interaction_only": trial.suggest_categorical("interaction_only", [True, False]),
        "include_bias": trial.suggest_categorical("include_bias", [True, False])
    }

# Encoding search space
def one_hot_encoder_search_space(trial):
    return {
        "drop": trial.suggest_categorical("drop", ["first", None]),
        "sparse_output": trial.suggest_categorical("sparse", [True, False])
    }

# Preprocessing dictionary
preprocessing_search_space = {
    "simpleimputer": {
        "estimator": SimpleImputer,
        "search_space": simple_imputer_search_space
    },
    "knnimputer": {
        "estimator": KNNImputer,
        "search_space": knn_imputer_search_space
    },
    "standardscaler": {
        "estimator": StandardScaler,
        "search_space": standard_scaler_search_space
    },
    "minmaxscaler": {
        "estimator": MinMaxScaler,
        "search_space": min_max_scaler_search_space
    },
    "robustscaler": {
        "estimator": RobustScaler,
        "search_space": robust_scaler_search_space
    },
    "maxabsscaler": {
        "estimator": MaxAbsScaler,
        "search_space": max_abs_scaler_search_space
    },
    "quantiletransformer": {
        "estimator": QuantileTransformer,
        "search_space": quantile_transformer_search_space
    },
    "powertransformer": {
        "estimator": PowerTransformer,
        "search_space": power_transformer_search_space
    },
    "pca": {
        "estimator": PCA,
        "search_space": pca_search_space
    },
    "kernelpca": {
        "estimator": KernelPCA,
        "search_space": kernel_pca_search_space
    },
    "truncatedsvd": {
        "estimator": TruncatedSVD,
        "search_space": truncated_svd_search_space
    },
    "fastica": {
        "estimator": FastICA,
        "search_space": fast_ica_search_space
    },
    "selectkbest": {
        "estimator": SelectKBest,
        "search_space": select_k_best_search_space
    },
    "selectpercentile": {
        "estimator": SelectPercentile,
        "search_space": select_percentile_search_space
    },
    "rfe": {
        "estimator": RFE,
        "search_space": rfe_search_space
    },
    "rfecv": {
        "estimator": RFECV,
        "search_space": rfecv_search_space
    },
    "polynomialfeatures": {
        "estimator": PolynomialFeatures,
        "search_space": polynomial_features_search_space
    },
    "onehotencoder": {
        "estimator": OneHotEncoder,
        "search_space": one_hot_encoder_search_space
    }
}
# search_spaces.py

import optuna

# Ensemble methods
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor, 
    AdaBoostRegressor, 
    BaggingRegressor, 
    ExtraTreesRegressor, 
    HistGradientBoostingRegressor
)

# Linear models
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    Lasso, 
    ElasticNet, 
    SGDRegressor, 
    BayesianRidge, 
    HuberRegressor
)

# Neighbors-based methods
from sklearn.neighbors import (
    KNeighborsRegressor, 
    RadiusNeighborsRegressor
)

# Tree-based models
from sklearn.tree import DecisionTreeRegressor

# Support Vector Machines
from sklearn.svm import SVR, LinearSVR

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessRegressor

# Neural Network-based model
from sklearn.neural_network import MLPRegressor

# Define search space functions for each estimator

def gradient_boosting_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }

def random_forest_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

def adaboost_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0)
    }

def bagging_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_samples": trial.suggest_uniform("max_samples", 0.1, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

def extra_trees_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }

def hist_gradient_boosting_search_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "l2_regularization": trial.suggest_loguniform("l2_regularization", 1e-5, 1.0)
    }

def ridge_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 10.0)
    }

def lasso_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 10.0)
    }

def elasticnet_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 10.0),
        "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1)
    }

def sgd_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 10.0),
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000)
    }

def knn_search_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"])
    }

def decision_tree_search_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

def svr_search_space(trial):
    return {
        "C": trial.suggest_loguniform("C", 1e-5, 10.0),
        "epsilon": trial.suggest_loguniform("epsilon", 1e-5, 1.0),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    }

def mlp_search_space(trial):
    return {
        "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 50, 200),
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e-1),
        "learning_rate_init": trial.suggest_loguniform("learning_rate_init", 1e-5, 1e-1)
    }


def linear_regression_search_space(trial):
    # No tunable parameters for basic LinearRegression, can just return an empty dict.
    return {}

def bayesian_ridge_search_space(trial):
    return {
        "alpha_1": trial.suggest_loguniform("alpha_1", 1e-6, 1e-1),
        "alpha_2": trial.suggest_loguniform("alpha_2", 1e-6, 1e-1),
        "lambda_1": trial.suggest_loguniform("lambda_1", 1e-6, 1e-1),
        "lambda_2": trial.suggest_loguniform("lambda_2", 1e-6, 1e-1)
    }

def huber_regressor_search_space(trial):
    return {
        "epsilon": trial.suggest_uniform("epsilon", 1.1, 2.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1.0)
    }

def radius_neighbors_search_space(trial):
    return {
        "radius": trial.suggest_uniform("radius", 0.5, 10.0),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2)  # p=1 for Manhattan, p=2 for Euclidean
    }   

def linear_svr_search_space(trial):
    return {
        "C": trial.suggest_loguniform("C", 1e-5, 10.0),
        "epsilon": trial.suggest_loguniform("epsilon", 1e-5, 1.0),
        "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"])
    }

def gaussian_process_regressor_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-10, 1e-1),
        "normalize_y": trial.suggest_categorical("normalize_y", [True, False])
    }


# Dictionary of estimators and search space functions
regression_search_space = {
    "gradientboostingregressor": {
        "estimator": GradientBoostingRegressor,
        "search_space": gradient_boosting_search_space
    },
    "randomforestregressor": {
        "estimator": RandomForestRegressor,
        "search_space": random_forest_search_space
    },
    "adaboostregressor": {
        "estimator": AdaBoostRegressor,
        "search_space": adaboost_search_space
    },
    "baggingregressor": {
        "estimator": BaggingRegressor,
        "search_space": bagging_search_space
    },
    "extratreesregressor": {
        "estimator": ExtraTreesRegressor,
        "search_space": extra_trees_search_space
    },
    "histgradientboostingregressor": {
        "estimator": HistGradientBoostingRegressor,
        "search_space": hist_gradient_boosting_search_space
    },
    "ridge": {
        "estimator": Ridge,
        "search_space": ridge_search_space
    },
    "lasso": {
        "estimator": Lasso,
        "search_space": lasso_search_space
    },
    "elasticnet": {
        "estimator": ElasticNet,
        "search_space": elasticnet_search_space
    },
    "sgdregressor": {
        "estimator": SGDRegressor,
        "search_space": sgd_search_space
    },
    "kneighborsregressor": {
        "estimator": KNeighborsRegressor,
        "search_space": knn_search_space
    },
    "decisiontreeregressor": {
        "estimator": DecisionTreeRegressor,
        "search_space": decision_tree_search_space
    },
    "svr": {
        "estimator": SVR,
        "search_space": svr_search_space
    },
    "mlpregressor": {
        "estimator": MLPRegressor,
        "search_space": mlp_search_space
    },
    "linearregression": {
        "estimator": LinearRegression,
        "search_space": linear_regression_search_space
    },
    "bayesianridge": {
        "estimator": BayesianRidge,
        "search_space": bayesian_ridge_search_space
    },
    "huberregressor": {
        "estimator": HuberRegressor,
        "search_space": huber_regressor_search_space
    },
    "radiusneighborsregressor": {
        "estimator": RadiusNeighborsRegressor,
        "search_space": radius_neighbors_search_space
    },
    "linearsvr": {
        "estimator": LinearSVR,
        "search_space": linear_svr_search_space
    },
    "gaussianprocessregressor": {
        "estimator": GaussianProcessRegressor,
        "search_space": gaussian_process_regressor_search_space
    },
}
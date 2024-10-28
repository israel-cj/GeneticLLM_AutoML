# search_spaces.py

import optuna
# Ensemble methods
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    AdaBoostClassifier, 
    BaggingClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier
)

# Linear models
from sklearn.linear_model import (
    LogisticRegression, 
    RidgeClassifier, 
    SGDClassifier, 
    PassiveAggressiveClassifier,
    Perceptron
)

# Neighbors-based methods
from sklearn.neighbors import (
    KNeighborsClassifier, 
    RadiusNeighborsClassifier, 
    NearestCentroid
)

# Naive Bayes methods
from sklearn.naive_bayes import (
    GaussianNB, 
    MultinomialNB, 
    BernoulliNB, 
    ComplementNB
)

# Tree-based models
from sklearn.tree import DecisionTreeClassifier

# Support Vector Machines
from sklearn.svm import (
    SVC, 
    NuSVC, 
    LinearSVC
)

# Discriminant Analysis
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, 
    QuadraticDiscriminantAnalysis
)

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessClassifier

# Neural Network-based model
from sklearn.neural_network import MLPClassifier

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
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "algorithm": trial.suggest_categorical("algorithm", ["SAMME"])
    }

def bagging_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

def extratrees_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }

def hist_gradient_boosting_search_space(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "l2_regularization": trial.suggest_loguniform("l2_regularization", 1e-10, 1.0)
    }

def logistic_regression_search_space(trial):
    return {
        "C": trial.suggest_loguniform("C", 1e-5, 10),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "sag", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 50, 500)
    }

def sgd_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e-1),
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000)
    }

def decision_tree_search_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

def svc_search_space(trial):
    return {
        "C": trial.suggest_loguniform("C", 1e-5, 10),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int("degree", 2, 5),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
    }

def ridge_classifier_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 10.0),
        "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"])
    }

def passive_aggressive_search_space(trial):
    return {
        "C": trial.suggest_loguniform("C", 1e-5, 10.0),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
        "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
    }

def kneighbors_classifier_search_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": trial.suggest_int("leaf_size", 10, 100),
        "p": trial.suggest_int("p", 1, 2)  # p=1 is Manhattan, p=2 is Euclidean
    }

def mlp_search_space(trial):
    return {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)]),
        "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e-1),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "max_iter": trial.suggest_int("max_iter", 200, 1000)
    }

def perceptron_search_space(trial):
    return {
        "penalty": trial.suggest_categorical("penalty", [None, "l2", "l1", "elasticnet"]),
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1.0),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1)
    }

def radius_neighbors_search_space(trial):
    return {
        "radius": trial.suggest_loguniform("radius", 1e-1, 10.0),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": trial.suggest_int("leaf_size", 10, 100),
        "p": trial.suggest_int("p", 1, 2)  # p=1 is Manhattan, p=2 is Euclidean
    }

def nearest_centroid_search_space(trial):
    return {
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"])
    }

def gaussian_nb_search_space(trial):
    # GaussianNB has no hyperparameters to tune in sklearn
    return {}

def multinomial_nb_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False])
    }

def bernoulli_nb_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        "binarize": trial.suggest_loguniform("binarize", 1e-3, 10.0)
    }

def complement_nb_search_space(trial):
    return {
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        "norm": trial.suggest_categorical("norm", [True, False])
    }

def nu_svc_search_space(trial):
    return {
        # "nu": trial.suggest_uniform("nu", 0.1, 0.9),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "coef0": trial.suggest_uniform("coef0", 0.0, 10.0)
    }

def linear_svc_search_space(trial):
    return {
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "C": trial.suggest_loguniform("C", 1e-5, 10.0),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
        "max_iter": trial.suggest_int("max_iter", 100, 1000)
    }

def linear_discriminant_analysis_search_space(trial):
    return {
        "solver": trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"]),
        "shrinkage": trial.suggest_categorical("shrinkage", [None, "auto"]) if trial.params["solver"] != "svd" else None
    }

def quadratic_discriminant_analysis_search_space(trial):
    return {
        "reg_param": trial.suggest_uniform("reg_param", 0.0, 1.0)
    }

def gaussian_process_classifier_search_space(trial):
    return {
        "max_iter_predict": trial.suggest_int("max_iter_predict", 100, 1000),
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 10),
        "warm_start": trial.suggest_categorical("warm_start", [True, False])
    }

# Dictionary of estimators and search space functions
classification_search_space = {
    "gradientboostingclassifier": {
        "estimator": GradientBoostingClassifier,
        "search_space": gradient_boosting_search_space
    },
    "randomforestclassifier": {
        "estimator": RandomForestClassifier,
        "search_space": random_forest_search_space
    },
    "adaboostclassifier": {
        "estimator": AdaBoostClassifier,
        "search_space": adaboost_search_space
    },
    "baggingclassifier": {
        "estimator": BaggingClassifier,
        "search_space": bagging_search_space
    },
    "extratreesclassifier": {
        "estimator": ExtraTreesClassifier,
        "search_space": extratrees_search_space
    },
    "histgradientboostingclassifier": {
        "estimator": HistGradientBoostingClassifier,
        "search_space": hist_gradient_boosting_search_space
    },
    "logisticregression": {
        "estimator": LogisticRegression,
        "search_space": logistic_regression_search_space
    },
    "sgdclassifier": {
        "estimator": SGDClassifier,
        "search_space": sgd_search_space
    },
    "decisiontreeclassifier": {
        "estimator": DecisionTreeClassifier,
        "search_space": decision_tree_search_space
    },
    "svc": {
        "estimator": SVC,
        "search_space": svc_search_space
    },
    "mlpclassifier": {
        "estimator": MLPClassifier,
        "search_space": mlp_search_space
    },
    "ridgeclassifier": {
        "estimator": RidgeClassifier,
        "search_space": ridge_classifier_search_space
    },
    "passiveaggressiveclassifier": {
        "estimator": PassiveAggressiveClassifier,
        "search_space": passive_aggressive_search_space
    },
    "kneighborsclassifier": {
        "estimator": KNeighborsClassifier,
        "search_space": kneighbors_classifier_search_space
    },
    "perceptron": {
        "estimator": Perceptron,
        "search_space": perceptron_search_space
    },
    "radiusneighborsclassifier": {
        "estimator": RadiusNeighborsClassifier,
        "search_space": radius_neighbors_search_space
    },
    "nearestcentroid": {
        "estimator": NearestCentroid,
        "search_space": nearest_centroid_search_space
    },
    "gaussiannb": {
        "estimator": GaussianNB,
        "search_space": gaussian_nb_search_space
    },
    "multinomialnb": {
        "estimator": MultinomialNB,
        "search_space": multinomial_nb_search_space
    },
    "bernoullinb": {
        "estimator": BernoulliNB,
        "search_space": bernoulli_nb_search_space
    },
    "complementnb": {
        "estimator": ComplementNB,
        "search_space": complement_nb_search_space
    },
    "nusvc": {
        "estimator": NuSVC,
        "search_space": nu_svc_search_space
    },
    "linearsvc": {
        "estimator": LinearSVC,
        "search_space": linear_svc_search_space
    },
    "lineardiscriminantanalysis": {
        "estimator": LinearDiscriminantAnalysis,
        "search_space": linear_discriminant_analysis_search_space
    },
    "quadraticdiscriminantanalysis": {
        "estimator": QuadraticDiscriminantAnalysis,
        "search_space": quadratic_discriminant_analysis_search_space
    },
    "gaussianprocessclassifier": {
        "estimator": GaussianProcessClassifier,
        "search_space": gaussian_process_classifier_search_space
    }
}

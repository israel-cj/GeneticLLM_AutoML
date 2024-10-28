import joblib
import os
import optuna
import time
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from .search_space import classification_search_space
from .search_space_regression import regression_search_space
from .search_space_preprocessing import preprocessing_search_space

# Split the pipeline into preprocessing steps and final estimator
def split_pipeline(pipe):
    steps = pipe.steps
    components = {}
    for step_name, step_obj in steps:
        components[step_name] = step_obj
    return components

def optimize(pipe, X, y, timeout= 300, task='classification'):
    if task == 'classification':
        search_space = classification_search_space
        scoring_metric = 'balanced_accuracy'
    elif task == 'regression':
        search_space = regression_search_space
        scoring_metric = 'r2'

    components = split_pipeline(pipe)

    # Separate preprocessing steps from the final estimator
    if len(components) > 1:
        preprocessing_steps = list(components.items())[:-1]  # All steps except the last one
        final_estimator_name, _ = list(components.items())[-1]  # Final estimator
    else:
        preprocessing_steps = []
        final_estimator_name, _ = list(components.items())[0]

    # Define the objective function for Optuna optimization
    def objective(trial):
        # Sample preprocessing steps
        sampled_preprocessing_steps = []
        for step_name, step_obj in preprocessing_steps:
            if step_name in preprocessing_search_space:
                step_info = preprocessing_search_space[step_name]
                step_hyperparams = step_info["search_space"](trial)
                step_instance = step_info["estimator"](**step_hyperparams)
                sampled_preprocessing_steps.append((step_name, step_instance))
            else:
                sampled_preprocessing_steps.append((step_name, step_obj))

        # Sample the estimator's hyperparameters
        estimator_info = search_space.get(final_estimator_name)
        if estimator_info is None:
            model = list(components.items())[-1]
        else:
            hyperparams = estimator_info["search_space"](trial)
            model = estimator_info["estimator"](**hyperparams)

        # Create the pipeline with both sampled preprocessing steps and model
        pipeline = Pipeline(sampled_preprocessing_steps + [(final_estimator_name, model)])

        # Measure inference time
        start_time = time.time()
        score = cross_val_score(
            pipeline, X, y, scoring=scoring_metric, cv=trial.suggest_int('cv', 2, 5)
        )
        inference_time = time.time() - start_time

        # Return multi-objective: Metric score (maximize) and inference time (minimize)
        metric_score = score.mean()
        return metric_score, inference_time

    # Use NSGA-II sampler for multi-objective optimization
    sampler = optuna.samplers.NSGAIISampler()

    # Add Hyperband for multi-fidelity pruning
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5)

    # Create the study with both NSGA-II and Hyperband pruning
    study = optuna.create_study(
        directions=["maximize", "minimize"], 
        sampler=sampler, 
        pruner=pruner
    )

    # Optimize the study
    study.optimize(objective, timeout=timeout) # n_trials
    # study.optimize(objective, n_trials=3)

    # Print the best trial results for both objectives
    best_trial = study.best_trials[0]
    print(f"Best trial Metric Score: {best_trial.values[0]}")
    print(f"Best trial Inference Time: {best_trial.values[1]}")


    # Extract the best parameters from the best trial
    best_params = best_trial.params
    print("Best parameters:")
    print(best_params)


    # Reconstruct the preprocessing steps based on the best parameters
    reconstructed_steps = []
    for step_name, step_obj in preprocessing_steps:
        if step_name in preprocessing_search_space:
            step_info = preprocessing_search_space[step_name]
            
            # Get the search space parameters for this step by checking the prefix in best_params
            step_hyperparams = {
                param_name: param_value for param_name, param_value in best_params.items()
                if param_name in step_info["search_space"](optuna.trial.FixedTrial(best_params)).keys()
            }

            # Create the step instance with the chosen hyperparameters
            step_instance = step_info["estimator"](**step_hyperparams)
            reconstructed_steps.append((step_name, step_instance))
        else:
            # If the step is not in preprocessing_search_space, use the original step
            reconstructed_steps.append((step_name, step_obj))

    # Rebuild the final estimator using the best parameters
    final_estimator_info = search_space[final_estimator_name]
    estimator_hyperparams = {
        param_name: param_value for param_name, param_value in best_params.items()
        if param_name in final_estimator_info["search_space"](optuna.trial.FixedTrial(best_params)).keys()
    }
    final_estimator = final_estimator_info["estimator"](**estimator_hyperparams)

    # Create the final pipeline with make_pipeline
    best_pipeline = make_pipeline(*[step[1] for step in reconstructed_steps], final_estimator)

    # Now `best_pipeline` is a sklearn object with the best parameters
    print(best_pipeline)

    return best_pipeline


if __name__ == "__main__":
    # Define file paths
    x_path = os.path.join(os.getcwd(), 'X.pkl')
    y_path = os.path.join(os.getcwd(), 'y.pkl')
    pipe_path = os.path.join(os.getcwd(), 'pipe.pkl')

    # Load objects
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    pipe = joblib.load(pipe_path)

    # Optimize the pipeline
    best_pipeline = optimize(pipe, X, y, timeout=30, task='classification')

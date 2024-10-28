import time
from .zero_shot_LLM import generated_zero_shot_pipeline
from .optuna_optimization import optimize

def generate_pipelines(
        X,
        y,
        model="gpt-4o-mini",
        task="classification",
        y_origin= None,
        timeout=300,
):
    for number_of_trials in range(4):
        try:
            first_individual = generated_zero_shot_pipeline(X=X, y=y_origin, task=task, model=model)
            if first_individual:
                # optimize with optimize(pipe, X, y, timeout= 300, task='classification')
                try:
                    best_pipe = optimize(first_individual, X, y, timeout=timeout, task=task)
                except Exception as e:
                    print("Error in Optuna optimization." + str(e))
                    time.sleep(60)
                break
        except Exception as e:
            print("Error in process PPL_LLM." + str(e))
            time.sleep(60)  # Wait 1 minute before next request

    return best_pipe



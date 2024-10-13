from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .pp_llm import generate_pipelines, list_pipelines
from .create_ensemble import create_ensemble_sklearn, create_ensemble_test, create_ensemble_sklearn_str
from .run_llm_code import run_llm_code
from .llmoptimization import optimize_LLM
from .llmensemble import generate_code_embedding, generate_ensemble_manually
from typing import Union
import numpy as np
import pandas as pd
import stopit
import uuid

class PP_LLM():
    """
    Parameters:
    """
    def __init__(
            self,
            dataset_name = None,
            task = 'classification',
            # 'llm_model': "gpt-4",
            llm_model =  "gpt-3.5-turbo",
            iterations = 5,
            max_total_time = 180,
    ) -> None:
        self.dataset_name = dataset_name
        self.llm_model = llm_model
        self.iterations = iterations
        self.task = task
        self.timeout = max_total_time
        self.uid = str(uuid.uuid4())

    def fit(
            self, X, y, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.

        """
        # global list_pipelines # To retrieve at least one result if the timeout is reached
        # Generate a unique UUID
        original_y = y.copy()
        print('uid', self.uid)
        if self.task == "classification":
            y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
            self._label_encoder = LabelEncoder().fit(y_)
            if any(isinstance(yi, str) for yi in y_):
                # If target values are `str` we encode them or scikit-learn will complain.
                y = self._label_encoder.transform(y_)
                self._decoding = True

        self.X_ = X
        self.y_ = y
        try:
            with stopit.ThreadingTimeout(self.timeout):
                list_codeblocks_generated, list_performance_pipelines = generate_pipelines(
                    X,
                    y,
                    # name = self.dataset_name,
                    model=self.llm_model,
                    display_method="markdown",
                    iterations=self.iterations,
                    task = self.task,
                    identifier=self.uid,
                    y_origin = original_y,
                )
            if len(list_pipelines)==1:
                index_best_pipeline = list_performance_pipelines.index(max(list_performance_pipelines))
                self.pipe = list_pipelines[index_best_pipeline] # We get at least 1 pipeline to return
            if len(list_pipelines)<1:
                # self.pipe = create_ensemble_sklearn(list_pipelines, self.task)
                self.pipe = create_ensemble_sklearn_str(X, y, list_codeblocks_generated, self.task)
                # self.pipe.fit(X, y)

            # self.base_models = list_pipelines
            if len(list_pipelines) == 0:
                raise ValueError("Not pipeline could be created")


            print('The model has been created, final fit')
            self.pipe.fit(X, y)

        except stopit.TimeoutException:
            print("Timeout expired")


    def predict(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        # if self.manually_success:  # Only applicable if the model was ensembled manually
        #     # This process is using mlxtend v1
        #     preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
        #     numeric_X = preprocessing_steps[0].fit_transform(X)
        #     X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        if self.task == "classification":
            y = self.pipe.predict(X)  # type: ignore
            # Decode the predicted labels - necessary only if ensemble is not used.
            if y[0] not in list(self._label_encoder.classes_):
                y = self._label_encoder.inverse_transform(y)
            return y
        else:
            return self.pipe.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        # if self.manually_success:  # Only applicable if the model was ensembled manually
        #     # This process is using mlxtend v1
        #     preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
        #     numeric_X = preprocessing_steps[0].fit_transform(X)
        #     X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        return self.pipe.predict_log_proba(X)

    def predict_proba(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        # if self.manually_success: # Only applicable if the model was ensembled manually
        #     # This process is using mlxtend v1
        #     preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
        #     numeric_X = preprocessing_steps[0].fit_transform(X)
        #     X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        return self.pipe.predict_proba(X)







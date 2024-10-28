from sklearn.preprocessing import LabelEncoder
from .pp_llm import generate_pipelines
import pandas as pd
import stopit
import uuid

class PP_LLM():
    """
    Parameters:
    """
    def __init__(
            self,
            task = 'classification',
            llm_model =  "gpt-4o-mini",
            max_total_time = 300,
    ) -> None:
        self.llm_model = llm_model
        self.task = task
        self.timeout = max_total_time
        self.uid = str(uuid.uuid4())

    def fit(
            self, X, y, disable_caafe=False
    ):
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
                self.pipe = generate_pipelines(
                    X,
                    y,
                    model=self.llm_model,
                    task = self.task,
                    y_origin = original_y,
                    timeout = self.timeout-10,
                )

            print('The model has been created, final fit')
            self.pipe.fit(X, y)

        except stopit.TimeoutException:
            print("Timeout expired")


    def predict(self, X):
        if self.task == "classification":
            y = self.pipe.predict(X)  # type: ignore
            # Decode the predicted labels - necessary only if ensemble is not used.
            if y[0] not in list(self._label_encoder.classes_):
                y = self._label_encoder.inverse_transform(y)
            return y
        else:
            return self.pipe.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        return self.pipe.predict_log_proba(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)







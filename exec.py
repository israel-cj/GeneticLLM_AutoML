import logging
import os
import sys
import tempfile as tmp
import pandas as pd
from typing import Union

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


from packaging import version
import sklearn

from frameworks.shared.callee import call_run, result, measure_inference_times
from frameworks.shared.utils import Timer 

###

from ppllm import PP_LLM

###

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** LLM AutoML ****")
    log.info("sklearn == %s", sklearn.__version__)

    log.info('Running LLM AutoML with a maximum time of %ss', config.max_runtime_seconds)
    
    X_train, y_train = dataset.train.X, dataset.train.y
      
    is_classification = (config.type == 'classification')
    if is_classification:
        type_task = 'classification'
        kwargs = {
            'task': type_task,
            'llm_model': "gpt-4o-mini", 
            'max_total_time': config.max_runtime_seconds,
        }
    else:
        type_task = 'regression'
        kwargs = {
            'task': type_task,
            'llm_model': "gpt-4o-mini", 
            'max_total_time': config.max_runtime_seconds,
        }
        
    
    
    automl = PP_LLM(**kwargs)
    with Timer() as training_timer:
        automl.fit(X_train, y_train)
    log.info(f"Finished fit in {training_timer.duration}s.")

    log.info('Predicting on the test set.')
    def infer(data: Union[str, pd.DataFrame]):
        test_data = pd.read_parquet(data) if isinstance(data, str) else data
        predict_fn = automl.predict_proba if is_classification else automl.predict
        return predict_fn(test_data)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, dataset.test.X.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")
  
    X_test, y_test = dataset.test.X, dataset.test.y

    with Timer() as predict_timer:
        predictions = automl.predict(X_test)
    if config.type == 'classification':
        probabilities = automl.predict_proba(X_test)
    else:
        probabilities = None
    log.info(f"Finished predict in {predict_timer.duration}s.")

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        probabilities=probabilities,
        truth=y_test,
        target_is_encoded=False,
        training_duration=training_timer.duration,
        predict_duration=predict_timer.duration,
        inference_times=inference_times,
    )
    
    
if __name__ == '__main__':
    call_run(run)
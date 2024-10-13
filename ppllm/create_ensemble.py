from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

def run_llm_code_stacker(code):
    try:
        globals_dict = {}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        pipe = output['pipe']
        print(pipe)

    except Exception as e:
        pipe = None
        print("Code could not be executed", e)

    return pipe

def create_ensemble_test(list_models, task):
    if task=="classification":
        ensemble = StackingCVClassifier(
            classifiers=list_models,
            meta_classifier=LogisticRegression(),
            use_features_in_secondary=True,
        )

    if task=="regression":
        ensemble = StackingRegressor(
            regressors=list_models,
            meta_regressor=SVR(kernel='rbf'),
            use_features_in_secondary=True,
        )

    return ensemble

def create_ensemble_sklearn(list_models, task):
    if task == "classification":
        ensemble = StackingClassifier(
            estimators=list_models,
            final_estimator=LogisticRegression(),
            stack_method='auto',
            passthrough=True,
        )

    if task == "regression":
        ensemble = StackingRegressor(
            estimators=list_models,
            final_estimator=SVR(kernel='rbf'),
            passthrough=True,
        )

    return ensemble

def create_ensemble_sklearn_str(X, y, list_models_str, task):
    list_models = []
    for i, element in enumerate(list_models_str):
        element = element.replace('pipe.fit(X_train, y_train)', '')
        this_model = run_llm_code_stacker(element)
        if this_model is not None:
            list_models.append((f'model{i}', this_model))

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols)>0:
        ordinal_cols = [col for col in categorical_cols if X[col].nunique() <= 2]
        onehot_cols = [col for col in categorical_cols if 2 < X[col].nunique() <= 10]

        preprocessor = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
            ], sparse_threshold=0)

    if task == "classification":
        ensemble = StackingClassifier(
            estimators=list_models,
            final_estimator=LogisticRegression(),
            stack_method='auto',
            passthrough=True,
        )

    if task == "regression":
        ensemble = StackingRegressor(
            estimators=list_models,
            final_estimator=SVR(kernel='rbf'),
            passthrough=True,
        )
    final_pipeline = ensemble
    if len(categorical_cols) > 0:
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('ensemble', ensemble)])
    final_pipeline.fit(X,y)

    return final_pipeline

# def create_ensemble_sklearn_str(X, y, list_models_str, task):
#     # list_models = []
#     # for element in list_models_str:
#     #     element = element.replace('pipe.fit(X_train, y_train)', '')
#     #     this_model = run_llm_code_stacker(element)
#     #     if this_model is not None:
#     #         list_models.append(this_model)
#     list_models = []
#     for i, element in enumerate(list_models_str):
#         element = element.replace('pipe.fit(X_train, y_train)', '')
#         this_model = run_llm_code_stacker(element)
#         if this_model is not None:
#             list_models.append((f'model{i}', this_model))
#
#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns
#     ordinal_cols = [col for col in categorical_cols if X[col].nunique() <= 2]
#     onehot_cols = [col for col in categorical_cols if 2 < X[col].nunique() <= 10]
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
#             ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
#         ])
#
#     if task == "classification":
#         ensemble = StackingClassifier(
#             estimators=list_models,
#             final_estimator=LogisticRegression(),
#             stack_method='auto',
#             passthrough=True,
#         )
#
#     if task == "regression":
#         ensemble = StackingRegressor(
#             estimators=list_models,
#             final_estimator=SVR(kernel='rbf'),
#             passthrough=True,
#         )
#
#     final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                      ('ensemble', ensemble)])
#     final_pipeline.fit(X,y)
#
#     return final_pipeline
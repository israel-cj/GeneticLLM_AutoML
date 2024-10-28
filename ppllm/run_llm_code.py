import pandas as pd

def run_llm_code_preprocessing(code, X_train, y_train):
    try:
        globals_dict = {'X_train': X_train, 'y_train': y_train}
        output = {}
        exec(code, globals_dict, output)
        pipe = output['pipe']
        pipe.fit(X_train, y_train)
        print(pipe)

    except Exception as e:
        print("Code could not be executed", e)
        raise e

    return pipe

# def run_llm_code_preprocessing(code, X_train, y_train):
#     try:
#         globals_dict = {'X_train': X_train, 'y_train': y_train}
#         output = {}
#         exec(code, globals_dict, output)
#         pipe = output['pipe']
#         pipe.fit(X_train, y_train)
#         print(pipe)
#
#     except Exception as e:
#         print("Code could not be executed", e)
#         raise e
#
#     return pipe

# def run_llm_code_preprocessing(code, X_train, y_train):
#     X_transformed = X_train.copy() # If we don't requiere encoders
#     categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
#     if len(categorical_cols)>0:
#         ordinal_cols = [col for col in categorical_cols if X_train[col].nunique() <= 2]
#         onehot_cols = [col for col in categorical_cols if 2 < X_train[col].nunique() <= 10]
#
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
#             ], sparse_threshold=0)
#
#         # Fit the preprocessor and transform the data
#         X_transformed = preprocessor.fit_transform(X_train)
#
#     # X_transformed = pd.DataFrame(X_transformed, columns=ordinal_cols + onehot_cols)
#     try:
#         globals_dict = {'X_train': X_transformed, 'y_train': y_train}
#         output = {}
#         exec(code, globals_dict, output)
#         #output = {}
#         #exec(code, None, output)
#         pipe = output['pipe']
#         # Prepend preprocessor to the pipe's steps
#         if len(categorical_cols) > 0:
#             pipe.steps.insert(0, ('preprocessor_basic', preprocessor))
#         pipe.fit(X_train, y_train)
#         print(pipe)
#
#     except Exception as e:
#         print("Code could not be executed", e)
#         raise (e)
#
#     return pipe

# def run_llm_code_preprocessing(code, X_train, y_train, pipe=None):
#     """
#     Executes the given code on the given dataframe and returns the resulting dataframe.
#
#     Parameters:
#     code (str): The code to execute.
#     df (pandas.DataFrame): The dataframe to execute the code on.
#     convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
#     fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.
#
#     Returns:
#     pandas.DataFrame: The resulting dataframe after executing the code.
#     """
#     # Define preprocessing for categorical columns
#     # categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
#     # ordinal_cols = [col for col in categorical_cols if X_train[col].nunique() <= 2 or (X_train[col].nunique() > 10 and y_train.dtype == 'object')]
#     #onehot_cols = [col for col in categorical_cols if 2 < X_train[col].nunique() <= 10]
#     categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
#     ordinal_cols = [col for col in categorical_cols if X_train[col].nunique() <= 2]
#     onehot_cols = [col for col in categorical_cols if 2 < X_train[col].nunique() <= 10]
#
#     # preprocessor = ColumnTransformer(
#     #     transformers=[
#     #         ('ord', OrdinalEncoder(), ordinal_cols),
#     #         ('onehot', OneHotEncoder(), onehot_cols)
#     #     ])
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
#             ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
#         ])
#
#     # Fit the preprocessor and transform the data
#     X_transformed = preprocessor.fit_transform(X_train)
#     # X_transformed = pd.DataFrame(X_transformed, columns=ordinal_cols + onehot_cols)
#     try:
#         globals_dict = {'X_train': X_transformed, 'y_train': y_train}
#         output = {}
#         exec(code, globals_dict, output)
#         #output = {}
#         #exec(code, None, output)
#         pipe = output['pipe']
#         # Prepend preprocessor to the pipe's steps
#         pipe.steps.insert(0, ('preprocessor_basic', preprocessor))
#         pipe.fit(X_train, y_train)
#         print(pipe)
#
#     except Exception as e:
#         print("Code could not be executed", e)
#         raise (e)
#
#     return pipe

def run_llm_code(code, X_train, y_train, pipe=None):
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:

        globals_dict = {'X_train': X_train, 'y_train': y_train}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        # Use the resulting pipe object
        pipe = output['pipe']
        print(pipe)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return pipe


def run_llm_code_ensemble(code, X_train, y_train, list_pipelines, model=None):
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:

        globals_dict = {'X_train': X_train, 'y_train': y_train, 'list_pipelines':list_pipelines}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        # Use the resulting pipe object
        model = output['model']
        print(model)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return model

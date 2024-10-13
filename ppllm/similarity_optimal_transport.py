import random
import pickle
import ott
import gdown
import re
from ott.geometry import pointcloud
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from ott.problems.quadratic import quadratic_problem
from sklearn.decomposition import FastICA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dirty_cat import TableVectorizer
from scipy.sparse import isspmatrix

from .data_automlbenchmark import data_automl
from .data_automlbenchmark_regression import data_automl_regression

def extract_preprocessing_from_pipe(whole_pipeline):
    pipeline_string = whole_pipeline
    # Find the indices of the first "(" and the second to last ")"
    start_index = pipeline_string.find('(') + 1
    indices = []
    index = pipeline_string.rfind(')')

    while index != -1:
        indices.append(index)
        index = pipeline_string.rfind(')', 0, index)
    if len(indices) > 1:
        indices = sorted(indices)
        end_index = indices[-2]
        # Extract the information between the indices
        extracted_info = pipeline_string[start_index:end_index + 1]

        return extracted_info
    else:
        return " "


# computing the valur for optimal transport
def compute_value_ot(X_train, y_train):
    max_samples = 5000
    if X_train.shape[1] > 1000:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 1000 and X_train.shape[
        1] > 500:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1000, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > 2000 and X_train.shape[
        1] > 100:  # If the dimensionality of the dataset is to large we limited even more
        min_size_sample = min(1500, len(X_train) - 1)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=min_size_sample)

    if len(X_train) > max_samples:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=max_samples)

        # y_train = y_train.sparse.to_dense() # There is at least 1 dataset in openml with y as sparse float pandas
    table_vec = TableVectorizer(auto_cast=True)
    try:
        transformed_data = table_vec.fit_transform(X_train, y_train)

        if isspmatrix(transformed_data):
            transformed_data = transformed_data.toarray()
        imp_mean = SimpleImputer(strategy='mean')
        imp_mean.fit(transformed_data)
        transformed_data = imp_mean.transform(transformed_data)
        # The transformed_data is now a NumPy array with all preprocessing applied
        transformed_dataset = FastICA().fit_transform(transformed_data)
        geom_xx = pointcloud.PointCloud(transformed_dataset)
        # Print the joint probability
        print(f"Optimal transport value for this dataset: {geom_xx}")
        return geom_xx
    except Exception as e:
        print(e)
        return None

def TransferedPipelines(X_train, y_train, number_of_pipelines=5, task='classification'):
    # Define how many similar task we are going to consider
    search_similar_datasets = 3
    # Load the JSON data which contains the pipelines from the AutoMLBenchmark
    if task=='classification':
        url = 'https://drive.google.com/uc?id=1in9vbqghlaqFXAqiRoF7qnbthOq9hm4n'
        output = 'classification_optimal_transport_light.pickle'
        gdown.download(url, output, quiet=False)
        with open('classification_optimal_transport_light.pickle', 'rb') as handle:
            dictionary_classification = pickle.load(handle)
        automl_benchmark_data = data_automl
        automl_description = dictionary_classification
    else:
        url = 'https://drive.google.com/uc?id=1fh8lNMb5iLADwiQ0vJeEnH-TGeFFNuHY'
        output = 'regression_optimal_transport_light.pickle'
        gdown.download(url, output, quiet=False)
        with open('regression_optimal_transport_light.pickle', 'rb') as handle:
            dictionary_regression = pickle.load(handle)
        automl_benchmark_data = data_automl_regression
        automl_description = dictionary_regression

    list_name_datasets = list(automl_description.keys())
    list_geom_yy_tuple = list(automl_description.items())
    list_geom_yy = [t[1] for t in list_geom_yy_tuple]

    geom_xx  = compute_value_ot(X_train, y_train)
    if geom_xx is None:
        names_other_similar_datasets = list_name_datasets[:3] # three first datasets if there is a weird bug I need to fix
    else:
        # Look for the cost of each dataset vs the new dataset
        costs = []
        for geom_yy in list_geom_yy[:5]: # change this
        # for geom_yy in list_geom_yy:  # change this
            prob = ott.problems.quadratic.quadratic_problem.QuadraticProblem(geom_xx, geom_yy)
            try:
                solver = gromov_wasserstein.GromovWasserstein(rank=6) # this is the proposal of prabhant
            except Exception as e:
                print(e)
                solver = gromov_wasserstein_lr.LRGromovWasserstein(rank=6) # This is cuz we got the error "AssertionError: For low-rank GW, use `ott.solvers.quadratic.gromov_wasserstein_lr.LRGromovWasserstein"
            ot_gwlr = solver(prob)
            cost = ot_gwlr.costs[ot_gwlr.costs > 0][-1]
            costs.append(cost)

        distance = min(costs)
        similar_task_name = list_name_datasets[costs.index(distance)]  # index with the lowest distance
        print('The most similar dataset name is: ', similar_task_name)
        min_other_similar_datasets = sorted(costs)[:search_similar_datasets]
        names_other_similar_datasets = [list_name_datasets[costs.index(index)] for index in min_other_similar_datasets]

    most_similar_dataset = []
    for similar_datasets in names_other_similar_datasets:
        if similar_datasets == 'numerai28.6':
            similar_datasets = 'numerai28_6'
        this_similar = automl_benchmark_data[similar_datasets][:3]
        most_similar_dataset += this_similar

    if len(most_similar_dataset) <= number_of_pipelines:
        this_list = most_similar_dataset
        this_list_preprocessing = [extract_preprocessing_from_pipe(element) for element in this_list] # Extract only the preprocessing steps
        return '\n'.join(this_list_preprocessing)
    else:
        # Random pipelines to add exploration :)
        this_list = random.sample(most_similar_dataset, number_of_pipelines)
        this_list_preprocessing = [extract_preprocessing_from_pipe(element) for element in this_list]  # Extract only the preprocessing steps
        return '\n'.join(this_list_preprocessing)

if __name__=='__main__':
    import openml
    dataset = openml.datasets.get_dataset(40983) # 40983 = 'Wilt'
    # dataset = openml.datasets.get_dataset(990)  # 990 = 'eucalyptus'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    this_pipelines = TransferedPipelines(X_train=X, y_train=y, task='classification', number_of_pipelines=3)
    print(this_pipelines)
    
    
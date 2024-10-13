import random
from .data_automlbenchmark import data_automl
from .data_automlbenchmark_regression import data_automl_regression
from .joint_probability_classification import dictionary_classification
from .joint_probability_regression import dictionary_regression

def compute_probability(X):
    df = X
    num_rows = df.shape[0]
    # Calculate the joint probability of the dataset
    joint_prob = 1
    for col in df.columns:
        unique_values = df[col].nunique()
        prob = unique_values / num_rows
        joint_prob *= prob
    return joint_prob

def TransferedPipelines(dataset_X=None, number_of_pipelines=5, task='classification'):
    # Define how many similar task we are going to consider
    search_similar_datasets = 10
    # Load the JSON data which contains the pipelines from the AutoMLBenchmark
    if task=='classification':
        automl_benchmark_data = data_automl
        automl_description = dictionary_classification
    else:
        automl_benchmark_data = data_automl_regression
        automl_description = dictionary_regression

    list_name_datasets = list(automl_description.keys())
    list_joint_probability_tuple = list(automl_description.items())
    list_joint_probability = [t[1] for t in list_joint_probability_tuple]

    def closest_index(number, lst):
        return min(range(len(lst)), key=lambda i: abs(lst[i] - number))

    this_probability = compute_probability(dataset_X)
    # Calculate the absolute differences between each element and this_probability
    scores = [abs(p - this_probability) for p in list_joint_probability]
    #scores = sorted(range(len(list_joint_probability)), key=lambda i: abs(list_joint_probability[i] - this_probability))

    # Combine docs & scores by names
    score_pairs_names = list(zip(list_name_datasets, scores))

    # Sort by decreasing score
    score_pairs = sorted(score_pairs_names, key=lambda x: x[1])
    # Retrieve the pipelines of the task name with the most similar description
    similar_task_name = score_pairs[0][0]
    print('The most similar dataset name is: ', similar_task_name)
    list_other_similar_datasets = [score_pairs[i+1][0] for i in range(search_similar_datasets)] # Not including the most similar
    # let's use 2 (arbirtray) elements for each one of the 'number_of_pipelines' not first most similar pipelines
    most_similar_dataset =  []
    for similar_datasets in list_other_similar_datasets:
        if similar_datasets == 'numerai28.6':
            similar_datasets = 'numerai28_6'
        this_similar = automl_benchmark_data[similar_datasets][:3]
        most_similar_dataset+=this_similar

    if len(most_similar_dataset) <= number_of_pipelines:
        this_list = most_similar_dataset
        return '\n'.join(this_list)
    else:
        # Random pipelines to add exploration :)
        this_list = random.sample(most_similar_dataset, number_of_pipelines)
        return '\n'.join(this_list)

if __name__=='__main__':
    import openml
    # dataset = openml.datasets.get_dataset(40983) # 40983 = 'Wilt'
    dataset = openml.datasets.get_dataset(990)  # 990 = 'eucalyptus'
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    this_pipelines = TransferedPipelines(dataset_X=X, number_of_pipelines=5)
    print(this_pipelines)
    
    
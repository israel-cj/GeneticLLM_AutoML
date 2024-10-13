import requests
import random
from sentence_transformers import SentenceTransformer, util
from .data_automlbenchmark import data_automl
from .data_automlbenchmark_regression import data_automl_regression
from .description_classification import dictionary_classification
from .description_regression import dictionary_regression

def TransferedPipelines(description_dataset=None, number_of_pipelines=5, task='classification'):
    # Load the JSON data which contains the pipelines from the AutoMLBenchmark
    if task=='classification':
        automl_benchmark_data = data_automl
        automl_description = dictionary_classification
    else:
        automl_benchmark_data = data_automl_regression
        automl_description = dictionary_regression

    list_name_datasets = list(automl_description.keys())
    list_description_datasets_tuples = list(automl_description.items())
    list_description_datasets = [t[1] for t in list_description_datasets_tuples]

    query = description_dataset
    docs = list_description_datasets
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    # Encode query and documents
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    # Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    # Combine docs & scores by names
    doc_score_pairs_names = list(zip(list_name_datasets, scores))

    # Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs_names, key=lambda x: x[1], reverse=True)

    # Retrieve the pipelines of the task name with the most similar description
    similar_task_name = doc_score_pairs[0][0]
    print('The most similar dataset name is: ', similar_task_name)
    most_similar_dataset =  automl_benchmark_data[similar_task_name]

    if len(most_similar_dataset)<=number_of_pipelines:
        this_list = most_similar_dataset
        return '\n'.join(this_list)
    else:
        this_list= random.sample(most_similar_dataset, number_of_pipelines)
        return '\n'.join(this_list)
    
        
if __name__=='__main__':
    import openml
    dataset = openml.datasets.get_dataset(1111) # 1111 = 'KDDCup09_appetency'
    this_pipelines = TransferedPipelines(description_dataset=dataset.description, number_of_pipelines=5)
    print(this_pipelines)
    
    
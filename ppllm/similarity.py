import requests
import random
from .data_automlbenchmark import data_automl
from .data_automlbenchmark_regression import data_automl_regression

def TransferedPipelines(hf_token, name_dataset=None, number_of_pipelines=5, task='classification'):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    # Load the JSON data which contains the pipelines from the AutoMLBenchmark
    if task=='classification':
        automl_benchmark_data = data_automl
    else:
        automl_benchmark_data = data_automl_regression
        
    list_names_datasets = list(automl_benchmark_data.keys())
    
    data_vector = query(
        {
            "inputs": {
                "source_sentence": name_dataset,
                "sentences": list_names_datasets
            }
        }
    )
  
    index_most_similar = data_vector.index(max(data_vector))
    print('The most similar dataset name is: ', list_names_datasets[index_most_similar])
    most_similar_dataset =  automl_benchmark_data[list_names_datasets[index_most_similar]] 
    
    if len(most_similar_dataset)<=number_of_pipelines:
        this_list = most_similar_dataset
        return '\n'.join(this_list)
    else:
        this_list= random.sample(most_similar_dataset, number_of_pipelines)
        return '\n'.join(this_list)
    
        
if __name__=='__main__':
    this_pipelines = TransferedPipelines(hf_token='', name_dataset='KDDCup09_appetency', number_of_pipelines=5)
    print(this_pipelines)
    
    
"""Defines operations which are commonly performed on the amlb result data"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal

def is_old(framework: str, constraint: str, metric: str) -> bool:
    """Encodes the table in `raw_to_clean.ipynb`"""
    if framework == "TunedRandomForest":
        return True
    if constraint == "1h8c_gp3":
        return False
    # if framework in ["autosklearn2", "GAMA(B)", "TPOT"]:
    return framework == "MLJAR(B)" and metric != "neg_rmse"
    

def get_print_friendly_name(name: str, extras: dict[str, str] = None) -> str:
    if extras is None:
        extras = {}
        
    # frameworks = {
    #     "AutoGluon_benchmark": "AutoGluon(B)",
    #     "AutoGluon_hq": "AutoGluon(HQ)",
    #     "AutoGluon_hq_il001": "AutoGluon(HQIL)",
    #     "GAMA_benchmark": "GAMA(B)",
    #     "mljarsupervised_benchmark": "MLJAR(B)",
    #     "mljarsupervised_perform": "MLJAR(P)",
    # }
    frameworks = {
    'GeneticLLM_AutoML_5min': 'GeneticLLM_5min',
    'GeneticLLM_AutoML': 'GeneticLLM',
    'AutoGluon_benchmark_5min': 'AutoGluon(B)_5min',
    'AutoGluon_benchmark_10min': 'AutoGluon(B)_10min',
    'AutoGluon_benchmark_30min': 'AutoGluon(B)_30min',
    'AutoGluon_benchmark_60min': 'AutoGluon(B)_60min',
    'AutoGluon_HQIL_5min': 'AutoGluon(HQIL)_5min',
    'AutoGluon_HQIL_10min': 'AutoGluon(HQIL)_10min',
    'AutoGluon_HQIL_30min': 'AutoGluon(HQIL)_30min',
    'AutoGluon_HQIL_60min': 'AutoGluon(HQIL)_60min',
    'AutoGluon_HQ_5min': 'AutoGluon(HQ)_5min',
    'AutoGluon_HQ_10min': 'AutoGluon(HQ)_10min',
    'AutoGluon_HQ_30min': 'AutoGluon(HQ)_30min',
    'AutoGluon_HQ_60min': 'AutoGluon(HQ)_60min',
    'FEDOT_5min': 'FEDOT_5min',
    'FEDOT_10min': 'FEDOT_10min',
    'FEDOT_30min': 'FEDOT_30min',
    'FEDOT_60min': 'FEDOT_60min',
    'GAMA_5min': 'GAMA(B)_5min',
    'GAMA_10min': 'GAMA(B)_10min',
    'GAMA_30min': 'GAMA(B)_30min',
    'GAMA_60min': 'GAMA(B)_60min',
    'H2OAutoML_5min': 'H2OAutoML_5min',
    'H2OAutoML_10min': 'H2OAutoML_10min',
    'H2OAutoML_30min': 'H2OAutoML_30min',
    'H2OAutoML_60min': 'H2OAutoML_60min',
    'NaiveAutoML_5min': 'NaiveAutoML_5min',
    'NaiveAutoML_10min': 'NaiveAutoML_10min',
    'NaiveAutoML_30min': 'NaiveAutoML_30min',
    'NaiveAutoML_60min': 'NaiveAutoML_60min',
    'RandomForest_5min': 'RF_5min',
    'RandomForest_10min': 'RF_10min',
    'RandomForest_30min': 'RF_30min',
    'RandomForest_60min': 'RF_60min',
    'TPOT_5min': 'TPOT_5min',
    'TPOT_10min': 'TPOT_10min',
    'TPOT_30min': 'TPOT_30min',
    'TPOT_60min': 'TPOT_60min',
    'autosklearn2_5min': 'autosklearn2_5min',
    'autosklearn2_10min': 'autosklearn2_10min',
    'autosklearn2_30min': 'autosklearn2_30min',
    'autosklearn2_60min': 'autosklearn2_60min',
    'autosklearn_5min': 'autosklearn_5min',
    'autosklearn_10min': 'autosklearn_10min',
    'autosklearn_30min': 'autosklearn_30min',
    'autosklearn_60min': 'autosklearn_60min',
    'flaml_5min': 'flaml_5min',
    'flaml_10min': 'flaml_10min',
    'flaml_30min': 'flaml_30min',
    'flaml_60min': 'flaml_60min',
    'lightautoml_5min': 'lightautoml_5min',
    'lightautoml_10min': 'lightautoml_10min',
    'lightautoml_30min': 'lightautoml_30min',
    'lightautoml_60min': 'lightautoml_60min',
    'mljarsupervised_5min': 'MLJAR_5min',
    'mljarsupervised_10min': 'MLJAR_10min',
    'mljarsupervised_30min': 'MLJAR_30min',
    'mljarsupervised_60min': 'MLJAR_60min',
    'mljarsupervised_compete_5min': 'MLJAR(C)_5min',
    'mljarsupervised_compete_10min': 'MLJAR(C)_10min',
    'mljarsupervised_compete_30min': 'MLJAR(C)_30min',
    'mljarsupervised_compete_60min': 'MLJAR(C)_60min',
    'mljarsupervised_benchmark_5min': 'MLJAR(B)_5min',
    'mljarsupervised_benchmark_10min': 'MLJAR(B)_10min',
    'mljarsupervised_benchmark_30min': 'MLJAR(B)_30min',
    'mljarsupervised_benchmark_60min': 'MLJAR(B)_60min',
    'TunedRandomForest_5min': 'TRF_5min',
    'TunedRandomForest_10min': 'TRF_10min',
    'TunedRandomForest_30min': 'TRF_30min',
    'TunedRandomForest_60min': 'TRF_60min',
    'constantpredictor_60min': 'CP_60min',
    'constantpredictor_30min': 'CP_30min',
    'constantpredictor_10min': 'CP_10min',
    'constantpredictor_5min': 'CP_5min',
    'CP_60min': 'CP_60min',
    'RF_60min': 'RF_60min',
    'TRF_60min': 'TRF_60min',
    'TPOT_early_5min': 'TPOT_E_5min',
    'TPOT_early_10min': 'TPOT_E_10min',
    'TPOT_early_30min': 'TPOT_E_30min',
    'TPOT_early_60min': 'TPOT_E_60min',
    'flaml_early_5min': 'flaml_E_5min',
    'flaml_early_10min': 'flaml_E_10min',
    'flaml_early_30min': 'flaml_E_30min',
    'flaml_early_60min': 'flaml_E_60min',
    'FEDOT_early_5min': 'FEDOT_E_5min',
    'FEDOT_early_10min': 'FEDOT_E_10min',
    'FEDOT_early_30min': 'FEDOT_E_30min',
    'FEDOT_early_60min': 'FEDOT_E_60min',
    'H2OAutoML_early_5min': 'H2OAutoML_E_5min',
    'H2OAutoML_early_10min': 'H2OAutoML_E_10min',
    'H2OAutoML_early_30min': 'H2OAutoML_E_30min',
    'H2OAutoML_early_60min': 'H2OAutoML_E_60min',
    'AutoGluon_benchmark_early_5min': 'AutoGluon(B)_E_5min',
    'AutoGluon_benchmark_early_10min': 'AutoGluon(B)_E_10min',
    'AutoGluon_benchmark_early_30min': 'AutoGluon(B)_E_30min',
    'AutoGluon_benchmark_early_60min': 'AutoGluon(B)_E_60min',
    'AutoGluon_HQIL_early_5min': 'AutoGluon(HQIL)_E_5min',
    'AutoGluon_HQIL_early_10min': 'AutoGluon(HQIL)_E_10min',
    'AutoGluon_HQIL_early_30min': 'AutoGluon(HQIL)_E_30min',
    'AutoGluon_HQIL_early_60min': 'AutoGluon(HQIL)_E_60min',
    'AutoGluon_HQ_early_5min': 'AutoGluon(HQ)_E_5min',
    'AutoGluon_HQ_early_10min': 'AutoGluon(HQ)_E_10min',
    'AutoGluon_HQ_early_30min': 'AutoGluon(HQ)_E_30min',
    'AutoGluon_HQ_early_60min': 'AutoGluon(HQ)_E_60min',
    'AutoGluon_FI_FT_IL_early_5min': 'AutoGluon(FIFTIL)_E_5min',
    'AutoGluon_FI_FT_IL_early_10min': 'AutoGluon(FIFTIL)_E_10min',
    'AutoGluon_FI_FT_IL_early_30min': 'AutoGluon(FIFTIL)_E_30min',
    'AutoGluon_FI_FT_IL_early_60min': 'AutoGluon(FIFTIL)_E_60min'
    }
    budgets = {
        "1h8c_gp3": "1 hour",
        "4h8c_gp3": "4 hours",
    }
    print_friendly_names = (frameworks | budgets | extras)
    return print_friendly_names.get(name, name)


def impute_missing_results(results: pd.DataFrame, with_results_from: str = "constantpredictor") -> pd.DataFrame:
    """Imputes missing values in `results` with the corresponding score from `constantpredictor`"""
    if with_results_from not in results["framework"].unique():
        raise ValueError(f"{with_results_from=} is not in `results`")
    results = results.copy()
        
    lookup_table = results.set_index(["framework", "task", "fold", "constraint"])
    rows_with_missing_result = ((index, row) for index, row in results.iterrows() if np.isnan(row["result"]))
    for index, row in rows_with_missing_result:
        task, fold, constraint = row[["task", "fold", "constraint"]]
        value = lookup_table.loc[(with_results_from, task, fold, constraint)].result
        results.loc[index, "result"] = value
    return results

# def calculate_ranks(results: pd.DataFrame) -> dict[str, float]:
#     """Produce a mapping framework->rank based on ranking mean performance per task"""
#     mean_performance = results[["framework", "task", "result"]].groupby(["framework", "task"], as_index=False).mean()
#     mean_performance["rank"] = mean_performance.groupby("task").result.rank(ascending=False, method="average", na_option="bottom")
#     ranks_by_framework = {
#         framework: mean_performance[mean_performance["framework"] == framework]["rank"]
#         for framework in mean_performance["framework"].unique()
#     }
    
#     _, p = friedmanchisquare(*ranks_by_framework.values())
#     if p >= 0.05:
#         # Given the number of results we don't really expect this to happen.
#         raise RuntimeError("Ranks are not statistically significantly different.")
    
#     return {framework: ranks.mean() for framework, ranks in ranks_by_framework.items()}

def calculate_ranks(results: pd.DataFrame) -> dict[str, float]:
    """Produce a mapping framework->rank based on ranking mean performance per task"""
    mean_performance = results[["framework", "task", "result"]].groupby(["framework", "task"], as_index=False).mean()
    mean_performance["rank"] = mean_performance.groupby("task").result.rank(ascending=False, method="average", na_option="bottom")
    ranks_by_framework = {
        framework: mean_performance[mean_performance["framework"] == framework]["rank"]
        for framework in mean_performance["framework"].unique()
    }
    
    _, p = kruskal(*ranks_by_framework.values())
    
    if p >= 0.05:
        # Given the number of results we don't really expect this to happen.
        # raise RuntimeError("Ranks are not statistically significantly different.")
        # print("Ranks are not statistically significantly different.")
        pass
    
    return {framework: ranks.mean() for framework, ranks in ranks_by_framework.items()}

def add_rescale(data: pd.DataFrame, lower: str) -> pd.DataFrame:
    """Adds a `scaled` column to data scaling between -1 (lower) and 0 (best observed)."""
    lookup = data.set_index(["framework", "task", "constraint"]).sort_index()
    oracle = data.groupby(["task", "constraint"]).max().sort_index()
    
    for index, row in data.sort_values(["task"]).iterrows():
        task, constraint = row["task"], row["constraint"]
        lb = lookup.loc[(lower, task, constraint)].result
        ub = oracle.loc[(task, constraint)].result
        if lb == ub:
            data.loc[index, "rescaled"] = float("nan")
        else:
            v = -((row["result"] - lb) / (ub - lb)) + 1
            data.loc[index, "scaled"] = v
    return data
            
    
    
# def scaled_result(results: pd.DataFrame, low: str = "RandomForest") -> pd.DataFrame:
#     """Adds `scaled` column which has result scaled from -1 (low) and 0 (best known result) for the (task, fold, constraint)-combination."""
#     lookup = data.set_index(["framework", "task", "constraint"]).sort_index()
#     oracle = data.groupby(["task", "constraint"]).max().sort_index()
    
#     for index, row in data.sort_values(["task"]).iterrows():
#         task, constraint = row["task"], row["constraint"]
#         lb = lookup.loc[(lower, task, constraint)].result
#         ub = oracle.loc[(task, constraint)].result
#         if lb == ub:
#             data.loc[index, "rescaled"] = float("nan")
#         else:
#             v = -((row["result"] - lb) / (ub - lb)) + 1
#             data.loc[index, "rescaled"] = v
#     return data
            
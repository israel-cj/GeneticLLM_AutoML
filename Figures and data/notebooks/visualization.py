import seaborn as sns
import matplotlib.colors as mcolors


frameworks_with_constraints = ['AutoGluon_benchmark_5min', 'AutoGluon_benchmark_10min', 'AutoGluon_benchmark_30min', 'AutoGluon_benchmark_60min', 'AutoGluon_HQIL_5min', 'AutoGluon_HQIL_10min', 'AutoGluon_HQIL_30min', 'AutoGluon_HQIL_60min', 'AutoGluon_HQ_5min', 'AutoGluon_HQ_10min', 'AutoGluon_HQ_30min', 'AutoGluon_HQ_60min', 'FEDOT_5min', 'FEDOT_10min', 'FEDOT_30min', 'FEDOT_60min', 'GAMA_5min', 'GAMA_10min', 'GAMA_30min', 'GAMA_60min', 'H2OAutoML_5min', 'H2OAutoML_10min', 'H2OAutoML_30min', 'H2OAutoML_60min', 'NaiveAutoML_5min', 'NaiveAutoML_10min', 'NaiveAutoML_30min', 'NaiveAutoML_60min', 'RandomForest_5min', 'RandomForest_10min', 'RandomForest_30min', 'RandomForest_60min', 'TPOT_5min', 'TPOT_10min', 'TPOT_30min', 'TPOT_60min', 'autosklearn2_5min', 'autosklearn2_10min', 'autosklearn2_30min', 'autosklearn2_60min', 'autosklearn_5min', 'autosklearn_10min', 'autosklearn_30min', 'autosklearn_60min', 'flaml_5min', 'flaml_10min', 'flaml_30min', 'flaml_60min', 'lightautoml_5min', 'lightautoml_10min', 'lightautoml_30min', 'lightautoml_60min', 'mljarsupervised_5min', 'mljarsupervised_10min', 'mljarsupervised_30min', 'mljarsupervised_60min', 'mljarsupervised_compete_5min', 'mljarsupervised_compete_10min', 'mljarsupervised_compete_30min', 'mljarsupervised_compete_60min', 'mljarsupervised_benchmark_5min', 'mljarsupervised_benchmark_10min', 'mljarsupervised_benchmark_30min', 'mljarsupervised_benchmark_60min', 'TunedRandomForest_5min', 'TunedRandomForest_10min', 'TunedRandomForest_30min', 'TunedRandomForest_60min', 'constantpredictor_60min', '5min', '10min', '30min', '60min', 'CP_60min', 'RF_60min', 'TRF_60min', 'GeneticLLM_AutoML', 'GeneticLLM_5min', 'GeneticLLM_b', 'GeneticLLM_m', 'GeneticLLM_r', 'CP_b', 'CP_m', 'CP_r'] 

early_stopping_frameworks = ['TPOT_early_5min', 'TPOT_early_10min', 'TPOT_early_30min', 'TPOT_early_60min', 'flaml_early_5min', 'flaml_early_10min',  'flaml_early_30min', 'flaml_early_60min', 'FEDOT_early_5min', 'FEDOT_early_10min', 'FEDOT_early_30min', 'FEDOT_early_60min', 'H2OAutoML_early_5min', 'H2OAutoML_early_10min', 'H2OAutoML_early_30min', 'H2OAutoML_early_60min', 'AutoGluon_benchmark_early_5min', 'AutoGluon_benchmark_early_10min', 'AutoGluon_benchmark_early_30min', 'AutoGluon_benchmark_early_60min', 'AutoGluon_HQIL_early_5min', 'AutoGluon_HQIL_early_10min', 'AutoGluon_HQIL_early_30min', 'AutoGluon_HQIL_early_60min', 'AutoGluon_HQ_early_5min', 'AutoGluon_HQ_early_10min', 'AutoGluon_HQ_early_30min', 'AutoGluon_HQ_early_60min', 'AutoGluon_FI_FT_IL_early_5min', 'AutoGluon_FI_FT_IL_early_10min', 'AutoGluon_FI_FT_IL_early_30min', 'AutoGluon_FI_FT_IL_early_60min',]

frameworks_with_constraints += early_stopping_frameworks

# Generate a color palette
colors = sns.color_palette("Paired", len(frameworks_with_constraints))

# Create a dictionary mapping each framework_with_constraints to a color
framework_colors = {framework: color for framework, color in zip(frameworks_with_constraints, colors)}

# Convert colors to hexadecimal
FRAMEWORK_TO_COLOR = {framework: mcolors.to_hex(color) for framework, color in framework_colors.items()}

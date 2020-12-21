from train_test import load_models, evaluate, plot_pr_curve, plot_roc_curve
import pandas as pd
from data_preparation import read_data

cm_path = 'E:/project/'
model_path = "E:/project/models"
data_path = "E:/project/training_data.csv"
data,labels = read_data(data_path)
models = load_models(model_path)

result_table = pd.DataFrame(columns=['classifiers','accuracy', 'fpr','tpr','auc', 'precision', 'recall', 'f1 score', 'AP'])
cm_path = "E:/project/results/"

for name, model in models:
    acc, fpr, tpr, roc_auc, precision, recall, f1 , log_los, ap = evaluate(name, model, data, labels, cm_path)
    result_table  = result_table.append({'classifiers': name,
                                           'accuracy' : f"{acc*100}%",
                                           'fpr':fpr,
                                           'tpr':tpr,
                                           'auc':roc_auc,
                                           'precision' : precision,
                                           'recall' : recall,
                                           'f1 score' : f1,
                                          'log_loss' : log_los,
                                          'AP' : ap}, ignore_index=True)
result_table.set_index('classifiers', inplace=True)
#PR curve
path_pr = 'E:/project/PR_curve.png'
plot_pr_curve(result_table, path_pr)
# ROC curve
path_roc = 'E:/project/ROC_curve.png'
plot_roc_curve(result_table, path_roc)
# Save the result table
result = single_result_table_test[['accuracy', 'auc', 'f1 score', 'AP', 'log_loss']]
result.to_csv('E:/project/result_table.csv',index = True)

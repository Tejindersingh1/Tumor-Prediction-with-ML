import pandas as pd
from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV


def read_data(path):
    """
    reads the csv data

    Args:
    path - path to the csv file

    Returns:
    A pandas dataframe and its labels
    """

    data = pd.read_csv(path)
    data.set_index("index",inplace = True)
    labels = data.pop("labels")
    return data, labels

def get_DEG(p_value, dataset, labels, log_FC):
    """
    Finds differentially expressed genes with the given p_value and fold_change using independent t-test

    Args:
    p_value - int value for specifying the p value for stastical significance
    dataset - csv dataset on which the differential expression analysis is to be performed
    fold_change - int value depicting fold change

    Returns:
    A pandas dataframe containing genes that are differentially expressed with given p value and fold change
    """

    dataset['labels'] = labels
    groups = dataset.groupby("labels")
    normal_sample = groups.get_group("normal")
    tumor_sample = groups.get_group("tumor")
    tumor_sample = tumor_sample.drop(["labels"], axis = 1)
    normal_sample = normal_sample.drop(["labels"], axis = 1)

    columns = normal_sample.columns
    print(columns)
    t_data = []
    for i in range(0,(len(columns)-1)):
        a = normal_sample[columns[i]]
        b = tumor_sample[columns[i]]
        # T-test
        t,p = ttest_ind(a,b)
        if p<p_value:
            t_data.append(columns[i])

    p_data = pd.DataFrame(dataset, columns = t_data)
    p_data["samples"] = labels

    #for fold change
    group = p_data.groupby("samples")
    norm_sample = group.get_group("normal")
    tumr_sample = group.get_group("tumor")
    tumr_sample.drop("samples", axis = 1, inplace = True)
    norm_sample.drop("samples", axis = 1, inplace = True)

    fold = []
    for i in tumr_sample.columns:
        f = tumr_sample[i].mean() - norm_sample[i].mean()
        if f < -(log_FC) or f >(log_FC):
            fold.append(i)
    deg = pd.DataFrame(dataset, columns = fold)
    return deg


def svc_rfe_cv(dataset, label):
    """
    Performing recursive feature elimination using support vector classifier with 10 fold cross validation

    Args:
    dataset - training data
    label - trainig data labels

    Returns:
    A  list of most informative columns according to SVC_RFE
    """
    estimator = SVC(kernel="linear")
    selector = RFECV(estimator, min_features_to_select=100, step=1, cv = 10)
    selector = selector.fit(dataset, label)
    training_data = dataset[dataset.columns[selector.get_support()]]

    return training_data


if __name__ == '__main__':
    save_data_path = "E:/project"
    read_data_path = "E:/project/gene_data.csv"
    print("\nReading data\n")
    data,labels = read_data(read_data_path)
    data['labels'] = labels
    print("\nPerforming Differential expression analysis using independent t__test\n")
    deg_data = get_DEG(0.0001,data,labels, 1)
    print(f"\nDataset shape{deg_data.shape}\nPerforming recursive feature eliminiation using SVC\n")
    print(deg_data.columns,"\n")
    print("\nNull values in data: ",deg_data.isnull().values.any(),"\nNull values in Labels: ", labels.isnull().values.any())
    training_data = svc_rfe_cv(deg_data, labels)
    training_data['labels'] = labels
    print(f"\nDataset shape{training_data.shape}\nSaving data...\n")
    training_data.to_csv(f"{save_data_path}/training_data.csv", index = True)
    print("Done")

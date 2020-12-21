import joblib
import os
from sklearn.metrics import log_loss, ConfusionMatrixDisplay, average_precision_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def train_models(models, data, labels):
    """
    Function to train the models from the given list of model objects

    Args:
    models = list of tuples of models objects and their names
    data = traing dataset rows as samples and columns as genes
    labels = labels for the given samples in the dataset

    Returns:
    list of trained models
    """
    for name, model in models:
        model.fit(data, labels)
    return models

def save_models(models, path):
    """
    Function to save the trained models

    Args:
    models = list of tuples of trained models and their names
    path = path to save the models

    Returns: None
    """
    for name, model in models:
        joblib.dump(model, f'{path}{name}.pkl')

def load_models(path):
    """
    Function to load the trained models from the directory

    Args:
    Path - path to the directory containing trained models

    Returns:
    models = list of tuples of trained models and their names
    """
    model_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path)
                          if file.endswith('pkl')])
    file_name = sorted([file_name for file_name in os.listdir(path)])

    models = [(name ,joblib.load(model)) for name, model in zip(file_name, model_files)]
    return models

def _binary_labels(labels):
    """
    Function to convert the labels into binary ("1" for tumor and "0" for Normal sample)

    Args:
    labels - labels to be converted into binary format

    Returns:
    labels for the data set in binary form ("0" for normal sample and "1" for tumour sample)
    """
    y_labels =[]
    for i in labels:
        if i == "tumor":
            y_labels.append(1)
        else:
            y_labels.append(0)
    return y_labels

def evaluate(name, model, test_data, test_label, path):
    """
    This function makes prediction on the test dataset and returns the results

    Args:
    name - String ,name of the classifier
    model - classifier object
    test_dataset - test dataframe
    test_label - labels for the test dataframe
    path - path to save confusion matrix

    Returns:
    accuracy - accuracy of the classifier on test dataset
    fpr1 - False positive rate
    tpr1 - True positive rate
    roc_auc - Area under the ROC curve
    precision - Precision of classifier on test dataset
    recall - Recall of the classifier on test dataset
    f1 - F1 score
    log_los - Log loss
    ap - Average precision
    """
    binary_labels = _binary_labels(test_label)
    prediction = model.predict(test_data)
    accuracy = accuracy_score(test_label, prediction)
    pred_binary = _binary_labels(prediction)
    probs = model.predict_proba(test_data)[:,1]
    fpr1, tpr1, _ = roc_curve(binary_labels, probs)
    roc_auc = auc(fpr1, tpr1)
    log_los = log_loss(binary_labels, probs)
    precision, recall, _ = precision_recall_curve(binary_labels, probs)
    f1 = f1_score(binary_labels, pred_binary)

    #Confusion Matrix
    label = ["tumor", "normal"]
    cm = confusion_matrix(test_label, prediction,labels = label)
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm,display_labels = label).plot(cmap = "Reds")
    plt.savefig(f'{path}{name}.png')
    plt.show()
    ap = average_precision_score(binary_labels, pred_binary)

    return accuracy, fpr1, tpr1, roc_auc, precision, recall,  f1, log_los, ap


def plot_pr_curve(result, path):
    """
    Plots the Precision recall curve

    Args:
    result - Result dataframe containing the following columns
             'recall'
             'precision'
             'AP'
    path - path to save the PR plot

    Returns:
    None
    """

    fig = plt.figure(figsize=(8,6))
    for i in result.index:
        plt.plot(result.loc[i]['recall'],
                 result.loc[i]['precision'],
                 label = f"{i}, AP={result.loc[i]['AP']}")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("recall", fontsize=15)
    plt.yticks(np.arange(0.0, 1.15, step=0.1))
    plt.ylabel("precision", fontsize=15)
    plt.title('Precision Recall Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='center left', bbox_to_anchor=(1, 0.5))
    ax = fig.add_subplot(111)
    lgd = ax.legend(prop={'size':13}, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(path,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def plot_roc_curve(result, path):
    """
    Plots the ROC curve

    Args:
    result - Result dataframe containing the following columns
             'fpr'
             'tpr'
             'AUC'
    path - path to save the ROC plot

    Returns:
    None
    """

    fig = plt.figure(figsize=(8,6))
    for i in result.index:
        plt.plot(result.loc[i]['fpr'],
                 result.loc[i]['tpr'],
                 label = f"{i}, AUC={result.loc[i]['auc']}" )
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='center left', bbox_to_anchor=(1, 0.5))
    ax = fig.add_subplot(111)
    lgd = ax.legend(prop={'size':13}, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(path,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

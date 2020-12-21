# Machine learning based prediction of tumour from microarray data
Microarrays help to study the expression of thousands of genes simultaneously. The genes responsible for the tumour growth can be identified by analyzing changes in the gene expressions related to the tumour growth with microarray data from the normal and the tumour samples. There is a need for highly efficient computational techniques to analyze these large numbers of gene expressions and find out the most significant differentially expressed genes related to the particular disease. Many computational methods have difficulties in selecting the optimal set of genes because of the small number of samples compared to the thousands of genes. For machine learning classifiers to accurately classify the tumour and the normal samples, gene selection is a very important step. 

## Steps followed
#### 1. Data Preparation -
    1. Differential expression analysis using independent t-test
    2. Recursive feature elimination using SVC classifier
    
#### 2. Traing 10 classifiers from scikit-learn
    1. Support Vector machine classifier
    2. Logistic regression
    3. Linear Discriminant analysis
    4. Quaderatic discriminant analysis
    5. Decision Tree
    6. Gaussian naive bayes
    7. Random forest
    8. Gaussian process classifier
    9. Adaboost
    10. XGBoost
    
#### 3. Testing and Evaluation
     Tested the classifiers based on following Evaluation matrics on Microoaray data containing 228 samples from different independent experiments
     The evaluation metrics used were – 
    1. ROC curve
    2. Precision recall curve
    3. Confusion matrix
    4. Accuracy
    5. Area under the curve
    6. F1 score
    7. Average precision
    8. Log loss

## Project files
####   1. Models
        Folder containing all the trained models. These models are trained on 400 samples from microarray data from 8 independent experiments for clear cell Renal cell carcinoma.
####   2. Results
        This folder contains the test results.
####   3. data_preparation.py
        This python file contains the codes for data preparation i.e. feature extraction.
####   4. training.py
       This python file contains the codes for training ML classifiers 
####   5. evaluation.py
       This python file contains the codes for testing and evaluation classifiers 
####   6. train_test.py       
       This python file contains the functions used in training and testing
## Dataset shape
    The dataset should be in the following shape for these classiers
    columns - gene names or probe ids
    rows    - samples
    last column should be the 'labels' containg labels as 'tumor' for tumor samples and 'normal' for normal samples.
### Notes
    These classifiers are traind on the whole genome clear cell renal cell carcinoma microarray meta-dataset. 
    Using and further training of these clasifiers on more diverse datasets for different diseases is encouraged.
    

       
        
       

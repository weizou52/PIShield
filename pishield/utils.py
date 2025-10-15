from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import json
import os
import pickle
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def jdump(obj, filename, indent_flag=1):
    directory = os.path.dirname(filename)
    if directory:  # Check if directory is not an empty string
        os.makedirs(directory, exist_ok=True)
    json_dict = json.dumps(obj, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(f"{filename}.json", 'w') as f:
        if indent_flag:
            json.dump(dict_from_str, f, indent=4)
        else:
            json.dump(dict_from_str, f)

def jload(filename):
    with open(f"{filename}.json", 'r') as f:
        return json.load(f)
    

def load_pickle(filename):
    with open(f"{filename}.pkl", 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(obj, f)


def create_logistic_regression_model(data, labels, split_idx):
    '''
    Input:
        data: Features (e.g., numpy array or pandas DataFrame)
        labels: Corresponding labels for the data
        split_idx: Index at which the training data ends and the testing data begins
    Output:
        log_model: Trained Logistic Regression model
        train_accuracy: Accuracy of the model on the training set
        test_accuracy: Accuracy of the model on the test set
    '''

    # Initialize the Logistic Regression model
    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    # Split the data into training and testing sets
    x_train = data[:split_idx]
    y_train = labels[:split_idx]
    x_test = data[split_idx:]
    y_test = labels[split_idx:]

    # Train the model
    t0=time.time()
    log_model.fit(x_train, y_train)
    t1=time.time()
    print(f"Time taken to train: {t1-t0} seconds")

    # Calculate training and testing accuracy
    train_accuracy = log_model.score(x_train, y_train)
    test_accuracy = log_model.score(x_test, y_test)

    # Return model and accuracies
    return log_model, train_accuracy, test_accuracy

def get_evaluation_metrics(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    if len(y_test) != len(y_pred):
        raise ValueError("The lengths of true labels and predicted labels must be the same.")
    
    for label in y_test + y_pred.tolist():
        if label not in {0, 1}:
            raise ValueError("Labels must be either 0 or 1.")
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for pred, true in zip(y_pred, y_test):
        if pred == 1:
            if true == 1:
                TP += 1
            else:
                FP += 1
        else:
            if true == 1:
                FN += 1
            else:
                TN += 1
    
    # Calculate False Positive Rate (FPR)
    denominator_fpr = FP + TN
    fpr = 0.0 if denominator_fpr == 0 else FP / denominator_fpr
    
    # Calculate False Negative Rate (FNR)
    denominator_fnr = FN + TP
    fnr = 0.0 if denominator_fnr == 0 else FN / denominator_fnr
    
    return cm, accuracy, fpr, fnr

def get_log_predictions(log_model, X_test, threshold=0.5):
    '''
    input: 
        log_model (LogisticRegression)
        X_test (array)
        y_test (array)
        threshold (float): prediction threshold (default: 0.5)
    output: y_pred (array)
    '''
    # Get probability predictions
    print(f"len(data): {len(X_test)}")
    y_prob = log_model.predict_proba(X_test)[:, 1]
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred



    
    
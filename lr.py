from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np


class LogisticRegression:
    def __init__(self, training_dataset, validation_dataset, learning_rate = 0.08, epochs = 3000):
        """Logistic Regression algorithm designed to work on spambase dataset"""
        self.__training_dataset = training_dataset.to_numpy()
        self.__validation_dataset = validation_dataset.to_numpy()
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        
        #Get the features and labels from training dataset
        self.__training_features = self.__training_dataset[:, :-1]
        self.__training_labels = self.__training_dataset[:, -1].reshape(-1, 1)
        
        #Add one more column to features
        self.__training_prime = np.c_[np.ones((len(self.__training_features), 1)), self.__training_features]
        self.__y = self.__training_labels
    
    def gradient_descent(self):
        """Uses gradient descent to find the best model and performance"""
        M = np.random.randn(self.__training_features.shape[1] + 1, 1)
        best_model = M
        best_performance = 0
        for _ in range(self.__epochs):
            pred_y = self.sigmoid(np.dot(self.__training_prime, M))
            gradient = np.dot(self.__training_prime.T, (pred_y - self.__y)) * 2 / len(self.__training_features)
            _p = self.performance(M, self.__validation_dataset)
            if(_p > best_performance):
                best_performance = _p
                best_model = M
            M = M - (self.__learning_rate * gradient)
        return best_model


    def sigmoid(self, z):
        """Applies sigmoid function to z"""
        z = np.clip(z, -500, 500)
        return 1.00/(1.00+np.exp(-z))
            
    def performance(self, model, test_dataset):
        """Measures accuracy of model using test dataset"""
        if(isinstance(test_dataset, DataFrame)):
            test_dataset = test_dataset.to_numpy()
        test_features = test_dataset[:, :-1]
        test_labels = test_dataset[:, -1].reshape(-1, 1)
        test_features_prime = np.c_[np.ones((len(test_features), 1)), test_features]
        pred_y = self.sigmoid(np.dot(test_features_prime, model))
        pred_y_val = (pred_y > 0.5).astype(int)
        return accuracy_score(pred_y_val, test_labels)

#Load dataset
data = pd.read_csv("spambase.csv")
data = data.sample(frac = 1).reset_index(drop = True) #Shuffle the dataset

#Split into training and test sets
size = int(len(data) * 0.8)
training = data[:size]
test = data[size:]

def performance(clf, dataset):
    model_accuracy = clf.gradient_descent()
    model = model_accuracy
    accuracy = clf.performance(model, dataset)
    return accuracy



#Split using 5-cross validation
fold5 = KFold(5)
for train_idx, val_idx in fold5.split(training):
    sub_train = training.iloc[train_idx]
    sub_val = training.iloc[val_idx]
    clf = LogisticRegression(sub_train, sub_val)
    #Run performance on test dataset
    print("Test dataset accuracy: {:.2%}".format(performance(clf, test)))
    print("Validation dataset accuracy: {:.2%}".format(performance(clf, sub_val)))
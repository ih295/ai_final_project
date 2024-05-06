
from sklearn.model_selection import KFold
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

class NaiveBayes:
    def __init__(self, training_dataset, validation_dataset):
        """Naive Bayes algorithm designed to work on spambase dataset"""
        #Preserve training and validation datasets
        self.__training_dataset = training_dataset
        self.__validation_dataset = validation_dataset
        if(isinstance(self.__training_dataset, DataFrame)):
            self.__numpy_training_dataset = self.__training_dataset.to_numpy()
        else:
            self.__numpy_training_dataset = self.__training_dataset
        
        if(isinstance(self.__validation_dataset, DataFrame)):
            self.__numpy_validation_dataset = self.__validation_dataset.to_numpy()
        else:
            self.__numpy_validation_dataset = self.__validation_dataset

        #Get the features and labels from training dataset
        self.__training_features = self.__numpy_training_dataset[:, :-4]
        self.__training_labels = self.__numpy_training_dataset[:, -1]

        #Calculate odds of spam and not spam
        self.__ham_prob = self.calculate_total_probabilities(0)
        self.__spam_prob = self.calculate_total_probabilities(1)

        #And get number of spam and not spam
        self.__num_ham = self.__ham_prob * len(self.__numpy_training_dataset)
        self.__num_spam = self.__spam_prob * len(self.__numpy_training_dataset)
        #Probabilities dictionary
        self.__probabilities = dict()
        self.__probabilities["spam"] = [0 for i in range(self.__training_features.shape[1])]
        self.__probabilities["ham"] = [0 for i in range(self.__training_features.shape[1])]

        #Now calculate the probabilities of spam / not spam
        self.__probabilities = self.calculate_probabilities()

    def calculate_total_probabilities(self, _class):
        """Calculates the odds of any feature being spam or ham, depending on class code. _class = 1 is for spam, 0 is for ham."""
        return np.sum(self.__training_labels == _class) / len(self.__training_labels)

    def calculate_probabilities(self):
        """Calculates probabilities for spam and ham for every feature in dataset"""
        for label_index in range(len(self.__training_labels)):
            feature_label = self.__training_labels[label_index]
            features = self.__training_features[label_index]
            for feature_index in range(len(features)):
                feature = features[feature_index]
                if(feature > 0.0):
                    #Word exists
                    if((feature_index < len(self.__probabilities["spam"])) and (feature_index < len(self.__probabilities["ham"]))):
                        if(feature_label == 0):
                            #Word is not spam, increment count at feature_index
                            not_spam_count = self.__probabilities["ham"][feature_index] + 1
                            self.__probabilities["ham"][feature_index] = not_spam_count
                        elif(feature_label == 1):
                            #Word is spam, increment count at feature_index
                            spam_count = self.__probabilities["spam"][feature_index] + 1
                            self.__probabilities["spam"][feature_index] = spam_count
        #Now calculate actual probabilities for spam and not spam
        hams = self.__probabilities["ham"]
        spams = self.__probabilities["spam"]
        for ham_index in range(len(hams)):
            hams[ham_index] = (hams[ham_index] / self.__num_ham) * self.__ham_prob
        for spam_index in range(len(spams)):
            spams[spam_index] = (spams[spam_index] / self.__num_spam) * self.__spam_prob
        self.__probabilities["ham"] = hams
        self.__probabilities["spam"] = spams
        return self.__probabilities
    
    def classify(self, test_features):
        """Classifies features from a test dataset"""
        probabilities_ham = self.__probabilities["ham"]
        probabilities_spam = self.__probabilities["spam"]
        prob_ham = self.__ham_prob
        prob_spam = self.__spam_prob
        for feature_index in range(len(test_features)):
            feature = test_features[feature_index]
            if(feature > 0.00):
                prob_ham *= probabilities_ham[feature_index]
                prob_spam *= probabilities_spam[feature_index]
        if(prob_ham > prob_spam):
            return 0
        return 1

    def evaluate(self, test_dataset):
        """Evaluates test dataset and returns accuracy"""
        if(isinstance(test_dataset, DataFrame)):
            test_dataset = test_dataset.to_numpy()
        test_features = test_dataset[:, :-4]
        test_labels = test_dataset[:, -1]
        num_correct = 0
        for test_index in range(len(test_dataset)):
            test_feature = test_features[test_index]
            prediction = self.classify(test_feature)
            if(test_labels[test_index] == prediction):
                num_correct += 1
        return num_correct / len(test_dataset)


    def performance(self, test_dataset):
        """Measures performance by calculating model accuracy using test dataset"""
        return self.evaluate(test_dataset)


#Load dataset
data = pd.read_csv("spambase.csv")
data = data.sample(frac = 1).reset_index(drop = True) #Shuffle the dataset

#Split into training and test sets
size = int(len(data) * 0.8)
training = data[:size]
test = data[size:]

def performance(clf, dataset):
    return clf.performance(dataset)



#Split using 5-cross validation
fold5 = KFold(5)
for train_idx, val_idx in fold5.split(training):
    sub_train = training.iloc[train_idx]
    sub_val = training.iloc[val_idx]
    clf = NaiveBayes(sub_train, sub_val)
    #Run performance on validation dataset
    print("Validation dataset accuracy: {:.2%}".format(clf.performance(sub_val)))
    print("Test dataset accuracy: {:.2%}".format(clf.performance(test)))
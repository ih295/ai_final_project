
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from multiprocessing import Pool
from pandas.core.frame import DataFrame
import numpy as np

class KNN:
    def __init__(self, training_dataset, validation_dataset, k = 3):
        """K-Nearest Neighbor algorithm designed to work on spambase dataset"""
        self.__training = training_dataset.to_numpy()
        self.__validation = validation_dataset.to_numpy()
        self.__k = k
    
    def calculate_similarity(self, row_1, row_2):
        """Calculates the similarity between two rows"""
        return (np.dot(row_1, row_2)) / ((np.linalg.norm(row_1) * np.linalg.norm(row_2)))
    
    def get_neighbors(self, test_row):
        """Gets neighbors using similarity between test row and training features"""
        neighbors = list()
        for index in range(len(self.__training)):
            row = self.__training[index][:-1]
            similarity = self.calculate_similarity(test_row, row)
            neighbors.append((similarity, set(row)))
        neighbors.sort(key = lambda sim_row: sim_row[0])
        labels = [neighbor for similarity, neighbor in neighbors]
        return max(labels, key=labels.count)

    
    def predict(self, test_row):
        """Creates a prediction list using test dataset"""
        return self.get_neighbors(test_row)
    
    def pooled_performance(self, test_dataset):
        """Measures accuracy of test and validation datasets"""
        if(type(test_dataset) is DataFrame):
            test_dataset = test_dataset.to_numpy()
        for row_index in range(len(test_dataset)):
            self.predict(test_dataset[row_index][:-1])
        test_predictions = [self.predict(row[:-1]) for row in test_dataset]
        test_accuracy = np.mean(test_predictions == test_dataset[:, -1])
        validation_predictions = [self.predict(row[:-1]) for row in self.__validation]
        validation_accuracy = np.mean(validation_predictions == self.__validation[:, -1])
        return [test_accuracy, validation_accuracy]
        
    
    def performance(self, test_dataset):
        pool = Pool(100)
        data = pool.map(self.pooled_performance, [test_dataset])
        pool.close()
        return data


#Load dataset
data = pd.read_csv("spambase.csv")
data = data.sample(frac = 1).reset_index(drop = True) #Shuffle the dataset

#Split into training and test sets
size = int(len(data) * 0.8)
training = data[:size]
test = data[size:]

def performance(clf, dataset):
    performance_accuracy = clf.performance(dataset)[0]
    test_accuracy = performance_accuracy[0]
    validation_accuracy = performance_accuracy[1]
    print("Test accuracy: {:.2%}".format(test_accuracy))
    print("Validation accuracy: {:.2%}".format(validation_accuracy))



#Split using 5-cross validation
fold5 = KFold(5)
for train_idx, val_idx in fold5.split(training):
    sub_train = training.iloc[train_idx]
    sub_val = training.iloc[val_idx]
    clf = KNN(sub_train, sub_val)
    #Run performance on test dataset
    performance(clf, test)
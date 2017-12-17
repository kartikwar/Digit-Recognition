import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#ignore warnings
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_data(X, y):
	rcf = RandomForestClassifier().fit(X, y)
	return rcf

def predict_test(classifier, X_test):
	X_test_predict = classifier.predict(X_test)
	X_test["ImageId"] = X_test.index.values
	X_test["Label"] = X_test_predict
	X_test = X_test[["Label", "ImageId"]]
	X_test.to_csv("Predicitons.csv", index=False) 		

if __name__ == '__main__':
	train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
	y_train = train["label"]
	X_train, X_test = train.drop('label', axis=1), test
	classifier = train_data(X_train, y_train)
	predict_test(classifier, X_test)

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
from sklearn.cross_validation import KFold

def round_to_int(number):
	return int(round(number))

def train_in_folds(clf, X_train, y_train, X_test):
	ntrain = X_train.shape[0]
	ntest = X_test.shape[0]
	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	SEED = 0 # for reproducibility
	NFOLDS = 15 # set folds for out-of-fold prediction
	kf = KFold(len(X_train), n_folds= NFOLDS, random_state=SEED)
	oof_test_skf = np.empty((NFOLDS, ntest))

	for i, (train_index, test_index) in enumerate(kf):
		x_tr = X_train.iloc[train_index]
		y_tr = y_train.iloc[train_index]
		x_te = X_train.iloc[test_index]
		clf.fit(x_tr, y_tr)
		oof_train[test_index] = clf.predict(x_te)
		oof_test_skf[i, :] = clf.predict(X_test)

	oof_test[:] = oof_test_skf.mean(axis=0)
	predictions = oof_test.tolist()
	predictions = map(round_to_int, predictions)
	return predictions

def train_data(X, y, test):
	rcf = RandomForestClassifier()
	predictions = train_in_folds(rcf, X, y, test)
	return predictions

def save_predictions(predictions, X_test):
	X_test["ImageId"] = X_test.index.values
	X_test["ImageId"] = X_test["ImageId"] + 1
	X_test["Label"] = predictions
	# print X_test["Label"].dtype
	# X_test["Label"] = round(X_test["Label"])
	X_test = X_test[["Label", "ImageId"]]
	X_test.to_csv("Predicitons.csv", index=False) 		

if __name__ == '__main__':
	train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
	y_train = train["label"]
	X_train, X_test = train.drop('label', axis=1), test
	predictions = train_data(X_train, y_train, X_test)
	save_predictions(predictions, X_test)

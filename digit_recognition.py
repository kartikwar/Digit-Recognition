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
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def determine_best_params_random_forest(X_train, y_train):
	grid_values = {'n_estimators' : [1, 5, 25],
	'max_features': [1,  2 , 3, 4 , 5 ] 	 
	}
	clf = RandomForestClassifier(random_state = 0)
	grid_clf_accuracy = GridSearchCV(clf, param_grid=grid_values, 
		n_jobs=-1, scoring='accuracy')
	grid_clf_accuracy.fit(X_train, y_train)
	best_params =grid_clf_accuracy.best_params_
	return best_params

def train_data(X, y):
	# best_params = determine_best_params_random_forest(X, y)
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
		hidden_layer_sizes=(10, 10), random_state=1).fit(X, y)
	# clf = RandomForestClassifier(max_features=5, n_estimators=25).fit(X, y)
	return clf

def predict_test(classifier, X_test, test):
	X_test_predict = classifier.predict(X_test)
	test["ImageId"] = test.index.values
	test["ImageId"] = test["ImageId"] + 1
	test["Label"] = X_test_predict
	test = test[["Label", "ImageId"]]
	test.to_csv("Predicitons.csv", index=False) 			

def preprocessing(train, validation):
	y_train = train["label"]
	X_train, X_validation = train.drop('label', axis=1), validation.copy()
	X_train = StandardScaler().fit_transform(X_train)
	pca = PCA(n_components=20)
	X_train = pca.fit_transform(X_train)
	# print (pca.explained_variance_ratio_)
	validation = StandardScaler().fit_transform(validation)
	validation = pca.fit_transform(validation)
	return X_train, y_train, validation, X_validation

if __name__ == '__main__':
	train, validation = pd.read_csv('train.csv'), pd.read_csv('test.csv')
	X_train, y_train, validation, X_validation = preprocessing(train,
		validation)
	classifier = train_data(X_train, y_train)
	predict_test(classifier, validation, X_validation)


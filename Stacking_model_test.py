from get_data import *
from predict import *
from test_stacking import stacking
import yfinance as yf
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from test_stacking import stacking
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import math
from plot_comfusion_mat import plot_confusion_matrix
from stacking import *


if __name__ == "__main__":
	# corpus
	stack_train = pd.read_csv('tweet_sentiment.csv').dropna(axis=0, how='any')
	stack_train.columns = ['Content', 'Sentiment']
	stack_train.reset_index(drop=True, inplace=True)
	# data process
	x_training = stack_train['Content'][:10000]
	y_training = stack_train['Sentiment'][:10000]
	df1 = pd.read_csv('all-data.csv', encoding='ISO-8859-1', header=None).dropna(axis=0, how='any')
	df1.columns = ['Sentiment', 'Content']
	x_test_set = df1.iloc[-1000:-500, :]
	x_test_set.reset_index(drop=True, inplace=True)
	x_testing = x_test_set['Content']
	y_testing = x_test_set['Sentiment'].replace('negative', -1)
	y_testing = y_testing.replace('positive', 1)
	y_testing = y_testing.replace('neutral', 0)

	# classifier
	classifiers = [svm.SVC(kernel='linear'),
				   RandomForestClassifier(),
				   GradientBoostingClassifier(random_state=10),
				   KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
				   MultinomialNB()]

	stacking_model = StackedGeneralization(classifiers)
	stacking_model.fit(x_training, y_training)
	df_predict_1 = stacking_model.predict(x_testing)

	# df_predict_1 = stacking(classifiers, x_training, y_training, x_testing)
	np.array(df_predict_1).reshape(-1, 1)
	df_predict_1[df_predict_1 < 0] = -1
	df_predict_1[df_predict_1 > 0] = 1
	# df_predict_1[np.logical_and(-0.01 < df_predict_1, df_predict_1 < 0.01)] = 0
	set(df_predict_1)
	cm = confusion_matrix(y_testing, df_predict_1, sample_weight=None)
	print(cm)
	plot_confusion_matrix(cm)


	np.sum(np.diag(cm))/np.sum(cm)  # (18+33+5+2+263+12+28+139)

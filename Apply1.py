from get_data import *
from predict import *
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from stacking import *

#
# class StackedGeneralization:
# 	def __init__(self, classifier, meta=DecisionTreeClassifier(), kf_n=5):
# 		self.classifier = classifier
# 		self.meta = meta
# 		self.to_vector = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
# 		self.kf_n = kf_n
# 		self.kf = KFold(n_splits=kf_n, random_state=49, shuffle=True)
#
# 	def Base_learner_output(self, classifier, x_train, y_train, x_test):
# 		second_train = np.zeros((len(x_train)))
# 		second_test = np.zeros((len(x_test)))
# 		second_test_prep = np.empty((5, len(x_test)))
# 		temp_x_train = self.to_vector.fit_transform(x_train)
# 		temp_y_train = y_train
# 		temp_x_test = self.to_vector.transform(x_test)
# 		for i, (train_index, test_index) in enumerate(self.kf.split(temp_x_train)):
# 			kf_x_train = temp_x_train[train_index]
# 			kf_y_train = temp_y_train[train_index]
# 			kf_x_test = temp_x_train[test_index]
# 			classifier.fit(kf_x_train, kf_y_train)
# 			second_train[test_index] = classifier.predict(kf_x_test)
# 			second_test_prep[i, :] = classifier.predict(temp_x_test)
#
# 		second_test[:] = second_test_prep.mean(axis=0)
# 		return second_train.reshape(-1, 1), second_test.reshape(-1, 1)
#
# 	def stacking(self, x_train, y_train, x_test):
# 		train_sets, test_sets = [], []
# 		# run base leaner
# 		for classifier in classifiers:
# 			train_set, test_set = self.Base_learner_output(classifier, x_train, y_train, x_test)
# 			train_sets.append(train_set)
# 			test_sets.append(test_set)
#
# 		# get input of second layer
# 		second_layer_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
# 		second_layer_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)
#
# 		# second level model
# 		self.meta.fit(second_layer_train, y_train)
# 		prediction = self.meta.predict(second_layer_test)
# 		return prediction


def polarize(seq):
	out = seq
	out[out < 0] = -1
	out[out > 0] = 1
	return out


def polarize_2(seq):
	out = seq
	out[out < 0.5] = -1
	out[out == 0.5] = 0
	out[out > 0.5] = 1
	return out


if __name__ == "__main__":
	# corpus
	stack_train = pd.read_csv('tweet_sentiment.csv').dropna(axis=0, how='any')
	stack_train.columns = ['Content', 'Sentiment']
	stack_train.reset_index(drop=True, inplace=True)

	# data process
	x_training = stack_train['Content'][:10000]
	y_training = stack_train['Sentiment'][:10000]

	stock_df = yf.download("^GSPC", start='2019-10-01', end='2020-09-30')
	timespan = [i.strftime("%Y-%m-%d") for i in stock_df.index.to_series()]

	# classifier
	classifiers = [svm.SVC(kernel='linear'),
				   RandomForestClassifier(),
				   GradientBoostingClassifier(random_state=10),
				   KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
				   MultinomialNB()]

	# -----------------------------------------------------------------------------------

	twitter_id = ['MarketWatch', 'business', 'WSJ', 'TheEconomist', 'nytimes', 'MorganStanley']
	stacking_model = StackedGeneralization(classifiers)
	stacking_model.fit(x_training, y_training)

	sentiment_df = pd.DataFrame(data=None, columns=twitter_id)
	for twitter in twitter_id:
		temp_df = clean_csv(twitter)
		x_testing = temp_df['Content']
		predictions = stacking_model.predict(x_testing)
		temp_df['Sentiment_stacking'] = predictions

		mark, stack = [], []
		day = '2020-08-28'
		for day in timespan:
			lst = temp_df[temp_df['Date'].isin([day])]
			if len(lst) == 0:
				mark.append(-1)
			else:
				mark.append(np.min(lst.index))
				print(str(day) + str(np.min(lst.index)))
		mark.reverse()
		mark.append(len(temp_df))
		for i in range(len(mark)):
			if mark[i] == -1:
				mark[i] = mark[i-1]
		for i in range(len(mark) - 1):
			span = [i for i in range(mark[i], mark[i + 1])]
			lst = temp_df.iloc[span, :]
			if lst is not None:
				stack.append(np.mean(lst['Sentiment_stacking']))
			else:
				stack.append(0)
		sentiment_df[twitter] = stack

	# sentiment_df.to_csv('./output/sentiments.csv', index=False)

	# -----------------------------------------------------------------------------------

	sentiment_df = pd.read_csv('./output/sentiments.csv')
	sentiment_df.index = timespan
	# sentiment_df = pd.read_csv('./output/snow_sent.csv')

	market_return = stock_df['Adj Close'].apply(np.log).diff().dropna()
	market_movement = list(polarize(market_return))

	correct_records = []
	for twitter in twitter_id:
		pred = list(polarize_2(sentiment_df[twitter][1:]))
		cm = confusion_matrix([int(i) for i in market_movement], [int(i) for i in pred], sample_weight=None)
		correct_records.append(np.diag(cm).sum())

	plt.bar(range(len(correct_records)), correct_records, color='darkseagreen', tick_label=twitter_id)
	plt.title('Twitter sentiment that have the same direction as stock market')
	plt.show()

	plt.bar(range(len(correct_records)), [i/251 for i in correct_records], color='darkseagreen', tick_label=twitter_id)
	plt.title('Twitter sentiment that have the same direction as stock market')
	plt.ylim(ymax=0.404, ymin=0.392)
	plt.show()

	plt.scatter(np.arange(1,253,1), sentiment_df['MorganStanley'], color='orange')
	# plt.title('My score')
	plt.ylabel('sentiment')
	plt.title('Snow score')
	plt.show()
















# from get_data import *
# from predict import *
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix
# from plot_comfusion_mat import plot_confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB
#
#
# # def sentiment_pred(tweeters):
# # 	for ID in tweeters:
# # 		df = clean_csv(pd.read_csv('./intermediate/' + ID + '.csv'))
# # 		df.to_csv('./intermediate/' + ID + '_cleaned.csv')
# # 		return df
#
#
# class StackedGeneralization:
# 	def __init__(self, classifier, meta=DecisionTreeClassifier(), kf_n=5):
# 		self.classifier = classifier
# 		self.meta = meta
# 		self.to_vector = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
# 		self.kf_n = kf_n
# 		self.kf = KFold(n_splits=kf_n, random_state=49, shuffle=True)
# 		self.meta_train = None
# 		self.meta_test = None
# 		self.base_memory = None
# 		self.y_train = None
#
# 	def Base_learner_train(self, classifier, x_train, y_train):
# 		second_train = np.zeros((len(x_train)))
# 		temp_x_train = self.to_vector.fit_transform(x_train)
# 		temp_y_train = y_train
# 		for i, (train_index, test_index) in enumerate(self.kf.split(temp_x_train)):
# 			kf_x_train = temp_x_train[train_index]
# 			kf_y_train = temp_y_train[train_index]
# 			kf_x_test = temp_x_train[test_index]
# 			classifier.fit(kf_x_train, kf_y_train)
# 			second_train[test_index] = classifier.predict(kf_x_test)
# 		self.meta_train = second_train.reshape(-1, 1)
# 		return second_train.reshape(-1, 1)
#
# 	def Base_learner_test(self, classifier, x_test):
# 		second_test = np.zeros((len(x_test)))
# 		second_test_prep = np.empty((5, len(x_test)))
# 		temp_x_test = self.to_vector.transform(x_test)
# 		for i in range(self.kf_n):
# 			second_test_prep[i, :] = classifier.predict(temp_x_test)
# 		second_test[:] = second_test_prep.mean(axis=0)
# 		self.meta_test = second_test.reshape(-1, 1)
# 		return second_test.reshape(-1, 1)
#
# 	def fit(self, x_test, x_train=None, y_train=None, new: str = 'no'):
# 		self.y_train = y_train
# 		train_sets, test_sets = [],[]
# 		for classifier in self.classifier:
# 			train_set = self.Base_learner_train(classifier, x_train, y_train)
# 			train_sets.append(train_set)
# 			test_set = self.Base_learner_test(classifier, x_test)
# 			test_sets.append(test_set)
# 		second_layer_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
# 		second_layer_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)
# 		self.base_memory = second_layer_train
#
# 		# second level model
# 		self.meta.fit(self.base_memory, self.y_train)
# 		prediction = self.meta.predict(second_layer_test)
# 		return prediction
#
# 	def Predict(self, x_test):
# 		test_sets = []
# 		for classifier in classifiers:
# 			test_set = self.Base_learner_test(classifier, x_test)
# 			test_sets.append(test_set)
# 			second_layer_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)
#
# 		# second level model
# 		self.meta.fit(self.base_memory, self.y_train)
# 		prediction = self.meta.predict(second_layer_test)
# 		return prediction
#
#
# # np.array(df_predict_1).reshape(-1, 1)
# # df_predict_1[df_predict_1 < 0] = -1
# # df_predict_1[df_predict_1 > 0] = 1
#
# # market_movement = np.array(market_movement)
# # market_movement[market_movement < 0] = -1
# # market_movement[market_movement > 0] = 1
#
#
#
#
#
#
#
#
# 	#
# 	#
# 	#
# 	# sentiment_df = pd.DataFrame(data=None, columns=twitter_id)
# 	# for twitter in twitter_id:
# 	# 	temp_df = clean_csv(twitter)
# 	# 	temp_df = temp_df[temp_df['Content'] != '']
# 	# 	temp_df.reset_index(drop=True, inplace=True)
# 	# 	x_testing = temp_df['Content']
# 	# 	predictions = []
# 	# 	for cont in temp_df['Content']:
# 	# 		if len(cont) > 0:
# 	# 			predictions.append(SnowNLP(cont).sentiments)
# 	#
# 	# 	temp_df['Sentiment_stacking'] = predictions
# 	#
# 	# 	mark, stack = [], []
# 	# 	day = '2020-08-28'
# 	# 	for day in timespan:
# 	# 		lst = temp_df[temp_df['Date'].isin([day])]
# 	# 		if len(lst) == 0:
# 	# 			mark.append(-1)
# 	# 		else:
# 	# 			mark.append(np.min(lst.index))
# 	# 			print(str(day) + str(np.min(lst.index)))
# 	# 	mark.reverse()
# 	# 	mark.append(len(temp_df))
# 	# 	for i in range(len(mark)):
# 	# 		if mark[i] == -1:
# 	# 			mark[i] = mark[i-1]
# 	# 	print('mark 1')
# 	# 	for i in range(len(mark) - 1):
# 	# 		print(i)
# 	# 		span = [i for i in range(mark[i], mark[i + 1])]
# 	# 		print('nest' + str(i))
# 	# 		lst = temp_df.iloc[span, :]
# 	# 		print('third' + str(i))
# 	# 		if lst is not None:
# 	# 			stack.append(np.mean(lst['Sentiment_stacking']))
# 	# 		else:
# 	# 			stack.append(0)
# 	# 	print('mark 2')
# 	# 	sentiment_df[twitter] = stack
# 	#
# 	# 	sentiment_df.to_csv('./output/snow_sent.csv', index=False)
# 	#
# 	#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

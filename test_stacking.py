import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from plot_comfusion_mat import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


# method prepare
to_vector = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
kf = KFold(n_splits=5, random_state=49, shuffle=True)


# function prepare
def Base_learner_output(classifier, x_train, y_train, x_test):
	second_train = np.zeros((len(x_train)))
	second_test = np.zeros((len(x_test)))
	second_test_prep = np.empty((5, len(x_test)))
	temp_x_train = to_vector.fit_transform(x_train)
	temp_y_train = y_train
	temp_x_test = to_vector.transform(x_test)
	for i, (train_index, test_index) in enumerate(kf.split(temp_x_train)):
		kf_x_train = temp_x_train[train_index]
		kf_y_train = temp_y_train[train_index]
		kf_x_test = temp_x_train[test_index]
		classifier.fit(kf_x_train, kf_y_train)
		second_train[test_index] = classifier.predict(kf_x_test)
		second_test_prep[i, :] = classifier.predict(temp_x_test)

	second_test[:] = second_test_prep.mean(axis=0)
	return second_train.reshape(-1, 1), second_test.reshape(-1, 1)


# second level model
dt_model = DecisionTreeClassifier()


def stacking(classifiers, x_train, y_train, x_test, second_layer_model=DecisionTreeClassifier()):
	train_sets, test_sets = [], []
	# run base leaner
	for classifier in classifiers:
		train_set, test_set = Base_learner_output(classifier, x_train, y_train, x_test)
		train_sets.append(train_set)
		test_sets.append(test_set)

	# get input of second layer
	second_layer_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
	second_layer_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

	# second level model
	second_layer_model.fit(second_layer_train, y_train)
	prediction = second_layer_model.predict(second_layer_test)
	return prediction


if __name__ == '__main__':
	# data prepare(read))
	corpus02 = pd.read_csv('./corpus/all-data.csv', encoding='ISO-8859-1', header=None).dropna(axis=0, how='any')
	corpus02.columns = ['Sentiment', 'Content']
	corpus02.reset_index(drop=True, inplace=True)
	model_test_data = pd.read_csv('./corpus/2/tweet_sentiment.csv').dropna(axis=0, how='any')
	model_test_data.reset_index(drop=True, inplace=True)
	model_test_data.columns = ['Content', 'Sentiment']

	# data prepare(clean and split)
	x_training = corpus02['Content']
	y_training = corpus02['Sentiment'].replace('positive', 1)
	y_training = y_training.replace('negative', -1)
	y_training = y_training.replace('neutral', 0)

	# base learner prepare
	svm_model = svm.SVC(kernel='linear')
	rf_model = RandomForestClassifier()
	gbm_model = GradientBoostingClassifier(random_state=10)
	kmm_model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
	nb_model = MultinomialNB()
	classifiers_ = [svm_model, rf_model, gbm_model, kmm_model, nb_model]

	# performance of stacking model
	df_predict_1 = stacking(classifiers_, x_training, y_training, model_test_data['Content'])
	df_predict_1[df_predict_1 < -0.01] = -1
	df_predict_1[df_predict_1 > 0.01] = 1
	df_predict_1[np.logical_and(-0.01 < df_predict_1, df_predict_1 < 0.01)] = 0
	cm = confusion_matrix(model_test_data['Sentiment'], df_predict_1, sample_weight=None)
	print(cm)
	plot_confusion_matrix(cm)

	# more information
	FP = cm.sum(axis=0) - np.diag(cm)
	FN = cm.sum(axis=1) - np.diag(cm)
	TP = np.diag(cm)
	TN = cm.sum() - (FP + FN + TP)
	# Sensitivity, hit rate, recall, or true positive rate
	TPR = TP / (TP + FN)
	# Specificity or true negative rate
	TNR = TN / (TN + FP)
	# Precision or positive predictive value
	PPV = TP / (TP + FP)
	# Negative predictive value
	NPV = TN / (TN + FN)
	# Fall out or false positive rate
	FPR = FP / (FP + TN)
	# False negative rate
	FNR = FN / (TP + FN)
	# False discovery rate
	FDR = FP / (TP + FP)
	precision = TP / (TP + FP)  # 查准率
	recall = TP / (TP + FN)  # 查全率






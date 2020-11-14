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


# function prepare
def get_return(symbol: list, startDay: str = "2020-06-01", endDay: str = "2020-10-01"):
	out = []
	for i in symbol:
		temp_df = yf.download(i, start=startDay, end=endDay)
		temp_return = temp_df['Adj Close'].apply(np.log).diff().dropna()
		out.append(np.array(temp_return).reshape(1, -1))
	return out


def polarize(ser):
	out = []
	for i in ser:
		if i > 0:
			out.append(1)
		if i < 0:
			out.append(-1)
		else:
			out.append(0)
	return out


def fin_model(df, sentiment: int = 1):
	bound1, bound2 = int(df.shape[0]*0.65), int(df.shape[0]*0.8)
	print(bound1)
	print(bound2)
	train = df[:bound2]
	test = df[bound2:]
	y_train = pd.DataFrame()
	y_train['KSS'] = train['KSS']
	train.drop(labels="KSS", axis=1, inplace=True)
	train.drop(labels=train.columns[0], axis=1, inplace=True)

	for i in range(len(y_train)):
		if y_train['KSS'][i] < 0:
			y_train['KSS'][i] = -1
		else:
			y_train['KSS'][i] = 1
	test.drop(labels="KSS", axis=1, inplace=True)
	test.drop(labels=train.columns[0], axis=1, inplace=True)

	if sentiment == 1:
		n = 5
		test.drop(labels="sentiment_snow", axis=1, inplace=True)
		train.drop(labels="sentiment_snow", axis=1, inplace=True)
	elif sentiment == 2:
		n = 5
		test.drop(labels="sentiment_stack", axis=1, inplace=True)
		train.drop(labels="sentiment_stack", axis=1, inplace=True)
	else:
		n = 4
		test.drop(labels="sentiment_stack", axis=1, inplace=True)
		train.drop(labels="sentiment_stack", axis=1, inplace=True)
		test.drop(labels="sentiment_snow", axis=1, inplace=True)
		train.drop(labels="sentiment_snow", axis=1, inplace=True)

	scaler = MinMaxScaler()
	x_train = scaler.fit_transform(train)
	# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
	x_train_1 = x_train[:bound1]
	x_val = x_train[bound1:]
	y_train_1 = [i[0] for i in y_train.values[:bound1].tolist()]
	y_val = [i[0] for i in y_train.values[bound1:].tolist()]

	learning, accuracy = 0, 0
	lr_list = np.arange(.01, 1, 0.001)
	for learning_rate in lr_list:
		gb_clf = GradientBoostingClassifier(n_estimators=n, learning_rate=learning_rate,
											max_features=2, max_depth=6, random_state=0)
		gb_clf.fit(x_train_1, y_train_1)
		if gb_clf.score(x_val, y_val) > accuracy:
			learning = learning_rate
			accuracy = gb_clf.score(x_val, y_val)

		# print("Learning rate: ", learning_rate)
		# print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
		# print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_val, y_val)))

	gb_clf2 = GradientBoostingClassifier(n_estimators=n, learning_rate=learning,
										 max_features=2, max_depth=6, random_state=0)
	gb_clf2.fit(x_train_1, y_train_1)
	prediction = gb_clf2.predict(x_val)

	print("Confusion Matrix:")
	print(confusion_matrix(y_val, prediction))

	print("Classification Report")
	print(classification_report(y_val, prediction))


if __name__ == "__main__":

	# stock data prepare
	symbols = ["^GSPC", "^IXIC", "^DJI", "VIXY", "KSS"]
	intermediate = get_return(symbols, "2019-10-01", "2020-09-30")
	SP500, NASDAQ, DOW, VIXY, KSS = intermediate[0][0], intermediate[1][0], intermediate[2][0], \
									intermediate[3][0], intermediate[4][0]
	stock_df = yf.download("KSS", start="2019-10-01", end="2020-09-30")
	timespan = [i.strftime("%Y-%m-%d") for i in stock_df.index.to_series()]

	# Prepare a data frame to predict
	Analyze = pd.DataFrame(data=KSS, columns=['KSS'])
	Analyze['S&P 500'] = SP500
	Analyze['NASDAQ'] = NASDAQ
	Analyze['DOW'] = DOW
	Analyze['VIXY'] = VIXY
	sentiment = pd.read_csv('./output/sentiments.csv')
	snow = pd.read_csv('./output/snow_sent.csv')
	Analyze['sentiment_snow'] = np.array(snow['MorganStanley'][:-1]).reshape(-1, 1)
	Analyze['sentiment_stack'] = np.array(sentiment['MorganStanley'][:-1]).reshape(-1, 1)
	Analyze.index = timespan[1:]
	Analyze.to_csv('./output/analyze.csv')

	# ----------------------------------------------------------------------

	df_cor = pd.read_csv("./output/analyze.csv")
	# make dependent variable one day ahead
	df_1ahead = pd.DataFrame(Analyze.iloc[:-1, 1:])
	df_1ahead['KSS'] = KSS[1:]
	# df = pd.read_csv("./output data/analyze.csv")

	fin_model(df_cor,0)
	fin_model(df_cor)
	fin_model(df_cor, 2)
	fin_model(df_1ahead, 0)
	fin_model(df_1ahead)
	fin_model(df_1ahead, 2)



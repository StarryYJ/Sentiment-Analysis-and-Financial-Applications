from get_data import *
from Apply2 import *
from test_stacking import stacking
import numpy as np


# get stock information in longer time period
startDay, endDay = '2020-01-01', '2020-10-01'
symbols = ["^GSPC", "^IXIC", "^DJI", "VIXY", "KSS"]
intermediate = get_return(symbols, startDay, endDay)
SP500, NASDAQ, DOW, VIXY, KSS = intermediate[0][0], intermediate[1][0], intermediate[2][0], \
								intermediate[3][0], intermediate[4][0]
stock_df = yf.download("KSS", start=startDay, end=endDay)
timespan = [i.strftime("%Y-%m-%d") for i in stock_df.index.to_series()]

# get twitter in longer time period
market_watch = get_tweet_csv(['MarketWatch'], start='2020-01-01', end='2020-10-01', save=0)
for i in range(market_watch.shape[0]):
	raw_content = extract_url(market_watch['Content'][i])
	market_watch['Content'][i] = ' '.join(raw_content)
market_watch.reset_index(drop=True, inplace=True)
market_watch['Sentiment_snow'] = [SnowNLP(cont).sentiments for cont in market_watch['Content']]

corpus02 = pd.read_csv('./corpus/all-data.csv', encoding='ISO-8859-1', header=None).dropna(axis=0, how='any')
corpus02.columns = ['Sentiment', 'Content']
corpus02.reset_index(drop=True, inplace=True)

x_training = corpus02['Content']
y_training = corpus02['Sentiment'].replace('positive', 1)
y_training = y_training.replace('negative', -1)
y_training = y_training.replace('neutral', 0)
x_testing = market_watch['Content']


# text to vector
classifiers = [svm.SVC(kernel='linear'),
			   RandomForestClassifier(),
			   GradientBoostingClassifier(random_state=10),
			   KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
			   MultinomialNB()]

predictions = stacking(classifiers, x_training, y_training, x_testing)
market_watch['Sentiment_stacking'] = predictions

mark = []
for day in timespan:
	lst = market_watch[market_watch['Date'].isin([day])]
	if lst is None:
		mark.append(0)
	else:
		mark.append(np.min(lst.index))
mark.reverse()
mark.append(len(market_watch))

snow, stack = [], []
for i in range(len(mark) - 1):
	span = [i for i in range(mark[i], mark[i + 1])]
	lst = market_watch.iloc[span, :]
	if lst is not None:
		snow.append(np.mean(lst['Sentiment_snow']))
		stack.append(np.mean(lst['Sentiment_stacking']))
	else:
		snow.append(0)
		stack.append(0)

Analyze = pd.DataFrame(data=KSS, columns=['KSS'])
Analyze['S&P 500'] = SP500
Analyze['NASDAQ'] = NASDAQ
Analyze['DOW'] = DOW
Analyze['VIXY'] = VIXY
Analyze['sentiment_snow'] = np.array(snow[:-1]).reshape(-1, 1)
Analyze['sentiment_stack'] = np.array(stack[:-1]).reshape(-1, 1)
Analyze.index = timespan[1:]


print(Analyze)

# ----------------------------------------------------------------------

df_cor = Analyze
# make dependent variable one day ahead
df_1ahead = pd.DataFrame(Analyze.iloc[:-1, 1:])
df_1ahead['KSS'] = KSS[1:]
# df = pd.read_csv("./output data/analyze.csv")

fin_model(df_cor, 0)
fin_model(df_cor)
fin_model(df_cor, 2)
fin_model(df_1ahead, 0)
fin_model(df_1ahead)
fin_model(df_1ahead, 2)







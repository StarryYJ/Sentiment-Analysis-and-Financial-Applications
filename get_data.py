import yfinance as yf
import pandas as pd
import snscrape.modules.twitter as tw
import sys
import string


# SP500 = yf.download("^GSPC", start="2019-01-01", end="2019-12-31")
# NASDAQ = yf.download("^IXIC", start="2019-01-01", end="2019-12-31")
# DOW = yf.download("^DJI", start="2019-01-01", end="2019-12-31")

# financial_text_01 = pd.read_csv('./corpus/corpus01.csv')
#
# row = pd.read_csv('./corpus/2/stockerbot-export1.csv')
# twitter = pd.read_csv('./corpus/2/tweet_sentiment.csv')

candidates = ['MarketWatch', 'business', 'YahooFinance', 'TechCrunch', 'WSJ', 'Forbes', 'FT', 'TheEconomist', 'nytimes',
              'Reuters', 'GerberKawasaki', 'jimcramer', 'TheStreet', 'TheStalwart', 'TruthGundlach', 'CarlCIcahn',
              'ReformedBroker', 'benbernanke', 'bespokeinvest', 'BespokeCrypto', 'stlouisfed', 'federalreserve',
              'GoldmanSachs', 'ianbremmer', 'MorganStanley', 'AswathDamodaran', 'mcuban', 'muddywatersre', 'StockTwits',
              'SeanaNSmith']


def get_tweet_txt(id_list: list, start: str = '2020-06-01', end: str = '2020-10-01'):
    for ID in id_list:
        f = open('./intermediate file/' + ID + '.txt', 'w', encoding='utf-8')

        for tweet in tw.TwitterSearchScraper(
                query="from:" + ID + " since:" + start + " until:" + end).get_items():
            date_str = tweet.date.strftime("%Y-%m-%d %H:%M:%S%z")
            date_str = date_str[:-2] + ":" + date_str[-2:]
            f.write(date_str + "|" + tweet.content + "\n")
        f.close()


def get_tweet_csv(id_list: list, start: str = '2020-06-01', end: str = '2020-10-01', save=1):
    for ID in id_list:
        output_df = pd.DataFrame(data=None, columns=['Twitter', 'Date', 'Content'])
        date, content = [], []
        for tweet in tw.TwitterSearchScraper(
                query="from:" + ID + " since:" + start + " until:" + end).get_items():
            date_str = tweet.date.strftime("%Y-%m-%d %H:%M:%S%z")
            date_str = date_str[:10]
            date.append(date_str)
            content.append(tweet.content)
        output_df['Date'], output_df['Content'] = date, content
        output_df['Twitter'] = ID
        if save == 1:
            output_df.to_csv('./intermediate/' + ID + '.csv', index=False)
    return output_df


def extract_url(content: str):
    out_index, n = [], 0
    lst = content.split()
    for i in range(len(lst)):
        if lst[i].find('https://') == 0:
            out_index.append(i)
    for j in out_index:
        lst.pop(j-n)
        n += 1
    return lst


def clean_punctuation(content: str):
    lst = content.split()
    for i in range(len(lst)):
        while lst[i][0] in string.punctuation:
            lst[i] = lst[i][1:]
        while lst[i][-1] in string.punctuation:
            lst[i] = lst[i][:-1]
    return ' '.join(lst)


def combine_csv(name_list: list):
    merged_df, temp_df = pd.DataFrame(data=None, columns=['Twitter', 'Date', 'Content']), None
    for ID in name_list:
        try:
            temp_df = pd.read_csv('./intermediate/' + ID + '.csv')
        except IOError:
            sys.stderr.write("You don't have corresponding file.")
            exit(1)
        for i in range(temp_df.shape[0]):
            raw_content = extract_url(temp_df['Content'][i])
            temp_df['Content'][i] = ' '.join(raw_content)
        merged_df = merged_df.append(temp_df)
    merged_df.to_csv('./output data/merged.csv', index=False)
    return merged_df


def clean_punctuation_new(content):
    out = []
    for word in content:
        while word[0] in string.punctuation:
            word = word[1:]
        while word[-1] in string.punctuation:
            word = word[:-1]
        out.append(word)
    return out


def clean_csv(ID: str):
    temp_df = pd.read_csv('./intermediate/' + ID + '.csv')
    for i in range(temp_df.shape[0]):
        raw_content = extract_url(temp_df['Content'][i])
        temp_df['Content'][i] = ' '.join(raw_content)
    return temp_df


if __name__ == "__main__":
    tweet_id = ['MarketWatch', 'business', 'WSJ', 'TheEconomist', 'nytimes', 'MorganStanley']
    get_tweet_csv(tweet_id, '2019-10-01', '2020-10-01')
    combine_tweet_csv = combine_csv(tweet_id)





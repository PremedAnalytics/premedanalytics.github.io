from functools import reduce
from textblob import TextBlob
import pandas as pd

data = pd.read_csv("sdnThreads.csv", encoding='UTF-8')


def pol(x):
    return TextBlob(x).sentiment.polarity


def sub(x):
    return TextBlob(x).sentiment.subjectivity


schoolList = pd.DataFrame(data['school'].unique(), columns=["school"])
ratings = list()
ratings.append(schoolList)

for i in range(0, 7):
    year = i + 2014
    yearlyData = data[data['cycle'] == year]
    corpus = yearlyData.loc[:, ("school", "post")]
    corpus['post'] = corpus['post'].astype(str)

    corpus['polarity'+str(year)] = corpus['post'].apply(pol)
    corpus['subjectivity'+str(year)] = corpus['post'].apply(sub)
    mean_df = corpus.groupby("school").mean()
    ratings.append(mean_df)

df = reduce(lambda df1, df2: pd.merge(
    df1, df2, on='school', how="outer"), ratings)
df.to_csv("sentimentRatings.csv")

longString = corpus.groupby(['school'], as_index=False).agg({'post': ' '.join})
longString['polarity'] = longString['post'].apply(pol)
longString['subjectivity'] = longString['post'].apply(sub)

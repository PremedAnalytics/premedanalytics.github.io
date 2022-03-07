from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Get files
files = glob.glob("./sdnSchoolYear/" + "*.csv")

# Define functions that analyze for polarity and subjectivity


def pol(x): return TextBlob(x).sentiment.polarity
def sub(x): return TextBlob(x).sentiment.subjectivity


# Read 2015 thread
df = pd.read_csv(files[1], encoding='UTF-8')
byPost = list()
totalText = list()
schoolList = df['school'].unique()
byPost[1] = schoolList


for file in files:
    df = pd.read_csv(file, encoding='UTF-8')
    corpus = df.loc[:, ("school", "post")]
    corpus['post'] = corpus['post'].astype(str)
    longString = corpus.groupby(
        ['school'], as_index=False).agg({'post': ' '.join})

    corpus['polarity'] = corpus['post'].apply(pol)
    corpus['subjectivity'] = corpus['post'].apply(sub)
    longString['polarity'] = longString['post'].apply(pol)
    longString['subjectivity'] = longString['post'].apply(sub)

    grouped_df = corpus.groupby("school")
    mean_df = grouped_df.mean()

    mean_df['polarity']

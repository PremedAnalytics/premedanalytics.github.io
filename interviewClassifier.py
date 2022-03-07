import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from gensim.parsing.preprocessing import remove_stopwords
import contractions
import re
import string

# Import data
df = pd.read_csv("./labeledSDNPosts/interviewSDN.csv")
annotated = df.loc[:2823, ("school", "post", "interviewKeys")]
posts = annotated.loc[:, "post"]
labels = annotated.loc[:, "interviewKeys"]

# Expand Contractions
posts = posts.apply(lambda x: contractions.fix(x))

# Remove Punctuation
posts = posts.apply(lambda x: re.sub(
    '[%s]' % re.escape(string.punctuation), '', x))

# Remove digits
posts = posts.apply(lambda x: re.sub('[0-9]+', '', x))

# Change all the words to lowercase
posts = posts.str.lower()

# Remove Stop Words
stopwords = stopwords.words('english')
newposts = posts.apply(lambda x: remove_stopwords("".join(x)))

# Lemmatize
lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


newposts = newposts.apply(lambda text: lemmatize_words(text))

# Split out train/test dataset
X = newposts
Y = labels
posts_train, posts_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=1000)

# Create Vectors out of Text (TF-IDF)
tfidfvectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidfvectorizer.fit_transform(posts_train)
X_test = tfidfvectorizer.transform(posts_test)

# Create Vectors out of text (TF-IDF n-gram)
tfidf_vect_ngram = TfidfVectorizer(
    analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=10000)
xtrain_tfidf_ngram = tfidf_vect_ngram.fit_transform(posts_train)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(posts_test)

# Make a List of Models
models = [
    ('LogReg', LogisticRegression()),
    ('RF', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC()),
    ('GNB', GaussianNB()),
    ('XGB', XGBClassifier()),
    ('NaiveBayes', MultinomialNB())
]

# SVM Model
svm = SVC()
svm.fit(xtrain_tfidf_ngram, Y_train)
svm_predict = svm.predict(xtest_tfidf_ngram)
confusion_matrix(Y_test, svm_predict)
f1_score(svm_predict, Y_test)

# Test a GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain_tfidf_ngram.toarray(), Y_train)
gnb_predict = gnb.predict(xtest_tfidf_ngram.toarray())
confusion_matrix(Y_test, gnb_predict)
f1_score(gnb_predict, Y_test)

# Test a KNN Classifier
knn = KNeighborsClassifier()
knn.fit(xtrain_tfidf_ngram, Y_train)
knn_predict = knn.predict(xtest_tfidf_ngram)
confusion_matrix(Y_test, knn_predict)
f1_score(knn_predict, Y_test)

# Test a Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(xtrain_tfidf_ngram, Y_train)
rf_predict = rf.predict(xtest_tfidf_ngram)
confusion_matrix(Y_test, rf_predict)
f1_score(rf_predict, Y_test)

# Test a XGBClassifier
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(xtrain_tfidf_ngram, Y_train)
xgb_predict = xgb.predict(xtest_tfidf_ngram)
confusion_matrix(Y_test, xgb_predict)
f1_score(xgb_predict, Y_test)

# Test a logistic regression model
classifier = LogisticRegression()
classifier.fit(xtrain_tfidf_ngram, Y_train)
classifier_predict = classifier.predict(xtest_tfidf_ngram)
confusion_matrix(Y_test, classifier_predict)
f1_score(classifier_predict, Y_test)

# Naive Bayes Model
naive_bays_classifier = MultinomialNB()
naive_bays_classifier.fit(xtrain_tfidf_ngram, Y_train)
naive_bays_pred = naive_bays_classifier.predict(xtest_tfidf_ngram)

# Find F1 Score
confusion_matrix(Y_test, naive_bays_pred)
confusion_matrix(Y_test, classifier_predict)
confusion_matrix(Y_test, )
f1_score(naive_bays_pred, Y_test)
f1_score(classifier_predict, Y_test)

sum(xgb_predict)
confusion_matrix(Y_test, xgb_predict)


print(f1_score(svm_predict, Y_test))
print(f1_score(gnb_predict, Y_test))
print(f1_score(knn_predict, Y_test))
print(f1_score(rf_predict, Y_test))
print(f1_score(xgb_predict, Y_test))
print(f1_score(naive_bays_pred, Y_test))
print(f1_score(classifier_predict, Y_test))


testResults = pd.DataFrame(posts_test)
testResults['label'] = xgb_predict.tolist()
pd.set_option('display.max_rows', 565)
pd.set_option('display.max_rows', 10)

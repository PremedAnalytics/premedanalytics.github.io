from wordcloud import WordCloud
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./sdnSchoolYear/sdn2014.csv")
posts = df['post'].astype(str)
postID = df['Unnamed: 0']

# Vectorize text with TFIDF Method
vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(posts)

# Elbow Method to decide number of clusters (Picked to have 9 clusters)
Sum_of_squared_distances = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10).fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Build K Cluster Model
true_k = 8
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
labels = model.labels_
wiki_cl = pd.DataFrame(list(zip(postID, labels)),
                       columns=['postID', 'cluster'])
print(wiki_cl.sort_values(by=['cluster']))


finaldf = df.set_index('Unnamed: 0').join(wiki_cl.set_index('postID'))
finaldf[finaldf['cluster'] == 3]

# Conclusion: Well, I filtered the posts, but they've filtered into things like "Good Luck!"
# and not actually by the root content of the post.

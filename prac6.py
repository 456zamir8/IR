'''
Clustering for Information Retrieval 
Implement a clustering algorithm (e.g., K-means or hierarchical clustering). 
Apply the clustering algorithm to a set of documents and evaluate the clustering results.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample documents
documents = [
    "Cats are known for their agility and grace",  
    "Dogs are often called ‘man’s best friend’.",  
    "Some dogs are trained to assist people with disabilities.",  
    "The sun rises in the east and sets in the west.",  
    "Many cats enjoy climbing trees and chasing toys.",  
]

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 2: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Step 3: Dimensionality Reduction (PCA to 2D)
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Step 4: Plotting
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
labels = kmeans.labels_

for i in range(len(documents)):
    plt.scatter(reduced_X[i][0], reduced_X[i][1], color=colors[labels[i]], label=f'Cluster {labels[i]}')

# Annotate each point with the document index
for i, txt in enumerate(documents):
    plt.annotate(f'Doc {i+1}', (reduced_X[i][0], reduced_X[i][1]), fontsize=9)

# Remove duplicate legends
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("K-Means Clustering of Text Documents (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

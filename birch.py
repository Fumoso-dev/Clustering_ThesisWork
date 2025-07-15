import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns

# Reading csv
realDataset = "mydataset/wed_benign77.csv"

# Extracting data from txt
data = np.genfromtxt(
    realDataset,
    delimiter=",",
    usecols=range(0, 77),
    skip_header=1
  )
  
# Extracting label names  
true_label_names = np.genfromtxt(
    realDataset,
    delimiter=",",
    usecols=77,
    skip_header=1,
    dtype="str"
  )

# In order to use the label names, they have to be converted into integers
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_label_names)
print(true_labels[:5])

print(label_encoder.classes_)
n_clusters = len(label_encoder.classes_)

# This is a pipeline to process the data before using Birch, because
# they are not in the optimal format otherwise
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

#This is a separate pipeline to perform AffinityPropagation clustering
clusterer = Pipeline(
   [
       (
           "Birch",
           Birch(
                n_clusters=12
           ),
       ),
   ]
)

#The Pipeline class can be chained to form a larger pipeline.
#Build an end-to-end k-means clustering pipeline by passing the "preprocessor" and "clusterer" pipelines to Pipeline:
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

#The pipeline performs all the necessary steps to execute k-means clustering on the data:
pipe.fit(data)

#Evaluate the performance by calculating the silhouette coefficient:
preprocessed_data = pipe["preprocessor"].transform(data)

predicted_labels = pipe["clusterer"]["Birch"].labels_

#silhouette_score(preprocessed_data, predicted_labels)

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["Birch"].labels_
print("PCADF:")
print(pcadf)
pcadf.to_csv("wed_afterBirch12.csv")
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

# Visualization of the graph
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    data=pcadf,
    x="component_1",
    y="component_2",
    s=50,    
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering results"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

#plt.show()
plt.savefig("wed_afterBirch12.png", bbox_inches='tight')



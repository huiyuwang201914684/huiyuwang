from time import time
import numpy as np
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import mixture

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)
digits = load_digits()
data = scale(digits.data)
labels_true= digits.target




print(50 * '*')
print('The evalution result is as followed')
print(50 * '-')
print('name'.ljust(15) +  '  NMI      homogeneity completeness  ')

def bench(name, labels_true, labels_pred):
    print('%-15s%.4f\t\t%.4f\t\t%.4f'
          % (name,
             metrics.normalized_mutual_info_score(labels_true, labels_pred ,average_method='arithmetic'),
             metrics.homogeneity_score(labels_true, labels_pred),
             metrics.completeness_score(labels_true,labels_pred),
             ))


km = KMeans().fit(data)
labels_pred= km.labels_
bench('Kmeans',labels_true, labels_pred)

af = AffinityPropagation().fit(data)
labels_pred= af.labels_
bench('AP',labels_true, labels_pred)

ms = MeanShift().fit(data)
labels_pred= ms.labels_
bench('MeanShift',labels_true, labels_pred)

sc = SpectralClustering(n_clusters=6, eigen_solver='arpack', affinity="nearest_neighbors").fit(data)
labels_pred= sc.labels_
bench('Spectral',labels_true, labels_pred)

wh = AgglomerativeClustering(linkage="ward").fit(data)
labels_pred= wh.labels_
bench('WH',labels_true, labels_pred)

ag = AgglomerativeClustering(linkage="average").fit(data)
labels_pred= ag.labels_
bench('Agglomerative',labels_true, labels_pred)

db = DBSCAN(eps=4, min_samples=8).fit(data)
labels_pred= db.labels_
bench('DBSCAN',labels_true, labels_pred)

labels_pred = mixture.GaussianMixture(n_components=6).fit_predict(data)
bench('GaussianMixture',labels_true, labels_pred)
print(50 * '-')
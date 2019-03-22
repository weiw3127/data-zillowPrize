
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
from sklearn.model_selection import train_test_split

prop = pd.read_csv('properties_2016.csv')
train = pd.read_csv("train_2016_v2.csv")

#### geographic with PCA and clustering

# only choose the sample with geographic data
test_yes = prop[prop.latitude.isnull() == False]
test_no = prop[prop.latitude.isnull() == True]

# PCA
coords = np.vstack((test_yes[['latitude', 'longitude']].values))
pca = PCA().fit(coords)
test_yes['lat_pca'] = pca.transform(test_yes[['latitude', 'longitude']])[:, 0]
test_yes['lon_pca'] = pca.transform(test_yes[['latitude', 'longitude']])[:, 1]

# cluster
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
test_yes.loc[:, 'cluster'] = kmeans.predict(test_yes[['latitude', 'longitude']])

prop = pd.concat([test_yes, test_no])
del test_yes, test_no

## feaure selection with simple backward elimination, delect the useless data
dropping = ['architecturalstyletypeid', 'basementsqft', 'buildingclasstypeid', 'decktypeid', 'finishedsquarefeet13',
            'finishedsquarefeet6', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'storytypeid',
            'typeconstructiontypeid', 'yardbuildingsqft26', 'fireplaceflag', 'taxdelinquencyflag',
            'taxdelinquencyyear', 'buildingclasstypeid', 'decktypeid', 'hashottuborspa', 'poolcnt',
            'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'storytypeid', 'fireplaceflag', 'assessmentyear',
            'taxdelinquencyflag']
prop = prop.drop(dropping, axis=1)

## dealing object type and missing value
for c in prop.columns:
    prop[c] = prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

#### dealing the training set
prop.to_csv('prop1004.csv', index=False)

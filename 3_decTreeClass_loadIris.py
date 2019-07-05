import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(dir(iris))

# ==========================
# create dataframe

dfIris = pd.DataFrame(
    iris['data'],
    columns = iris['feature_names']
)
dfIris['target'] = iris['target']
dfIris['jenis'] = dfIris['target'].apply(
    lambda z: iris['target_names'][z]
)
# print(dfIris.head())

# ==========================
# split: train 80% & test 20%

from sklearn.model_selection import train_test_split
x_train, x_tes, y_train, y_tes = train_test_split(
    dfIris[[
        'sepal length (cm)', 
        'sepal width (cm)', 
        'petal length (cm)', 
        'petal width (cm)'
    ]],
    dfIris['jenis'],
    test_size = .2
)

# print(len(x_train))
# print(len(x_tes))

# ==========================
# decision tree

from sklearn import tree
model = tree.DecisionTreeClassifier()

# train
model.fit(x_train, y_train)

# predict
prediksi = model.predict([x_tes.iloc[0]])
aktual = y_tes.iloc[0]
print(prediksi)
print(aktual)

# accuracy
print(model.score(x_train, y_train) * 100, '%')
print(model.score(x_tes, y_tes) * 100, '%')
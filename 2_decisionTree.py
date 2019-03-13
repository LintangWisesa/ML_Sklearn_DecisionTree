import pandas as pd
import numpy as np

# =================================
# load csv & create dataframe
df = pd.read_csv('0_data.csv')
# print(df)

# =================================
# convert nominal data => ordinal data
from sklearn.preprocessing import LabelEncoder

labelKantor = LabelEncoder()
df['kantorLE'] = labelKantor.fit_transform(df['kantor'])
labelJabatan = LabelEncoder()
df['jabatanLE'] = labelJabatan.fit_transform(df['jabatan'])
labelTitel = LabelEncoder()
df['titelLE'] = labelTitel.fit_transform(df['titel'])

df = df.drop(
    ['kantor', 'jabatan', 'titel'],
    axis = 'columns'    
)
# print(df)

# ===============================
# kantorLE    : 0 Facebook, 1 Google, 2 Tesla
# jabatanLE   : 0 GM, 1 Manager, 2 Staf
# titelLE     : 0 S1, 1 S2
# ===============================

# split: train 80% & test 20%
from sklearn.model_selection import train_test_split
x_train, x_tes, y_train, y_tes = train_test_split(
    df[['kantorLE', 'jabatanLE', 'titelLE']], 
    df['gaji>50'],
    test_size = .2,
    random_state = 1
)
print(x_train)
# print(len(x_tes))
print(y_train)
# print(len(y_tes))

# ===============================
# decision tree algo
from sklearn import tree
model = tree.DecisionTreeClassifier()

# train
model.fit(x_train, y_train)

# accuracy
acc = model.score(x_train, y_train)
print(acc * 100, '%')
acc2 = model.score(x_tes, y_tes)
print(acc2 * 100, '%')

# predict kantor, jabatan, titel
print(model.predict([[1, 1, 0]]))
print(model.predict([[1, 1, 1]]))
# print(model.predict([[1, 3, 0]]))

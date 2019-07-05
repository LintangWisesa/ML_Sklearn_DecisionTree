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
# kantor    : 0 Facebook, 1 Google, 2 Tesla
# jabatan   : 0 GM, 1 Manager, 2 Staf
# titel     : 0 S1, 1 S2
# ===============================

# decision tree algo
from sklearn import tree
model = tree.DecisionTreeClassifier()

# train
model.fit(
    df[['kantorLE', 'jabatanLE', 'titelLE']], 
    df['gaji>50']
)

# accuracy
acc = model.score(
    df[['kantorLE', 'jabatanLE', 'titelLE']], 
    df['gaji>50']
)
print(acc * 100, '%')

# predict kantor, jabatan, titel
print(model.predict([[0, 0, 0]]))
print(model.predict([[2, 0, 0]]))
print(model.predict([[1, 3, 0]]))

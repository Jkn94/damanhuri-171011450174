import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('data_set.csv')

df.produktif[df.produktif == 'good' ] = 1
df.produktif[df.produktif == 'bad' ] = 2

y = df['produktif'].values
y = y.astype('int')

x = df.drop(labels=['produktif'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 20)

model = RandomForestClassifier(n_estimators = 10, random_state = 30)

model.fit(x_train, y_train)
prediksi_tes = model.predict(x_test)

print(" Akurasi = ", metrics.accuracy_score(y_test, prediksi_tes))
list_fitur = list(x.columns)
fitur_impo = pd.Series(model.feature_importance_, index = list_fitur).sort_values(ascending=False)

print(fitur_impo)

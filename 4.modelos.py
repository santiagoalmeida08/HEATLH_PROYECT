import pandas as pd
import sqlite3 as sql
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression

# conexión a la base de datos

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# cargar tabla

df = pd.read_sql('SELECT *  from hrmin', conn)
df.info()

# Variables explicativas

x = df.drop(['readmitted'], axis=1)
y = df['readmitted']

# seleccionar variables representativas


def recursive_feature_selection(x,y,model,k): #model=modelo que me va a servir de estimador para seleccionar las variables
                                              # K = numero de variables a seleccionar
  rfe = RFE(model, n_features_to_select=k, step=1) # step=1 significa que se eliminara una variable en cada iteracion
  fit = rfe.fit(x, y) # ajustar el modelo
  b_var = fit.support_ # seleccionar las variables
  print("Num Features: %s" % (fit.n_features_)) # numero de variables seleccionadas
  print("Selected Features: %s" % (fit.support_)) # variables seleccionadas
  print("Feature Ranking: %s" % (fit.ranking_)) # ranking de las variables

  return b_var 

#recursive_feature_selection(x,y,LogisticRegression(),8) # seleccionar 5 variables
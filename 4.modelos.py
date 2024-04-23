
#Paquetes necesarios
import pandas as pd
import numpy as np
import sqlite3 as sql
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import funciones as fn
import matplotlib.pyplot as plt


# conexión a la base de datos

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# cargar tabla

df = pd.read_sql('SELECT *  from hrmin', conn)
df.info()

# Variables explicativas

x = df.drop(['readmitted'], axis=1)
y = df['readmitted']
y_mod = y.replace({'yes':1, 'no':0})

# Encoding variables #

#Dummies -medical speciality--AC1 TEST
df1=x.copy()

df1.dtypes
df1['edad'] = df1['edad'].astype('object')

list_dummies =['medical_specialty','diag_1','diag_2','diag_3','glucose_test','A1Ctest']

#Ordinal - edad
list_ordinal = ['edad']  

#Label - change y diabetes med
list_label = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]


x_encoded = fn.encode_data(df1, list_label, list_dummies,list_ordinal)

#Escalado de variables 
scaler = StandardScaler()
Xesc = scaler.fit_transform(x_encoded) 
Xesc1 = pd.DataFrame(Xesc, columns = x_encoded.columns)

#Definir modelos a evaluar ____ usar bagging

mod_tree = DecisionTreeClassifier()
mod_rand = RandomForestClassifier()
#mos_xgbosting = XGBClassifier()
mod_log = LogisticRegression()
modelos = list([mod_tree,mod_rand,mod_log])

import seaborn as sns


forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(Xesc, y_mod)

# Obtenemos la importancia de las características
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#TERMINAR GRAFICO, SINO USAR FUNCION PROFESROR
plt.figure()
plt.title("Importancia de las características")
plt.bar(range(Xesc.shape[1]), importances[indices], align="center")
plt.xticks(range(Xesc.shape[1]), indices)
plt.xlabel("Características")

var = recursive_feature_selection(Xesc,y_mod,LogisticRegression(),8) # seleccionar 5 variables

def medir_modelos(modelos,scoring,X,y,cv):
    "Recibe como parametros una lista de modelos, la metrica con la cual se quiere evaluar su desempeño, la base de datos escalada y codificada, la variable objetivo y el numero de folds para la validación cruzada."
    os = RandomOverSampler() # Usamos random oversampling para balancear la base de datos ya que la variable objetivo esta desbalanceada
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        pipeline = make_pipeline(os, modelo)
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["decision_tree","random_forest","reg_logistic"]
    return metric_modelos   

medir_modelos(modelos,"f1",Xesc,y_mod,5)

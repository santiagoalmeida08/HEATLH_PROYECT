 
 # paquetes 
 
import pandas as pd
import sqlite3 as sql
import funciones as fn 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import OrdinalEncoder

# conexión a la base de datos

r = pd.read_csv('data//hospital_readmissions.csv')
r

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# insertar el DataFrame en la base de datos
r.to_sql('hr', conn, if_exists='replace', index=False)

cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()

# cargar base de datos 

hr = pd.read_sql('SELECT * FROM hr', conn)
hr.isnull().sum()

hr[['glucose_test']].value_counts()
hr[['A1Ctest']].value_counts()

###   Preprocesamiento   ###

hr[['age']].value_counts()# volver categoria la variable edad
encoder= OrdinalEncoder(categories=[['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']])
hr2=hr.copy()
hr2['edad']=encoder.fit_transform(hr2[['age']])#se uso ordinal encoder porque tienen la misma distancia

hr2.dtypes 
#Se debe borrar la columna age y cambiar de float a categorica la columna edad 



# Preprocesamiento variables 9--17

fn.ejecutar_sql('1.preprocesamiento.sql',conn)

bf = pd.read_sql('SELECT * FROM hrmin', conn)
bf['medical_specialty'].value_counts()


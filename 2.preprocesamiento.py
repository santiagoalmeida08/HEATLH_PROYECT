#En este apartado se realizará la carga de datos,creación de la base de datos y preprocesamiento de los datos
# para el análisis de los datos.

#1.Paquetes requeridos
#2.Carga de datos
#3.Preprocesamiento

 
#1.Paquetes requeridos 
import pandas as pd
import sqlite3 as sql
import funciones as fn 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import OrdinalEncoder

#2.Carga de datos
r = pd.read_csv('data//hospital_readmissions.csv')
r

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# insertar el DataFrame en la base de datos
r.to_sql('hr', conn, if_exists='replace', index=False)

cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()

#Cargar base de datos 

hr = pd.read_sql('SELECT * FROM hr', conn)
hr.isnull().sum()

hr[['glucose_test']].value_counts()
hr[['A1Ctest']].value_counts()

#3. Preprocesamiento 

hr[['age']].value_counts()# volver categoria la variable edad
encoder= OrdinalEncoder(categories=[['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']])
re2=hr.copy()
re2['edad']=encoder.fit_transform(re2[['age']])#se uso ordinal encoder porque tienen la misma distancia

re2.dtypes 
re2.to_csv('data/re2.csv', index= False)
base = pd.read_csv('data//re2.csv')
base.to_sql('basecambios', conn, if_exists='replace', index=False)


fn.ejecutar_sql('1.preprocesamiento.sql',conn)

bf = pd.read_sql('SELECT * FROM hrmin', conn)
bf['medical_specialty'].value_counts()



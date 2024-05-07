
"""En este apartado se realizara las predicciones con el modelo entrenado y se exportara a excel las predicciones"""

#1.Paquetes requeridos
#2.Importar elementos necesarios para despliegue
#3.Cargar base de datos con la que se quiere hacer predicciones
#4.Transformación de datos para realizar predicciones
#5.Cargar modelo y realizar predicciones con el umbral
#6.Importancia de las variables del modelo
#7.Predicciones
#8.Exportar predicciones e importancia de variables a excel

#1.Paquetes requeridos
import sqlite3 as sql

import funciones as fn  #archivo de funciones propias
import pandas as pd ### para manejo de datos
import joblib
import numpy as np

#2. Importar elementos necesarios para despliegue
final = joblib.load("salidas\\final.pkl")
list_dumies=joblib.load("salidas\\list_dumies.pkl")
list_label=joblib.load("salidas\\list_label.pkl")
list_ordinal=joblib.load("salidas\\list_ordinal.pkl")
#var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 

#3. Cargar base de datos con la que se quiere hacer predicciones
conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()

# Cargar tabla

df = pd.read_sql('SELECT *  from hr_full', conn)
df.info()
df.isnull().sum() # Verificar valores nulos y variables
df.columns

#4. Transformación de datos para realizar predicciones
df_t=fn.preparar_datos(df) 
df_t.columns

#5.Cargar modelo y realizar predicciones con el umbral
probabilidades = final.predict_proba(df_t)[:, 1]#definir las probabilidades asociadas a la clase 1 (readmitido)
umbral = 0.35  # Puedes ajustar este valor según tus necesidades
predicciones= (probabilidades > umbral).astype(int)

#6.Importancia de las variables del modelo
coeficientes = final.coef_[0]
feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'coef': coeficientes})
feature_importances_df = feature_importances_df.sort_values(by='coef', ascending=False) #Base de datos con la importancia de mayor a menor

#7.Predicciones
predicciones=pd.DataFrame(predicciones)
predicciones = predicciones.replace({1:'si',0:'no'})

#8.Exportar predicciones e importancia de variables a excel

#Asumimos que el index de la tabla original es el id de cada paciente 
predicciones.to_excel("salidas\\predicciones.xlsx")  #Exportar todas las  predicciones 
feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") #Exportar importancia de variables
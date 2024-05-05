# Importar librerias necesarias #
"""En este apartado se realizara las predicciones con el modelo entrenado y se exportara a excel las predicciones"""

#1. Importar elementos necesarios para despliegue
#2. Cargar base de datos con la que se quiere hacer predicciones
#3. Transformación de datos para realizar predicciones
#4. Predicciones
#5. Importancia de las variables del modelo
#6. Exportar predicciones a excel

import sqlite3 as sql

import funciones as funciones  #archivo de funciones propias
import pandas as pd ### para manejo de datos
import joblib
import numpy as np

# Importar elementos necesarios para despliegue
final = joblib.load("salidas\\final.pkl")
list_dumies=joblib.load("salidas\\list_dumies.pkl")
list_label=joblib.load("salidas\\list_label.pkl")
list_ordinal=joblib.load("salidas\\list_ordinal.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 

# Cargar base de datos con la que se quiere hacer predicciones


conn = sql.connect('data//readmissions.db')
cur = conn.cursor()


cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()
# cargar tabla

df = pd.read_sql('SELECT *  from hr_full', conn)
df.info()
df.isnull().sum() # Verificar valores nulos y variables
df.columns

# Transformación de datos para realizar predicciones
df_t=funciones.preparar_datos(df) 
df_t.columns

#Cargar modelo y predecir
final1 = joblib.load("salidas\\rf_final.pkl")
predicciones=final1.predict(df_t) # se realiza la predicción
#pd_pred=pd.DataFrame(predicciones, columns=['Attrition_17']) # se agrega la variable attrition_17 que es la predicción referente al abandono de los empleados

#Crear base con predicciones
#perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)

#perf_pred['Attrition_17'].value_counts() # Verificar valores nulos
   
# Importancia de las variables del modelo
importances = final1.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'Importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False) #Base de datos con la importancia de mayor a menor
#En la tabla podemos observar que el salario es la variable más importante para predecir la rotación de empleados; esto es importante ya que como se 
#mencionó en el análisis exploratorio, los empleados que ganan menos eran los que abandonaban la empresa.
 
# Exportar predicciones e importancia de variables a excel
predicciones.to_excel("salidas\\predicciones.xlsx")  #Exportar todas las  predicciones 
feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") #Exportar importancia de variables
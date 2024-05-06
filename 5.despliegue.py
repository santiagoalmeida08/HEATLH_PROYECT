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
#var_names=joblib.load("salidas\\var_names.pkl")
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

def preparar_datos (df):
    
    #Se realizan los cambios necesarios para que la base nueva tenga el mismo formato que la base con la que se entreno el modelo
    df['edad'] = df['edad'].astype('object')
    # Cargar las listas, escalador y variables necesarias para realizar las predicciones
    list_dumies=joblib.load("salidas\\list_dumies.pkl")
    list_label=joblib.load("salidas\\list_label.pkl")
    list_ordinal=joblib.load("salidas\\list_ordinal.pkl")
    #var_names=joblib.load("salidas\\var_names.pkl")
    scaler=joblib.load("salidas\\scaler.pkl") 

    #Ejecutar funciones de transformaciones 
    df_dummies = df.copy()
    df_dummies= funciones.encode_data(df, list_label, list_dumies,list_ordinal)
    
    #Escalar variables
    df_dummies=df_dummies.drop(['readmitted'],axis=1)
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    #X=X[var_names]
    
    return X

df_t=preparar_datos(df) 
df_t.columns

#Cargar modelo y realizar predicciones con el umbral
probabilidades = final.predict_proba(df_t)[:, 1]#definir las probabilidades asociadas a la clase 1 (readmitido)
umbral = 0.35  # Puedes ajustar este valor según tus necesidades
predicciones= (probabilidades > umbral).astype(int)

# Importancia de las variables del modelo
coeficientes = final.coef_[0]
feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'coef': coeficientes})
feature_importances_df = feature_importances_df.sort_values(by='coef', ascending=False) #Base de datos con la importancia de mayor a menor

# Exportar predicciones e importancia de variables a excel
predicciones=pd.DataFrame(predicciones)
predicciones = predicciones.replace({1:'si',0:'no'})

#Asumimos que el index de la tabla original es el id de cada paciente 
predicciones.to_excel("salidas\\predicciones.xlsx")  #Exportar todas las  predicciones 
feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") #Exportar importancia de variables
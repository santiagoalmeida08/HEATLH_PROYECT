# Importar librerias necesarias #
"""En este apartado se realizara las predicciones con el modelo entrenado y se exportara a excel las predicciones"""

#1. Importar elementos necesarios para despliegue
#2. Cargar base de datos con la que se quiere hacer predicciones
#3. Transformación de datos para realizar predicciones
#4. Predicciones
#5. Importancia de las variables del modelo
#6. Exportar predicciones a excel


import funciones as funciones  #archivo de funciones propias
import pandas as pd ### para manejo de datos
import joblib
import numpy as np

# Importar elementos necesarios para despliegue
rf_final = joblib.load("salidas\\rf_final.pkl")
list_oe=joblib.load("salidas\\list_oe.pkl")
list_le=joblib.load("salidas\\list_le.pkl")
list_dd=joblib.load("salidas\\list_dd.pkl")
list_cat=joblib.load("salidas\\list_cat.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 

# Cargar base de datos con la que se quiere hacer predicciones

df_pred= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/baseprediccion.csv'
df = pd.read_csv(df_pred,sep=',')
df.isnull().sum() # Verificar valores nulos y variables
df.columns

# Transformación de datos para realizar predicciones
df_t=funciones.preparar_datos(df) 
df_t.columns

#Cargar modelo y predecir
rf_final = joblib.load("salidas\\rf_final.pkl")
predicciones=rf_final.predict(df_t) # se realiza la predicción
pd_pred=pd.DataFrame(predicciones, columns=['Attrition_17']) # se agrega la variable attrition_17 que es la predicción referente al abandono de los empleados

#Crear base con predicciones
perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)

perf_pred['Attrition_17'].value_counts() # Verificar valores nulos
   
# Importancia de las variables del modelo
importances = rf_final.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'Importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False) #Base de datos con la importancia de mayor a menor
#En la tabla podemos observar que el salario es la variable más importante para predecir la rotación de empleados; esto es importante ya que como se 
#mencionó en el análisis exploratorio, los empleados que ganan menos eran los que abandonaban la empresa.
 
# Exportar predicciones e importancia de variables a excel
perf_pred.to_excel("salidas\\predicciones.xlsx")  #Exportar todas las  predicciones 
feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") #Exportar importancia de variables
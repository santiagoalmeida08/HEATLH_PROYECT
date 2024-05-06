import sqlite3 as sql
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.feature_selection import SelectFromModel
import joblib


def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)
    
def chi_square_test(dataframe, target): 
    for col in dataframe.columns:
        print(col)
        print("----------")
        cross_tab = pd.crosstab(dataframe[col], dataframe[target], margins = False)
        stat, p, dof, expected = chi2_contingency(cross_tab)
        print(f"Chi-Square Statistic: {stat}, p-value: {p}")
        print("===========")
        

def encode_data(df, list_le, list_dd,list_oe): 
    df_encoded = df.copy()   
    "Recibe como parametros la base de datos y las listas de variables que se quieren codificar"
    #Get dummies
    df_encoded=pd.get_dummies(df_encoded,columns=list_dd)
    
    # Ordinal Encoding
    oe = OrdinalEncoder()
    for col in list_oe:
        df_encoded[col] = oe.fit_transform(df_encoded[[col]])
    
    # Label Encoding
    le = LabelEncoder()
    for col in list_le:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

def sel_variables(modelo,X,y,threshold): 
    """Recibe como parametros una lista de modelos, la base de datos escalada y codificada, treshold para seleccionar variables"""
    var_names_ac=np.array([])
    #for modelo in modelos:
    modelo.fit(X,y)
    sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
    var_names= modelo.feature_names_in_[sel.get_support()]
    
    var_names_ac=np.append(var_names_ac, var_names)
    var_names_ac=np.unique(var_names_ac)
        
        
    
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):
    "Recibe como parametros una lista de modelos, la metrica con la cual se quiere evaluar su desempeño, la base de datos escalada y codificada, la variable objetivo y el numero de folds para la validación cruzada."
    os = RandomOverSampler() # Usamos random oversampling para balancear la base de datos ya que la variable objetivo esta desbalanceada
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        pipeline = make_pipeline(os, modelo)
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["gard_boost","decision_tree","random_forest","reg_logistic"]
    return metric_modelos   




# FUNCION 6# Esta función prepara la base de datos con informacion nueva para hacer predicciones

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
    df_dummies= encode_data(df, list_label, list_dumies,list_ordinal)
    
    #Escalar variables
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    #X=X[var_names]
    
    return X
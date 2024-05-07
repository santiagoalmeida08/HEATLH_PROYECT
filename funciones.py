#En este script se encuentran las funciones que se utilizaran en los scripts de analisis exploratorio,
#modelos y despliegue

#1.Paquetes requeridos
#2.Funcion lectura de datos preprocesamiento
#3.Funcion prueba chi-cuadrado
#4.Funcion para codificar variables
#5.Funcion para medir el rendimiento de los modelos
#6. Funcion optimización de red neuronal
#7.Funcion de preparacion de datos

#1.Paquetes requeridos
import sqlite3 as sql
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib
from tensorflow import keras  
from sklearn import metrics  


#2.Funcion lectura de datos preprocesamiento
def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

#3.Funcion prueba chi-cuadrado
def chi_square_test(dataframe, target): 
    for col in dataframe.columns:
        print(col)
        print("----------")
        cross_tab = pd.crosstab(dataframe[col], dataframe[target], margins = False)
        stat, p, dof, expected = chi2_contingency(cross_tab)
        print(f"Chi-Square Statistic: {stat}, p-value: {p}")
        print("===========")
        
#4.Funcion para codificar variables
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

#5.Funcion para medir el rendimiento de los modelos
def modelos(list_mod, xtrain, ytrain, xtest, ytest):
    metrics_mod = pd.DataFrame()
    list_train = []
    list_test = []
    for modelo in list_mod:
        modelo.fit(xtrain,ytrain)
        y_pred = modelo.predict(xtest)
        score_train = metrics.accuracy_score(ytrain,modelo.predict(xtrain)) #metrica entrenamiento  
        score_test = metrics.accuracy_score(ytest,y_pred) #metrica test
        z= ['mod_tree','mod_rand','mod_log','mod_svm']
        modelos = pd.DataFrame(z)
        list_test.append(score_test)
        list_train.append(score_train)
        pdscores_train = pd.DataFrame(list_train)
        pdscroestest = pd.DataFrame(list_test)
        
        metrics_mod = pd.concat([modelos, pdscores_train, pdscroestest], axis=1)
        metrics_mod.columns = ['modelo','score_train','score_test']
    return metrics_mod

#6. Funcion optimización de red neuronal
def mejor_m(hp):
    opti = hp.Choice('OPTI', ['adam','rd2'])
    fa = hp.Choice('FA', ['relu','tanh','sigmoid'])
    
    ann2 = keras.models.Sequential([
    keras.layers.Dense(512, input_shape = (43,), activation=fa), 
    
    keras.layers.Dense(256,activation=fa),
    
    keras.layers.Dense(128,activation=fa),
    
    keras.layers.Dense(64,activation=fa),
   
    keras.layers.Dense(32, activation=fa),
 
    keras.layers.Dense(2, activation='sigmoid')
    ])

    if opti == 'adam':
        opti2 = keras.optimizers.Adam(learning_rate=0.001)
    else:
        opti2 = keras.optimizers.RMSprop(learning_rate=0.001) 
    
    ann2.compile(optimizer= opti2, loss= l, metrics= m)
    
    return ann2

#7.Funcion de preparacion de datos
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
    df_dummies=df_dummies.drop(['readmitted'],axis=1)
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    #X=X[var_names]
    
    return X
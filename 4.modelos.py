
#Paquetes necesarios.
import pandas as pd
import numpy as np
import sqlite3 as sql
import seaborn as sns
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import funciones as fn
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
import funciones as fn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# conexión a la base de datos

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()


cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()
# cargar tabla

df = pd.read_sql('SELECT *  from hr_full', conn)
df.info()
#df['edad'] = df['edad'].astype('object')
# Variables explicativas

x = df.drop(['readmitted'], axis=1)
y = df['readmitted']
y_mod = y.replace({'yes':1, 'no':0})

y_mod.value_counts()


df.info()


#############################################################################
#importar train_test_split


# DATAFRAME #

df

x = df.drop(['readmitted'], axis=1)
y = df['readmitted']
y_mod = y.replace({'yes':1, 'no':0})

#encoding variables

list_dumies = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) > 2]
x['edad'] = x['edad'].astype('object')
list_ordinal = ['edad']
list_label = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) == 2]

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

x_en = encode_data(x, list_label, list_dumies,list_ordinal)

#escalar
scaler = StandardScaler()
x_esc = scaler.fit_transform(x_en)

xtrain, xtest, ytrain, ytest = train_test_split(x_esc, y_mod, test_size=0.2, random_state=42)


#### Evaluacion varios modelos 

mod_tree= DecisionTreeClassifier()
mod_rand = RandomForestClassifier()
#mod_xgbosting = XGBClassifier()
mod_log = LogisticRegression( max_iter=1000)
mod_svm = svm.SVC()

list_mod = ([mod_tree,mod_rand,mod_log,mod_svm])

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

modelos(list_mod, xtrain, ytrain, xtest, ytest)

#MEJOR MODELO REGRESION LOGISTICA 

mod_reg = LogisticRegression( max_iter=1000)

mod_reg.fit(xtrain, ytrain)

y_pred = mod_reg.predict(xtest)

# Classification report

print(metrics.classification_report(ytest, y_pred))

# Matriz de confusion

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = metrics.confusion_matrix(ytest, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['no','yes'])
cmd.plot()
print(cm)


# Ajuste de hiperparametros

params = {
          'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
          'max_iter' : [100, 1000, 10000]}

from sklearn.model_selection import RandomizedSearchCV

h1 = RandomizedSearchCV(LogisticRegression(), params, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='recall')

h1.fit(xtrain,ytrain)


resultados = h1.cv_results_
h1.best_params_
pd_resultados=pd.DataFrame(resultados)
pd_resultados[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False) 

mod_hiper = h1.best_estimator_

### DESEMPEÑO 

y_pred2 = mod_hiper.predict(xtest)

rs_accuracy = metrics.accuracy_score(ytest,y_pred2)
rs_precision = metrics.precision_score(ytest,y_pred2)
rs_recall = metrics.recall_score(ytest,y_pred2)
rs_f1_score = metrics.f1_score(ytest,y_pred2)

print('accuracy score: %.2f' % rs_accuracy)
print('precision score: %.2f' % rs_precision)
print('recall score: %.2f' % rs_recall)
print('f1 score: %.2f' %  rs_f1_score)

#DEFINIR PROBABILIDAD PARA LA CLASIFICACIÓN
probabilidades = mod_hiper.predict_proba(xtest)[:, 1]#definir las probabilidades asociadas a la clase 1 (readmitido)

umbral = 0.35  # Puedes ajustar este valor según tus necesidades
predicciones_con_umbral = (probabilidades > umbral).astype(int)
print(classification_report(ytest, predicciones_con_umbral))
#matrix de confusión con las nuevas probabilidades
cm2 = metrics.confusion_matrix(ytest, predicciones_con_umbral)
cmd2 = ConfusionMatrixDisplay(cm, display_labels=['no','yes'])
cmd2.plot()
print(cm2)

######## RED NEURONAL ###################

#importar paquetes para redes neuronales
from tensorflow import keras    

ann1 = keras.models.Sequential([
    keras.layers.Dense(128,activation ='sigmoid'),
    keras.layers.Dense(64,activation ='relu'),
    keras.layers.Dense(32,activation ='tanh'),
    keras.layers.Dense(3,activation ='sigmoid') # en este casp utilizamos 3 neuronas en la capa de salida ya que la variable respuesta es categorica y cuenta con 3 categorias
                                                #ademas la funcion activation depende si trabajamos regresion o clasificacion (softmax)
])

l = keras.losses.SparseCategoricalCrossentropy()
m = keras.metrics.SparseCategoricalAccuracy()

ann1.compile(loss=l,metrics = m)

ann1.fit(xtrain,ytrain,epochs = 15,validation_data=(xtest,ytest))

#PRIMER DIAGNOSTICO -- UNDERFITTING 

#Diagnostico por grilla 
import keras_tuner as kt 

hp = kt.HyperParameters() #se definen hiperparametros

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


# ANALISIS MODELO GANADOR 

search_model = kt.RandomSearch(
    hypermodel= mejor_m ,
    hyperparameters= hp,
    objective= kt.Objective('val_sparse_categorical_accuracy',direction= 'max'),
    max_trials= 10,
    overwrite = True,
    project_name = 'rest_afin' 
)

search_model.search(xtrain,ytrain,epochs = 20,validation_data=(xtest,ytest)) 
search_model.results_summary()

##MAXIMO RENDIMIENTO, accuracy de 61%

#Selección mejor modelo
win_model = search_model.get_best_models(1)[0]
win_model.build()
win_model.summary()


#porque las predicciones no estan en 1 y 0 ???
#Predicciones
predicciones = win_model.predict(xtest)

#Transformación
y_pred = np.array(predicciones)[:,0]
y_test = np.array(ytest)

y_pred.shape
y_test.shape

#Metricas

# Exportar modelo ganador #
import joblib
var_names= df.drop(['readmitted']).columns
joblib.dump(predicciones_con_umbral, "salidas\\final.pkl") # Modelo ganador con afinamiento de hipermarametros 
joblib.dump(list_label, "salidas\\list_label.pkl") 
joblib.dump(list_dumies, "salidas\\list_dumies.pkl") 
joblib.dump(list_ordinal, "salidas\\list_ordinal.pkl")  
joblib.dump(var_names, "salidas\\var_names.pkl") ### para variables con que se entrena modelo
joblib.dump(scaler, "salidas\\scaler.pkl") ## para normalizar datos con MinMaxScaler

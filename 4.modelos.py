#En script se realiza el modelado de los datos utilizando algoritmos y tecnicas de machine learning

#1.Paquetes necesarios.
#2.Cargar datos
#3.División de variables predictoras y variable respuesta
#4.Encoding variables
#5.Escalado de variables
#6.Division train-test / 80-20
#7.Evaluacion varios modelos
#8.Seleccion del mejor modelo
#9.Ajuste de modelo con umbral de probabilidad
#10.Exportar modelo ganador
#11.Redes Neuronales

#1.Paquetes necesarios.
import pandas as pd
import numpy as np
import sqlite3 as sql
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
import funciones as fn
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import  ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
import keras_tuner as kt 
from tensorflow import keras    

#2.Cargar datos

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()


cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()

# cargar tabla

df = pd.read_sql('SELECT *  from hr_full', conn)
df.info()

#3.División de variables predictoras y variable respuesta
x = df.drop(['readmitted'], axis=1)
y = df['readmitted']
y_mod = y.replace({'yes':1, 'no':0})

#4.Encoding variables

list_dumies = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) > 2]
x['edad'] = x['edad'].astype('object')
list_ordinal = ['edad']
list_label = [x.columns[i] for i in range(len(x.columns)) if x[x.columns[i]].dtype == 'object' and len(x[x.columns[i]].unique()) == 2]


x_en = fn.encode_data(x, list_label, list_dumies,list_ordinal)

#5.Escalado de variables 
scaler = StandardScaler()
x_esc = scaler.fit_transform(x_en)

#6.Division train-test / 80-20
xtrain, xtest, ytrain, ytest = train_test_split(x_esc, y_mod, test_size=0.2, random_state=42)


#7.Evaluacion varios modelos 
mod_tree= DecisionTreeClassifier()
mod_rand = RandomForestClassifier()
mod_log = LogisticRegression( max_iter=1000)
mod_svm = svm.SVC()
list_mod = ([mod_tree,mod_rand,mod_log,mod_svm])
fn.modelos(list_mod, xtrain, ytrain, xtest, ytest)

#Best model: Logistic Regression
mod_reg = LogisticRegression( max_iter=1000)
mod_reg.fit(xtrain, ytrain)
y_pred = mod_reg.predict(xtest)

# Classification report & confusion matrix
print(metrics.classification_report(ytest, y_pred))
cm = metrics.confusion_matrix(ytest, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['no','yes'])
cmd.plot()

# Afinamiento de hiperparametros
params = {
          'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
          'max_iter' : [100, 1000, 10000]}

h1 = RandomizedSearchCV(LogisticRegression(), params, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='recall')
h1.fit(xtrain,ytrain)
resultados = h1.cv_results_
h1.best_params_
pd_resultados=pd.DataFrame(resultados)
pd_resultados[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False) 

#8.Seleccion del mejor modelo
mod_hiper = h1.best_estimator_ #modelo con ajuste de hiperparametros 

# Desempeño modelo con ajuste de hiperparametros

y_pred2 = mod_hiper.predict(xtest)

rs_accuracy = metrics.accuracy_score(ytest,y_pred2)
rs_precision = metrics.precision_score(ytest,y_pred2)
rs_recall = metrics.recall_score(ytest,y_pred2)
rs_f1_score = metrics.f1_score(ytest,y_pred2)

print('accuracy score: %.2f' % rs_accuracy)
print('precision score: %.2f' % rs_precision)
print('recall score: %.2f' % rs_recall)
print('f1 score: %.2f' %  rs_f1_score)

#9.Ajuste de modelo con umbral de probabilidad
probabilidades = mod_hiper.predict_proba(xtest)[:, 1]#definir las probabilidades asociadas a la clase 1 (readmitido)

umbral = 0.35  # Puedes ajustar este valor según tus necesidades
predicciones_con_umbral = (probabilidades > umbral).astype(int)
print(classification_report(ytest, predicciones_con_umbral))
#matrix de confusión con las nuevas probabilidades
cm2 = metrics.confusion_matrix(ytest, predicciones_con_umbral)
cmd2 = ConfusionMatrixDisplay(cm2, display_labels=['no','yes'])
cmd2.plot()


r_recall = metrics.recall_score(ytest,predicciones_con_umbral)
print('recall score: %.2f' % r_recall)


#10.Exportar modelo ganador 
joblib.dump(mod_hiper, "salidas\\final.pkl") # Modelo ganador con afinamiento de hipermarametros 
joblib.dump(list_label, "salidas\\list_label.pkl") 
joblib.dump(list_dumies, "salidas\\list_dumies.pkl") 
joblib.dump(list_ordinal, "salidas\\list_ordinal.pkl")  
joblib.dump(scaler, "salidas\\scaler.pkl") ## para normalizar datos con MinMaxScaler


#11.Redes Neuronales

#importar paquetes para redes neuronales

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
hp = kt.HyperParameters() #se definen hiperparametros

# ANALISIS MODELO GANADOR 
search_model = kt.RandomSearch(
    hypermodel= fn.mejor_m ,
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



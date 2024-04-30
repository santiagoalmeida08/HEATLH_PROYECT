
#Paquetes necesarios
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

"""
#### ----- #####
x=x.drop(['diag_1','diag_2','diag_3','change'], axis=1)
x=x[['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications']]

df['change'].value_counts()
# Encoding variables #

df1=x.copy()

df1.dtypes

#Dummies
#list_dummies =['medical_specialty','diag_1','diag_2','diag_3','glucose_test','A1Ctest']
list_dummies =['medical_specialty','glucose_test','A1Ctest']
#Ordinal - edad
list_ordinal = ['edad']  

#Label - change y diabetes med
list_label = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]


x_encoded = fn.encode_data(df1, list_label, list_dummies,list_ordinal)

#Escalado de variables 

scaler = StandardScaler()
Xesc = scaler.fit_transform(x) 
Xesc1 = pd.DataFrame(Xesc, columns = x.columns)

Xesc1.info()
#Seleccion e importancia de variables 

modelo_importance = LogisticRegression().fit(Xesc1,y_mod)
selector  = SelectFromModel(estimator=modelo_importance, prefit=True)
selector.fit(Xesc1,y_mod)
select_features = Xesc1.columns[selector.get_support()]
coef = modelo_importance.coef_[0] 
importance = pd.DataFrame(coef, index = Xesc1.columns, columns = ['coef']).sort_values(by='coef', ascending=False).head(5).reset_index()
sns.barplot(x='coef', y='index', data=importance,palette='viridis')


var_select = fn.sel_variables(LogisticRegression(),Xesc1,y_mod,threshold="2*mean")


# Desempeño de los modelos 

mod_rand= DecisionTreeClassifier(min_samples_leaf=2)
mod_rand = RandomForestClassifier(min_samples_leaf=2)
#mod_xgbosting = XGBClassifier()
mod_log = LogisticRegression()
modelos = list([mod_tree,mod_rand,mod_log])


mod_rand.fit(Xesc1, y_mod)
mod_rand.feature_importances_
mod_rand.feature_names_in_

y_pred=mod_rand.predict(Xesc1)

y_mod2=np.array(y_mod)

metrics.accuracy_score(y_mod2,y_pred)

scores=cross_val_score(mod_rand,Xesc1,y_mod, scoring='accuracy', cv=10)
pdscores=pd.DataFrame(scores)



def medir_modelos(modelos,scoring,X,y,cv):
    "Recibe como parametros una lista de modelos, la metrica con la cual se quiere evaluar su desempeño, la base de datos escalada y codificada, la variable objetivo y el numero de folds para la validación cruzada."
    os = RandomOverSampler() # Usamos random oversampling para balancear la base de datos ya que la variable objetivo esta desbalanceada
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        pipeline = make_pipeline(os, modelo)
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["decision_tree","random_forest",    "reg_logistic"]
    return metric_modelos   

mod = medir_modelos(modelos,"accuracy",Xesc,y_mod,3)


f1s= mod

f1s.columns=[ 'dt_sel', 'rf_sel', 'rl_Sel']
f1s.plot(kind='box') # Boxplot de f1 score para cada modelo con todas las variables y con las variables seleccionadas
f1s.mean()  # Media de rendimiendo para cada variable 
"""


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



######## RED NEURONAL ###################

#importar paquetes para redes neuronales
from tensorflow import keras    

ann1 = keras.models.Sequential([
    keras.layers.Dense(128,activation ='sigmoid'),
    keras.layers.Dense(64,activation ='sigmoid'),
    keras.layers.Dense(32,activation ='sigmoid'),
    keras.layers.Dense(3,activation ='softmax') # en este casp utilizamos 3 neuronas en la capa de salida ya que la variable respuesta es categorica y cuenta con 3 categorias
                                                #ademas la funcion activation depende si trabajamos regresion o clasificacion (softmax)
])

l = keras.losses.SparseCategoricalCrossentropy()
m = keras.metrics.SparseCategoricalAccuracy()

ann1.compile(loss=l,metrics = m)

ann1.fit(xtrain,ytrain,epochs = 5,validation_data=(xtest,ytest))
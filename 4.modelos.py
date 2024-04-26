
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

# conexión a la base de datos

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# cargar tabla

df = pd.read_sql('SELECT *  from hr_full', conn)
df.info()
df['edad'] = df['edad'].astype('object')
# Variables explicativas

x = df.drop(['readmitted'], axis=1)
y = df['readmitted']
y_mod = y.replace({'yes':1, 'no':0})

y_mod.value_counts()


df.info()

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
    
    metric_modelos.columns=["decision_tree","random_forest","reg_logistic"]
    return metric_modelos   

mod = medir_modelos(modelos,"accuracy",Xesc,y_mod,3)


f1s= mod

f1s.columns=[ 'dt_sel', 'rf_sel', 'rl_Sel']
f1s.plot(kind='box') # Boxplot de f1 score para cada modelo con todas las variables y con las variables seleccionadas
f1s.mean()  # Media de rendimiendo para cada variable 


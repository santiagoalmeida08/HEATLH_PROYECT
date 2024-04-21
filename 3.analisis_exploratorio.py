 
import pandas as pd
import sqlite3 as sql
import funciones as fn 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import OrdinalEncoder
#visualización
import plotly.express as px

# conexión a la base de datos

r = pd.read_csv('data//hospital_readmissions.csv')
r

conn = sql.connect('data//readmissions.db')
cur = conn.cursor()

# insertar el DataFrame en la base de datos
r.to_sql('hr', conn, if_exists='replace', index=False)

cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
cur.fetchall()

# cargar base de datos 

hr = pd.read_sql('SELECT * FROM tabla1', conn)

hr.dtypes

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        plt.figure(figsize=(12,6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_cols = [col for col in hr.columns if hr[col].dtypes == "O"]
for col in cat_cols:
    cat_summary(hr, col, plot=False)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)

        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_cols = [col for col in hr.columns if hr[col].dtypes != "O"]
num_cols.pop(0)

for col in num_cols:
    num_summary(hr, col, plot=False)


#Variable edad

# crear dataset
base = hr.groupby(['edad'])[['readmitted']].count().reset_index().sort_values('readmitted', ascending = False).rename(columns ={'readmitted':'count'})

# crear gráfica
fig = px.bar(base, x='departamento', y='count', barmode ='group', title ='<b>Departamento<b>')

# agregar detalles a la gráfica
fig.update_layout(
    xaxis_title = 'Edad',
    yaxis_title = 'Cantidad',
    template = 'simple_white',
    title_x = 0.5)

fig.show()

#--Analisis de variables categoricas

hr_cat = hr.iloc[:, 8:] #seleccionamos variables categoricas

for column in  hr_cat.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(x=column, data=hr_cat)
    plt.title(column)
    
# agrupar injury y musculoskeletal en variables de diagnostico
# Missing wn medical_specialty se refiere a que no se especifico la especialidad
# Agrupar medicina interna y emergencia en especialidades de medicina
    

# Prueba chi-cuadrado
from scipy.stats import chi2_contingency


def chi_square_test(dataframe, target): 
    for col in dataframe.columns:
        print(col)
        print("----------")
        cross_tab = pd.crosstab(dataframe[col], dataframe[target], margins = False)
        stat, p, dof, expected = chi2_contingency(cross_tab)
        print(f"Chi-Square Statistic: {stat}, p-value: {p}")
        print("===========")
        
chi_square_test(hr_cat, 'readmitted')


#Las variables referentes a als pruebas no representativas ; sin embargo es importante tener estos 
#datos en cuenta al momento de proponer estrategias y concluir
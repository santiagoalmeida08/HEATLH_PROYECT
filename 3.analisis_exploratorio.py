#Paquetes requeridos  
import pandas as pd
import sqlite3 as sql
import funciones as fn 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import OrdinalEncoder
#visualización
import plotly.express as px
#Prueba chi-cuadrado
from scipy.stats import chi2_contingency

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

hr = pd.read_sql('SELECT * FROM hrmin', conn)

hr.dtypes


#Resumen de variables categoricas 

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


#Resumen variables numericas
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [ 0.20, 0.50, 0.80, 0.959]
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


# Analisis Variables numericas 
hr.columns
hr_num = hr.iloc[:, 0:7] #seleccionamos variables numericas 


for column in hr_num.columns:

    #crear base
    base = hr.groupby([column])[['readmitted']].count().reset_index().sort_values(column, ascending = False)

    # crear gráfica
    fig = px.bar(base, x=column, y='readmitted', title =f'<b>Frecuencia de {column}<b>' ,text_auto = True)
    fig.update_traces(marker_color = 'aqua',textfont_size = 14, textangle = 0, textposition = "outside")

    # agregar detalles a la gráfica
    fig.update_layout(
        xaxis_title = column,
        yaxis_title = 'Frecuencia',
        template = 'simple_white',
        title_x = 0.5)


    fig.show()
    
for column in hr_num.columns:
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].hist(x = hr_num[column], bins=70)
    ax[0].set_title(f'Histograma de {column}')
    ax[1].boxplot(hr_num[column])
    ax[1].set_title(f'Boxplot de {column}')


"""
- Frecuencia de tiempo en hospital 
    Se observa que hay una menor cantidad de pacientes a medida que aumenta el tiempo 
    en el hospital, ademas se observa que hay gran numero de datos en cada valor por lo 
    cual la variable sigue siendo signficativa
    
- Frecuencia de numero de procedimientos de laboratorio
    Se puede observar que los datos tienen una distribución normal a pesar de que la mayor cantidad 
    de pacientes solos se les realiza un procedimiento de laboratorio 
    
- Frecuencia del numero de procedimientos
    La varible es representativa ya que hay una cantidad de datos representativa para 
    los diferentes numeros de procedimiento, tambien se puede ver que mayor numero de procedimientos
    disminuye el numero de pacientes
    
- Frecuencia de numero de medicamentos 
    Los valores se distribuyen de forma normal con un leve sesgo a la derecha, con poca frecuencia ya que 
    son pocos los pacientes que necesitan un gran numero de medicamentos
    
- Frecuencia de numero de visitas ambulatorias 
    Se observa que la gran mayoria de los datos estan en el valor de cero por lo cual esta variable 
    puede considerarse no representativa
    
- Frecuencia de n_inpatient 
    En cuanto a el numero de visitas el año anterior la mayoria de datos tienen un valor de cero, pero 
    aun asi esto no es suficiente para descartar esta variable 
    
- Frecuencia de n_emergency
    La variable de pacientes que estuvieron en emergencia el año anterior no es representativa segun el grafico
    a pesar de esto se tendra en cuenta para aplicar un modelo de seleccion de variables
    
"""


#Tratamiento de valores atipicos
# tratar variables n_medication,n_outpatient,n_inpatient,n_emergency

hr_at = hr.copy()

hr_at = hr_at[(hr_at['n_lab_procedures']<=100)] # 101 registros
hr_at = hr_at[(hr_at['n_medications']<=35)] #637 registros
hr_at = hr_at[hr_at['n_outpatient']<=6]
hr_at = hr_at[(hr_at['n_emergency']<=3)] #  854 registros
hr_at = hr_at[(hr_at['n_inpatient']<=5)] #  854 registros


hr_full = hr_at.copy()
hr_full['n_medications'].describe()
hr_full['n_inpatient'].describe()
hr_full['n_outpatient'].describe()
hr_full['n_emergency'].describe()

for column in hr_full.columns:
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].hist(x = hr_full[column], bins=70)
    ax[0].set_title(f'Histograma de {column}')
    ax[1].boxplot(hr_full[column])
    ax[1].set_title(f'Boxplot de {column}')
    


# Analisis de correlacion entre variables numericas #
correlation = hr_num.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""No hay una correlación fuerte entre las variables numericas, el valor mas alto lo las variables numero de medicamentos y el tiempo en el hospital, 
mientras mas tiempo esta un paciente en el hospital mayor es el numero de medicamentos a pesar de la clara relación
la correlación no supera un valor de 0,5 por lo cual ambas variables se seguiran teniendo en cuenta"""


# Analisis de variables categoricas #

hr_cat = hr.iloc[:, 7:] #seleccionamos variables categoricas

for column in  hr_cat.columns:
    if column != 'readmitted':
        plt.figure(figsize=(12,6))
        sns.countplot(x=column, data=hr_cat,palette='viridis',hue=column)
        plt.title(column)
        
c = hr_cat.loc[:, ['diag_1','diag_2','diag_3']]
c[(c['diag_1'] == 'circulatory') & (c['diag_2'] == 'circulatory')]

#Se observa que la cantidad de test no realizados es proporcional a la cantidad de pacientes diabeticos 
hr_cat['glucose_test'].value_counts()   
c['diag_1'].value_counts()

"""
-Diagnosticos : se observa que los diagnosticos que reciben los pacientes varian mucho en los reingresos que se tienen
    ademas de ello se observa la presencia de la categoria 'missing' en los diagnosticos con pocos registros, por lo cual se eliminara

-Tests : es una categoria que no es representativa ya que la mayoria de los pacientes no se les realiza la prueba

- change : indica si se cambio la medicación para la diabetes, se observa un leve equilibrio entre ambas categorias por lo cual se considera representativa

- diabetes_med :  observamos que a la mayoria de los pacientes se les prescribio una orden para diabetes

-edad :  Los pacientes tienen desde 40 años hasta los 100; la mayoria de los pacientes se encuentran en el rango de 70-80 años.

"""    

#ELiminar categoria Missing de diag_1,|diag_2,diag_3

hr_full = hr_full[(hr_full['diag_1'] != 'missing')]
hr_full = hr_full[(hr_full['diag_2'] != 'missing')]
hr_full = hr_full[(hr_full['diag_3'] != 'missing')]


#Analisis de variable obejtiivo
sns.countplot(x= 'readmitted', data = hr_cat)
plt.title('Readmitted')
plt.show()

# Prueba chi-cuadrado

fn.chi_square_test(hr_cat, 'readmitted')

"""Podemos observar que las variables que no tienen una relacion significativa con la variable objetivo son:
    glucose_test y A1C_test ya que el p-valor es mayor a 0.05, por lo cual se eliminara de la base de datos; respecto a las demas
    variables observamos que existe un tipo de relacion, sin embargo no podemos determinar la magnitud de dicha relacion; por lo cual nos apoyaremos
    del uso de un algoritmo para seleccionar las variables las importantes"""


#Las variables referentes a als pruebas no representativas ; sin embargo es importante tener estos 
#datos en cuenta al momento de proponer estrategias y concluir

hr_full.to_sql('hr_full', conn, if_exists='replace', index=False)
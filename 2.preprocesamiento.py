 
 # paquetes 
 
import pandas as pd
import sqlite3 as sql
import funciones as fn 
import matplotlib.pyplot as plt
import seaborn as sns  

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

hr = pd.read_sql('SELECT * FROM hr', conn)
hr.isnull().sum()

hr[['glucose_test']].value_counts()
hr[['A1Ctest']].value_counts()

# Preprocesamiento variables 9--17

s1 = pd.read_sql("""SELECT medical_specialty as especialidad, count(*) AS ingresos FROM hr
                    GROUP BY especialidad
                    ORDER BY ingresos DESC""", conn)
sns.barplot(x='ingresos', y='especialidad', data=s1, color='blue')

s2 = pd.read_sql("""SELECT diag_1 as diagnostico, count(*) AS pacientes FROM hr
                    GROUP BY diagnostico
                    ORDER BY pacientes DESC""", conn)
sns.barplot(x='pacientes', y='diagnostico', data=s2, color='blue')


s3 = pd.read_sql("""SELECT diag_2 as diagnostico, count(*) AS conteo FROM hr
                    GROUP BY diagnostico
                    ORDER BY conteo DESC""", conn)


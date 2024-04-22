import sqlite3 as sql
import pandas as pd
from scipy.stats import chi2_contingency


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
import numpy as np
import pandas as pd

baseDeDados = pd.read_csv('svbr.csv', delimiter=';')
x = baseDeDados.iloc[:,:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median') #substitui os valores nan usando a estratédgia median
imputer = imputer.fit(x[:,1:3]) #seleciona as colunas 1 e 2 como base para os calculos dos novos valores (obs ao definir deve-se somar um ao indice da ultima coluna)
x = imputer.transform(x[:,1:3]).astype(str) #transforma os dados em string para ser possível mostrá-los na tela
x = np.insert(x, 0, baseDeDados.iloc[:,0].values, axis=1) #adiciona novamente a coluna de nomes

print(x)
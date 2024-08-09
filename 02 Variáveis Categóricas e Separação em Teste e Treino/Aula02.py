import numpy as np
import pandas as pd

baseDeDados = pd.read_csv('admission.csv', delimiter=';')
x = baseDeDados.iloc[:,:-1].values # contém todas as colunas, exceto a última (as variáveis independentes)
y = baseDeDados.iloc[:,-1].values # y contém apenas a última coluna (as variáveis dependentes)

from sklearn.impute import SimpleImputer # Importa SimpleImputer para substituir valores ausentes.
imputer = SimpleImputer(missing_values=np.nan, strategy='median') # imputer é configurado para substituir valores ausentes pela mediana.
imputer = imputer.fit_transform(x[:,1:]) # Aplica o imputer às colunas de x, exceto a primeira.

from sklearn.preprocessing import LabelEncoder #importa o módulo responsável por converter strings em numeros inteiros
labelencoder_x = LabelEncoder() #cria a instancia labelencoder_x
x[:, 0] = labelencoder_x.fit_transform(x[:, 0]) #Substitui a primeira coluna pelos valores transformados

x = x[:,1:] # Descarta a coluna 0, a primeira coluna de x (que já foi transformada).
d = pd.get_dummies(x[:,0], dtype=int) # Cria variáveis dummy (one hot encoding) para a nova primeira coluna.
x = np.insert(x, 0, d.values, axis=1) #insere em x as variaveis dummy, ou seja, as colunas novas em binario

from sklearn.model_selection import train_test_split #Importa train_test_split para dividir os dados.
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2) #Divide x e y em conjuntos de treino e teste, com 20% dos dados reservados para teste.

print('Computando Normalização...')
from sklearn.preprocessing import StandardScaler # O StandardScaler é usado para padronizar os recursos
scaleX = StandardScaler() # cria uma instância de StandardScaler chamada scaleX
xTrain = scaleX.fit_transform(xTrain) # Ajuste e transformação dos dados de treino
xTest = scaleX.fit_transform(xTest) # Transformação dos dados de teste
print(xTrain)
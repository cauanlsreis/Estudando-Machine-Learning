import numpy as np
import pandas as pd

def loadDataset(filename):
    print("Carregando a base de dados...")
    baseDeDados = pd.read_csv(filename, delimiter=';')
    x = baseDeDados.iloc[:,:-1].values
    y = baseDeDados.iloc[:,-1].values
    print('Ok!')
    return x,y

def fillMissingData(x):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x[:,1:] = imputer.fit_transform(x[:,1:])
    return x 

def computeCategorization(x):
    from sklearn.preprocessing import LabelEncoder
    labelencoder_x = LabelEncoder()
    x[:,0] = labelencoder_x.fit_transform(x[:, 0])

    #one hot encoding
    D = pd.get_dummies(x[:,0])
    x = x[:,1:]
    x = np.insert(x, 0, D.values, axis=1)
    return x

def splitTrainTestSets(x,y, testSize):
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = testSize)
    return xTrain, xTest, yTrain, yTest

def computeScaling(train, test):
    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    test = scaleX.transform(test)
    return train, test

def computeLinearRegressionModel(xTrain, yTrain, xTest, yTest):
    from sklearn.linear_model import LinearRegression 
    regressor = LinearRegression()
    regressor.fit(xTrain, yTrain)
    #yPred = regressor.predict(xTest)

    #Gerar gráfico
    import matplotlib.pyplot as plt
    plt.scatter(xTest[:,-1], yTest, color="red")
    plt.plot(xTest[:,-1], regressor.predict(xTest), color="blue")
    plt.title("Inscritos x Visualizações (SVBR)")
    plt.xlabel("Total de Inscritos")
    plt.ylabel("Total Visualizações")
    plt.show()

def runLinearRegressionExample(filename):
    x,y = loadDataset(filename)
    x = fillMissingData(x)
    x = computeCategorization(x)
    xTrain, xTest, yTrain, yTest = splitTrainTestSets(x,y,0.8)
    computeLinearRegressionModel(xTrain, yTrain, xTest, yTest)
    
if __name__ == "__main__":
    runLinearRegressionExample("svbr.csv")
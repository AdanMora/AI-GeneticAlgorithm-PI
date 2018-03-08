import numpy as np

setosa, versicolor, virginica = range(3)
sepal_length_i, sepal_width_i, petal_length_i, petal_width_i = range(4)
mini, maxi = range(1)

# Valores obtenidos a partir del set de datos iris
setosa_values = [[4.3,5.8],[2.3,4.4],[1.0,1.9],[0.1,0.6]]
versicolor_values = [[4.9,7.0],[2.0,3.4],[3.0,5.1],[1.0,1.8]]
virginica_values = [[4.9,7.9],[2.2,3.8],[4.5,6.9],[1.4,2.5]]
iris_values = [setosa_values,versicolor_values,virginica_values]

def generateRoW(flower):
    values = iris_values[flower]
    sepal_length = np.random.uniform(low=values[sepal_length_i][mini],high=values[sepal_length_i][maxi],size=(1,))
    sepal_width = np.random.uniform(low=values[sepal_width_i][mini],high=values[sepal_width_i][maxi],size=(1,))
    petal_length = np.random.uniform(low=values[petal_length_i][mini],high=values[petal_length_i][maxi],size=(1,))
    petal_width = np.random.uniform(low=values[petal_width_i][mini],high=values[petal_width_i][maxi],size=(1,))

    return np.concatenate((sepal_length,sepal_width,petal_length,petal_width))

def generateW():
    row1 = generateRoW(setosa)
    row2 = generateRoW(versicolor)
    row3 = generateRoW(virginica)

    return np.array([row1,row2,row3])

def generateWs(n):
    ws = []
    for i in range(n):
        ws.append(generateW())
    return np.array(ws)

#print(len(generateWs(150)))
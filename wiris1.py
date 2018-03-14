import numpy as np

tipo = np.dtype([("w",np.float32,(3,4)),("E",np.float32),("L",np.float32),("E_i",np.float32,(3,))])

def generateRoW():
    #values = iris_values[flower]
    minimo = 0.0
    maximo = 10.0
    sepal_length = np.random.uniform(low=minimo,high=maximo,size=(1,))
    sepal_width = np.random.uniform(low=minimo,high=maximo,size=(1,))
    petal_length = np.random.uniform(low=minimo,high=maximo,size=(1,))
    petal_width = np.random.uniform(low=minimo,high=maximo,size=(1,))

    return np.concatenate((sepal_length,sepal_width,petal_length,petal_width))

def generateW():
    row1 = generateRoW()
    row2 = generateRoW()
    row3 = generateRoW()

    return np.array([row1,row2,row3])

def generateWs(n):
    ws = []
    for i in range(n):
        ws.append((generateW(),0,0,[0,0,0]))
    return np.array(ws,dtype=tipo)

def generateIndividuos(ws):
    return np.array(ws,dtype=tipo)

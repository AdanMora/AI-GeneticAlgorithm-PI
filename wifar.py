import numpy as np

mini = 0
maxi = 255

tipo = np.dtype([("w",np.float32,(4,1024)),("E",np.float32),("L",np.float32),("E_i",np.float32,(4,)), ("R",bool)])


def generateRoW():
    return np.random.uniform(low=mini,high=maxi,size=(1024,))
    #return 

def generateW(classes):
    w = []
    for i in range(0,classes):
        w.append(generateRoW())
    return np.array(w)

def generateWs(n,classes):
    ws = []
    for i in range(n):
        ws.append((generateW(classes),0,0,[0,0,0,0],False))
    return np.array(ws,dtype=tipo)

def generateIndividuos(ws):
    return np.array(ws,dtype=tipo)

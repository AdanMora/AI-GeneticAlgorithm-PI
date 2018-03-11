import numpy as np

mini = 0
maxi = 255

# self.type = np.dtype([("image",np.float64), ("label", np.str_,16)])
tipo = np.dtype([("w",np.float32,(4,1024)),("E",np.float32),("L",np.float32),("Li",np.float32,(4,))])


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
        ws.append((generateW(classes),0,0,[0,0,0,0]))
    return np.array(ws,dtype=tipo)

def generateIndividuos(ws):
    return np.array(ws,dtype=tipo)

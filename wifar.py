import numpy as np

mini = 0
maxi = 255

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
        ws.append(generateW(classes))
    return np.array(ws)

a = generateWs(4,3)
#print(a[0])
print(len(a))
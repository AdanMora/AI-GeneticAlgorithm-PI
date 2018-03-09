import numpy as np
import pickle
from sklearn.datasets import load_iris
import wifar
import wiris



#-------------------------------------------------------------------------------------------------------#

class Iris_Enum:
    """Clase que funciona como enumeración clásica de C para manejar fácilmente el nombre de la clase en Iris"""
    Setosa, Versicolor, Virginica = range(3)

class CIFAR_Enum:
    """Clase que funciona como enumeración clásica de C para manejar fácilmente el nombre de la clase en CIFAR-10"""
    Airplane, Automobile, Bird, Cat = range(4)

class GenAlgorithm:
    X = None
    Y = None
    W = None
    W_s = []
    hyperparameters = []
    actual_hyper = {}
    tipoD = None

    def addHyperparameter(self, tipo, cantWs, gen, cant_menosAptos, minAceptacion):
        """Recibe tipo, cantidad de W's, cant de generaciones, cant de cruces con menos aptos,
           mínimo de aceptación para terminar."""
        nHyperParameter = {}
        nHyperParameter["tipo"] = tipo        
        nHyperParameter["cant_W's"] = cantWs
        nHyperParameter["cant_Gen"] = gen
        nHyperParameter["cant_MenosAptos"] = cant_menosAptos
        nHyperParameter["min_Aceptacion"] = minAceptacion

        self.hyperparameters.append(nHyperParameter)

    def hingeLoss_i(self, w, x, y):
        resultF_WX = np.dot(w,x)
        individual_Loss = 0
        for i in range(resultF_WX.shape[0]):
            if (i != y):
                individual_Loss += max(0, resultF_WX[i] - resultF_WX[y] + 1)
                
        return individual_Loss
    
    def hingeLoss_W(self, w):
        lossClases = [(0,0)] * w.shape[0]
        lossTotal = 0
        N = self.X.shape[0]
        for i in range(N):
            L_i = self.hingeLoss_i(w, self.X[i], self.Y[i])
            lossTotal += L_i
            lossClases[self.Y[i]][0] += L_i  # Suma del Loss para la clase sub_i
            lossClases[self.Y[i]][1] += 1    # Contador de X's de esa clase

        for i in range(len(lossClases)):
            lossClases[i][0] /= lossClases[i][1]
        
        
        return lossTotal / N

    def genW_s(self):
        # tipo 0 -> iris
        if not (actual_hyper["tipo"]):
            self.W_s = wiris.generateWs(actual_hyper["cant_W's"])
            self.tipoD = wiris.tipo
        else:
            self.W_s = wifar.generateWs(actual_hyper["cant_W's"],4)
            self.tipoD = wifar.tipo

    def mkCruce(self, W1, W2):
        N = W1["w"].shape[0]
        nW = [0]*N
        for i in range(N):
            if (W1["Li"][i] > W2["Li"][i]):
                nW[i] = W1["w"][i]
            else:
                nW[i] = W2["w"][i]
        return np.array((nW,0,[0]*N), dtype = self.tipoD)

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y

        # Generar W's

        self.genW_s(actual_hyper["tipo"])

        # Calcular Loss, general y por clase
        ## ...
        for i in range(actual_hyper["cant_Gen"]):
            newW_s = []
            
            self.hingeLoss_W()
            self.W_s = np.sort(self.W_s, order="L")

            masAptos = self.W_s[int(:self.W_s.shape[0]*0.5)]

            if (masAptos[0]["L"] <= actual_hyper["min_Aceptacion"]) or (masAptos.shape[0] < 3):
                self.W = masAptos[0]["w"]
                break

            menosAptos = self.W_s[int(self.W_s.shape[0]*actual_hyper["cant_MenosAptos"]:)]

            N_masAptos = masAptos.shape[0]
            N_menosAptos = menosAptos.shape[0]

            diferencia = N_masAptos - N_menosAptos
            
            for i in range(N_masAptos):
                newW_s.append(mkCruce(masAptos[i], masAptos[i + 1]))
                
            for i in range(diferencia, diferencia + N_menosAptos):
                newW_s.append(mkCruce(masAptos[i], menosAptos[i - (N_menosAptos + 1)]))


            self.W_s = np.array(newW_s)

        
                
                
        
        #self.
        # Selección de W's para cruce.

        # Hacer cruce

        # Hacer mutación

        # Una vez obtenida la nueva generación de W's, empezar el proceso de nuevo.

    def classify(self, X):
        """Función que implementa el proceso de clasificación una vez obtenida la W definiva, recibe el vector con los datos de prueba.
           Retorna un vector con las respuetas predecidas correspondientes a cada dato de prueba"""
        predict_Y = []
        for i in range(X.shape[0]):
            predict_Y.append(np.argmax(np.dot(self.W,X[i])))
        return predict_Y

#-------------------------------------------------------------------------------------------------------#

def unpickle(file):
    """Función para descomprimir los archivos de imágenes de CIFAR-10"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def RGBtoGrayscale(img):
    lenV = img.size // 3
    resultado = []
    R = img[:1024]
    img = img[1024:]
    G = img[:1024]
    B = img[1024:]
    for i in range(lenV):
        resultado.append((R[0]*0.299) + (G[0]*0.587) + (B[0]*0.114))
        R = R[1:]
        G = G[1:]
        B = B[1:]
    return np.array(resultado, dtype= np.float32)

def plotGrayImage(img):
    from pylab import imshow, show, get_cmap

    ind = 0
    m = []
    print(img)
    for i in range(32):
        f = []
        for j in range(32):
            f.append(img[ind])
            ind += 1
        m.append(f)

    m = np.array(m)
    imshow(m, cmap="gray")
    show()

def main():

    genAlg = GenAlgorithm

    #----- Iris -----#

    iris = load_iris()

    X = iris['data']
    Y = iris['target']

    testX = X[50:55]
    X = np.concatenate((X[:50],X[55:]))
    testX = np.concatenate((testX,X[:5]))
    X = X[5:]
    testX = np.concatenate((testX,X[X.shape[0] - 5:]))
    X = X[:X.shape[0] - 5]
    

    testY = Y[50:55]
    Y = np.concatenate((Y[:50],Y[55:]))
    testY = np.concatenate((testY,Y[:5]))
    Y = Y[5:]
    testY = np.concatenate((testY,Y[Y.size - 5:]))
    Y = Y[:Y.size - 5]

    genAlg.addHyperparameter(0, 200, 10, 0.1, 2)
    genAlg.addHyperparameter(0, 100, 10, 0.05, 2)

    genAlg.train(X, Y)

    genAlg.classify(testX)

    #----- CIFAR-10 -----#

    """train = unpickle("train.p")
    test = unpickle("test.p")


    X = train['data']
    X = np.apply_along_axis(RGBtoGrayscale, 1, X)
    Y = train['labels']

    testX = test['data']
    testX = np.apply_along_axis(RGBtoGrayscale, 1, X)
    testY = test['labels']

    genAlg.addHyperparameter(1, 4000, 10, 0.1, 2)
    genAlg.addHyperparameter("1, 2000, 10, 0.05, 2)

    genAlg.train(X, Y)

    genAlg.classify(testX)"""

main()

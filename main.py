import numpy as np
import pickle

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

    def addHyperparameter(name, cantWs, gen, cant_menosAptos, minAceptacion):
        """Recibe nombre, cantidad de W's, cant de generaciones, cant de cruces con menos aptos,
           mínimo de aceptación para terminar."""
        nHyperParameter = {}
        nHyperParameter["name"] = name
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

    def genW_s(self, tipo):
        if(tipo == 0):
            self.W_s = None
        else:
            self.W_s = None

    def mkCruce(W1, W2):

        N = W1.shape[0]
        nW = [0]*N
        for i in (N):
            nw[i] = W1[i]


        return nW

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y

        # Generar W's

        # Calcular Loss, general y por clase

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


    #----- CIFAR-10 -----#

    train = unpickle("train.p")
    test = unpickle("test.p")


    X = train['data']
    Y = train['labels']

    testX = test['data']
    testY = test['labels']

    #genAlg.train(X,Y)

    #plotGrayImage(RGBtoGrayscale(X[123]))

    #genAlg.classify(test)

main()

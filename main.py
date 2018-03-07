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

    def hingeLoss_i(self, w, x, y):
        resultF_WX = np.dot(w,x)
        individual_Loss = 0
        for i in range(resultF_WX.shape[0]):
            if (i != y):
                individual_Loss += max(0, resultF_WX[i] - resultF_WX[y] + 1)
                
        return individual_Loss
    
    def hingeLoss_W(self, w):
        lossTotal = 0
        N = self.X.shape[0]
        for i in range(N):
            lossTotal = hingeLoss_i(w, self.X[i], self.Y[i])
            
        return lossTotal / N

    def genW_s(self, tipo):
        if(tipo == 0):
            self.W_s = None
        else:
            self.W_s = None

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y

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

    #----- Iris -----#


    #----- CIFAR-10 -----#

    train = unpickle("train.p")
    test = unpickle("test.p")


    X = train['data']
    Y = train['labels']

    testX = test['data']
    testY = test['labels']

    genAlg = GenAlgorithm
    #genAlg.train(X,Y)

    #plotGrayImage(RGBtoGrayscale(X[123]))

    #genAlg.classify

main()

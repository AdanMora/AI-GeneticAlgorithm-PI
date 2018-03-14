import numpy as np
import pickle
from sklearn.datasets import load_iris
import wifar
import wiris1 as wiris



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
    EficienciaW_Apto = []

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
        #print("\n#########################")
        lossTotal = 0
        N = self.X.shape[0]
        for i in range(N):
            L_i = self.hingeLoss_i(w["w"], self.X[i], self.Y[i])
            lossTotal += L_i
            
        w["L"] = lossTotal / N
        #print(w["w"])
        #print(w["L"])
        #print(w["Li"])
        #print("\n#########################")

    def genW_s(self):
        # tipo 0 -> iris
        if not (self.actual_hyper["tipo"]):
            self.W_s = wiris.generateWs(self.actual_hyper["cant_W's"])
            self.tipoD = wiris.tipo
        else:
            self.W_s = wifar.generateWs(self.actual_hyper["cant_W's"],4)
            self.tipoD = wifar.tipo

    def mkCruce(self, W1, W2):
        N = W1["w"].shape[0]
        nW = [0]*N
        for i in range(N):
            nW[i] = np.concatenate((W1["w"][i][:N//2],W2["w"][i][N//2:]))
        return np.array((nW,0,0,[0]*N), dtype = self.tipoD)

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y


        # Generar W's

        self.genW_s()

        # Calcular Loss, general y por clase
        ## ...
        count = 0
        for i in range(self.actual_hyper["cant_Gen"]):
            count += 1
            newW_s = []

            for w in range(self.W_s.shape[0]):
                self.hingeLoss_W(self.W_s[w])
                predicted_Y = self.classifyTrain(self.X, self.W_s[w]["w"])
                self.W_s[w]["E"] = (np.sum(np.equal(predicted_Y, self.Y)) / len(self.Y))
                
                for j in range(predicted_Y.size):
                    if(predicted_Y[j] == self.Y[j]):
                        self.W_s[w]["E_i"][predicted_Y[j]] += 1

                for k in range(self.W_s[w]["E_i"].size):
                    self.W_s[w]["E_i"][k] /= (self.X.shape[0] // self.W_s[w]["E_i"].size)
                
            self.W_s = np.sort(self.W_s, order="E")

            self.W_s = self.W_s[::-1]

            print(self.W_s.shape[0])

            masAptos = self.W_s[:int(self.W_s.shape[0]*0.5)]

            if (self.W_s[0]["E"] >= self.actual_hyper["min_Aceptacion"]) or (self.W_s.shape[0] < 10):
                break

            indMenosAptos = self.W_s.shape[0] - int(self.W_s.shape[0]*self.actual_hyper["cant_MenosAptos"])
            menosAptos = self.W_s[indMenosAptos:]

            N_masAptos = masAptos.shape[0]
            N_menosAptos = menosAptos.shape[0]

            diferencia = N_masAptos - N_menosAptos

            cruce1 = masAptos[:diferencia]
            cruce2 = masAptos[diferencia:]

            
            for i in range(cruce1.shape[0] - 1):
                newW_s.append(self.mkCruce(cruce1[i], cruce1[i + 1]))
                
            for i in range(N_menosAptos):
                newW_s.append(self.mkCruce(cruce2[i], menosAptos[i]))


            self.W_s = np.array(newW_s)
            
        self.W = masAptos[0]

    def classifyTrain(self, X, W):
        """Función que implementa el proceso de clasificación una vez obtenida la W definiva, recibe el vector con los datos de prueba.
           Retorna un vector con las respuetas predecidas correspondientes a cada dato de prueba"""
        predict_Y = []
        for i in range(X.shape[0]):
            predict_Y.append(np.argmax(np.dot(W,X[i])))

        return np.array(predict_Y)

    def classify(self, X):
        """Función que implementa el proceso de clasificación una vez obtenida la W definiva, recibe el vector con los datos de prueba.
           Retorna un vector con las respuetas predecidas correspondientes a cada dato de prueba"""
        predict_Y = []
        for i in range(X.shape[0]):
            predict_Y.append(np.argmax(np.dot(self.W["w"],X[i])))

        return np.array(predict_Y)

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

def main(prueba):

    genAlg = GenAlgorithm()

    #----- Iris -----#

    if(prueba):

        iris = load_iris()

        X = iris['data']
        Y = iris['target']

        genAlg.addHyperparameter(0, 1000, 10, 0.1, 0.85)
        genAlg.addHyperparameter(0, 100, 10, 0.05, 2.0)

        genAlg.actual_hyper = genAlg.hyperparameters[0]

        genAlg.train(X, Y)

        predict_Y = genAlg.classify(X)

        print("W: ",genAlg.W["w"])
        print("Eficiencia: ", genAlg.W["E"])
        print("Loss: ", genAlg.W["L"])
        print("Eficiencia i: ", genAlg.W["E_i"])

        
        print(genAlg.W["E_i"][0]*50)
        
        print(genAlg.W["E_i"][1]*50)
        
        print(genAlg.W["E_i"][2]*50)

        print(np.array(predict_Y))
        print(Y)
    

    #----- CIFAR-10 -----#

    else: 
        train = unpickle("train.p")
        test = unpickle("test.p")


        X = train['data']
        testX = test['data']
        nX = []
        ntX = []
        for i in range(X.shape[0]):
            nX.append(RGBtoGrayscale(X[i]))
            if(i < testX.shape[0]):
                ntX.append(RGBtoGrayscale(testX[i]))

        X = np.array(nX)
        testX = np.array(ntX)

        Y = train['labels']
        testY = test['labels']

        print("Gray")

        genAlg.addHyperparameter(1, 500, 20, 0.1, 0.8)
        genAlg.addHyperparameter(1, 2000, 10, 0.05, 0.8)

        genAlg.actual_hyper = genAlg.hyperparameters[0]

        genAlg.train(X, Y)

        predict_Y = genAlg.classify(testX)

        print(genAlg.W["w"])
        print(genAlg.W["E"])
        print(genAlg.W["L"])
        print(genAlg.W["E_i"])

        print(np.array(predict_Y))
        print(testY)

        plotGrayImage(genAlg.W["w"])

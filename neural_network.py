import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error,log_loss, accuracy_score
import pandas as pd


def config_dataset(dataset,y,sep=',',test_size=0.3,scaler=StandardScaler):
    df = pd.read_csv(dataset,sep=sep)
    X = df.drop(columns=[y])
    y_label = df[y].values.reshape(X.shape[0], 1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=test_size, random_state=2)
   
    sc = scaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.fit_transform(Xtest)
    
    return Xtrain, Xtest, ytrain, ytest

class NeuralNetwork():  # 2 hidden layers neural network

    def __init__(self, training_inputs, activation='sigmoid', iterations = 50000):
        np.random.seed(1)
        self.lr = 0.02
        self.iter = iterations
        self.weights = np.random.rand(training_inputs.shape[1], 8)
        self.weights2 = np.random.rand(8, 4)
        self.weights3 = np.random.rand(4, 1)
        self.bias = np.random.rand()
        self.bias2 = np.random.rand()
        self.bias3 = np.random.rand()
        self.output = 0
        self.cost = []
        self.i = []
        
    def softmax(self,x):
        pass
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def Relu(self, x):
        return np.maximum(0, x)

    def dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def think(self, x):
        self.z1 = np.dot(x, self.weights) + self.bias
        self.a1 = self.sigmoid(self.z1) 
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.weights3) + self.bias3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def train(self, training_inputs, training_outputs):
        training_inputs = np.array(training_inputs)

        for i in range(self.iter):
            
            self.i.append(i)
            output = self.think(training_inputs)
            self.output = output

            cost = mean_squared_error(training_outputs,self.output)
            #cost = log_loss(training_outputs,a3)
            self.cost.append(cost)

            dLoss_Yh =  - 2 * (training_outputs - self.output)
            #dLoss_Yh =  - (training_outputs - self.output) / self.output * (1 - self.output)
     
            dYh_Z3 = self.sigmoid_derivative(self.z3)
            dZ3_A2 = self.weights3
            dLoss_Z3 = dLoss_Yh * dYh_Z3

            dZ3_W3 = self.a2.T
            dLoss_W3 = np.dot(dZ3_W3, dLoss_Z3)
            adjustmentw3 = self.lr * dLoss_W3

            dZ3_B3 = 1
            dLoss_B3 = np.sum(dLoss_Z3, axis=0)
            adjustmentb3 = self.lr * dLoss_B3

            # 3
            # ------------------------------------------------------#

            dLoss_A2 = np.dot(dLoss_Z3, self.weights3.T)

            dA2_Z2 = self.sigmoid_derivative(self.z2)
            dLoss_Z2 = dLoss_A2 * dA2_Z2

            dZ2_W2 = self.a1.T
            dLoss_W2 = np.dot(dZ2_W2, dLoss_Z2)
            adjustmentw2 = self.lr * dLoss_W2

            dZ2_B2 = 1
            dLoss_B2 = np.sum(dLoss_Z2, axis=0)
            adjustmentb2 = self.lr * dLoss_B2

            # 2
            # -------------------------------------------
            # 1

            dLoss_A1 = np.dot(dLoss_Z2, self.weights2.T)

            dA1_Z1 = self.sigmoid_derivative(self.z1)
            dLoss_Z1 = dLoss_A1 * dA1_Z1

            dZ1_W1 = training_inputs.T
            dLoss_W1 = np.dot(dZ1_W1, dLoss_Z1)
            adjustmentw1 = self.lr * dLoss_W1

            dZ1_B1 = 1
            dLoss_B1 = np.sum(dLoss_Z1, axis=0)
            adjustmentb1 = self.lr * dLoss_B1

            self.weights -= adjustmentw1
            self.bias -= adjustmentb1
            self.weights2 -= adjustmentw2
            self.bias2 -= adjustmentb2
            self.weights3 -= adjustmentw3
            self.bias3 -= adjustmentb3

            if i%5000 == 0:
                print('After 5000 iterations')
                print('Cost: ', cost)
            #print(self.output[0])
        

    def test(self, test, y):
        #output = self.think(inputs)
        trues = 0
        falses = 0
        for i,n in enumerate(test):
    
            if n > 0.5:
                test[i] = 1
                
            if n < 0.5:
                test[i] = 0

            if test[i] == ytest[i]:
                print(test[i],ytest[i],'CORRECT')
                trues+=1
            else:
                print(test[i],ytest[i],'ERROR')
                falses+=1

        accuracy = trues / (trues + falses)

        return accuracy


Xtrain, Xtest, ytrain, ytest = config_dataset('divorce.csv', 'Class',test_size=0.33)

nn = NeuralNetwork(Xtrain, iterations=50_000)
nn.train(Xtrain, ytrain)
pred = nn.think(Xtest)

accuracy = nn.test(pred,ytest)
print(f"\nAccuracy = {accuracy}")

precision = precision_score(ytest, pred)
print(f"Precision = {precision}")

recall = recall_score(ytest, pred)
print(f"Recall = {recall}")

f1 = f1_score(ytest, pred)
print(f"F1 score = {f1}")
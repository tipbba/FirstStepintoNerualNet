import numpy as np
import scipy.special as special

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, widths : list[int], learningRate : float) -> None:

        self.Widths  = widths
        self.Depth   = len(widths)
        self.lr      = learningRate
        self.weights = []
        self.inputs  = []
        self.layers  = []

        if self.Depth <= 1:
            raise ValueError("depth must be a integer larger then 1, please check and retry") 
        if learningRate <=0:
            raise ValueError("learningRate must be a postive float, please check and retry")
        

        #Create proper weight-matrices between consecutive layers
        #ToDo:change this into xaiver initialization in future
        for i in range(self.Depth - 1):
            self.weights.append(np.random.normal(0.0, pow(self.Widths[i+1], -0.5), 
                                                 size=(self.Widths[i+1], self.Widths[i])))
            # print(self.weights[i])
    #ToDo:extend this to support more activate function
    def activate(self, value) -> float:
        return special.expit(value)
    #难点在需要保存上一次训练每层的输出，从而给反向传播使用，使用一个list，每个元素都是一个numpy array？这么写还是有问题，因为下一轮计算的结果会直接接到后面
    def forward_prop(self, input):
        if len(self.layers) != 0:
            self.layers.clear()
        temp = np.array(input, ndmin=2).T
        self.layers.append(temp)
        for weight in self.weights: 
            temp = self.activate(weight.dot(temp))
            self.layers.append(temp)
        return self.layers[-1] #the last element should be output of forward propaganda 


    def backward_prop(self, outputError):
        error = outputError
        nextError  = None
        backwardIndex = -1
        for weight in reversed(self.weights):
            nextError = np.dot(weight.T, error)
            weight += self.lr * np.dot((error * self.layers[backwardIndex]) * (1.0 - self.layers[backwardIndex]), 
                                       np.transpose(self.layers[backwardIndex - 1])) 
            error = nextError
            backwardIndex -= 1

            # 问题在于如何更新重构误差


    def query(self, testData):
        # self.load_data(testingSet)
        # values = self.inputs[0].split(',')
        # print(values[0])
        input  = (np.asfarray(testData) / 255.0 * 0.99) + 0.01

        return self.forward_prop(input)

    def train(self, trainingSet : str):
        self.load_data(trainingSet)
        for data in self.inputs:
            values = data.split(',')
            input  = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            output = self.forward_prop(input)            
            targets = np.zeros(self.Widths[-1]) + 0.01
            targets[int(values[0])] = 0.99
            targets = np.array(targets, ndmin=2).T
            
            outputError = np.array(targets - output)
            self.backward_prop(outputError)


    def load_data(self,filePath : str):
        with open(filePath,'r') as data:
            self.inputs = data.readlines()


    def get_accuracy(self, testDataPath: str) -> float:
        scoreCard = []

        self.load_data(testDataPath)
        for data in self.inputs:
            values = data.split(',')
            label  = int(values[0])
            output = self.query(values[1:])
            answer = np.argmax(output)

            if (label == answer):
                scoreCard.append(1)
            else:
                scoreCard.append(0)
        scoreArray = np.asarray(scoreCard)
        return scoreArray.sum() / scoreArray.size

        

net = NeuralNetwork([784, 400, 10], 0.15)
epoches = 5

for epoch in range(epoches):
    net.train("data/mnist_train.csv")
    accuracy = net.get_accuracy("data/mnist_test.csv")
    print("the accuracy of epoch{} = {}".format(epoch, accuracy))
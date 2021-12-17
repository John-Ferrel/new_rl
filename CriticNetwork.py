from Debug import logDebug
import os
import pickle
import numpy as np
import random

# NeuralNetwork
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output) 
    
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
 
    def backward(self, output):
        return 1 - output * output
class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        #self.input_size：输入维度
        #self.output_size：输出维度
        #self.activator：激活函数
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        #权值矩阵self.w和偏置self.b
        self.w = np.random.uniform(-0.5, 0.5, (output_size, input_size))
        self.b = np.zeros((output_size, 1))

        self.w_grad_total = np.zeros((output_size, input_size))
        self.b_grad_total = np.zeros((output_size, 1))

    
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = self.activator.forward(np.dot(self.w, self.input_data) + self.b)
 
 
    def backward(self, input_delta):
        #input_delta_为后一层传入的误差
        #output_delta为传入前一层的误差
        self.sen = input_delta * self.activator.backward(self.output_data)
        output_delta = np.dot(self.w.T, self.sen)
        self.w_grad = np.dot(self.sen, self.input_data.T)
        self.b_grad = self.sen
        self.w_grad_total += self.w_grad
        self.b_grad_total += self.b_grad
        return output_delta
 
    def update(self, lr,MBGD_mode = 0):
        #梯度下降法进行权值更新,有几种更新权值的算法。
        #MBGD_mod==0指SGD模式，即随机梯度下降
        #MBGD_mod==1指mnni_batch模式，即批量梯度下降, 当选取batch为整个训练集时，为BGD模式，即批量梯度下降
        if MBGD_mode == 0:
            self.w -= lr * self.w_grad
            self.b -= lr * self.b_grad
        elif MBGD_mode == 1:
            self.w -= lr * self.w_grad_add
            self.b -= lr * self.b_grad_add
            self.w_grad_add = np.zeros((self.output_size, self.input_size))
            self.b_grad_add = np.zeros((self.output_size, 1))
            
class Network():
    def __init__(self, params_array, activator):
        #params_array为层维度信息超参数数组
        #layers为网络的层集合
        self.layers = []
        for i in range(len(params_array) - 1):
            self.layers.append(FullConnectedLayer(params_array[i], params_array[i+1], activator))

    #网络前向迭代
    def predict(self, sample):
        #下面一行的output可以理解为输入层输出
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return output

    #网络反向迭代
    def calc_gradient(self, label):
        delta = (self.layers[-1].output_data - label)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    #一次训练一个样本 ，然后更新权值          
    def train_one_sample(self, sample, label, lr):
        self.predict(sample)
        Loss = self.loss(self.layers[-1].output_data, label)
        self.calc_gradient(label)
        self.update(lr)
        return Loss

    #一次训练一批样本 ，然后更新权值  
    def train_batch_sample(self, sample_set, label_set, lr, batch):
        Loss = 0.0
        for i in range(batch):
            self.predict(sample_set[i])
            Loss += self. loss(self.layers[-1].output, label_set[i])
            self.calc_gradient(label_set[i])
        self.update(lr, 1)
        return Loss

    def update(self, lr, MBGD_mode = 0):
        for layer in self.layers:
            layer.update(lr, MBGD_mode)

    def loss(self, pred, label):
        return 0.5 * ((pred - label) * (pred - label)) .sum()

    def gradient_check(self, sample, label):
        self.predict(sample)
        self.calc_gradient(label)
        incre = 10e-4
        for layer in self.layers:
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    layer.w[i][j] += incre
                    pred = self.predict(sample)
                    err1 = self.loss(pred, label)
                    layer.w[i][j] -= 2 * incre
                    pred = self.predict(sample)
                    err2 = self.loss(pred, label)
                    layer.w[i][j] += incre
                    pred_grad = (err1 - err2) / (2 * incre)
                    calc_grad = layer.w_grad[i][j]
                    print ('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, pred_grad, calc_grad))
                    
# CriticNetwork                    
class CriticNetwork():
    def __init__(self, params, pattern_finder):
        self.ALPHA = 1
        self.LEARNING_RATE = 0.1
        self.MAGNIFY = 1
        activator = SigmoidActivator()
        self.layers = []
        for i in range(len(params) - 1):
            self.layers.append(FullConnectedLayer(params[i], params[i+1], activator))
        
        self.losing_encodings = [
            pattern_finder.get_encoding('-xxxx'), 
            pattern_finder.get_encoding('x-xxx'), 
            pattern_finder.get_encoding('xx-xx'),
        ]
        self.winning_encodings = [
            pattern_finder.get_encoding('-oooo'), 
            pattern_finder.get_encoding('o-ooo'), 
            pattern_finder.get_encoding('oo-oo'),
        ]

    def load_layers(self, filepath):
        if os.path.exists(filepath):
            self.layers = pickle.load(open(filepath, 'rb'))
            logDebug('Loaded existing model at '+filepath)
        else:
            logDebug('File '+filepath+' does not exist')

    def extract_features(self, board):
        #logDebug('board features: '+ str(board.features))
        flattened_features = []
        for feature in board.features:
            for i in range(4):
                flattened_features.append(int(feature > i))
            flattened_features.append( (feature-4)/2 if feature>4 else 0 )
        return np.asarray(flattened_features + [board.whose_turn-1, 2-board.whose_turn]).reshape(-1, 1)
        #return np.random.rand(200, 1)

    def forward(self, board):
        if board.win is not None:
            return 1 if board.win else 0
        
        if board.whose_turn == 1:
            for index in self.winning_encodings:
                if board.features[index] > 0:
                    return 1
        if board.whose_turn == 2:
            for index in self.losing_encodings:
                if board.features[index] > 0:
                    return 0

        output = self.extract_features(board)
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return float(output)

    def calc_gradient(self, error):
        delta = -self.ALPHA*error #注意！！！
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        #return error * delta
        return delta

    def back_propagation(self, board, board_next, reward):
        next_V = self.forward(board_next)
        V = self.forward(board)
        learning_rate = self.LEARNING_RATE*self.MAGNIFY if next_V == 0 else self.LEARNING_RATE
        error = self.ALPHA* (reward + next_V - V)
        logDebug('error: '+str(error))
        self.calc_gradient(error)
        
        for layer in self.layers:
            layer.update(learning_rate, MBGD_mode=0)
        logDebug('value previous: '+str(V))
        logDebug('value updated: '+str(self.forward(board)))
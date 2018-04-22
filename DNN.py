# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 21:33:19 2017

@author: myazi
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
import math

def load_dataset():
    """
    加载数据，并完成对数据的预处理，由于每一张图片表示形式为64*64*3，
    这里需要将每张图片拉成列向量，多张图片一起构成一个大矩阵
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    print(np.shape(train_set_x_orig))
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    """
    test=([[[[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]],
            [[13,14,15],[16,17,18]]],
            [[[19,20,21],[22,23,24]],
            [[25,26,27],[28,29,30]],
            [[31,32,33],[34,35,36]]]
    ])
    test=np.array(test)
    print(test.shape)
    test1=test.reshape((test.shape[1]*test.shape[2]*test.shape[3],test.shape[0]))
    
    print(test1)
    
    test2=test.reshape((test.shape[0],-1)).T
    
    print(test2)
    
    print(test)
    
    """
    
    """
    完成图片拉成列向量
    /255防止在迭代过程中出现nan，因为激活函数收敛速度惊人
    """
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255.0 
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255.
    
    return train_set_x_flatten, train_set_y_orig, test_set_x_flatten, test_set_y_orig, classes

def load_dataset_2():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    #plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

"""
范数

"""
def L1(yhat,y):
    m=y.shape[1]
    loss=sum(np.abs(yhat-y))
    return loss/m

def L2(yhat,y):
    m=y.shape[1]
    loss=np.dot((y-yhat),(y-yhat).T)
    return loss/m
 
def init_parameters(in_n,hid_n,out_n):
    
    """
    一个简单的一个隐藏层的神经网络初始化
    """ 
    
    np.random.seed(1)
    W1 = np.random.randn(hid_n, in_n)*0.01
    b1 = np.zeros((hid_n, 1))
    W2 = np.random.randn(out_n, hid_n)*0.001
    b2 = np.zeros((out_n, 1))
    assert(W1.shape==(hid_n, in_n))
    assert(b1.shape==(hid_n, 1))
    assert(W2.shape==(out_n, hid_n))
    assert(b2.shape==(out_n, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
 
def init_parameters_deep(layer_dims,initialization):
    
    """
    可调整超参的神经网络初始化
    网络层数
    每层神经元个数
    当然每一层使用的激活函数也可以放到这里来设置
    """ 
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    if initialization == "zeros":
        for l in range(1, L):
            parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1])) 
            parameters['b' + str(l)] = np.zeros(((layer_dims[l], 1)))
        
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    elif initialization == "random":
        np.random.seed(3) 
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])  #/ np.sqrt((layer_dims[l-1]+layer_dims[l])/2)
            parameters['b' + str(l)] = np.zeros(((layer_dims[l], 1)))
        
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    elif initialization == "he":
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #* np.sqrt(2.0/layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros(((layer_dims[l], 1)))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def line_forword(A,W,b,keep_prob):
    
    """
    前向传导
    线性函数计算
    其中A是上一层神经元的激活值
    并将当前层W，b，A保存下来是为了在反向传播中需要使用到
    """
    if(keep_prob!=1):
        D=np.random.rand(A.shape[0], A.shape[1])     # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        
        #D=np.random.rand(A.shape[0],1)
        D = (D < keep_prob)
        #print(D)
        A=A*D;
        A=A/keep_prob
    
    Z=W.dot(A)+b
    
    assert(Z.shape==(W.shape[0],A.shape[1]))
    
    if(keep_prob!=1):
        cache=(A,W,b,D)
    else:
        cache=(A,W,b)
    return Z,cache

def sigmoid(Z):
    
    """
    前向传导
    sigmoid函数计算，且是矩阵计算，
    保存Z是因为在反向传播中需要计算dA，
    为了代码简洁而保存Z,完全可以只保存A,W,b
    """
    
    A=1/(1+np.exp(-Z))
    
    assert(A.shape==Z.shape)
    cache=Z
    return A,cache

def relu(Z):
    
    """
    前向传导
    relu修正单元函数计算，矩阵计算，
    """
    
    A=np.maximum(0,Z)
    
    
    assert(A.shape == Z.shape)
    cache = Z 
    
    return A, cache

def line_active_forword(Apre,W,b,keep_prob,activation):
    
    """
    前向传导
    线性函数，激活函数计算
    并且保存线性函数左边的值，与激活函数左边的值用于反向传播
    """
    
    Z,line_cache=line_forword(Apre,W,b,keep_prob)
    
   # print("===================")
    #print(Z.shape)
    
    if(activation=="sigmoid"):
        A,active_cache=sigmoid(Z)
        
    if(activation=="rule"):
        A,active_cache=relu(Z)
        
    assert(A.shape==(W.shape[0],Apre.shape[1]))
    
    cache=(line_cache,active_cache)
    
    return A,cache

def model_forward(X,parameters,keep_prob):
    
    """
    前向传导
    将输入特征给A0，根据参数，线性函数，激活函数计算
    计算过程中保存每一层中的函数左边的值
    即A,W,b,Z
    """
    
    caches=[];
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        Apre=A
        A , cache = line_active_forword(Apre,parameters["W" + str(l)],parameters["b" + str(l)],keep_prob[l-1],"rule")
        caches.append(cache)
    
    AL , cache = line_active_forword(A,parameters["W" + str(L)],parameters["b" + str(L)],keep_prob[L],"sigmoid")
    caches.append(cache)
    
    assert(AL.shape==(1,X.shape[1]))
    
    return AL,caches

def relu_backword(dA,cache):
    
    """
    relu函数的导数0,1
    而导数值取决于Z是否大于0
    """
    
    Z=cache
    
    #dZ = np.array(dA, copy=True)
    #dZ[Z <= 0] = 0
    
    dZ = np.multiply(dA, np.int64(Z > 0))
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backword(dA,cache):
    
    """
    反向传播
    sigmoid函数的导数是s（1-s）
    而损失函数对Z的偏导为dAs（1-s）
    这里，如果保存了当前层的激活值，完全可以不用再计算一遍s
    但是当前层cache中保存的是A,W,b,Z,其中有Z=wA+b，不是A=g（z）,
    所以之前sigmoid中保存了z，且在反向传播中再计算一遍A
    """
    Z=cache
    
    s=1/(1+np.exp(-Z))
    
    dZ=dA*s*(1-s)

    
    assert (dZ.shape == Z.shape)
    
    return dZ

def line_backward(dZ,cache,lambd,keep_prob):
    
    """
    反向传播
    线性函数求导同时乘上链式求导下来的dZ
    dA=W.T*dZ
    dW=1.0/m*dZ*A
    db=1.0/m*dZ
    """
    if(keep_prob!=1):
        Apre, W, b, D = cache
    else:
        Apre, W, b = cache
    m = Apre.shape[1]

    if lambd!=0:
        dW=1.0 / m * np.dot(dZ,Apre.T) + 1.0/m * lambd * W
    else:
        dW=1.0 / m * np.dot(dZ,Apre.T)    
    db = 1.0 / m * np.sum(dZ, axis = 1, keepdims = True)
    
    dApre=np.dot(W.T,dZ)
    if(keep_prob!=1):
        ### START CODE HERE ### (≈ 2 lines of code)
        dApre = D * dApre           # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        dApre = dApre / keep_prob           # Step 2: Scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
    
    assert (dApre.shape == Apre.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dApre,dW,db

def line_active_backword(dA,cache,activation,lambd,keep_prob):
    
    """
    反向传播
    激活函数，线性函数求导
    """
    
    line_cache,activation_cache=cache
    
    if(activation=="relu"):
        
        dZ=relu_backword(dA,activation_cache)
        dApre,dW,db=line_backward(dZ,line_cache,lambd,keep_prob)
        
    if(activation=="sigmoid"):
        
        dZ=sigmoid_backword(dA,activation_cache)
        dApre,dW,db=line_backward(dZ,line_cache,lambd,keep_prob)
        
    return dApre,dW,db

def model_backward(AL,Y,caches,lambd,keep_prob):
        
    """
    反向传播
    1损失函数对AL的导数单独计算
    2重复计算激活函数，线性函数的导数，
    但值得注意的是，实际上还是损失函数对参数求偏导
    保存每一层参数的导数用于更新参数
    """
    
    grads={}
    L=len(caches)

    Y = Y.reshape(AL.shape)
    
    dAL= - (np.divide(Y, AL) - np.divide((1 - Y ), (1 - AL)))#dF/dAL视损失函数而定，对数线性模型损失，最小均方差损失    
    
    current_cache=caches[L-1]
    
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)]=line_active_backword(dAL, current_cache, "sigmoid",lambd,keep_prob[L])
    
    for l in reversed(range(L-1)):#l L-2,L-3,...L=0
    
        current_cache=caches[l]
        
        dA_pre, dW, db=line_active_backword(grads["dA" + str(l+2)],current_cache,"relu",lambd,keep_prob[l])
        
        grads["dA" + str(l+1)]=dA_pre
        grads["dW" + str(l+1)]=dW
        grads["db" + str(l+1)]=db
        
    return grads
        
def cumpute_loss(AL,Y,parameters,lambd):
    
    """
    计算损失值
    """
    L=len(parameters)//2
    
    m = Y.shape[1]
    #print("AL==========\n")
    #print(AL)
    cost = (1.0/m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    L2_regularization_cost = 0
    if lambd!=0:
        for l in range(L):
            L2_regularization_cost += 1.0/(2*m) * lambd * (np.sum(np.square(parameters["W" + str(l+1)]),keepdims=True))
    
    cost+=L2_regularization_cost
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        
    assert(cost.shape == ())
    
    return cost

def update_parameters_with_gd(parameters,grads,learn_rate):
    
    L=len(parameters)//2
    
    for l in range(L):
        parameters["W" + str(l+1)]-=learn_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)]-=learn_rate*grads["db" + str(l+1)]
        
    return parameters

def initialize_velocity(parameters):
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        ### END CODE HERE ###
        
    return v

def update_parameters_with_momentum(parameters, grads, beta, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural networks
    
    v = initialize_velocity(parameters)
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * v["db" + str(l+1)]
        ### END CODE HERE ###
        
    return parameters, v

def initialize_adam(parameters) :

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    ### END CODE HERE ###
    
    return v, s
def update_parameters_with_adam(parameters, grads, t, learning_rate,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    
    v, s = initialize_adam(parameters)
    
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(beta1,t))
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * grads['dW' + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * grads['db' + str(l+1)]**2
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2,t))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - np.power(0.9,t/1000)*learning_rate * v_corrected["dW"+str(l+1)] / (np.sqrt(s_corrected["dW"+str(l+1)])+epsilon)
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - np.power(0.9,t/1000)*learning_rate * v_corrected["db"+str(l+1)] / (np.sqrt(s_corrected["db"+str(l+1)])+epsilon)
        ### END CODE HERE ###

    return parameters, v, s

def update_parameters(parameters,grads,learn_rate,t,optimizer,beta1,beta2,epsilon):
    
    """
    上一次参数值，梯度，学习速率
    更新参数
    """    
    
    if optimizer == "gd":
        parameters = update_parameters_with_gd(parameters, grads, learn_rate)
        
    elif optimizer == "momentum":
        parameters, v = update_parameters_with_momentum(parameters, grads, beta1, learn_rate)
    elif optimizer == "adam":
        #t = t + 1 # Adam counter
        parameters, v, s = update_parameters_with_adam(parameters, grads,t,learn_rate, beta1, beta2,epsilon)           
    
    
    return parameters

def two_layer_modle(X,Y,layer_dims,optimizer="adam",learn_rate=0.01,initialization="he",lambd=0.01,keep_prob = 0.9,
    mini_batch_size = 64,beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_iterations = 30000, print_cost=True):
    
    """
    神经网络模型
    输入特征，输出特征，网络结构(可以包括每一层的激活函数类型)，学习速率，迭代次数，损失值
    """
    L = len(layer_dims) 
    
    costs = []
    
    parameters=init_parameters_deep(layer_dims,initialization)
    
    if keep_prob<1:
        keep_probs=np.ones(L)
        for i in range(0,L):
            if(i==0 or i==L-1):
                keep_probs[i]=1
            else:
                keep_probs[i]=0.9
    else:
       keep_probs=np.ones(L)
    seed=10    
   # Xj=np.zeros((X.shape[0],1));
   # Yj=np.zeros((Y.shape[0],1));
    for i in range(num_iterations):
        
        """ Stochastic Gradient Descent """
        
#        for j in range(0,Y.shape[1]):
#            
#            ###传递过去的是一个一维数组
#            Xj[:,0]=X[:,j]
#            Yj[:,0]=Y[:,j]
#            AL,caches=model_forward(Xj,parameters,keep_prob)
#        
#            cost=cumpute_loss(AL,Yj,parameters,lambd)
#        
#            grads=model_backward(AL,Yj,caches,lambd,keep_prob)
#        
#            parameters=update_parameters(parameters,grads,learn_rate)
            
            
        """ Gradient Descent """
#        AL,caches=model_forward(X,parameters,keep_prob)
#        
#        cost=cumpute_loss(AL,Y,parameters,lambd)
#        
#        grads=model_backward(AL,Y,caches,lambd,keep_prob)
#        
#        parameters=update_parameters(parameters,grads,learn_rate)

        """ mini-batch Gradient Descent """

        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        
        for minibatch in minibatches:
         
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = model_forward(minibatch_X, parameters,keep_probs)

            # Compute cost
            cost = cumpute_loss(AL, minibatch_Y,parameters,lambd)

            # Backward propagation
            grads = model_backward(AL, minibatch_Y, caches,lambd,keep_probs)
            
            # update_parameters
            parameters=update_parameters(parameters,grads,learn_rate,i+1,optimizer,beta1,beta2,epsilon)
            
        if print_cost and i%1000==0:
            print(cost)
#            line_cache,activation_cache=caches[2]
#            Z=activation_cache
#            print(Z)
#            A,W,b,D=line_cache
#            print(A,W,b,D)
            costs.append(cost)
        i+=1
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learn_rate))
    plt.show()
    
    return parameters

def predict(X,Y,parameters):
    
    """
    预测
    测试集X,Y,训练参数

    """
    m=X.shape[1]
    pro=np.zeros((1,m))
    
    A,cache=model_forward(X,parameters,[1,1,1,1])
    print(A.shape)
    
    for i in range(0, A.shape[1]):
        if(A[0,i]>0.5):
            pro[0,i]=1
        else:
            pro[0,i]=0
    print(pro)
    print(Y)
    acc=0
    for i in range(0,A.shape[1]):
        if(pro[0,i]==Y[0,i]):
            acc+=1
    print(acc)    
    
    print("Accuracy: "  + str(np.mean((pro[0,:] == Y[0,:]))))
        
    return pro
  
def main():
        
    """
    主函数
    1加载训练集，预处理
    2初始化网络结构，超参，参数
    3训练
        1前向传导
        2计算损失值
        3反向传播
        4更新参数
    4预测

    """  
    a=np.zeros([4,3])
    print(a)
    #train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()
    #train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig=load_dataset_2()
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig=load_2D_dataset()
    
    train=np.concatenate([train_set_x_orig, train_set_y_orig],axis=0)
    np.savetxt("text.txt",train)
    
    in_n=train_set_x_orig.shape[0]
    hid_n=7
    out_n=1
    layer_dims=(in_n,hid_n,out_n)
    
    layer_dims=(in_n,20,3,1)
    parameters=two_layer_modle(train_set_x_orig,train_set_y_orig,layer_dims)
    
    predict(train_set_x_orig,train_set_y_orig,parameters)
    
    predict(test_set_x_orig,test_set_y_orig,parameters)
    print(test_set_y_orig)
    
    
    
    
    

    
    

    
    

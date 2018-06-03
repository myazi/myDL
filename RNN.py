# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:00:25 2018

@author: Administrator
"""
import numpy as np
import csv
import emoji

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_glove_vecs_3(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map    
    
def read_csv(filename = 'datasets/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

def embedding_layer(X,word_to_vec_map):
    
    x_dim=word_to_vec_map['a'].shape[0]
    
    m = X.shape[0]                                   # number of training examples
    
    T_x = len(max(X, key=len).split())
    
    X_train=np.zeros((x_dim,m,T_x))
      
    for i in range(m):
        sentence_words=X[i].lower().split()
        j=0
        for w in sentence_words:
            X_train[:,i,j]=word_to_vec_map[w]
            j+=1
            
    return X_train


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    ba = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba,"by": by}
    
    return parameters

def initialize_parameters_gru(n_a, n_x, n_y):
    np.random.seed(1)
    Wf = np.random.randn(n_a, n_x+n_a)*0.01
    bf = np.random.randn(n_a,1)
    Wo = np.random.randn(n_a, n_x+n_a)*0.01
    bo = np.random.randn(n_a,1)
    Wa = np.random.randn(n_a, n_x+n_a)*0.01
    ba = np.random.randn(n_a,1)
    Wy = np.random.randn(n_y,n_a)*0.01
    by = np.random.randn(n_y,1)

    parameters = {"Wf": Wf, "Wo": Wo, "Wa": Wa, "Wy": Wy, "bf": bf, "bo": bo, "ba": ba, "by": by}
    
    return parameters

def initialize_parameters_lstm(n_a, n_x, n_y):
    np.random.seed(1)
    Wf = np.random.randn(n_a, n_a+n_x)*0.01
    bf = np.random.randn(n_a,1)
    Wi = np.random.randn(n_a, n_a+n_x)*0.01
    bi = np.random.randn(n_a,1)
    Wo = np.random.randn(n_a, n_a+n_x)*0.01
    bo = np.random.randn(n_a,1)
    Wc = np.random.randn(n_a, n_a+n_x)*0.01
    bc = np.random.randn(n_a,1)
    Wy = np.random.randn(n_y,n_a)*0.01
    by = np.random.randn(n_y,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
    
    return parameters

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)   
    ### END CODE HERE ###
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, yt_pred, xt, parameters)
    
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    ### START CODE HERE ###
    
    # initialize "a" and "y" with zeros (≈2 lines)
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
        
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches

def gru_cell_forward(xt, a_prev, parameters):
    
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wa = parameters["Wa"]
    ba = parameters["ba"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###
    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros([n_a + n_x, m])
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    
    
    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    
    at=ft * a_prev    
    concat_ft = np.zeros([n_a + n_x, m])
    concat_ft[: n_a, :] = at
    concat_ft[n_a :, :] = xt
    
    aat=np.tanh(np.dot(Wa, concat_ft) + ba)
    
    a_next = ot * aat + (1-ot) * a_prev
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    cache = (a_next, yt_pred, a_prev, ft, ot, aat, xt, parameters)

    return a_next, yt_pred, cache

def gru_forward(x, a0, parameters):
    
    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    ### START CODE HERE ###
    # Retrieve dimensions from shapes of xt and Wy (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, yt, cache = gru_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)
        
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the save gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the focus gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the focus gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilda),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###
    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros([n_a + n_x, m])
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    cache = (a_next, yt_pred, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the save gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the focus gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the focus gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    ### START CODE HERE ###
    # Retrieve dimensions from shapes of xt and Wy (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros([n_a, m])
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)
        
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_step_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    
    # Retrieve values from cache
    (a_next, a_prev,y, xt, parameters) = cache 
    
    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
#    Wya = parameters["Wya"]
#    ba = parameters["ba"]
#    by = parameters["by"]

    ### START CODE HERE ###
    # compute the gradient of tanh with respect to a_next (≈1 line)
    dtanh = (1-a_next * a_next) * da_next  

    # compute the gradient of the loss with respect to Wax (≈2 lines)
    dxt = np.dot(Wax.T,dtanh) #可以不求
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to Waa (≈2 lines)
    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # compute the gradient with respect to b (≈1 line)
    dba = np.sum(dtanh, keepdims=True, axis=-1) 

    ###对dwy的导数需要每时刻都有真实的yt值，与前向传递中计算的yt进行比较，求导出dy，然后对dwy进行链式求导
    
    ### END CODE HERE ###
    
    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients

def rnn_backward(da, caches,Y):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
        
    ### START CODE HERE ###
    
#     Retrieve values from the first cache (t=1) of caches (≈2 lines)
    (caches, x) = caches
    (a1, a0, y, x1, parameters) = caches[0]  #参数从cache中获得，这里只是检查caches[0],
                                             #同时得到下面值
                                             #反向传播还是从最后一个开始
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈6 lines)
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    
    #########################################
    #many-to-one 目标损失只有最后一个，则只对da[::T_x-1]求导，其他都为0，
    #如果是many-to-many，则可以把下面代码放到for循环内，每次先求da，然后将da和da_prevt加和进行反向传播
    
    a_next,a_pred,y,x,parameters=caches[T_x-1]
    Wya=parameters["Wya"]
    
    dWya=np.dot((y-Y),a_next.T)
    #dby=1.0 / m * np.sum((y-Y), axis = 1, keepdims = True)
    dby = np.sum((y-Y), axis = 1, keepdims = True)
    da[:,:,T_x-1]=np.dot(Wya.T,(y-Y))#yt对at的导数
    
    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        # Retrieve derivatives from gradients (≈ 1 line)
        
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line) 
    da0 = da_prevt
    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba, "dWya": dWya, "dby": dby}
    
    return gradients

def gru_cell_backward(da_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the input gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, y, a_prev, ft, aat, ot, xt, parameters) = cache
    
    ### START CODE HERE ###
    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = da_next * (aat - a_prev)
    daat = da_next * ot
    
    dft = daat * (1 - np.square(aat))
    dft = np.dot(parameters["Wa"][:,:n_a].T, dft)
    dft = dft * a_prev
    
    dot_sigmod=dot*ot*(1-ot)
    dft_sigmod=dft*ft*(1-ft)
    daat_tanh=daat*(1-np.square(aat))
    
    concat = np.concatenate((a_prev, xt), axis=0).T
    concat_ft = np.concatenate(((ft * a_prev), xt), axis=0).T
    
    dWf = np.dot(dft_sigmod, concat)
    dWa = np.dot(daat_tanh, concat_ft)###
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft_sigmod,axis=1,keepdims=True)  
    dba = np.sum(daat_tanh,axis=1,keepdims=True)  
    dbo = np.sum(dot_sigmod,axis=1,keepdims=True)  

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft_sigmod) + np.dot(parameters["Wa"][:, :n_a].T, daat_tanh) + np.dot(parameters["Wo"][:, :n_a].T, dot_sigmod) +  (1-ot) * da_next
    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft_sigmod) + np.dot(parameters["Wa"][:, n_a:].T, daat_tanh)  + np.dot(parameters["Wo"][:, n_a:].T, dot_sigmod)
    ### END CODE HERE ###
    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWf": dWf,"dbf": dbf, "dWa": dWa,"dba": dba, "dWo": dWo,"dbo": dbo}

    return gradients


def gru_backward(da, caches, Y):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, y1, a0, f1, o1, aa1, x1, parameters) = caches[0]
    
    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])
    
    
    #########################################
    #many-to-one 目标损失只有最后一个，则只对da[::T_x-1]求导，其他都为0，
    #如果是many-to-many，则可以把下面代码放到for循环内，每次先求da，然后将da和da_prevt加和进行反向传播
    
    a_next,y,a_prev,f1,o1,aa1,x1,parameters=caches[T_x-1]
    
    dWy=np.dot((y-Y),a_next.T)
    dby=np.sum((y-Y), axis = 1, keepdims = True) 
    da[:,:,T_x-1]=np.dot(parameters["Wy"].T,(y-Y))
    
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = gru_cell_backward(da[:,:,t],caches[t])
        # da_prevt, dc_prevt = gradients['da_prev'], gradients["dc_prev"]
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:,:,t] = gradients['dxt']
        dWf = dWf+gradients['dWf']
        dWa = dWc+gradients['dWa']
        dWo = dWo+gradients['dWo']
        dbf = dbf+gradients['dbf']
        dba = dbc+gradients['dba']
        dbo = dbo+gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']
    
    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf,
                "dWa": dWa,"dba": dba, "dWo": dWo,"dbo": dbo, "dWy": dWy, "dby": dby}
    
    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the input gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, y, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    ### START CODE HERE ###
    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
    
    ## Code equations (7) to (10) (≈4 lines)
    ##dit = None
    ##dft = None
    ##dot = None
    ##dcct = None
    ##
    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft,axis=1,keepdims=True)  
    dbi = np.sum(dit,axis=1,keepdims=True)  
    dbc = np.sum(dcct,axis=1,keepdims=True)  
    dbo = np.sum(dot,axis=1,keepdims=True)  

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)
    dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next
    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)
    ### END CODE HERE ###
    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients


def lstm_backward(da, caches, Y):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, y1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])
    
    
    #########################################
    #many-to-one 目标损失只有最后一个，则只对da[::T_x-1]求导，其他都为0，
    #如果是many-to-many，则可以把下面代码放到for循环内，每次先求da，然后将da和da_prevt加和进行反向传播
    
    a_next,y,c1,a,c0,f1,i1,cc1,o1,x1,parameters=caches[T_x-1]
    Wy=parameters["Wy"]
    
    dWy=np.dot((y-Y),a_next.T)
    dby=np.sum((y-Y), axis = 1, keepdims = True) 
    da[:,:,T_x-1]=np.dot(Wy.T,(y-Y))
    
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:,:,t],dc_prevt,caches[t])
        # da_prevt, dc_prevt = gradients['da_prev'], gradients["dc_prev"]
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:,:,t] = gradients['dxt']
        dWf = dWf+gradients['dWf']
        dWi = dWi+gradients['dWi']
        dWc = dWc+gradients['dWc']
        dWo = dWo+gradients['dWo']
        dbf = dbf+gradients['dbf']
        dbi = dbi+gradients['dbi']
        dbc = dbc+gradients['dbc']
        dbo = dbo+gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']
    
    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy, "dby": dby}
    
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def initialize_velocity(parameters):
    
    v = {}
    v["dWax"] = np.zeros(parameters['Wax'].shape)
    v["dWaa"] = np.zeros(parameters['Waa'].shape)
    v["dWya"] = np.zeros(parameters['Wya'].shape)
    v["dba"] = np.zeros(parameters['ba'].shape)
    v["dby"] = np.zeros(parameters['by'].shape)
        
    return v

def update_parameters_with_momentum(parameters,v, grads, beta, learning_rate):
    
    
    v["dWax"] = beta * v["dWax"] + (1-beta) * grads['dWax']
    v["dWaa"] = beta * v["dWaa"] + (1-beta) * grads['dWaa']
    v["dWya"] = beta * v["dWya"] + (1-beta) * grads['dWya'] 
    v["dba"] = beta * v["dba"] + (1-beta) * grads['dba']
    v["dby"] = beta * v["dby"] + (1-beta) * grads['dby']
        # update parameters
    parameters["Wax"]  -= learning_rate * v["dWax"]
    parameters["Waa"]  -= learning_rate * v["dWaa"]
    parameters["Wya"] -= learning_rate * v["dWya"]
    parameters["ba"] -= learning_rate * v["dba"]
    parameters["by"] -= learning_rate * v["dby"]
        ### END CODE HERE ###
        
    return parameters,v


def update_parameters_lstm(parameters, gradients, lr):

    parameters['Wf'] += -lr * gradients['dWf']
    parameters['Wi'] += -lr * gradients['dWi']
    parameters['Wc'] += -lr * gradients['dWc']
    parameters['Wo'] += -lr * gradients['dWo']
    parameters['Wy'] += -lr * gradients['dWy']
    parameters['bf']  += -lr * gradients['dbf']
    parameters['bi']  += -lr * gradients['dbi']
    parameters['bc']  += -lr * gradients['dbc']
    parameters['bo']  += -lr * gradients['dbo']
    parameters['by']  += -lr * gradients['dby']

    return parameters

def update_parameters_gru(parameters, gradients, lr):

    parameters['Wf'] += -lr * gradients['dWf']
    parameters['Wa'] += -lr * gradients['dWa']
    parameters['Wo'] += -lr * gradients['dWo']
    parameters['Wy'] += -lr * gradients['dWy']
    parameters['bf']  += -lr * gradients['dbf']
    parameters['ba']  += -lr * gradients['dba']
    parameters['bo']  += -lr * gradients['dbo']
    parameters['by']  += -lr * gradients['dby']

    return parameters

def model(X,Y,learn_rate=0.0002,num_iterations = 3000, print_cost=True):
    
    n_x=X.shape[0]
    m=X.shape[1]
    T_x=X.shape[2]
    n_a=4;
    n_y=Y.shape[0];
    
    parameters=initialize_parameters(n_a, n_x, n_y)
    #parameters=initialize_parameters_lstm(n_a, n_x, n_y)
    #parameters=initialize_parameters_gru(n_a, n_x, n_y)
    
    a0=np.random.randn(n_a, m)
    
    v = initialize_velocity(parameters)
    
    for i in range(num_iterations):
    
        a_next, y, cache =rnn_forward(X, a0, parameters)
        #a_next, y, c, cache =lstm_forward(X, a0, parameters)
        #a_next, y, cache =gru_forward(X, a0, parameters)
    
        da = np.zeros((n_a,m,T_x)) #该处应该是由损失函数求得，这里先用da表示，没有用到y
        
        cost = sum(sum(-np.multiply(Y,np.log(y[:,:,T_x-1]))))/m
        if(i%1000==0 and print_cost):
            print("cost=" + str(cost))
        
        gradients = rnn_backward(da, cache,Y)
        #gradients = lstm_backward(da, cache,Y)
        #gradients = gru_backward(da, cache,Y) 
        
        parameters=update_parameters(parameters, gradients, learn_rate) #这里更新的参数应该是保存在cache中的
        #parameters,v=update_parameters_with_momentum(parameters, v,gradients, 0.9, learn_rate)
        #parameters=update_parameters_lstm(parameters, gradients, learn_rate)
        #parameters=update_parameters_gru(parameters, gradients, learn_rate)
        
    return parameters
 
    
def predict(X, Y, parameters):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    n_a=4;
    m = X.shape[1]
    T_x=X.shape[2]
    pred = np.zeros((m, 1))
    
    a0=np.random.randn(n_a, m)
    a,y,cache=rnn_forward(X,a0,parameters)
    #a,y,c,cache=lstm_forward(X,a0,parameters)
    y_T=y[:,:,T_x-1]
    
    print(y_T.shape)
    pred = np.argmax(y_T,0).reshape(y_T.shape[1],1)
    print(pred.shape)
    
    
    print(Y.shape)
    pred_Y=np.argmax(Y,0).reshape(Y.shape[1],1)
    print(pred_Y.shape)
    
    print("Accuracy: "  + str(np.mean((pred[:] == pred_Y[:]))))
    
    return pred

def main():
#    x = np.random.randn(3,10,4)
#    y = np.random.randn(2,4)
    X_train, Y_train = read_csv('datasets/train_emoji.csv')
    X_test, Y_test = read_csv('datasets/tesss.csv')
    
    index = 1
    print(X_train[index], label_to_emoji(Y_train[index]))
    
    Y_oh_train = convert_to_one_hot(Y_train, C = 5)
    Y_oh_test = convert_to_one_hot(Y_test, C = 5)
    
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_3('datasets/glove.6B.50d.txt')
    
    
    X_train=embedding_layer(X_train,word_to_vec_map)
    
    print(X_train.shape)
    print(Y_oh_train.T.shape)
    
    Y_train=np.zeros(())
    
    parameters=model(X_train,Y_oh_train.T)
    
    X_test=embedding_layer(X_test,word_to_vec_map)
    
    pred_test = predict(X_train, Y_oh_train.T, parameters)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
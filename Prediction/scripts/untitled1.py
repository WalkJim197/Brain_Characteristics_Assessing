# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:59:19 2022

@author: Conles
"""

#implement LSTM with TensorFlow
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import matplotlib.pyplot as plt
import pandas as pd
tf.reset_default_graph()
import scipy.io as io 
from sklearn.decomposition import PCA

data_predictor = pd.read_excel('E:/ASD/cov.xlsx',sheet_name='Sheet3')
y = np.array(data_predictor['Age'])
data2 = io.loadmat('E:/ASD/PostPrep/prediction/features/ic/AUD_GIG_features.mat')
X2_1 = data2['AUD_GIG_features']

pc = PCA(n_components=0.95)
X = pc.fit_transform(X2_1)
# 7:3划分训练集和测试集
train_data,test_data,train_label,test_label = train_test_split(np.array(X),
                                                               np.array(y),
                                                               test_size = 0.3,
                                                               random_state = 30)

tf.set_random_seed(100)

num_epochs = 200
batch_size = 2
alpha = 0.01
hidden_nodes = 10
#regression 24 input----1 output
input_features = X.shape[1]
sequence_len = 1
output_class = 1 

# input placeholder
X = tf.placeholder("float", [None, sequence_len, input_features])
Y = tf.placeholder("float", [None, sequence_len, output_class])

# define weights, gaussian distribution
weights = {
    'out': tf.Variable(tf.random_normal([hidden_nodes, output_class]))
}

biases = {
    'out': tf.Variable(tf.random_normal([output_class]))
}

# define the RNN lstm network
def RNN(x):
    # reshape input tensor into batch x sequence length x # of features

    x = tf.reshape(x , [-1, sequence_len, input_features])

    # triple layer LSTM with same number of nodes each layer
#    lstm_cell1 = tf.nn.rnn_cell.LSTMCell(10)
#    lstm_cell2 = tf.nn.rnn_cell.LSTMCell(10)
    lstm_cell3 = tf.nn.rnn_cell.LSTMCell(hidden_nodes)

    #stack of those layers
#    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2, lstm_cell3])
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell3])

    #initialize state
    init_state = lstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

    #get the output of each state
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, initial_state=init_state)
    output_sequence = tf.matmul(tf.reshape(outputs, [-1, hidden_nodes]), weights['out']) + biases['out']

    return tf.reshape(output_sequence, [-1, sequence_len, output_class])


#initialization
logits = RNN(X)
loss = tf.losses.mean_squared_error(predictions = logits, labels = Y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
                alpha,
                global_step,
                num_epochs, 0.99,
                staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-10).minimize(loss,global_step=global_step)
init = tf.global_variables_initializer()



#lists to keep track of training, validation and test loss at each epoch
train = []
test = []

with tf.Session() as sess:
    sess.run(init)

    #get # of inputs
    N = train_data.shape[0]

    # training cycle
    for epoch in range(num_epochs):
        # Apply SGD, each time update with one batch and shuffle randomly
        total_batch = int(math.ceil(N / batch_size))
        indices = np.arange(N)
        np.random.shuffle(indices)
        avg_loss = 0

        for i in range(total_batch):
            #get one batch
            rand_index = indices[batch_size*i:batch_size*(i+1)]
            x = train_data[rand_index]
            x = x.reshape(-1,sequence_len, input_features)
            y = train_label[rand_index]
            y = y.reshape(-1, sequence_len, output_class)
            _, cost = sess.run([optimizer, loss],
                                feed_dict={X: x, Y: y})
            avg_loss += cost / total_batch


        #take the square root
        avg_loss = np.sqrt(avg_loss)


        #append to list for drawing
        train.append(avg_loss)
        print('epoch:',epoch,' ,train loss ',avg_loss)

    #calculate result each dataset
    train_data=train_data.reshape(-1,sequence_len,input_features)
    train_pred = sess.run(logits, feed_dict={X: train_data})
    train_pred = train_pred.reshape(-1, output_class)

    
    test_data=test_data.reshape(-1,sequence_len,input_features)
    test_pred = sess.run(logits, feed_dict={X: test_data})
    test_pred = test_pred.reshape(-1, output_class)
    

# calculate mse

test_mse=np.mean(np.square(test_pred-test_label))
test_RMSE = math.sqrt(test_mse)
print('test RMSE:',test_RMSE)

# In[] plot RMSE vs epoch

g = plt.figure(1)
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.plot( train, label='training')
#plt.plot( valid, label='validation')
#plt.plot( test, label='test')
plt.title('train loss')
plt.legend()
plt.show()    
##plot test_set result
for i in range(1):
    plt.figure()
    plt.plot(pd.DataFrame(test_label)[i],c='r', label='true')
    plt.plot(pd.DataFrame(test_pred)[i],c='b',label='predict')
    string='the'+str(i+1)+'output'
    plt.title(string)
    plt.legend()
    plt.show()
# plt.figure()
# plt.plot(test_label,c='r', label='true')
# plt.plot(test_pred,c='b',label='predict')
# #string='the '+str(i+1)+' output'
# plt.title('预测精度曲线')
# plt.legend()
# plt.show()




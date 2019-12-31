# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:56:24 2019

@author: WJDT
"""

# mechine learning model, output as est. mkt rate of ret 

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.reset_default_graph() # tf variable dissemvles

# 입력부분
nodes = 10
keep_prob = 0.5
learning_rate = 0.0005
split_ratio = 0.5
training_epochs = 1000
batch_size = 6
lamb = 0.05

NUM = 2
# 입력부분 끝
"""
xls = pd.ExcelFile('KSData.xlsx')
df1 = pd.read_excel(xls, 'MoRet')
df2 = pd.read_excel(xls, 'MktWeight_Sort')
df3 = pd.read_excel(xls, 'MoMktRet')

df1.index = df1['Symbol']
del df1['Symbol']
df1 = df1/100
df2.index = df2['Symbol']
del df2['Symbol']
df3.index = df3['Symbol']
del df3['Symbol']
df3 = df3/100

Ini = df2.columns[:-51] # -51번째부터 시점당시 상장

for i in range(len(df1)):
    if i == 0:
        A0 = np.log(df1+1)
    else:
        A0.iloc[i] = A0.iloc[i-1] + A0.iloc[i]

df1 = np.exp(A0)-1

for i in range(len(df3)):
    if i == 0:
        A1 = np.log(df3+1)
    else:
        A1.iloc[i] = A1.iloc[i-1] + A1.iloc[i]

df3 = np.exp(A1)-1
"""
x_data = np.array(df1[Ini[:NUM]], dtype = np.float32).T
x_data = x_data.T
y_data = np.array([df3['^KS11']])
y_data = y_data.T

train_size = int(len(y_data) * split_ratio)
test_size = len(y_data) - train_size
trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])

X = tf.placeholder(tf.float32, shape=[None, len(x_data.T)])
Y = tf.placeholder(tf.float32, shape=[None, 1])

init = tf.constant_initializer(np.array(df2[Ini[:NUM]].iloc[0])/np.sum(np.array(df2[Ini[:NUM]].iloc[0])))

w1 = tf.get_variable('weight1', shape=[len(x_data.T),1],
                     initializer = init, constraint = lambda x:tf.clip_by_value(x, 0, 1))
w1 = w1 / tf.reduce_sum(w1)

logits1 = tf.matmul(X, w1)

hypothesis = tf.nn.relu(logits1)
cost = tf.reduce_sum(tf.square(hypothesis - Y))
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_count = len(trainX)
    print('Learning started. It takes while..')
    for epoch in range(training_epochs):
        total_batch = int(train_size / batch_size)
        for start, end in zip(range(0, train_count, batch_size),
                              range(batch_size, train_count + 1, batch_size)):
            feed_dict = {X: trainX[start:end],
                         Y: trainY[start:end]}
            l, _ = sess.run([cost, train], feed_dict = feed_dict)
            w = sess.run(w1).T.reshape(-1)
        
        pred = sess.run(hypothesis, feed_dict={X: testX})
        for p, y in zip(pred, testY.flatten()):
            pred_l = sess.run(cost, feed_dict={X: testX, Y: testY})
        if epoch == 0:
            st = np.array([1, np.sqrt((pred_l + l)/(len(x_data.T)-1))])
            wst = w
        else:
            st = np.vstack([st,
                            np.array([epoch + 1, np.sqrt((pred_l + l)/(len(x_data.T)-1))])])
            wst = np.vstack([wst, w])
    
        #if epoch % 20 == 19:
            #print('Epoch', '%02d'%(epoch + 1), "loss= {:.6f}".format(np.sqrt(l/(len(trainX)-1))))

plt.title("Portfolio Tracking error(# of stock: {})".format(NUM))
plt.scatter(st[:, 0], st[:, 1], color = 'k', s = 3., label = 'Tracking Error')
legend = plt.legend(loc = 'upper right')
plt.show()

# np.argmin(st[:,1]), min(st[:,1])
# wst[np.argmin(st[:, 1])]


    


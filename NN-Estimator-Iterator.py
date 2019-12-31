# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:51:56 2019

@author: WJDT
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 입력부분
nodes = 10
split_ratio = 0.5
batch_size = 6
lamb = 0.05
num = 200
# 입력부분 끝

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

for i in range(int(len(df1)/2)):
    if i == 0:
        A0 = np.log(df1+1)
    else:
        A0.iloc[i] = A0.iloc[i-1] + A0.iloc[i]
        
for i in range(int(len(df1)/2), len(df1)):
    if i == len(df1)/2:
        A0.iloc[i] = np.log(df1+1).iloc[i]
    else:
        A0.iloc[i] = A0.iloc[i-1] + A0.iloc[i]

df1 = np.exp(A0)-1

for i in range(int(len(df3)/2)):
    if i == 0:
        A1 = np.log(df3+1)
    else:
        A1.iloc[i] = A1.iloc[i-1] + A1.iloc[i]

for i in range(int(len(df3)/2), len(df3)):
    if i == len(df3)/2:
        A1.iloc[i] = np.log(df3+1).iloc[i]
    else:
        A1.iloc[i] = A1.iloc[i-1] + A1.iloc[i]

df3 = np.exp(A1)-1

print('Learning started. It takes while..')
for NUM in range(2, num+1):
    if NUM <= 5:
        learning_rate = 0.00025
        training_epochs = 2000

    elif NUM <= 33:
        learning_rate = 0.0001
        training_epochs = 1500

    elif NUM <= 100:
        learning_rate = 0.0001
        training_epochs = 100

    else:
        learning_rate = 0.0001
        training_epochs = 500
    
    tf.reset_default_graph() # tf variable dissemvles
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
    
    k = df1[df2.columns[:NUM]].iloc[int(len(df2)/2-1)] + 1
    
    logits1 = tf.matmul(X, w1)
    
    hypothesis = tf.nn.relu(logits1)
    loss = hypothesis - Y
    cost = tf.reduce_sum(tf.square(loss))
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_count = len(trainX)
        for epoch in range(training_epochs):
            total_batch = int(train_size / batch_size)
            for start, end in zip(range(0, train_count, batch_size),
                                  range(batch_size, train_count + 1, batch_size)):
                feed_dict = {X: trainX[start:end],
                             Y: trainY[start:end]}
                l, _ = sess.run([cost, train], feed_dict = feed_dict)
                w = sess.run(w1).T.reshape(-1)
            
            hypo = np.dot(w*k/sum(w*k), df1[int(len(x_data)/2):][df2.columns[:NUM]].T)
            actY = np.array(df3[int(len(x_data)/2):]).reshape(-1)
            a = hypo - actY
            pred_l = sum(a**2)
            if epoch == 0:
                st = np.array([1, np.sqrt(pred_l/(int(len(x_data)/2-1)))])
                wst = w
                alpha = a
            else:
                st = np.vstack([st, np.array([epoch + 1, np.sqrt(pred_l/(int(len(x_data)/2-1)))])])
                wst = np.vstack([wst, w])
                alpha = np.vstack([alpha, a])
        
    plt.title("Portfolio Tracking error(# of stock: {})".format(NUM))
    plt.scatter(st[:, 0], st[:, 1], color = 'k', s = 3., label = 'Tracking Error')
    legend = plt.legend(loc = 'upper right')
    plt.show()
    
    if NUM == 2:
        optweight_ini = []
        optweight_fn = []
        opt = np.array([NUM, np.argmin(st[:,1]), min(st[:,1])])
        ALPHA = alpha[np.argmin(st[:,1])]
        optweight_ini.append(wst[np.argmin(st[:, 1])])
        optweight_inter = np.array(wst[np.argmin(st[:, 1])] * k)
        optweight_fn.append(optweight_inter / sum(optweight_inter))        
    else:
        opt = np.vstack([opt, np.array([NUM, np.argmin(st[:,1]), min(st[:,1])])])
        ALPHA = np.vstack([ALPHA, alpha[np.argmin(st[:,1])]])
        optweight_ini.append(wst[np.argmin(st[:, 1])])
        optweight_inter = np.array(wst[np.argmin(st[:, 1])] * k)
        optweight_fn.append(optweight_inter / sum(optweight_inter))


plt.title("Portfolio Tracking error")
plt.scatter(opt[:,0], opt[:,2], color = 'k', s = 3., label = 'Tracking Error')
legend = plt.legend(loc = 'upper right')
plt.show()

pd.DataFrame(optweight_fn, columns = df2.columns[:len(opt)+1], index = np.arange(2, len(opt)+2)).to_excel('Optweight.xlsx')
pd.DataFrame(opt[:, 1:], columns = ['argmin(T_E)', 'T_E'], index = np.arange(2, len(opt)+2)).to_excel('T_E.xlsx')
pd.DataFrame(ALPHA, columns = df2.index[int(len(df2)/2):], index = np.arange(2, len(opt)+2)).to_excel('ALPHA.xlsx')


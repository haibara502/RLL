import pickle
import pandas as pd
from scipy import sparse
import collections
import random
import time
import numpy as np
import os
import logging
import tensorflow as tf
import sys
from tensorflow.core.protobuf import saver_pb2
import tensorflow.contrib.layers as layers


import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def inferWeight(data, alpha=None, beta=None):    
    votes = data[:,1]
    maxVote = max(votes)
    weights = []
    for i in range(votes.shape[0]):
        v = votes[i]
        if(v>=(1+maxVote)/2):
            if(alpha==None or beta==None):
                weights.append(float(v/maxVote))
            else:
                weights.append(float((v+alpha)/(maxVote+alpha+beta)))
        else:
            if(alpha==None or beta==None):
                weights.append(1-float(v/maxVote))
            else:
                weights.append(float((maxVote-v+alpha)/(maxVote+alpha+beta)))
    data[:,1] = weights
    return data

def splitFeatureWeight(x):
    return x[:,1:], x[:,0]

def createGroupsRandom(data, groupSize=int(4e5)):
    positive = data[np.where(data[:,0]==1)]
    negative = data[np.where(data[:,0]==0)]
    posNum = positive.shape[0]
    negNum = negative.shape[0]
    idx = np.random.randint(low=0, high=posNum, size=groupSize)
    query = np.array([positive[i,1:] for i in idx])
    posDoc = shuffle(query)
    idx = np.random.randint(low=0, high=negNum, size=groupSize)
    negDoc0 = np.array([negative[i,1:] for i in idx])
    negDoc1 = shuffle(negDoc0)
    negDoc2 = shuffle(negDoc0)
    
    query, _ = splitFeatureWeight(query)
    posDoc, posDocW = splitFeatureWeight(posDoc)
    negDoc0, negDoc0W = splitFeatureWeight(negDoc0)
    negDoc1, negDoc1W = splitFeatureWeight(negDoc1)
    negDoc2, negDoc2W = splitFeatureWeight(negDoc2)
    
    groups = (query, posDoc, negDoc0, negDoc1, negDoc2)
    weights = (posDocW, negDoc0W, negDoc1W, negDoc2W)
    return groups, weights



'''
This code implements the RLL framework and its variants
paper published on "Learning Effective Embeddings From Crowdsourced Labels: An Educational Case Study", ICDE 2019
'''

'''
Parameters

bs: batch size
lr_rate: learning rate
l1_n: number of neurons in the first layer
l2_n: number of neurons in the second layer
max_iter: max interation number for training
reg_scale: regularization penalty
dropout_rate: ratio to drop neurons in each layer
gamma: a held-out hyperparameter in loss function, we set it to 10.0 in our experiments
save_path: where you save the model
model_name: you can name your model however you like
'''


def helper(df, groupSize=int(4e5)):
    tmp= [df.columns[1]] + [df.columns[0]] +list(df.columns[2:])
    df = df[tmp]
    data = np.array(df)
    data = inferWeight(data)
    groups, weights = createGroupsRandom(data, groupSize)
    return groups, weights


#compute cosine distance between two vectors
def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1) 
    cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=1)
    return cos_similarity
    
class RLL(object):
    def __init__(self, dimension, l1_n, l2_n, gamma):
        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
    
    def feedBatch(self, groups, weights, batchSize, is_training):
        batchIndex = np.random.randint(low=0, high=groups[0].shape[0], size=batchSize)
        batchGroups = [groups[i][batchIndex] for i in range(len(groups))]
        batchWeights = [weights[i][batchIndex] for i in range(len(weights))]
        batchData = {
                            self.is_training: is_training,
                            self.query: batchGroups[0], 
                            self.posDoc : batchGroups[1], 
                            self.negDoc0 :batchGroups[2], 
                            self.negDoc1 : batchGroups[3], 
                            self.negDoc2: batchGroups[4],
                            self.posDocW: batchWeights[0].reshape(-1, ),    
                            self.negDoc0W: batchWeights[1].reshape(-1, ),
                            self.negDoc1W: batchWeights[2].reshape(-1, ),
                            self.negDoc2W: batchWeights[3].reshape(-1,) 
            }
        return batchData
    
    
    def buildRLL(self, lr_rate, max_iter, reg_scale, dropout_rate):
        tf.reset_default_graph()

        self.is_training = tf.placeholder_with_default(False, shape=(), name='isTraining')
        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension], name='queryInput')
        self.posDoc = tf.placeholder(tf.float32, shape=[None, self.dimension], name='posDocInput')
        self.negDoc0 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc0Input')
        self.negDoc1 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc1Input')
        self.negDoc2 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc2Input')
        self.posDocW = tf.placeholder(tf.float32, shape=[None], name='posDocWeight')
        self.negDoc0W = tf.placeholder(tf.float32, shape=[None], name='negDoc0Weight')
        self.negDoc1W = tf.placeholder(tf.float32, shape=[None], name='negDoc1Weight')
        self.negDoc2W = tf.placeholder(tf.float32, shape=[None], name='negDoc2Weight')

        with tf.name_scope('fc_l1_query'):
            outputQuery = tf.contrib.layers.fully_connected(self.query, self.l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                 activation_fn = tf.nn.sigmoid, scope='fc_l1_query')
            outputQuery = tf.layers.dropout(outputQuery, dropout_rate, training=self.is_training)
        with tf.name_scope('fc_l1_doc'):
            outputPosDoc =  tf.contrib.layers.fully_connected(self.posDoc, self.l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                activation_fn = tf.nn.sigmoid, scope='fc_l1_doc')
            outputPosDoc = tf.layers.dropout(outputPosDoc, dropout_rate, training=self.is_training)

            outputNegDoc0 =  tf.contrib.layers.fully_connected(self.negDoc0, self.l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                 activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
            outputNegDoc0=tf.layers.dropout(outputNegDoc0, dropout_rate, training=self.is_training)

            outputNegDoc1 =  tf.contrib.layers.fully_connected(self.negDoc1, self.l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                  activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
            outputNegDoc1 = tf.layers.dropout(outputNegDoc1, dropout_rate, training=self.is_training)

            outputNegDoc2 =  tf.contrib.layers.fully_connected(self.negDoc2, self.l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                  activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)

            outputNegDoc2=tf.layers.dropout(outputNegDoc2, dropout_rate, training=self.is_training)


        with tf.name_scope('fc_l2_query'):
            outputQuery = tf.contrib.layers.fully_connected(outputQuery, self.l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l2_query')
            outputQuery = tf.layers.dropout(outputQuery, dropout_rate, training=self.is_training)
        with tf.name_scope('fc_l2_doc'):
            outputPosDoc = tf.contrib.layers.fully_connected(outputPosDoc, self.l2_n, weights_regularizer = layers.l2_regularizer(reg_scale), 
                                                               activation_fn = tf.nn.sigmoid, scope='fc_l2_doc')
            outputPosDoc = tf.layers.dropout(outputPosDoc, dropout_rate, training=self.is_training)

            outputNegDoc0 = tf.contrib.layers.fully_connected(outputNegDoc0, self.l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
            outputNegDoc0 = tf.layers.dropout(outputNegDoc0, dropout_rate, training=self.is_training)

            outputNegDoc1 = tf.contrib.layers.fully_connected(outputNegDoc1, self.l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
            outputNegDoc1 = tf.layers.dropout(outputNegDoc1, dropout_rate, training=self.is_training)

            outputNegDoc2 = tf.contrib.layers.fully_connected(outputNegDoc2, self.l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                                activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)

            outputNegDoc2 = tf.layers.dropout(outputNegDoc2, dropout_rate, training=self.is_training)

        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_query')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_doc')
            reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_query')
            reg_ws_3 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_doc')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)+tf.reduce_sum(reg_ws_2)+tf.reduce_sum(reg_ws_3)

            nominator = tf.multiply(self.posDocW, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputPosDoc))))
            doc0_similarity = tf.multiply(self.negDoc0W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc0))))
            doc1_similarity = tf.multiply(self.negDoc1W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc1))))
            doc2_similarity = tf.multiply(self.negDoc2W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc2))))
            prob = tf.add(nominator,tf.constant(1e-10))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator,tf.constant(1e-10))
            log_prob = tf.log(prob)
            self.loss = -tf.reduce_sum(log_prob) + reg_loss

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdadeltaOptimizer(lr_rate).minimize(self.loss)
        
    def train(self, groupsTr, weightsTr, groupsVal, weightsVal, batchSize):
        train_size = groupsTr[0].shape[0]
        print('training group size is {}'.format(train_size))
        val_size = groupsVal[0].shape[0]
        print('validation group size is {}'.format(val_size))

        best_val_loss = sys.maxsize
        num_batch = train_size//batchSize
        earlyStopCount = 0
        saver = tf.train.Saver(max_to_keep=1)
        model_name = 'RLL_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}'.format(self.l1_n, self.l2_n, lr_rate, reg_scale, batchSize)
        
        currentModelPath = os.path.join(save_path, model_name)
        if(not os.path.exists(currentModelPath)):
            os.makedirs(currentModelPath)
        
        start = time.time()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(0, max_iter):
                total_loss = 0
                for batch in range(num_batch):
                    batchData = self.feedBatch(groupsTr, weightsTr, batchSize, is_training=True)
                    _, batch_loss = sess.run([self.optimizer, self.loss], feed_dict=batchData)
                    total_loss += batch_loss
                print("Epoch {} train loss {}".format(epoch, total_loss/train_size))
                if(epoch%10==0):
                    valData = self.feedBatch(groupsVal, weightsVal, groupsVal[0].shape[0], is_training=False)
                    valLoss = sess.run(self.loss, feed_dict=valData)
                    print('*'*66)
                    print("Epoch {} validation loss {}".format(epoch, valLoss/val_size))
                    print('\n')
                    if(valLoss<best_val_loss):
                        best_val_loss = valLoss
                        earlyStopCount = 0
                        saver.save(sess, os.path.join(currentModelPath, model_name+'.ckpt'))
                    else:
                        earlyStopCount += 1
                if(earlyStopCount>=5):
                    print('Early stop at epoch {}!'.format(epoch))
                    break



train = pd.read_csv('../raw_data/train.csv').drop(columns=['id'])
validation = pd.read_csv('../raw_data/validation.csv').drop(columns=['id'])

groupsTr, weightsTr = helper(train)
groupsVal, weightsVal = helper(validation, groupSize=int(1e5))




dropout_rate = 0.5
dimension = 50
gamma = 10.0
max_iter = 888
lr_rate = 0.05

l1_n_lst = [512, 256, 128, 64]
l2_n_lst = [256, 128, 64, 32]
#reg_scale_lst = [1.0, 5.0, 10.0]
# l1_n_lst = [1024, 512]
# l2_n_lst = [512, 256, 128, 64, 32]
reg_scale_lst = [0.1, 1.0, 5.0, 10.0]
batchSize_lst = [1024, 512, 256, 128]
dropout_rate_lst = [0.3, 0.5]

for dropout_rate in dropout_rate_lst:
    for l1_n in l1_n_lst:
        for l2_n in l2_n_lst:
            for reg_scale in reg_scale_lst:
                for batchSize in batchSize_lst:
                    try:
                        if(l1_n<l2_n):
                            continue
                        model = RLL(dimension, l1_n, l2_n, gamma)
                        save_path = '/workspace/Guowei/rll/model'
                        model.buildRLL(lr_rate, max_iter, reg_scale, dropout_rate)
                        model.train(groupsTr, weightsTr, groupsVal, weightsVal, batchSize)
                    except Exception as e:
                        print(e)











import zmail
def send_alert_email(subject, content_text, address):
    user = 'send@tabchen.com'
    passwd = 'Hello2Me.com'
    server = zmail.server(
    user,
    passwd,
    smtp_host='smtp.exmail.qq.com',
    smtp_port=465,
    smtp_ssl=True,
    pop_host='pop.exmail.qq.com',
    pop_port=995)
    mail = {
    'subject': subject,  # Anything you want.
    'content_text': content_text,  # Anything you want.
    }
    return server.send_mail(address, mail)

send_alert_email('Task-training RLL', 'jobs done!', 'emmitt@live.cn')

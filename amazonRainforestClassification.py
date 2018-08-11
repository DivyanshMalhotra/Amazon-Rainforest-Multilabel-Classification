
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


import random as rand
import cv2 as cv


# In[3]:


import pandas as pd
import random
data=np.asarray(pd.read_csv('C:/Users/Divyansh/Desktop/data/train_v2.csv/train_v2.csv',header=0))
trainy=data[:,1]
#print(trainy)
check=[]
intermid=[]
trainie=[]
file1=[]
indused=[]
mhot=np.zeros(17)
for line in trainy:
    mhot=np.zeros(17)
    inter=line.split(" ")
    for interm in inter:
        if interm not in check:
            check.append(interm)
        #print(check)
        mhot[check.index(interm)]=1
    trainie.append(mhot)
#print(trainie[0:100])
train=np.reshape(trainie,(-1,17))
inc=np.sum(train,axis=0)
print(inc)
inter=inc
size=0
print("length initial",len(file1))
while(len(file1)<7500):
    for val in indused:
        inter[val]=40000
    ind=np.argmin(inter)
    indused.append(ind)
    for index,test in enumerate(train):
        #print(test)
        #print("outer")
        if(len(file1)==7500):
            break
        if(test[ind] == 1):
            if data[index][0] not in file1:
          #      print("further in")
                file1.append(data[index][0])
                intermid.append(train[index])
    inter=np.sum(intermid,axis=0)
print(inter)
c = list(zip(file1, intermid))
random.shuffle(c)
file1,intermid = zip(*c)
total_y=np.reshape(intermid,(-1,17))
ind1=len(file1)*80//100
ind2=len(file1)*90//100
file=file1[0:ind1]
train_y=total_y[0:ind1,]
valfile=file1[ind1:ind2]
val_y=total_y[ind1:ind2,]
testfile=file1[ind2:]
test_y=total_y[ind2:,]
acc_y=total_y[0:ind2,]
# print(len(file))
print(len(file1))
print(total_y.shape)
print(check)
# print(train_y.shape)


# In[4]:


start="C:/Users/Divyansh/Desktop/data/train-jpg/train-jpg/train-jpg/"
imag_list=[]
val_list=[]
test_list=[]
for i in file:
    imag_list.append(start+i+".jpg")
for i in valfile:
    val_list.append(start+i+".jpg")
for i in testfile:
    test_list.append(start+i+".jpg")
#check=cv.imread(imag_list[6000])
#print(check.shape)
#cv.imwrite("check.jpg",check)


# In[5]:


x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 17])
batch_size=64


# In[6]:


def evaluate(y,threshold):
    sub=tf.subtract(y,threshold)
#     shape=y.shape
#     print(shape)
#     for i in range (batch_size):
#         for j in range (shape[1]):
#             if(y[i][j]>=threshold):
#                 y[i][j]=1
#             else:
#                 y[i][j]=0
    hotpred=tf.ceil(sub)
    return hotpred


# In[7]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[8]:


def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[9]:


W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])


# In[10]:


h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[11]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[12]:


W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.tanh(conv2d(h_pool2, W_conv3) + b_conv3)


# In[13]:


W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv3, [-1, 8*8*64])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[14]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[15]:


W_fc2 = weight_variable([1024, 17])
b_fc2 = bias_variable([17])

y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# In[16]:


cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.MomentumOptimizer(1e-4,1e-5).minimize(cross_entropy)
sharingan=tf.placeholder(tf.float32,shape=[17])
correct_prediction = tf.equal(evaluate(y_conv, sharingan), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[18]:


batch_size=64
outTot=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20): #20
        #print("No. of epoch: %d"%i)
        ind=0
        steps=(len(imag_list) + batch_size) // batch_size
        #print("Steps:%d"%steps)
        for ind in range(steps):
            batch=[]
            #label=[]
            start=ind*(batch_size)
            end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
            if(end>len(imag_list)):
                end=len(imag_list)
            for ix in range(start,end):
                if ix == start:
                    batch=cv.resize(cv.imread(imag_list[ix]),(32,32))
                else:
                    batch=np.append(batch,cv.resize(cv.imread(imag_list[ix]),(32,32)),axis=0)
                
            batch1=np.reshape(batch,(-1,32,32,3))
            label1=train_y[start:end,]
            #print(batch1.shape)
            #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
            train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.7})
            #train_cost = cross_entropy.eval(feed_dict={x: batch1, y_: label1, keep_prob: 1.0})
            #print('step %d, training accuracy %g' % (ind, train_cost))
        train_cost=0
        calc=0
        for ind in range(steps):
            batch=[]
            #label=[]
            start=ind*(batch_size)
            end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
            if(end>len(imag_list)):
                end=len(imag_list)
            for ix in range(start,end):
                if ix == start:
                    batch=cv.resize(cv.imread(imag_list[ix]),(32,32))
                else:
                    batch=np.append(batch,cv.resize(cv.imread(imag_list[ix]),(32,32)),axis=0)
                
            batch1=np.reshape(batch,(-1,32,32,3))
            label1=train_y[start:end,]
            #print(batch1.shape)
            #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
            #train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.5})
            train_cost = train_cost+cross_entropy.eval(feed_dict={x: batch1, y_: label1, keep_prob: 1.0})
            #outTot=np.append(outTot,y_conv.eval(feed_dict={x: batch1, y_: label1, keep_prob: 1.0}))
            #print(outTot)
        #print(len(outTot))
        #output=np.reshape(outTot,(-1,17))
        #print(output.shape)
        print('epoch %d, training cost %g' % (i, train_cost/steps))
    for ind in range(steps):
        batch=[]
            #label=[]
        start=ind*(batch_size)
        end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
        if(end>len(imag_list)):
            end=len(imag_list)
        for ix in range(start,end):
            if ix == start:
                batch=cv.resize(cv.imread(imag_list[ix]),(32,32))
            else:
                batch=np.append(batch,cv.resize(cv.imread(imag_list[ix]),(32,32)),axis=0)

        batch1=np.reshape(batch,(-1,32,32,3))
        #label1=train_y[start:end,]
        #print(batch1.shape)
        #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
        #train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.5})
        #train_cost = train_cost+cross_entropy.eval(feed_dict={x: batch1, y_: label1, keep_prob: 1.0})
        outTot=np.append(outTot,y_conv.eval(feed_dict={x: batch1, keep_prob: 1.0}))
    for ind in range((len(val_list)+batch_size)//batch_size):
        batch=[]
            #label=[]
        start=ind*(batch_size)
        end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
        if(end>len(val_list)):
            end=len(val_list)
        for ix in range(start,end):
            if ix == start:
                batch=cv.resize(cv.imread(val_list[ix]),(32,32))
            else:
                batch=np.append(batch,cv.resize(cv.imread(val_list[ix]),(32,32)),axis=0)

        batch1=np.reshape(batch,(-1,32,32,3))
        #label1=train_y[start:end,]
        #print(batch1.shape)
        #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
        #train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.5})
        #train_cost = train_cost+cross_entropy.eval(feed_dict={x: batch1, y_: label1, keep_prob: 1.0})
        outTot=np.append(outTot,y_conv.eval(feed_dict={x: batch1,keep_prob: 1.0}))
    output=np.reshape(outTot,(-1,17))
    #print(output)
    threshold=np.zeros(17)
    for clas in range(0,17):
        acc=0
        for thresh1 in range(1,10):
            thresh=thresh1/10
            sub=np.subtract(output[:,clas],thresh)
            mhotout=np.ceil(sub)
            correct_prediction = np.equal(mhotout, acc_y[:,clas])
            accurate = np.mean(correct_prediction)
            if(accurate>acc):
                acc=accurate
                threshold[clas]=thresh
#         print('Class'+str(clas))
#         print(sub)
#         print(mhotout)
    val_acc=0
    #val_acc2=0
#    chec=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    for ind in range((len(val_list)+batch_size)//batch_size):
        batch=[]
            #label=[]
        start=ind*(batch_size)
        end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
        if(end>len(val_list)):
            end=len(val_list)
        for ix in range(start,end):
            if ix == start:
                batch=cv.resize(cv.imread(val_list[ix]),(32,32))
            else:
                batch=np.append(batch,cv.resize(cv.imread(val_list[ix]),(32,32)),axis=0)

        batch1=np.reshape(batch,(-1,32,32,3))
        label1=val_y[start:end,]
        #print(batch1.shape)
        #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
        #train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.5})
        val_acc = val_acc+accuracy.eval(feed_dict={x: batch1, y_: label1,sharingan:threshold ,keep_prob: 1.0})
 #       val_acc2 = val_acc2+accuracy.eval(feed_dict={x: batch1, y_: label1,sharingan:chec ,keep_prob: 1.0})
    print("Validation Accuracy :")
    print(val_acc/((len(val_list)+batch_size)//batch_size))
#    print(val_acc2/((len(val_list)+batch_size)//batch_size))
    print(threshold)
    test_acc=0
#    val_acc2=0
#    chec=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    for ind in range((len(test_list)+batch_size)//batch_size):
        batch=[]
            #label=[]
        start=ind*(batch_size)
        end=(ind+1)*(batch_size)
            #print("Start index:%d"%start)
            #print("End index:%d"%end)
        if(end>len(test_list)):
            end=len(test_list)
        for ix in range(start,end):
            if ix == start:
                batch=cv.resize(cv.imread(test_list[ix]),(32,32))
            else:
                batch=np.append(batch,cv.resize(cv.imread(test_list[ix]),(32,32)),axis=0)

        batch1=np.reshape(batch,(-1,32,32,3))
        label1=test_y[start:end,]
        #print(batch1.shape)
        #print(label1.shape)
#             cv.imwrite('dogCat.jpg',batch1[0])
        #train_step.run(feed_dict={x: batch1, y_: label1, keep_prob: 0.5})
        test_acc = test_acc+accuracy.eval(feed_dict={x: batch1, y_: label1,sharingan:threshold ,keep_prob: 1.0})
 #       val_acc2 = val_acc2+accuracy.eval(feed_dict={x: batch1, y_: label1,sharingan:chec ,keep_prob: 1.0})
    print("Testing Accuracy :")
    print(test_acc/((len(test_list)+batch_size)//batch_size))


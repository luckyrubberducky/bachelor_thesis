# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:35:19 2019

@author: zeno
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(784,)),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
    return model

def create_model_con():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
    return model
#load same labeled and unlabeled data as was used during training of the generative model
x_mul = np.load('x_mul.npy')
y_mul_bin = np.load('y_mul.npy')

x_gen2 = np.load('x_gen2.npy')
y_gen2_bin = np.load('y_gen2.npy')
x_gen3 = np.load('x_gen3.npy')
y_gen3_bin = np.load('y_gen3.npy')

x_ran1 = np.load('x_ran1.npy')
y_ran1_bin = np.load('y_ran1.npy')
x_ran2 = np.load('x_ran2.npy')
y_ran2_bin = np.load('y_ran2.npy')
x_ran3 = np.load('x_ran3.npy')
y_ran3_bin = np.load('y_ran3.npy')

x_ran1_mul = np.load('x_ran1_mul.npy')
y_ran1_mul_bin = np.load('y_ran1_mul.npy')
x_ran2_mul = np.load('x_ran2_mul.npy')
y_ran2_mul_bin = np.load('y_ran2_mul.npy')
x_ran3_mul = np.load('x_ran3_mul.npy')
y_ran3_mul_bin = np.load('y_ran3_mul.npy')

x_per1 = np.load('x_per1.npy')
y_per1_bin = np.load('y_per1.npy')
x_per2 = np.load('x_per2.npy')
y_per2_bin = np.load('y_per2.npy')
x_per3 = np.load('x_per3.npy')
y_per3_bin = np.load('y_per3.npy')

x_per1_mul = np.load('x_per1_mul.npy')
y_per1_mul_bin = np.load('y_per1_mul.npy')
x_per2_mul = np.load('x_per2_mul.npy')
y_per2_mul_bin = np.load('y_per2_mul.npy')
x_per3_mul = np.load('x_per3_mul.npy')
y_per3_mul_bin = np.load('y_per3_mul.npy')

x_avg1 = np.load('x_avg1.npy')
y_avg1_bin = np.load('y_avg1.npy')
x_avg2 = np.load('x_avg2.npy')
y_avg2_bin = np.load('y_avg2.npy')
x_avg3 = np.load('x_avg3.npy')
y_avg3_bin = np.load('y_avg3.npy')
x_avg4 = np.load('x_avg4.npy')
y_avg4_bin = np.load('y_avg4.npy')

x_avg1_mul = np.load('x_avg1_mul.npy')
y_avg1_mul_bin = np.load('y_avg1_mul.npy')
x_avg2_mul = np.load('x_avg2_mul.npy')
y_avg2_mul_bin = np.load('y_avg2_mul.npy')
x_avg3_mul = np.load('x_avg3_mul.npy')
y_avg3_mul_bin = np.load('y_avg3_mul.npy')
x_avg4_mul = np.load('x_avg4_mul.npy')
y_avg4_mul_bin = np.load('y_avg4_mul.npy')

x_lab = np.load('final_x_lab.npy')
y_lab_bin = np.load('final_y_lab.npy')
x_ulab = np.load('final_x_ulab.npy')
y_ulab_bin = np.load('final_y_ulab.npy')
x_valid = np.load('final_x_valid.npy')
y_valid_bin = np.load('final_y_valid.npy')
x_test = np.load('final_x_test.npy')
y_test_bin = np.load('final_y_test.npy')
#un-binarize the label data
y_mul = np.argmax(y_mul_bin, axis=1)
y_gen2 = np.argmax(y_gen2_bin, axis=1)
y_gen3 = np.argmax(y_gen3_bin, axis=1)

y_ran1 = np.argmax(y_ran1_bin, axis=1)
y_ran2 = np.argmax(y_ran2_bin, axis=1)
y_ran3 = np.argmax(y_ran3_bin, axis=1)

y_ran1_mul = np.argmax(y_ran1_mul_bin, axis=1)
y_ran2_mul = np.argmax(y_ran2_mul_bin, axis=1)
y_ran3_mul = np.argmax(y_ran3_mul_bin, axis=1)


y_per1 = np.argmax(y_per1_bin, axis=1)
y_per2 = np.argmax(y_per2_bin, axis=1)
y_per3 = np.argmax(y_per3_bin, axis=1)

y_per1_mul = np.argmax(y_per1_mul_bin, axis=1)
y_per2_mul = np.argmax(y_per2_mul_bin, axis=1)
y_per3_mul = np.argmax(y_per3_mul_bin, axis=1)

y_avg1 = np.argmax(y_avg1_bin, axis=1)
y_avg2 = np.argmax(y_avg2_bin, axis=1)
y_avg3 = np.argmax(y_avg3_bin, axis=1)
y_avg4 = np.argmax(y_avg4_bin, axis=1)

y_avg1_mul = np.argmax(y_avg1_mul_bin, axis=1)
y_avg2_mul = np.argmax(y_avg2_mul_bin, axis=1)
y_avg3_mul = np.argmax(y_avg3_mul_bin, axis=1)
y_avg4_mul = np.argmax(y_avg4_mul_bin, axis=1)


y_lab = np.argmax(y_lab_bin, axis=1)
y_ulab = np.argmax(y_ulab_bin, axis=1)
y_test = np.argmax(y_test_bin, axis=1)
y_valid = np.argmax(y_valid_bin, axis=1)



s = np.arange(x_lab.shape[0])
np.random.shuffle(s)
x_lab = x_lab[s]
y_lab = y_lab[s]
s = np.arange(x_ulab.shape[0])
np.random.shuffle(s)
x_ulab = x_ulab[s]
y_ulab = y_ulab[s]



n_lab1 = 5000
n_lab2 = 25000
n_lab3 = 50000
x_lab1 = np.ndarray((n_lab1, x_lab.shape[1]))
y_lab1 = np.ndarray((n_lab1))
x_lab2 = np.ndarray((n_lab2, x_lab.shape[1]))
y_lab2 = np.ndarray((n_lab2))
x_lab3 = np.ndarray((n_lab3, x_lab.shape[1]))
y_lab3 = np.ndarray((n_lab3))

x_lab1[0:100, :] = x_lab[:,:]
x_lab1[100:n_lab1, :] = x_ulab[0:n_lab1-100, :]
y_lab1[0:100] = y_lab
y_lab1[100:n_lab1] = y_ulab[0:n_lab1-100]
x_lab2[0:100, :] = x_lab[:,:]
x_lab2[100:n_lab2, :] = x_ulab[0:n_lab2-100, :]
y_lab2[0:100] = y_lab
y_lab2[100:n_lab2] = y_ulab[0:n_lab2-100]
x_lab3[0:100, :] = x_lab[:,:]
x_lab3[100:n_lab3, :] = x_ulab[0:n_lab3-100, :]
y_lab3[0:100] = y_lab
y_lab3[100:n_lab3] = y_ulab[0:n_lab3-100]
#shuffle data
s = np.arange(x_mul.shape[0])
np.random.shuffle(s)
x_mul = x_mul[s]
y_mul = y_mul[s]
s = np.arange(x_gen2.shape[0])
np.random.shuffle(s)
x_gen2 = x_gen2[s]
y_gen2 = y_gen2[s]
s = np.arange(x_gen3.shape[0])
np.random.shuffle(s)
x_gen3 = x_gen3[s]
y_gen3 = y_gen3[s]


s = np.arange(x_ran1.shape[0])
np.random.shuffle(s)
x_ran1 = x_ran1[s]
y_ran1 = y_ran1[s]
s = np.arange(x_ran2.shape[0])
np.random.shuffle(s)
x_ran2 = x_ran2[s]
y_ran2 = y_ran2[s]
s = np.arange(x_ran3.shape[0])
np.random.shuffle(s)
x_ran3 = x_ran3[s]
y_ran3 = y_ran3[s]

s = np.arange(x_ran1_mul.shape[0])
np.random.shuffle(s)
x_ran1_mul = x_ran1_mul[s]
y_ran1_mul = y_ran1_mul[s]
s = np.arange(x_ran2_mul.shape[0])
np.random.shuffle(s)
x_ran2_mul = x_ran2_mul[s]
y_ran2_mul = y_ran2_mul[s]
s = np.arange(x_ran3_mul.shape[0])
np.random.shuffle(s)
x_ran3_mul = x_ran3_mul[s]
y_ran3_mul = y_ran3_mul[s]


s = np.arange(x_per1.shape[0])
np.random.shuffle(s)
x_per1 = x_per1[s]
y_per1 = y_per1[s]
s = np.arange(x_per2.shape[0])
np.random.shuffle(s)
x_per2 = x_per2[s]
y_per2 = y_per2[s]
s = np.arange(x_per3.shape[0])
np.random.shuffle(s)
x_per3 = x_per3[s]
y_per3 = y_per3[s]

s = np.arange(x_per1_mul.shape[0])
np.random.shuffle(s)
x_per1_mul = x_per1_mul[s]
y_per1_mul = y_per1_mul[s]
s = np.arange(x_per2_mul.shape[0])
np.random.shuffle(s)
x_per2_mul = x_per2_mul[s]
y_per2_mul = y_per2_mul[s]
s = np.arange(x_per3_mul.shape[0])
np.random.shuffle(s)
x_per3_mul = x_per3_mul[s]
y_per3_mul = y_per3_mul[s]

s = np.arange(x_avg1.shape[0])
np.random.shuffle(s)
x_avg1 = x_avg1[s]
y_avg1 = y_avg1[s]
s = np.arange(x_avg2.shape[0])
np.random.shuffle(s)
x_avg2 = x_avg2[s]
y_avg2 = y_avg2[s]
s = np.arange(x_avg3.shape[0])
np.random.shuffle(s)
x_avg3 = x_avg3[s]
y_avg3 = y_avg3[s]
s = np.arange(x_avg4.shape[0])
np.random.shuffle(s)
x_avg4 = x_avg4[s]
y_avg4 = y_avg4[s]

s = np.arange(x_avg1_mul.shape[0])
np.random.shuffle(s)
x_avg1_mul = x_avg1_mul[s]
y_avg1_mul = y_avg1_mul[s]
s = np.arange(x_avg2_mul.shape[0])
np.random.shuffle(s)
x_avg2_mul = x_avg2_mul[s]
y_avg2_mul = y_avg2_mul[s]
s = np.arange(x_avg3_mul.shape[0])
np.random.shuffle(s)
x_avg3_mul = x_avg3_mul[s]
y_avg3_mul = y_avg3_mul[s]
s = np.arange(x_avg4_mul.shape[0])
np.random.shuffle(s)
x_avg4_mul = x_avg4_mul[s]
y_avg4_mul = y_avg4_mul[s]

s = np.arange(x_lab1.shape[0])
np.random.shuffle(s)
x_lab1 = x_lab1[s]
y_lab1 = y_lab1[s]
s = np.arange(x_lab2.shape[0])
np.random.shuffle(s)
x_lab2 = x_lab2[s]
y_lab2 = y_lab2[s]
s = np.arange(x_lab3.shape[0])
np.random.shuffle(s)
x_lab3 = x_lab3[s]
y_lab3 = y_lab3[s]
s = np.arange(x_valid.shape[0])
np.random.shuffle(s)
x_valid = x_valid[s]
y_valid = y_valid[s]

s = np.arange(x_test.shape[0])
np.random.shuffle(s)
x_test = x_test[s]
y_test = y_test[s]


def test_con(x_train, y_train, x_test, y_test):
    model = create_model_con()
    x_train_2 = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test_2 = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
    model.fit(x_train_2, y_train, epochs=3)
    res = model.evaluate(x_test_2, y_test)
    print('Accuracy:' + str(res[1]))

    
def test_dense(x_train, y_train, x_test, y_test):
    model = create_model()   
    model.fit(x_train, y_train, epochs=3)
    res = model.evaluate(x_test, y_test)
    print('Accuracy:' + str(res[1]))



#model trained on labeled + perturbed images (style perturbation)
#%%
print('per1_mul_dense:')
res_per1_mul = test_dense(x_per1_mul, y_per1_mul, x_test, y_test)
#%%
print('per1_mul_con:')
res_per1_mul_con = test_con(x_per1_mul, y_per1_mul, x_test, y_test)
#%%
print('per2_mul_dense:')
res_per2_mul = test_dense(x_per2_mul, y_per2_mul, x_test, y_test)
#%%
print('per2_mul_con:')
res_per2_mul_con = test_con(x_per2_mul, y_per2_mul, x_test, y_test)
#%%
print('per3_mul_dense:')
res_per3_mul = test_dense(x_per3_mul, y_per3_mul, x_test, y_test)
#%%
print('per3_mul_con:')
res_per3_mul_con = test_con(x_per3_mul, y_per3_mul, x_test, y_test)

#model trained on labeled + perturbed images (style perturbation)
#%%
print('per1_dense:')
res_per1 = test_dense(x_per1, y_per1, x_test, y_test)
#%%
print('per1_con:')
res_per1_con = test_con(x_per1, y_per1, x_test, y_test)
#%%
print('per2_dense:')
res_per2 = test_dense(x_per2, y_per2, x_test, y_test)
#%%
print('per2_con:')
res_per2_con = test_con(x_per2, y_per2, x_test, y_test)
#%%
print('per3_dense:')
res_per3 = test_dense(x_per3, y_per3, x_test, y_test)
#%%
print('per3_con:')
res_per3_con = test_con(x_per3, y_per3, x_test, y_test)


#model trained on labeled + average images (style combination)
#%%
print('avg1_dense:')
res_avg1 = test_dense(x_avg1, y_avg1, x_test, y_test)
#%%
print('avg1_con:')
res_avg1_con = test_con(x_avg1, y_avg1, x_test, y_test)
#%%
print('avg2_dense:')
res_avg2 = test_dense(x_avg2, y_avg2, x_test, y_test)
#%%
print('avg2_con:')
res_avg2_con = test_con(x_avg2, y_avg2, x_test, y_test)
#%%
print('avg3_dense:')
res_avg3 = test_dense(x_avg3, y_avg3, x_test, y_test)

print('avg3_con:')
res_avg3_con = test_con(x_avg3, y_avg3, x_test, y_test)

print('avg4_dense:')
res_avg4 = test_dense(x_avg4, y_avg4, x_test, y_test)

print('avg4_con:')
res_avg4_con = test_con(x_avg4, y_avg4, x_test, y_test)

#model trained on labeled + perturbed images (style perturbation)
#%%
print('avg1_mul_dense:')
res_avg1_mul = test_dense(x_avg1_mul, y_avg1_mul, x_test, y_test)
#%%
print('avg1_mul_con:')
res_avg1_mul_con = test_con(x_avg1_mul, y_avg1_mul, x_test, y_test)
#%%
print('avg2_mul_dense:')
res_avg2_mul = test_dense(x_avg2_mul, y_avg2_mul, x_test, y_test)
#%%
print('avg2_mul_con:')
res_avg2_mul_con = test_con(x_avg2_mul, y_avg2_mul, x_test, y_test)
#%%
print('avg3_mul_dense:')
res_avg3_mul = test_dense(x_avg3_mul, y_avg3_mul, x_test, y_test)
#%%
print('avg3_mul_con:')
res_avg3_mul_con = test_con(x_avg3_mul, y_avg3_mul, x_test, y_test)
#%%
print('avg4_mul_dense:')
res_avg4_mul = test_dense(x_avg4_mul, y_avg4_mul, x_test, y_test)
#%%
print('avg4_mul_con:')
res_avg4_mul_con = test_con(x_avg4_mul, y_avg4_mul, x_test, y_test)


#model trained on random sampled images (random sampling)
#%%
print('ran1_dense:')
res_ran1 = test_dense(x_ran1, y_ran1, x_test, y_test)

print('ran1_con:')
res_ran1_con = test_con(x_ran1, y_ran1, x_test, y_test)
#%%
print('ran2_dense:')
res_ran2 = test_dense(x_ran2, y_ran2, x_test, y_test)

print('ran2_con:')
res_ran2_con = test_con(x_ran2, y_ran2, x_test, y_test)
#%%
print('ran3_dense:')
res_ran3 = test_dense(x_ran3, y_ran3, x_test, y_test)
#%%
print('ran3_con:')
res_ran3_con = test_con(x_ran3, y_ran3, x_test, y_test)

#%%
print('ran1_mul_dense:')
res_ran1_mul = test_dense(x_ran1_mul, y_ran1_mul, x_test, y_test)
#%%
print('ran1_mul_con:')
res_ran1_mul_con = test_con(x_ran1_mul, y_ran1_mul, x_test, y_test)
#%%
print('ran2_mul_dense:')
res_ran2_mul = test_dense(x_ran2_mul, y_ran2_mul, x_test, y_test)

print('ran2_mul_con:')
res_ran2_mul_con = test_con(x_ran2_mul, y_ran2_mul, x_test, y_test)
#%%
print('ran3_mul_dense:')
res_ran3_mul = test_dense(x_ran3_mul, y_ran3_mul, x_test, y_test)
#%%
print('ran3_mul_con:')
res_ran3_mul_con = test_con(x_ran3_mul, y_ran3_mul, x_test, y_test)


#model trained on labeled + generated images (digit multiplication)


#%%
print('x_mul_dense:')
res_x_mul = test_dense(x_mul, y_mul, x_test, y_test)
#%%
print('x_mul_con:')
res_x_mul_con = test_con(x_mul, y_mul, x_test, y_test)
#%%
print('x_lab_dense:')
res_x_lab = test_dense(x_lab, y_lab, x_test, y_test)
#%%
print('x_lab_con:')
res_x_lab_con = test_con(x_lab, y_lab, x_test, y_test)
#%%
print('x_lab1_dense:')
res_x_lab1 = test_dense(x_lab1, y_lab1, x_test, y_test)
#%%
print('x_lab1_con:')
res_x_lab1_con = test_con(x_lab1, y_lab1, x_test, y_test)
#%%
print('x_lab2_dense:')
res_x_lab2 = test_dense(x_lab2, y_lab2, x_test, y_test)
#%%
print('x_lab2_con:')
res_x_lab2_con = test_con(x_lab2, y_lab2, x_test, y_test)
#%%
print('x_lab3_dense:')
res_x_lab3 = test_dense(x_lab3, y_lab3, x_test, y_test)

print('x_lab3_con:')
res_x_lab3_con = test_con(x_lab3, y_lab3, x_test, y_test)
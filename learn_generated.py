# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:28:51 2019

@author: zeno
"""
from genclass import GenerativeClassifier
from vae import VariationalAutoencoder
import numpy as np
import data.mnist as mnist
import tensorflow as tf

def encode_dataset( model_path, min_std = 0.0 ):

    VAE = VariationalAutoencoder( dim_x = 28*28, dim_z = 50 ) #Should be consistent with model being loaded
    with VAE.session:
        VAE.saver.restore( VAE.session, VAE_model_path )

        enc_x_lab_mean, enc_x_lab_var = VAE.encode( x_lab )
        enc_x_ulab_mean, enc_x_ulab_var = VAE.encode( x_ulab )
        enc_x_valid_mean, enc_x_valid_var = VAE.encode( x_valid )
        enc_x_test_mean, enc_x_test_var = VAE.encode( x_test )

        id_x_keep = np.std( enc_x_ulab_mean, axis = 0 ) > min_std

        #enc_x_lab_mean, enc_x_lab_var = enc_x_lab_mean[ :, id_x_keep ], enc_x_lab_var[ :, id_x_keep ]
        #enc_x_ulab_mean, enc_x_ulab_var = enc_x_ulab_mean[ :, id_x_keep ], enc_x_ulab_var[ :, id_x_keep ]
        #enc_x_valid_mean, enc_x_valid_var = enc_x_valid_mean[ :, id_x_keep ], enc_x_valid_var[ :, id_x_keep ]
        #enc_x_test_mean, enc_x_test_var = enc_x_test_mean[ :, id_x_keep ], enc_x_test_var[ :, id_x_keep ]

        data_lab = np.hstack( [ enc_x_lab_mean, enc_x_lab_var ] )
        data_ulab = np.hstack( [ enc_x_ulab_mean, enc_x_ulab_var ] )
        data_valid = np.hstack( [enc_x_valid_mean, enc_x_valid_var] )
        data_test = np.hstack( [enc_x_test_mean, enc_x_test_var] )

    return data_lab, data_ulab, data_valid, data_test

def n_choose_k_index(n, k):
    if k == 1:
        res = np.ndarray((n, 1))
        res[:, 0] = range(n)
        return res
    else:
        a = n_choose_k_index(n, k-1)
        res = np.ndarray((int((n+1-k)/(k)*a.shape[0]), k))
        l = 0
        for i in range(a.shape[0]):
            for j in range(n):
                if j > a[i,k-2]:
                    res[l, 0:k-1] = a[i,:]
                    res[l, k-1] = j
                    l += 1
        return res

#load the same labeled data as was used for training of the GenerativeClassifier
x_lab= np.load('ref_x_lab.npy')
y_lab= np.load('ref_y_lab.npy')
x_ulab= np.load('ref_x_ulab.npy')
y_ulab= np.load('ref_x_ulab.npy')
x_valid= np.load('ref_x_valid.npy')
y_valid= np.load('ref_x_valid.npy')
x_test= np.load('ref_x_test.npy')
y_test= np.load('ref_y_test.npy')


#set all parameters same for restoring of the GenerativeClassifier
VAE_model_path = 'checkpoints/final_model_VAE_0.0003-50_1553994729.7604325.cpkt'
min_std = 0.1 #Dimensions with std < min_std are removed before training with GC

data_lab, data_ulab, data_valid, data_test = encode_dataset( VAE_model_path, min_std )

num_lab = 100           #Number of labelnumpy led examples (total)
num_batches = 100       #Number of minibatches in a single epoch
dim_z = 50              #Dimensionality of latent variable (z)
epochs = 1001           #Number of epochs through the full dataset
learning_rate = 3e-4    #Learning rate of ADAM
alpha = 0.1             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
seed = 31415
dim_x = data_lab.shape[1] / 2
dim_y = y_lab.shape[1]
num_examples = data_lab.shape[0] + data_ulab.shape[0]

z1_lab = data_lab[:,0:int(dim_x)]

#%%
#restore GenerativeClassifier and use it to generate a new training set based on the labeled data
GC = GenerativeClassifier(  dim_x, dim_z, dim_y, num_examples, num_lab, num_batches )

with GC.session:
    GC.saver.restore(GC.session, 'checkpoints/final_model_GC_100-0.0003-500_1554018668.4256456.cpkt')

    #encode all labeled data to get style parameters
    z2_lab_mean, z2_lab_var = GC.encode( z1_lab, y_lab )

    #augment dataset
    n_samples = 4
    perturb_dev = .1
    sample_dev = 1.0
    dim_z2 = z2_lab_mean.shape[1]
    

    #z2_aug2: like z2_aug with added normally randomly distributed points for
    z2_aug2 = np.ndarray((num_lab*10*n_samples, dim_z2))
    y_gen2 = np.ndarray((num_lab*10*n_samples, 10))

    #z2_avg: within each label choose k and compute the average for all possible combinations and add those images as well
    k = 2
    num_class = num_lab//10
    ind = n_choose_k_index(num_class,k)
    z2_avg1 = np.ndarray((num_lab + 10*ind.shape[0], dim_z2))
    y_avg1 = np.ndarray((num_lab + 10*ind.shape[0], 10))
    l = 0
    for i in range(num_class):
        z2_class = z2_lab_mean[i*num_class:(i+1)*num_class, :]
        z2_avg1[l:l+num_class,:] = z2_class
        y_avg1[l:l+num_class,:] = y_lab[i*num_class:(i+1)*num_class, :]
        l += num_class
        for j in range(ind.shape[0]):
            sel = ind[j, :].astype(int)
            tup = z2_class[sel, :]
            avg = np.mean(tup, axis=0)
            z2_avg1[l, :] = avg
            y_avg1[l, :] = y_lab[i*num_class, :]
            l += 1
    del l
    
    k = 3
    num_class = num_lab//10
    ind = n_choose_k_index(num_class,k)
    z2_avg2 = np.ndarray((num_lab + 10*ind.shape[0], dim_z2))
    y_avg2 = np.ndarray((num_lab + 10*ind.shape[0], 10))
    l = 0
    for i in range(num_class):
        z2_class = z2_lab_mean[i*num_class:(i+1)*num_class, :]
        z2_avg2[l:l+num_class,:] = z2_class
        y_avg2[l:l+num_class,:] = y_lab[i*num_class:(i+1)*num_class, :]
        l += num_class
        for j in range(ind.shape[0]):
            sel = ind[j, :].astype(int)
            tup = z2_class[sel, :]
            avg = np.mean(tup, axis=0)
            z2_avg2[l, :] = avg
            y_avg2[l, :] = y_lab[i*num_class, :]
            l += 1
    del l
    
    k = 4
    num_class = num_lab//10
    ind = n_choose_k_index(num_class,k)
    z2_avg3 = np.ndarray((num_lab + 10*ind.shape[0], dim_z2))
    y_avg3 = np.ndarray((num_lab + 10*ind.shape[0], 10))
    l = 0
    for i in range(num_class):
        z2_class = z2_lab_mean[i*num_class:(i+1)*num_class, :]
        z2_avg3[l:l+num_class,:] = z2_class
        y_avg3[l:l+num_class,:] = y_lab[i*num_class:(i+1)*num_class, :]
        l += num_class
        for j in range(ind.shape[0]):
            sel = ind[j, :].astype(int)
            tup = z2_class[sel, :]
            avg = np.mean(tup, axis=0)
            z2_avg3[l, :] = avg
            y_avg3[l, :] = y_lab[i*num_class, :]
            l += 1
    del l
    
    k = 5
    num_class = num_lab//10
    ind = n_choose_k_index(num_class,k)
    z2_avg4 = np.ndarray((num_lab + 10*ind.shape[0], dim_z2))
    y_avg4 = np.ndarray((num_lab + 10*ind.shape[0], 10))
    l = 0
    for i in range(num_class):
        z2_class = z2_lab_mean[i*num_class:(i+1)*num_class, :]
        z2_avg4[l:l+num_class,:] = z2_class
        y_avg4[l:l+num_class,:] = y_lab[i*num_class:(i+1)*num_class, :]
        l += num_class
        for j in range(ind.shape[0]):
            sel = ind[j, :].astype(int)
            tup = z2_class[sel, :]
            avg = np.mean(tup, axis=0)
            z2_avg4[l, :] = avg
            y_avg4[l, :] = y_lab[i*num_class, :]
            l += 1
    del l


    #digit multiplication on z2_avg1
    z2_aug3 = np.ndarray((z2_avg1.shape[0]*10, dim_z2))
    y_gen3 = np.ndarray((y_avg1.shape[0]*10, 10))

    for i in range(z2_avg1.shape[0]):
        for j in range(10):
            z2_aug3[i*10 + j,:] = z2_avg1[i, :]
            y_gen3[i*10 + j, :] = y_lab[j*10,:]


    #digit multiplication z2_mul and random perturbation after digit multiplication
                
                
                
    #random perturbation
    per_samples = 50
    per_dev1 = 0.3
    per_dev2 = 0.6
    per_dev3 = 1.0
    z2_per1 = np.ndarray((per_samples*num_lab + num_lab, dim_z2))
    y_per1 = np.ndarray((per_samples*num_lab + num_lab, 10))
    
    
    z2_per2 = np.ndarray((per_samples*num_lab + num_lab, dim_z2))
    y_per2 = np.ndarray((per_samples*num_lab + num_lab, 10))
    
    
    z2_per3 = np.ndarray((per_samples*num_lab + num_lab, dim_z2))
    y_per3 = np.ndarray((per_samples*num_lab + num_lab, 10))
    
    l = 0
    for i in range(num_lab):
        z2_per1[l, :] = z2_lab_mean[i, :]
        z2_per2[l, :] = z2_lab_mean[i, :]      
        z2_per3[l, :] = z2_lab_mean[i, :]
        y_per1[l, :] = y_lab[i,:]
        y_per2[l, :] = y_lab[i,:]
        y_per3[l, :] = y_lab[i,:]
        
        l += 1
        for j in range(per_samples):
            z2_per1[l, :] = np.random.normal(z2_lab_mean[i, :], per_dev1) 
            z2_per2[l, :] = np.random.normal(z2_lab_mean[i, :], per_dev2)      
            z2_per3[l, :] = np.random.normal(z2_lab_mean[i, :], per_dev3) 
            y_per1[l, :] = y_lab[i,:]
            y_per2[l, :] = y_lab[i,:]
            y_per3[l, :] = y_lab[i,:]
            l += 1

    del l
    
    
    #random sampling
    ran_samples = 5000
    ran_dev1 = 0.8
    ran_dev2 = 1.6
    ran_dev3 = 2.4
    z2_ran1 = np.ndarray((ran_samples, z2_lab_mean.shape[1]))
    y_ran1 = np.ndarray((ran_samples, 10))
    
    
    z2_ran2 = np.ndarray((ran_samples, z2_lab_mean.shape[1]))
    y_ran2 = np.ndarray((ran_samples, 10))
    
    
    z2_ran3 = np.ndarray((ran_samples, z2_lab_mean.shape[1]))
    y_ran3 = np.ndarray((ran_samples, 10))
    
    for i in range(ran_samples):
            z2_ran1[i, :] = np.random.normal(np.zeros((1, dim_z2)), ran_dev1)
            z2_ran2[i, :] = np.random.normal(np.zeros((1, dim_z2)), ran_dev2)
            z2_ran3[i, :] = np.random.normal(np.zeros((1, dim_z2)), ran_dev3)
            y_ran1[i, :] = y_lab[(i*10)%num_lab,:]
            y_ran2[i, :] = y_lab[(i*10)%num_lab,:]
            y_ran3[i, :] = y_lab[(i*10)%num_lab,:]
            
            
    #digit multiplication
    
    
    #z2_mul: for all original datapoints create 10 digits with the same style parameters
    z2_mul = np.ndarray((num_lab*10, dim_z2))
    y_mul = np.ndarray((num_lab*10, 10))
    
    for i in range(num_lab):
        for j in range(10):
            z2_mul[i*10 + j,:] = z2_lab_mean[i, :]
            y_mul[i*10 + j, :] = y_lab[j*10,:]
            for l in range(n_samples):
                dev = np.random.normal(np.zeros((1, dim_z2)), sample_dev)
                z2_aug2[i*10*n_samples + j*n_samples + l, :] = z2_lab_mean[i, :] + dev
                y_gen2[i*10*n_samples + j*n_samples + l, :] = y_lab[j*10,:]
    
    z2_ran1_mul = np.ndarray((z2_ran1.shape[0]*10, dim_z2))
    y_ran1_mul = np.ndarray((z2_ran1.shape[0]*10, 10))
    for i in range(z2_ran1.shape[0]):
        for j in range(10):
            z2_ran1_mul[i*10 + j,:] = z2_ran1[i, :]
            y_ran1_mul[i*10 + j, :] = y_lab[j*10,:]
            
    z2_ran2_mul = np.ndarray((z2_ran2.shape[0]*10, dim_z2))
    y_ran2_mul = np.ndarray((z2_ran2.shape[0]*10, 10))
    for i in range(z2_ran2.shape[0]):
        for j in range(10):
            z2_ran2_mul[i*10 + j,:] = z2_ran2[i, :]
            y_ran2_mul[i*10 + j, :] = y_lab[j*10,:]
            
            
    z2_ran3_mul = np.ndarray((z2_ran3.shape[0]*10, dim_z2))
    y_ran3_mul = np.ndarray((z2_ran3.shape[0]*10, 10))
    for i in range(z2_ran3.shape[0]):
        for j in range(10):
            z2_ran3_mul[i*10 + j,:] = z2_ran3[i, :]
            y_ran3_mul[i*10 + j, :] = y_lab[j*10,:]
            
    
        
    z2_per1_mul = np.ndarray((z2_per1.shape[0]*10, dim_z2))
    y_per1_mul = np.ndarray((z2_per1.shape[0]*10, 10))
    for i in range(z2_per1.shape[0]):
        for j in range(10):
            z2_per1_mul[i*10 + j,:] = z2_per1[i, :]
            y_per1_mul[i*10 + j, :] = y_lab[j*10,:]
            
    z2_per2_mul = np.ndarray((z2_per2.shape[0]*10, dim_z2))
    y_per2_mul = np.ndarray((z2_per2.shape[0]*10, 10))
    for i in range(z2_per2.shape[0]):
        for j in range(10):
            z2_per2_mul[i*10 + j,:] = z2_per2[i, :]
            y_per2_mul[i*10 + j, :] = y_lab[j*10,:]
            
    z2_per3_mul = np.ndarray((z2_per3.shape[0]*10, dim_z2))
    y_per3_mul = np.ndarray((z2_per3.shape[0]*10, 10))
    for i in range(z2_per3.shape[0]):
        for j in range(10):
            z2_per3_mul[i*10 + j,:] = z2_per3[i, :]
            y_per3_mul[i*10 + j, :] = y_lab[j*10,:]
            
    z2_avg1_mul = np.ndarray((z2_avg1.shape[0]*10, dim_z2))
    y_avg1_mul = np.ndarray((z2_avg1.shape[0]*10, 10))
    for i in range(z2_avg1.shape[0]):
        for j in range(10):
            z2_avg1_mul[i*10 + j,:] = z2_avg1[i, :]
            y_avg1_mul[i*10 + j, :] = y_lab[j*10,:]
    
    z2_avg2_mul = np.ndarray((z2_avg2.shape[0]*10, dim_z2))
    y_avg2_mul = np.ndarray((z2_avg2.shape[0]*10, 10))
    for i in range(z2_avg2.shape[0]):
        for j in range(10):
            z2_avg2_mul[i*10 + j,:] = z2_avg2[i, :]
            y_avg2_mul[i*10 + j, :] = y_lab[j*10,:]
        
    z2_avg3_mul = np.ndarray((z2_avg3.shape[0]*10, dim_z2))
    y_avg3_mul = np.ndarray((z2_avg3.shape[0]*10, 10))
    for i in range(z2_avg3.shape[0]):
        for j in range(10):
            z2_avg3_mul[i*10 + j,:] = z2_avg3[i, :]
            y_avg3_mul[i*10 + j, :] = y_lab[j*10,:]
            
    z2_avg4_mul = np.ndarray((z2_avg4.shape[0]*10, dim_z2))
    y_avg4_mul = np.ndarray((z2_avg4.shape[0]*10, 10))
    for i in range(z2_avg4.shape[0]):
        for j in range(10):
            z2_avg4_mul[i*10 + j,:] = z2_avg4[i, :]
            y_avg4_mul[i*10 + j, :] = y_lab[j*10,:]
        
    #decode augmented datasets to M1
    z1_mul_mean, z1_mul_var = GC.decode( z2_mul, y_mul )
    z1_gen2_mean, z1_gen2_var = GC.decode(z2_aug2, y_gen2)
    z1_gen3_mean, z1_gen3_var = GC.decode(z2_aug3, y_gen3)
    
    
    z1_ran1_mean, z1_ran1_var = GC.decode( z2_ran1, y_ran1 )
    z1_ran2_mean, z1_ran2_var = GC.decode( z2_ran2, y_ran2 )
    z1_ran3_mean, z1_ran3_var = GC.decode( z2_ran3, y_ran3 )
    
    z1_ran1_mul_mean, z1_ran1_mul_var = GC.decode( z2_ran1_mul, y_ran1_mul )
    z1_ran2_mul_mean, z2_ran1_mul_var = GC.decode( z2_ran2_mul, y_ran2_mul )
    z1_ran3_mul_mean, z3_ran1_mul_var = GC.decode( z2_ran3_mul, y_ran3_mul )
    
    
    z1_per1_mean, z1_per1_var = GC.decode( z2_per1, y_per1 )
    z1_per2_mean, z1_per2_var = GC.decode( z2_per2, y_per2 )
    z1_per3_mean, z1_per3_var = GC.decode( z2_per3, y_per3 )
    
    z1_per1_mul_mean, z1_per1_mul_var = GC.decode( z2_per1_mul, y_per1_mul )
    z1_per2_mul_mean, z1_per2_mul_var = GC.decode( z2_per2_mul, y_per2_mul )
    z1_per3_mul_mean, z1_per3_mul_var = GC.decode( z2_per3_mul, y_per3_mul )
    
    z1_avg1_mean, z1_avg1_var = GC.decode( z2_avg1, y_avg1 )
    z1_avg2_mean, z1_avg2_var = GC.decode( z2_avg2, y_avg2 )
    z1_avg3_mean, z1_avg3_var = GC.decode( z2_avg3, y_avg3 )
    z1_avg4_mean, z1_avg4_var = GC.decode( z2_avg4, y_avg4 )

    z1_avg1_mul_mean, z1_avg1_mul_var = GC.decode( z2_avg1_mul, y_avg1_mul )
    z1_avg2_mul_mean, z1_avg2_mul_var = GC.decode( z2_avg2_mul, y_avg2_mul )
    z1_avg3_mul_mean, z1_avg3_mul_var = GC.decode( z2_avg3_mul, y_avg3_mul )
    z1_avg4_mul_mean, z1_avg4_mul_var = GC.decode( z2_avg4_mul, y_avg4_mul )

VAE = VariationalAutoencoder( dim_x = 28*28, dim_z = 50 ) #Should be consistent with model being loaded

with VAE.session:
    VAE.saver.restore( VAE.session, VAE_model_path )

    x_mul = VAE.decode(z1_mul_mean)[0]
    x_gen2 = VAE.decode(z1_gen2_mean)[0]
    x_gen3 = VAE.decode(z1_gen3_mean)[0]
    
    x_ran1 = VAE.decode(z1_ran1_mean)[0]
    x_ran2 = VAE.decode(z1_ran2_mean)[0]
    x_ran3 = VAE.decode(z1_ran3_mean)[0]
    
    x_ran1_mul = VAE.decode(z1_ran1_mul_mean)[0]
    x_ran2_mul = VAE.decode(z1_ran2_mul_mean)[0]
    x_ran3_mul = VAE.decode(z1_ran3_mul_mean)[0]

    
    x_per1 = VAE.decode(z1_per1_mean)[0]
    x_per2 = VAE.decode(z1_per2_mean)[0]
    x_per3 = VAE.decode(z1_per3_mean)[0]
    
    x_per1_mul = VAE.decode(z1_per1_mul_mean)[0]
    x_per2_mul = VAE.decode(z1_per2_mul_mean)[0]
    x_per3_mul = VAE.decode(z1_per3_mul_mean)[0]

    
    x_avg1 = VAE.decode(z1_avg1_mean)[0]
    x_avg2 = VAE.decode(z1_avg2_mean)[0]
    x_avg3 = VAE.decode(z1_avg3_mean)[0]
    x_avg4 = VAE.decode(z1_avg4_mean)[0]

    x_avg1_mul = VAE.decode(z1_avg1_mul_mean)[0]
    x_avg2_mul = VAE.decode(z1_avg2_mul_mean)[0]
    x_avg3_mul = VAE.decode(z1_avg3_mul_mean)[0]
    x_avg4_mul = VAE.decode(z1_avg4_mul_mean)[0]


np.save('x_mul', x_mul)
np.save('y_mul', y_mul)
np.save('x_gen2', x_gen2)
np.save('y_gen2', y_gen2)
np.save('x_gen3', x_gen3)
np.save('y_gen3', y_gen3)

np.save('x_ran1', x_ran1)
np.save('y_ran1', y_ran1)
np.save('x_ran2', x_ran2)
np.save('y_ran2', y_ran2)
np.save('x_ran3', x_ran3)
np.save('y_ran3', y_ran3)

np.save('x_ran1_mul', x_ran1_mul)
np.save('y_ran1_mul', y_ran1_mul)
np.save('x_ran2_mul', x_ran2_mul)
np.save('y_ran2_mul', y_ran2_mul)
np.save('x_ran3_mul', x_ran3_mul)
np.save('y_ran3_mul', y_ran3_mul)


np.save('x_per1', x_per1)
np.save('y_per1', y_per1)
np.save('x_per2', x_per2)
np.save('y_per2', y_per2)
np.save('x_per3', x_per3)
np.save('y_per3', y_per3)

np.save('x_per1_mul', x_per1_mul)
np.save('y_per1_mul', y_per1_mul)
np.save('x_per2_mul', x_per2_mul)
np.save('y_per2_mul', y_per2_mul)
np.save('x_per3_mul', x_per3_mul)
np.save('y_per3_mul', y_per3_mul)


np.save('x_avg1', x_avg1)
np.save('y_avg1', y_avg1)
np.save('x_avg2', x_avg2)
np.save('y_avg2', y_avg2)
np.save('x_avg3', x_avg3)
np.save('y_avg3', y_avg3)
np.save('x_avg4', x_avg4)
np.save('y_avg4', y_avg4)

np.save('x_avg1_mul', x_avg1_mul)
np.save('y_avg1_mul', y_avg1_mul)
np.save('x_avg2_mul', x_avg2_mul)
np.save('y_avg2_mul', y_avg2_mul)
np.save('x_avg3_mul', x_avg3_mul)
np.save('y_avg3_mul', y_avg3_mul)
np.save('x_avg4_mul', x_avg4_mul)
np.save('y_avg4_mul', y_avg4_mul)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:45:08 2018

@author: shahrzad
"""
import glob
import cv2
from scipy.spatial.qhull import QhullError
import pickle
import numpy as np
import tensorflow as tf
from rdkit.Chem import PandasTools
import random
import sys

#########################################################################
#arguments
########################################################################
import argparse
if not len(sys.argv) == 3 :
    print("This program requires the following 1 argument: test_id ")
    exit(-57)

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('integers', type=int, nargs=1,
                    help='test_set')
parser.add_argument('strings', type=str, nargs=1,
                    help='save_path')
args = parser.parse_args()
print("Command Line: " ,args.integers[0], args.strings[0])

test_id = args.integers[0]
print("test_set: "+str(test_id))
np_save_path = args.strings[0]
##########################################################################
#parameters
##########################################################################   
# Python optimisation variables
epochs = 50
batch_size = 256
batch_size_test = 4096

#now = datetime.datetime.now()
#date_str=now.strftime("%Y-%m-%d-%H%M")
date_str='2018-11-13-1221'
start_set=0
end_set=1
repeat=1
##########################################################################  
##################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################
        
##########################################################################
#generate images
##########################################################################
generate_images=False
generate_ligand=False
directory='/home/sshirzad/workspace/deepdrug/'
savepath=directory+'train_test_data/'+date_str
dpi=50
size=128
receptors_folder=directory + 'pockets_dude_tiff_'+str(size)+'/'+date_str
ligands_folder = glob.glob(directory + 'screen-libs-sdf/*.sdf')

################################################################################################
#receptor
################################################################################################ 

if generate_ligand:
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from gensim.models import word2vec

    model = word2vec.Word2Vec.load('/home/shahrzad/src/mol2vec/examples/models/model_300dim.pkl')

    d_mols={}
    l_num=1
    r_num=1
    for fname in ligands_folder:   
        if 'actives' in fname:
            receptor_name=fname.split('-actives')[0].split('/')[-1]   
            label=1           
        elif 'decoys' in fname:
            receptor_name=fname.split('-decoys')[0].split('/')[-1]
            label=0            
        if receptor_name+'_'+str(label) not in d_mols.keys():
            d_mols[receptor_name+'_'+str(label)]=[]
            
        df = PandasTools.LoadSDF(fname)
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
        df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
        X = np.array([x.vec for x in df['mol2vec']])
        d_mols[receptor_name+'_'+str(label)]=X

        print(str(l_num), " th receptor")
        l_num = l_num+1

    save_obj(d_mols, directory + 'train_test_data/'+date_str+'/ligand_dict_mols')
else:
    ligand_dict=load_obj(savepath+'/ligand_dict_mols')

#####################################################
#Data
#####################################################             
if generate_images:
    receptors = sorted(glob.glob(directory + 'pockets_dude_tiff_128/'+date_str+'/*.png'))
    d={}
    i=1
    for filename in receptors:
        receptor=filename.split('/')[-1].split('_')[0].split('.')[0]
        
        print(str(i)+"- voronoi image for "+receptor+" generated")
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        d[receptor]=np.reshape(img,[-1, size*size*3])
        i=i+1
    save_obj(d, savepath+'/receptors_dict_'+str(size))
image_dict=load_obj(savepath+'/receptors_dict_'+str(size))
#####################################################
#ligand
#####################################################
receptor_list=sorted(list(set([k.split('_')[0] for k in ligand_dict.keys()])))

all_labels=[]
for k in ligand_dict.keys():
    if k.split('_')[0] in image_dict.keys():
        if k.split('_')[1]=='1':
            all_labels=all_labels+np.shape(ligand_dict[k][0])[0]*[1]
        if k.split('_')[1]=='0':
            all_labels=all_labels+np.shape(ligand_dict[k][0])[0]*[0]

from sklearn.utils import class_weight
weights=class_weight.compute_class_weight('balanced', [0,1],all_labels)

###########################################################################################
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]
    
    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    
    # add the bias
    out_layer += bias
    
    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    
    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')
    
    return out_layer


# declare the training data placeholders
# input x - for 20 x 20 pixels = 400 - this is the flattened image data t
x = tf.placeholder(tf.float32, [None,size*size*3])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, size, size, 3])
# now declare the output data placeholder - 2 classes
y = tf.placeholder(tf.float32, [None, 2])

# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 3, 16, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 16, 32, [3, 3], [2, 2], name='layer2')
layer3 = create_new_conv_layer(layer2, 32, 64, [3, 3], [2, 2], name='layer3')

flattened = tf.reshape(layer3, [-1, size * size])  #(size/8)*(size/8)*64
# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([size * size , 512], stddev=0.03), name='wd1') #(size/8)*(size/8)*64
bd1 = tf.Variable(tf.truncated_normal([512], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

z = tf.placeholder(tf.float32, [None,300])
z_shaped = tf.reshape(z, [-1, 300])
x_final=tf.concat([dense_layer1,z_shaped],-1)


wd2 = tf.Variable(tf.truncated_normal([512+300, 1000], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(x_final, wd2) + bd2
dense_layer2 = tf.nn.relu(dense_layer2)

# another layer with softmax activations
wd3 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd3')
bd3 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd3')
dense_layer3 = tf.matmul(dense_layer2, wd3) + bd3
y_ = tf.nn.softmax(dense_layer3)

class_weights=tf.constant([[  0.51474333,  17.45682248]])
weights = tf.reduce_sum(class_weights * y, axis=1)
# compute your (unweighted) softmax cross entropy loss
unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer3, labels=y)
# apply the weights, relying on broadcasting of the multiplication
weighted_losses = unweighted_losses * weights
# reduce the result to get your final loss
loss = tf.reduce_mean(weighted_losses)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer3, labels=y))
learning_rate_tf = tf.placeholder(tf.float32)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate_tf, epsilon=0.1, beta1=0.99).minimize(loss)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred = tf.cast(tf.argmax(y_, 1),tf.float32)
label = tf.cast(tf.argmax(y,1),tf.float32)
auc=tf.metrics.auc(label,pred, weights, curve='ROC')
confusion = tf.confusion_matrix(labels=label, predictions=pred, num_classes=2)
###########################################################################################
receptor_sizes=[0]*len(receptor_list)
if 'hxk4' in receptor_list:
    receptor_list.remove('hxk4')
data_size=0
for i in range(len(receptor_list)):
    l=receptor_list[i]
    receptor_sizes[i]=np.shape(ligand_dict[l+'_0'][0])[0]+np.shape(ligand_dict[l+'_1'][0])[0]   
    data_size=data_size+receptor_sizes[i]

#num_data_sets=10
num_data_sets=len(receptor_list)
def data_gen(set_i, epochs):
    l=receptor_list[set_i]  
    test_size=receptor_sizes[set_i]
    test_set=np.zeros((test_size, size*size*3+300))
    test_labels=np.zeros((test_size, 2))

    l_count = 0
    test_set[:,300:] = image_dict[l]

    for r in ligand_dict[l+'_0'][0]:
        test_set[l_count, 0:300] = r
        test_labels[l_count,0] = 1
        l_count = l_count+1
    for r in ligand_dict[l+'_1'][0]:
        test_set[l_count,0:300] = r
        test_labels[l_count,1] = 1
        l_count = l_count+1
    per = np.random.permutation(test_size)

    test_set=test_set[per]    
    test_labels=test_labels[per]
    
    for epoch in range(epochs):
        for set_j in range(0, num_data_sets):        
            if set_j != set_i:
                training_size=receptor_sizes[set_j]
                training_set=np.zeros((training_size, size*size*3+300))
                training_labels=np.zeros((training_size, 2))
                l=receptor_list[set_j]  
                training_set[:,300:]=image_dict[l]
                count_0=np.shape(ligand_dict[l+'_0'][0])[0]
                count_1=np.shape(ligand_dict[l+'_1'][0])[0]
                l_count = 0
                for r in range(count_0):
                    training_set[l_count,0:300]=ligand_dict[l+'_0'][0][r,:]
                    training_labels[l_count,0]=1
                    l_count=l_count+1
                for r in range(count_1):
                    training_set[l_count,0:300]=ligand_dict[l+'_1'][0][r,:]
                    training_labels[l_count,1]=1
                    l_count=l_count+1
                    
                per = np.random.permutation(training_size)
            
                training_set=training_set[per]    
                training_labels=training_labels[per]        
                yield (training_set, training_labels, test_set, test_labels, receptor_list[set_i] , receptor_list[set_j])


applied_rate=0.001

# setup the initialisation operator            
init_op = tf.global_variables_initializer()
init_loc_op = tf.local_variables_initializer()
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    sess.run(init_loc_op)
    my_data=data_gen(test_id, epochs)
    step=0

    for epoch in range(epochs):
        avg_cost = 0
        for train_id in range(num_data_sets-1):   
            step=step+1
            if step%500==0:
                applied_rate *= 0.5

            (training_set, training_labels, test_set, test_labels,set_i, set_j)=next(my_data)
            total_batch = int(np.shape(training_set)[0] / batch_size)
            total_batch_test = int(np.shape(test_set)[0] / batch_size_test)
            for i in range(total_batch):
                batch_x = training_set[i*batch_size:(i+1)*batch_size,300:]
                batch_y =training_labels[i*batch_size:(i+1)*batch_size] 
                batch_z =training_set[i*batch_size:(i+1)*batch_size,0:300]
    
                _, c = sess.run([optimiser, loss], feed_dict={x: batch_x, y: batch_y, z: batch_z, learning_rate_tf:applied_rate})
    
                avg_cost += c / total_batch
            train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, z: batch_z,learning_rate_tf:applied_rate})
            print("Epoch:", (epoch + 1), " train accuracy: {:.3f}".format(train_acc))

        print("training finished for epoch ",epoch+1)
        for j in range(total_batch_test):
            batch_x_test = test_set[j*batch_size_test:(j+1)*batch_size_test,300:]
            batch_y_test =test_labels[j*batch_size_test:(j+1)*batch_size_test,:]
            batch_z_test =test_set[j*batch_size_test:(j+1)*batch_size_test,0:300]

            test_acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, z: batch_z_test})
            np.save(np_save_path+'/batch_'+str(test_id)+'_'+str(j)+'_true.npy',batch_y_test)
            pr=sess.run(y_, feed_dict={x: batch_x_test, y: batch_y_test, z: batch_z_test})
            np.save(np_save_path+'/batch_'+str(test_id)+'_'+str(j)+'_pred.npy', pr)

            print("Epoch:", (epoch + 1), "batch: ",(j+1),"cost =", "{:.5f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
#                print(sess.run(confusion, feed_dict={x: batch_x_test, y: batch_y_test, z: batch_z_test}))
#                print(sess.run(auc, feed_dict={x: batch_x_test, y: batch_y_test, z: batch_z_test}))
                
            #print("Epoch:", (epoch + 1), "confusion matrix: ",sess.run(confusion, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
    
        
        print("\nTraining complete for model ", test_id+1, "!")
    #        print(sess.run(accuracy, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
#        print(sess.run(auc, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))
    #        print(sess.run(confusion, feed_dict={x: test_set[:,300:], y: test_labels,  z: test_set[:,0:300]}))


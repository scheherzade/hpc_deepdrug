#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:31:01 2018

@author: shahrzad
"""

import glob
#import cv2
import pickle
import numpy as np
#from rdkit.Chem import PandasTools
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter        
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import sys
#from matplotlib import pyplot as plt
import statistics
import time
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
#
#now = datetime.datetime.now()
#date_str=now.strftime("%Y-%m-%d-%H%M")
date_str='2018-11-13-1221'
start_set=0
end_set=1
repeat=1
###########################################################################  
###################################
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
generate_hist=False
generate_ligand=False
directory='/home/sshirz1/runs/'
savepath=directory+'train_test_data/'+date_str
dpi=50
size=128
receptors_folder=directory + 'pockets_dude_tiff_'+str(size)+'/'+date_str
ligands_folder = glob.glob(directory + 'screen-libs-sdf/*.sdf')

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

####################################################
def extract_color_hist(input_folder):
    all_hist={}
    receptors = glob.glob(input_folder + '/*.png')
    for filename in receptors:
        receptor=filename.split('.')[0].split('/')[-1]
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("error")
            return
        all_hist[receptor] = cv2.calcHist([img],[0],None,[256],[0,256])[:,0].tolist()
    return all_hist
#####################################################
#image
#####################################################      
        
if generate_hist:
    image_dict=extract_color_hist(receptors_folder)
    save_obj(image_dict, savepath+'/receptors_histogram_dict_'+str(size))
else:
    image_dict=load_obj(savepath+'/receptors_histogram_dict_'+str(size))
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



if 'hxk4' in receptor_list:
    receptor_list.remove('hxk4')
receptor_sizes=[0]*len(receptor_list)
data_size=0
for i in range(len(receptor_list)):
    l=receptor_list[i]
    receptor_sizes[i]=np.shape(ligand_dict[l+'_0'][0])[0]+np.shape(ligand_dict[l+'_1'][0])[0]   
    data_size=data_size+receptor_sizes[i]

def data_gen(set_i):
   l=receptor_list[set_i]  
   test_size=receptor_sizes[set_i]
   test_set=np.zeros((test_size, 256+300))
   test_labels=np.zeros(test_size)

   l_count = 0
   test_set[:,300:] = image_dict[l]

   for r in ligand_dict[l+'_0'][0]:
       test_set[l_count, 0:300] = r
       test_labels[l_count] = 0
       l_count = l_count+1
   for r in ligand_dict[l+'_1'][0]:
       test_set[l_count,0:300] = r
       test_labels[l_count] = 1
       l_count = l_count+1
   per = np.random.permutation(test_size)

   test_set=test_set[per]    
   test_labels=test_labels[per]
   
   training_size=sum(receptor_sizes[:])-receptor_sizes[set_i]
   training_set=np.zeros((training_size, 256+300))
   training_labels=np.zeros(training_size)
   
   l_count = 0
   for set_j in range(0, len(receptor_list)):        
       if set_j != set_i:
           l=receptor_list[set_j]  
           training_set[l_count:l_count+receptor_sizes[set_j],300:]=image_dict[l]
           count_0=np.shape(ligand_dict[l+'_0'][0])[0]
           count_1=np.shape(ligand_dict[l+'_1'][0])[0]
           for r in range(count_0):
               training_set[l_count,0:300]=ligand_dict[l+'_0'][0][r,:]
               training_labels[l_count]=0
               l_count=l_count+1
           for r in range(count_1):
               training_set[l_count,0:300]=ligand_dict[l+'_1'][0][r,:]
               training_labels[l_count]=1
               l_count=l_count+1
               
   per = np.random.permutation(training_size)

   training_set=training_set[per]    
   training_labels=training_labels[per]        
   yield (training_set, training_labels, test_set, test_labels, receptor_list[set_i] , receptor_list[set_j])

target_names = ['decoy','active']
my_data=data_gen(test_id)

read_data_time=time.time()
(training_set, training_labels, test_set, test_labels,set_i, set_j)=next(my_data)
read_data_time=time.time()-read_data_time
print("read data (s): "+str(read_data_time))
run_model=time.time()
clf = RandomForestClassifier(verbose=1,n_estimators=500,n_jobs=16,max_features=None,max_depth=10,class_weight="balanced_subsample")    
#clf.fit(training_set, training_labels)
prob= clf.fit(training_set, training_labels).predict_proba(test_set)
p=clf.predict(test_set)
run_model=time.time()-run_model
print("time: "+str(run_model))
print("shape of p: ", np.shape(p))
print("#of 1s: ", np.sum(p))
print("set "+ str(test_id))
print("classification report")
print(classification_report(test_labels, p, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(test_labels,p))
print("auc:")       
try:            
    print(roc_auc_score(test_set, prob))
except ValueError:
        print('Only one class present in y_true. ROC AUC score is not defined in that case.')
np.save(np_save_path+'/test_id_'+str(test_id)+'_true.npy', test_labels)
np.save(np_save_path+'/test_id_'+str(test_id)+'_pred.npy', prob)
       
        
#        if roc_plot:  
#            #not complete
#            from sklearn.metrics import roc_curve
#            import sklearn.metrics 
#            fpr = dict()
#            tpr = dict()
#            roc_auc = dict()
#            for i in range(2):
#                fpr[i], tpr[i], _ = roc_curve(p[:], t[:])
#                roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])


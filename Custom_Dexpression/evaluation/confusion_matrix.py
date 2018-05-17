import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#takes a confusion matrix (n x n np.ndarray) and plots it in a large figure
def plot_conf_mat(conf_mat, title = 'No title',size= (20,14)):
    plt.figure(figsize = size)
    plt.title(title)
    ticks = ['Anger','Contempt','Disgust','Fear','Happy','Saddness','Surprise']
    sn.set(font_scale=1.4)#for label size
    fig = sn.heatmap(conf_mat, annot=True,annot_kws={"size": 16},cmap="YlGnBu"
        ,xticklabels=ticks, yticklabels=ticks)
    fig.set_xlabel('prediction')
    fig.set_ylabel('labels')

def save_conf_mat(conf_mat, title = 'No title',size= (20,14)):
	plot_conf_mat(conf_mat, title ,size)
	plt.savefig('figures/' + title,format='png')

def plot_norm_conf_mat(conf_mat, title = 'No title',size= (20,14)):
    sum = (conf_mat).astype(np.float).sum(axis=0)
    conf_mat_norm =np.round(conf_mat.astype(np.float)/sum.reshape(1,7),decimals=4)

    plot_conf_mat(conf_mat_norm,title,size)

def save_norm_conf_mat(conf_mat, title = 'No title',size= (20,14)):
	plot_norm_conf_mat(conf_mat, title ,size)
	plt.savefig('figures/' + title,format='png')
  
# takes the output of a DNN converts it to class labels(input= np.ndarray:[[0,1,0,0],[0,0,0,1],[0,1,0,0]] output=np.ndarray:[1,3,2] )
def extract_classes(DNN_output):
    classes = []  
    for lab in DNN_output:
        maxi = lab.max()
        for i in range(0,len(lab)):
            if(lab[i] == maxi):
                classes.append(i)
    return np.asarray(classes)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#takes a confusion matrix (n x n np.ndarray) and plots it in a large figure
def plot_conf_mat(conf_mat):
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4)#for label size
    fig = sn.heatmap(conf_mat, annot=True,annot_kws={"size": 16},cmap="YlGnBu")
    fig.set_xlabel('labels')
    fig.set_ylabel('prediction')

# takes the output of a DNN converts it to class labels(input= np.ndarray:[[0,1,0,0],[0,0,0,1],[0,1,0,0]] output=np.ndarray:[1,3,2] )
def extract_classes(DNN_output):
    classes = []  
    for lab in DNN_output:
        maxi = lab.max()
        for i in range(0,len(lab)):
            if(lab[i] == maxi):
                classes.append(i)
    return np.asarray(classes)
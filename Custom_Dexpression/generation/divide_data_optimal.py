
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

#chris library imports
# from matplotlib import pyplot as plt
import cv2
# import tensorflow as tf
from sklearn.model_selection import train_test_split


from test_recursive_image_load_V2 import load_CKP_data
from test_recursive_image_load_V2 import load_formated_data
from test_recursive_image_load_V2 import split_dataset
from test_recursive_image_load_V2 import split_subject_dataset
from test_recursive_image_load_V2 import divide_subjects
from test_recursive_image_load_V2 import divide_data_to_subject
from test_recursive_image_load_V2 import load_npy_files

import logging
from memory_profiler import memory_usage


# In[8]:


import logging
from memory_profiler import memory_usage

import argparse

logfile = "divide_log.txt"

if os.path.exists(logfile):
        os.remove(logfile)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("start logging to " + logfile)


# In[3]:


def logprint(*arg):
    string = "str"
    stre = ""
    for a in arg:
        if(type(a) == type(string)):
            stre = stre + " " +  a
        else:
            stre = stre + " " + repr(a)
    print(stre)
    logging.info(stre)

def showInfo(Var,VarName = "UnKnown"):
    logprint("----------------------")
    logprint("Name " + repr(VarName))
    logprint("type: " + repr(type(Var)))
    logprint("Dtype: " + repr(Var.dtype))
    logprint("shape: " + repr(Var.shape))


# In[9]:


def extract_classes(DNN_output):
    classes = []  
    for lab in DNN_output:
        maxi = lab.max()
        for i in range(0,len(lab)):
            if(lab[i] == maxi):
                classes.append(i)
    return np.asarray(classes)


# In[10]:


load_dir = 'CKP_all_neutral' #the directory that is created in the predivided_data directory

# load the data
data= load_npy_files(5,load_dir)
[X_data, Y_data, X_subID] = data


# In[11]:



# Y_classes = extract_classes(Y_data)

# for i in range(len(X_data)):
#         stre = repr(i) + " " + repr(X_subID[i]) + " e" + repr(Y_classes[i])
#         cv2.imshow(stre, X_data[i].astype('uint8'))
#         cv2.waitKey()
#         cv2.destroyAllWindows()
        


# In[12]:


print(X_subID)


# In[13]:



X_subID_unique = np.unique(X_subID)

print(type(repr(X_subID_unique[0])))
print(X_subID_unique[0])
print(repr(X_subID_unique[0]))

Y_classes = extract_classes(Y_data)


count_per_sub = {X_subID_unique[0]: np.zeros(7)}

for i in range(len(X_subID_unique)):
# i = 1
    emotions_sub_indices = np.where( X_subID == X_subID_unique[i])
    # it seems like there are far less images from the database but only a few of them have actual emotion labels
#     print(X_subID_unique[i])
#     print(emotions_sub_indices)
#     print(Y_classes[emotions_sub_indices])
#     print(np.bincount(Y_classes[emotions_sub_indices], minlength=7))
    count_per_sub[X_subID_unique[i]]= np.bincount(Y_classes[emotions_sub_indices], minlength=7)

# for i in X_data[emotions_sub_i]:
#         cv2.imshow('ok', i.astype('uint8'))
#         cv2.waitKey()
#         cv2.destroyAllWindows()

emotion_count_list = np.zeros((len(count_per_sub),7))
subID_count_list = np.zeros(len(count_per_sub))


# print(len(count_per_sub))
# 
c = 0
for i in count_per_sub :
    print( i ," ", count_per_sub[i])
    subID_count_list[c] = i
    emotion_count_list[c] = np.asarray(count_per_sub[i])
    c = c + 1

# sum of emotions 
sum_per_emo = np.sum(emotion_count_list, axis = 0)
total_sum = np.sum(emotion_count_list)

print(sum_per_emo)
print(total_sum ) 

# calculate percentages

emotion_count_list_percent = np.divide(emotion_count_list,sum_per_emo)*100
total_sum_percent =(sum_per_emo/total_sum)*100
subID_sum = np.sum(emotion_count_list_percent, axis = 1)

print("subID_count_list[0]",subID_count_list[0])
print("subID_count_list",subID_count_list)
print("emotion_count_list_percent.shape[0]",emotion_count_list_percent.shape[0])
print("emotion_count_list_percent",np.round(emotion_count_list_percent,decimals = 2))
print("total_sum_percent.shape[0]",total_sum_percent.shape[0])
print("total_sum_percent",np.round(total_sum_percent,decimals = 2))
print("subID_sum.shape[0]",subID_sum.shape[0])
print("subID_sum",np.round(subID_sum,decimals = 2))


# In[14]:


def whereAllElementsOf(x_array,y_array):
    indices = np.asarray([])
    for y in y_array:
        indices = np.append(indices,np.where(x_array == y))
    return indices.astype('uint16')


# In[15]:


# count the percentage of the emotion when 2 subID are summed together
def get_distri_from_subID_array(subID_test_list): 
    indice = whereAllElementsOf(subID_count_list,subID_test_list)
#     print(indice)
#     print(subID_count_list[indice])
#     print(emotion_count_list[indice])
    sum_per_emo_test = np.sum(emotion_count_list[indice], axis = 0)
#     print(sum_per_emo_test)
    sum_test_total = np.sum(sum_per_emo_test)
    sum_per_emo_test_percent = np.divide(sum_per_emo_test,sum_test_total)*100
#     print(np.round(sum_per_emo_test_percent ,decimals= 2))
    return sum_per_emo_test_percent


# In[16]:


get_distri_from_subID_array( [5.0])
get_distri_from_subID_array( [5.0,10.0])
get_distri_from_subID_array( [10.0])


# In[17]:


subID_sum_sort_indexing = np.argsort(subID_sum, kind = 'mergesort')


subID_sum_sort = subID_sum[subID_sum_sort_indexing]
subID_sort = subID_count_list[subID_sum_sort_indexing]

print(np.round(subID_sum_sort,decimals = 2))
print(subID_sort)


# In[18]:


reverse = np.arange(subID_sum_sort.shape[0]-1,-1,-1)
print(reverse)


# In[19]:


subID_sum_sort_reverse = subID_sum_sort[reverse]
subID_sort_reverse = subID_sort[reverse]
print(np.round(subID_sum_sort_reverse,decimals = 2))
print(subID_sort_reverse)


# In[20]:


# lists of indexes per part
def initialise_parts_array():
    parts_subID = []
    for i in range(0,11):
        parts_subID.append(np.asarray([]))
    print(parts_subID)

    #  fill each part with the first instances of the sorted and reversed list 
#     for i in range(0,11):
#         parts_subID[i]= np.append(parts_subID[i],subID_sort_reverse[i])
    
    for i in range(0,len(subID_sort_reverse)):
        parts_subID[i%11]= np.append(parts_subID[i%11],subID_sort_reverse[i])

    print(parts_subID)
    return parts_subID
initialise_parts_array()


# In[21]:


import itertools as iter

# testing permutation methods  === DOESN'T work!

# lists = list(iter.permutations((range(2))))
# print(lists)
# perm = list(iter.permutations(([1,2,3])))
# count = iter.count(10)
# print(perm)
# print(count)


# In[22]:


# test of mean sqaure errorr method
from sklearn.metrics import mean_squared_error
A = np.asarray([8.0,2.0])
B =np.asarray([16.0,24.0])
mse = mean_squared_error(A, B)
print(mse)
print(type(mse))


# In[23]:


def getSumOfPercentages(x_array,indices_array):
    return np.sum(x_array[indices_array],axis =0)
    


# In[24]:


parts_subID = initialise_parts_array()
x =subID_count_list
y = parts_subID[0]
y =  np.append(y, 54.0)
index = whereAllElementsOf(x,y)
print(x)
print(y)
print(index)
print(index.astype('uint16'))
print(subID_count_list[index])
sum1 = getSumOfPercentages(emotion_count_list_percent,[index[0]])
print(sum1)
sum2 = getSumOfPercentages(emotion_count_list_percent,index)
print(sum2)
mse1 = mean_squared_error(total_sum_percent,sum1)
mse2 = mean_squared_error(total_sum_percent,sum2)
print(mse1)
print(mse2)


# In[25]:


# method to check of the sum of the percentages per emotion has reached the total distribution on 1 emotion
def check_if_above_threshold(total_sum_percent, sum1):
    overThreshold = False
    for i in range(0,len(total_sum_percent)):
        if(sum1[i] > total_sum_percent[i]):
            overThreshold = True
    return overThreshold


# In[26]:


# makes a list of all mean sqaure errorrs
def calculate_MSE_arrays(parts_subID):
    part_MSE_array = np.asarray([])
    for subID_array in parts_subID:
        indices = whereAllElementsOf(subID_count_list,subID_array)
        sum1 = getSumOfPercentages(emotion_count_list_percent, indices)   
        mse = mean_squared_error(total_sum_percent,sum1)
#         if the sum of the percentage is already higher than one of the total_sum_percent than shouldn't receive more subjects
        if(check_if_above_threshold(total_sum_percent, sum1)):
            mse = 0.0
        part_MSE_array = np.append(part_MSE_array, mse)
#     print(part_MSE_array)
    return part_MSE_array


# In[27]:


# 


# In[28]:


parts_subID = initialise_parts_array()
count = 11
for count in range(11,len(subID_sort_reverse)):
    MSE_parts = calculate_MSE_arrays(parts_subID)
    
    mini = np.argmax(MSE_parts)
    parts_subID[mini] = np.append(parts_subID[mini],subID_sort_reverse[count]) 
    print("added ", count, " element from sorted_reversed subID list to index " , mini," making ", parts_subID[mini])



# In[29]:


parts_subID = initialise_parts_array()

def calculate_MSE_for_all_parts(parts_subID):
#     print("------  goal ---------")
#     print(np.round(total_sum_percent,decimals= 2))
    print("calculate all parts")

    sum_mse = 0
    for i in range(0,11):
#         print(parts_subID[i])
        sum1 = get_distri_from_subID_array(parts_subID[i])
        mse = mean_squared_error(total_sum_percent,sum1)
        sum_mse = sum_mse + mse

    #     print("------ part ", i, '---------')
    #     print(parts_subID[i])
    #     print('length of parts_subID[i] ', parts_subID[i].shape[0])
    # #     indices = whereAllElementsOf(subID_count_list,parts_subID[i])
    # #     sum1 = getSumOfPercentages(emotion_count_list_percent, indices) 
    #     print(np.round(total_sum_percent,decimals= 2))
    #     print(np.round(sum1,decimals= 2))
    #     print(np.round(mse,decimals= 2))

    
    return sum_mse

calculate_MSE_for_all_parts(parts_subID)


# In[30]:


def generate_perm_per_index(parts_subID,index):
    column_9 = np.zeros(len(parts_subID))
    for i in range(0, len(parts_subID)):
        column_9[i] = parts_subID[i][index]

    column_9 = column_9.astype('uint16')
    # when generateing parmutations don't go over array of lenghth 9 => otherwise takes too long
    perm1 = list(iter.permutations((column_9[0:6])))
    perm2 = list(iter.permutations((column_9[6:])))
    print(column_9)
    return perm1,perm2
# print(perm)


# In[31]:


perm1_8,perm2_8 = generate_perm_per_index(parts_subID,8)
perm1_7,perm2_7 = generate_perm_per_index(parts_subID,7)
perm1_6,perm2_6 = generate_perm_per_index(parts_subID,6)
perm1_5,perm2_5 = generate_perm_per_index(parts_subID,5)
perm1_4,perm2_4 = generate_perm_per_index(parts_subID,4)
perm1_3,perm2_3 = generate_perm_per_index(parts_subID,3)
perm1_2,perm2_2 = generate_perm_per_index(parts_subID,2)
perm1_1,perm2_1 = generate_perm_per_index(parts_subID,1)


# In[34]:


from copy import deepcopy
new_parts_subID = parts_subID

max_mse = 0
max_parts_subID = parts_subID
min_mse = 100000000
min_parts_subID = parts_subID

# totlength = len(perm1_8) * len(perm2_8)* len(perm1_7) *len(perm2_7)* len(perm1_6) *len(perm2_6)* len(perm1_5)* len(perm2_5) *len(perm1_4) *len(perm2_4) *len(perm1_3)* len(perm2_3)* len(perm2_2)* len(perm2_2)* len(perm2_1)* len(perm2_1) 
totlength = len(perm1_8) * len(perm2_8)* len(perm1_7) *len(perm2_7)* len(perm1_6) *len(perm2_6)
print( "totlength " ,totlength)
count = 0
publishCount =0

# for i1 in range(10):#len(perm1_3)):
#     for k1 in range(10):#len(perm2_3)):
#         sum_mse = 0
#         perm_1 = np.append(perm1_1[i1], perm2_1[k1])
#         for i2 in range(10):#len(perm1_2)):
#             for k2 in range(10):#len(perm2_2)):
#                 sum_mse = 0
#                 perm_2 = np.append(perm1_2[i2], perm2_4[k2])
#                 for i3 in range(10):#len(perm1_3)):
#                     for k3 in range(10):#len(perm2_3)):
#                         sum_mse = 0
#                         perm_3 = np.append(perm1_3[i3], perm2_3[k3])
#                         for i4 in range(10):#len(perm1_4)):
#                                     for k4 in range(10):#len(perm2_4)):
#                                         sum_mse = 0
#                                         perm_4 = np.append(perm1_4[i4], perm2_4[k4])
#                                         for i5 in range(10):#len(perm1_5)):
#                                             for k5 in range(10):#len(perm2_5)):
#                                                 sum_mse = 0
#                                                 perm_5 = np.append(perm1_5[i5], perm2_5[k5])
for i6 in range(10):#len(perm1_6)):
    for k6 in range(10):#len(perm2_6)):
        sum_mse = 0
        perm_6 = np.append(perm1_6[i6], perm2_6[k6])        
        for i7 in range(10):#len(perm1_7)):
            for k7 in range(10):#len(perm2_7)):
                sum_mse = 0
                perm_7 = np.append(perm1_7[i7], perm2_7[k7])
                for i8 in range(10):#len(perm1_8)):
                    for k8 in range(10):#len(perm2_8)):
                        sum_mse = 0
                        perm_8 = np.append(perm1_8[i8], perm2_8[k8])
                #         print(perm)
                        for j in range(0, len(parts_subID)):
                            new_parts_subID[j][8] = perm_8[j]
                            new_parts_subID[j][7] = perm_7[j]
                            new_parts_subID[j][6] = perm_6[j]
#                             new_parts_subID[j][5] = perm_5[j]
#                             new_parts_subID[j][4] = perm_4[j]
#                             new_parts_subID[j][3] = perm_3[j]
#                             new_parts_subID[j][2] = perm_2[j]
#                             new_parts_subID[j][1] = perm_1[j]

                #             print(new_parts_subID[j8])
                            sum1 = get_distri_from_subID_array(new_parts_subID[j])
                            mse = mean_squared_error(total_sum_percent,sum1)
                            sum_mse = sum_mse + mse

                #       mse = calculate_MSE_for_all_parts(new_parts_subID)
                        
                        if(max_mse < sum_mse):
                            max_mse = sum_mse
                            max_parts_subID = deepcopy(new_parts_subID)
                        if(min_mse > sum_mse):
                            min_mse = sum_mse
                            min_parts_subID = deepcopy(new_parts_subID)
                        if(88.0 > sum_mse):
                            selected_mse = sum_mse
                            selected_parts_subID = deepcopy(new_parts_subID)

                        if(count%10000 == 0):
                            publishCount =publishCount + 1
                            logprint("------------------- publishCount ", publishCount)
                            logprint(count, "/", totlength, " sum_mse is ", sum_mse)
                            logprint("max_mse " ,max_mse )
                            logprint("max_parts_subID ",max_parts_subID)
                            logprint("recalculated max",calculate_MSE_for_all_parts(max_parts_subID))
                            logprint("min_mse ",min_mse)
                            logprint("min_parts_subID ", min_parts_subID)
                            logprint("recalculated min",calculate_MSE_for_all_parts(min_parts_subID))
                            np.save("min_parts_subID.npy", min_parts_subID)

                        count = count + 1


# In[ ]:


logprint("max_mse " ,max_mse )
logprint("max_parts_subID ",max_parts_subID)
logprint("recalculated max",calculate_MSE_for_all_parts(max_parts_subID))
logprint("min_mse ",min_mse)
logprint("min_parts_subID ", min_parts_subID)
logprint("recalculated min",calculate_MSE_for_all_parts(min_parts_subID))
npy.save("min_parts_subID.npy", min_parts_subID)


# In[ ]:


# best results permuting the parts_subID[8]
min_mse = 87.42253
# the next min_part is wrong
min_parts = [np.asarray([501.,  46.,  22.,  97.,  57.,  93., 127., 134.,  86.,  29.]),
            np.asarray([506., 130.,  95., 133.,  64.,  92.,  70.,  99.,  63.,   5.]),
            np.asarray([11., 125., 108.,  89., 115., 100., 135., 126., 107., 101.]),
            np.asarray([132.,  14.,  81., 113.,  37., 109.,  85.,  88., 128.,  83.]),
            np.asarray([71.,  32., 119.,  54., 129.,  80.,  76.,  77.,  73., 105.]),
            np.asarray([999., 131., 136.,  78.,  44.,  35.,  10.,  53.,  96., 122.]),
            np.asarray([ 74.,  50., 124., 102.,  52.,  58.,  91.,  56.,  94., 110.]),
            np.asarray([502.,  65.,  67.,  75.,  87.,  61.,  82., 116., 112.]),
            np.asarray([503., 504., 138.,  60., 117.,  45.,  69.,  51.,  28.]),
            np.asarray([55., 42., 68., 66., 84., 59., 90., 72., 98.]),
            np.asarray([ 62., 137.,  26.,  34., 106.,  79., 111., 114., 505.])]
# print(min_parts)


# In[ ]:


# print(calculate_MSE_for_all_parts(min_parts))


# In[35]:


# load = np.load("min_parts_subID.npy")


# In[36]:


# print(load)


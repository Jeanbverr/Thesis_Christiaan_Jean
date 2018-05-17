
import numpy as np
import os
import sys

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

from test_recursive_image_load_V2 import  split_subject_dataset
from test_recursive_image_load_V2 import  divide_subjects
from test_recursive_image_load_V2 import  load_npy_files

integer = 1
string = print('test print method ', integer)
print(string, " again")

import logging

logfile = "test.txt"

logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("start logging to " + logfile)



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

logprint("oke testing logprint")
logprint("normal string with repr" + repr(1))
logprint("special string only compatible with print " , 1)
logprint("longer special string only compatible with print " , 1, "another", 5.0, "and another !!", 90)



def manyArgs(*arg):
	print ("I was called with", len(arg), "arguments:", arg)
manyArgs(1)
# I was called with 1 arguments: (1,)
manyArgs(1, 2,3)

# test the np.unique method
[X_data,Y_data,X_subID] = load_npy_files(5,'CKP')


print("X_subID \n ", X_subID)
indice = np.argsort(X_subID,kind='mergesort',axis = 0)
print("X_subID_indices \n ", indice)
print("X_subID_sorted \n ", X_subID[indice])


unique,indices = np.unique(X_subID,return_index=True)
print("X_subID unique \n ", unique)
print("X_subID unique indices \n ", indices)



a = np.array([[ 1, 5 ,6 ,7 ,10 ,19 ,2],[1,2,3,4,5,6,7]])
#wanted outcome = [[1,2,5,6,7,10,19],[1,7,2,3,4,5,6]]

print(repr(a.shape))
a.sort(kind='mergesort',axis = 0)

a.sort(kind='mergesort',axis = 1)

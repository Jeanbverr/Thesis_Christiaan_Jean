import numpy as np 
import os 
import sys

from generate_data_info import generate_data_info

dirlist = sys.listdir(../../data)

print(dirlist)

for name in dirlist:
	generate_data_info(dirlist)
	

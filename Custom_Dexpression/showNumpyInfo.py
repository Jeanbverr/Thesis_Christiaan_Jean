
# coding: utf-8

# In[ ]:


import numpy as np

def showInfo(Var,VarName = "UnKnown"):
    print("Name ", VarName)
    print("type: ", type(Var))
    if(type(Var)==list):
        if(len(Var) > 0):
            print("Element type: ", type(Var[0]))
        else:
            print("Element type: None",)
        print("shape: ", len(Var))
    else:    
        print("Dtype: ", Var.dtype)
        print("shape: ", Var.shape)
        # print("flags: ", Var.flags)


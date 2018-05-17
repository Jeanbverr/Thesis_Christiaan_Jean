import sys
import os

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

os.chdir(Lpath)
print("current working dir: ", os.getcwd())

from test_recursive_image_load_V2 import create_formated_data
from test_recursive_image_load_V2 import load_formated_data
from test_recursive_image_load_V2 import load_all_annotated_CKP_data
from test_recursive_image_load_V2 import create_all_CKP_formated_data
from test_recursive_image_load_V2 import create_all_CKP_formated_data_neutral
from test_recursive_image_load_V2 import create_complete_CKP_formated_data		

dataPath = 'G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+'

# create_formated_data(dataPath,0)
# load_all_annotated_CKP_data(dataPath,0)
# create_complete_CKP_formated_data(dataPath,1)
create_all_CKP_formated_data_neutral(dataPath,0)
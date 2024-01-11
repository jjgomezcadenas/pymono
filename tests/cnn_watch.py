import numpy as np 
from collections import namedtuple
from pymono.config import CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_NX
from pymono.plt_funcs import  plot_loss
from pymono.plt_funcs import  plotxyz
import sys
import os
import json
import hashlib

def get_hash(config):
    json_string = json.dumps(config)
    print(f" cnn_watch: Watch configuration ->{json_string}")
    encoded_string = json_string.encode()
    hash_object_md5 = hashlib.md5(encoded_string)
    hex_hash_md5 = hash_object_md5.hexdigest()
    return hex_hash_md5

def dircheck(dir_list, hash):
    for dir in dir_list:
        if dir == hash:
            print(f"cnn_watch: Found directory")
            dir_list2 = os.listdir(dir)
            for file in dir_list2:
                if file == "conflog.json":
                    print(f"cnn_watch: found config file:", file)
                    return read_json(os.path.join(hex_hash_md5,file))
    return None

def read_json(jfile): 
    with open(jfile) as json_file:
        return json.load(json_file)
        
def read_npy(dir, file): 
    return np.load(os.path.join(dir,file))

config={  "project" :"cnn-mon",
          "dataset": CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_NX,
          "epochs": 20,
          "dropout": True,
          "dropout_fraction":0.3,
          "architecture": "CNN3x3",
          "optimizer": "Adam",
          "criterion": "MSELos",
          "learning_rate": 0.001,
          "first_file": 0,  # initial file indx
          "last_file": 100,  # lasta file indx
          "norm": True, 
          "mean": 165.90, 
          "std": 93.3,
          "batch_size": 1000,
          "train_fraction" : 0.7, 
          "val_fraction": 0.2,
      }

hex_hash_md5 = get_hash(config)
 
path = os.getcwd() 
dir_list = os.listdir(path) 
jconf = dircheck(dir_list, hash=hex_hash_md5)

if jconf == None:
    print("Configuration file not found")
    sys.exit()

print(f"cnn_watch: Configuration file = {jconf}")

print(f"cnn_watch: read run results")

train_losses =read_npy(hex_hash_md5,"train_losses.npy")
val_losses   =read_npy(hex_hash_md5,"val_losses.npy")
delta_x_NN   =read_npy(hex_hash_md5,"delta_x_NN.npy")
delta_y_NN   =read_npy(hex_hash_md5,"delta_y_NN.npy")
delta_z_NN   =read_npy(hex_hash_md5,"delta_z_NN.npy")


# define a named tuple to hold the results

tdl = namedtuple('tdl', 'delta_x_NN delta_y_NN delta_z_NN')

print(f"cnn_watch: plot results")
plot_loss(config.get("epochs"), train_losses, val_losses,figsize=(10, 4))
plotxyz(tdl(delta_x_NN, delta_y_NN, delta_z_NN), nbins=50)

 
                    
            




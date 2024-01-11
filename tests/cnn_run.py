import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim

from pymono.config import CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_NX
from pymono.mono_dl import MonoDataset,mono_data_loader
from pymono.cnn_func import single_run,  CNN_3x3,train_cnn, evaluate_cnn

from tqdm.auto import tqdm
from enum import Enum

import wandb
import os
import json
import hashlib

class Debug(Enum):
    VVerbose = 1
    Verbose = 2
    Info = 3
    Quiet = 4
    Mute = 5.

def cnn_run(hyperparameters, project_name, verbose=Debug.Info):

    # tell wandb to get started
    with wandb.init(project=project_name, config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      dataset = MonoDataset(config.dataset, 
                            config.first_file, 
                            config.last_file,
                            norm=config.norm, 
                            mean=config.mean, 
                            std=config.std)

      _, train_loader, val_loader, test_loader = mono_data_loader(dataset, 
                                                            batch_size=config.batch_size, 
                                                            train_fraction=config.train_fraction, 
                                                            val_fraction=config.val_fraction) 
      
      model = CNN_3x3(dropout=config.dropout, 
                      dropout_fraction=config.dropout_fraction).to(device)
      
      if verbose == Debug.Quiet: print(model)
      optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
      criterion = nn.MSELoss() 
      

      # and use them to train the model
      train_losses, val_losses = train_cnn(train_loader, 
                                           val_loader, 
                                           model, 
                                           optimizer, 
                                           device, 
                                           criterion,
                                           config.epochs,
                                           verbose,
                                           log_freq=10,
                                           iprnt=200)
      
      

      # and test its final performance
      delta_x_NN, delta_y_NN, delta_z_NN = evaluate_cnn(test_loader, model, device)
      
    return model, train_losses, val_losses, delta_x_NN, delta_y_NN, delta_z_NN

def train_cnn(train_loader, val_loader, model, optimizer, device, criterion, epochs,
              verbose, log_freq=10,  iprnt=200):
    """
    train the CNN
    """

    def train_log(loss, example_ct, epoch, stage="loss_train"):
        loss = loss.data.item()
        wandb.log({"epoch": epoch, stage: loss}, step=example_ct)
    
    wandb.watch(model, criterion, log="all", log_freq=10)

    train_losses, val_losses = [], []
    example_ct = 0  # number of examples seen
    batch_ct = 0
    example_vt = 0  # number of examples seen
    batch_vt = 0

    if verbose == Debug.Info or Debug.Quiet: print(f" Running for {epochs} epochs")
    for epoch in tqdm(range(epochs)):
        train_losses_epoch, val_losses_epoch = [], []

        if verbose == Debug.Info: print(f" Info: EPOCH {epoch}")

        # Training step
        for i, (images, positions) in enumerate(train_loader):

            example_ct +=  len(images)
            batch_ct += 1

            images = images.to(device)
            positions = positions.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, positions) # compare labels with predictions

            # Report metrics 
            if ((batch_ct + 1) % log_freq) == 0:
                train_log(loss, example_ct, epoch, stage="loss_train")

            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            if((i+1) % (iprnt) == 0 and verbose == Debug.Verbose):
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            
            for i, (images, positions) in enumerate(val_loader):

                example_ct +=  len(images)
                batch_ct += 1

                images = images.to(device)
                positions = positions.to(device)

                outputs = model(images)
                loss = criterion(outputs, positions)

                if ((batch_vt + 1) % log_freq) == 0:
                    train_log(loss, example_vt, epoch, stage="loss_val")

                val_losses_epoch.append(loss.data.item())
                if((i+1) % (iprnt) == 0 and verbose == Debug.Verbose):
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        if verbose == Debug.Info: 
            print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
            print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    #logging.info(f"Out of loop after epoch ->{epoch}")
    return np.array(train_losses), np.array(val_losses)


def evaluate_cnn(test_loader, model, device):
    
    true_x, true_y, true_z = [],[],[]
    predicted_x, predicted_y, predicted_z = [],[],[]
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)


    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    predicted_x = np.array(predicted_x); predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    
    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    wandb.log({"delta_x_NN_mean": np.mean(delta_x_NN)})
    wandb.log({"delta_x_NN_std": np.std(delta_x_NN)})
    wandb.log({"delta_y_NN_mean": np.mean(delta_y_NN)})
    wandb.log({"delta_y_NN_std": np.std(delta_y_NN)})
    wandb.log({"delta_z_NN_mean": np.mean(delta_z_NN)})
    wandb.log({"delta_z_NN_std": np.std(delta_z_NN)})

    return delta_x_NN, delta_y_NN, delta_z_NN

def get_hash(config):
    json_string = json.dumps(config)
    print(json_string)
    encoded_string = json_string.encode()
    hash_object_md5 = hashlib.md5(encoded_string)
    hex_hash_md5 = hash_object_md5.hexdigest()
    return hex_hash_md5

def savef_files(hex_hash_md5,config,model, 
               train_losses, val_losses, 
               delta_x_NN, delta_y_NN, delta_z_NN):
    
    with open(os.path.join(hex_hash_md5,"conflog.json"), 'w') as json_file:
        json.dump(config, json_file)

    np.save(os.path.join(hex_hash_md5,"train_losses.npy"), train_losses)
    np.save(os.path.join(hex_hash_md5,"val_losses.npy"), val_losses)
    np.save(os.path.join(hex_hash_md5,"delta_x_NN.npy"), delta_x_NN)
    np.save(os.path.join(hex_hash_md5,"delta_y_NN.npy"), delta_y_NN)
    np.save(os.path.join(hex_hash_md5,"delta_z_NN.npy"), delta_z_NN)
    torch.save(model.state_dict(), os.path.join(hex_hash_md5,'model_state.pth'))


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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

print(f"Configuration ->", config)

model, train_losses, val_losses, delta_x_NN, delta_y_NN, delta_z_NN =cnn_run(config, 
                                                                     config["project"],
                                                                     verbose=Debug.Info)

hex_hash_md5 = get_hash(config)
   
if not os.path.exists(hex_hash_md5):
    os.mkdir(hex_hash_md5)

savef_files(hex_hash_md5,config,model, 
               train_losses, val_losses, 
               delta_x_NN, delta_y_NN, delta_z_NN)


import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from collections import namedtuple

from pymono.aux_func import weighted_mean_and_sigma

def single_run(train_loader, device, model, optimizer, criterion):
    """
    Run the model for a single event

    """
    print(f"** Run for 1 event**")

    for epoch in range(1):
        print(f"epoch = {epoch}")
    
        for i, (images, labels) in enumerate(train_loader):  
            if i>1: break
            print(f"i = {i}")
            print(f"images = {images.shape}")
            print(f"labels = {labels.shape}")
            images = images.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)

            print(f"outputs = {outputs.data.shape}")
           
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
    
            print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")


def single_runx(train_loader, device, model, optimizer, criterion):
    """
    Runs for one event applying a sum_pool2d to images (e.g, 16 x 16 --> 8 x 8, summing blocks of 4 pixels)
    """
    print(f"** Run for 1 event**")

    for epoch in range(1):
        print(f"epoch = {epoch}")
    
        for i, (images, labels) in enumerate(train_loader):  
            if i>1: break
            print(f"i = {i}")
            print(f"images before conv = {images.shape}")
            print(f"labels = {labels.shape}")
            imgsr = sum_pool2d(images, 2)
            print(f"images after conv = {imgsr.shape}")
            images = imgsr.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)

            print(f"outputs = {outputs.data.shape}")
           
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
    
            print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")




def train_cnn(train_loader, val_loader, model, optimizer, device, criterion, 
              batch_size, epochs=10, iprnt=100):
    """
    train the CNN
    """

    print(f"Training with  ->{len(train_loader)*batch_size} images")
    print(f"size of train loader  ->{len(train_loader)} images")
    print(f"Evaluating with  ->{len(val_loader)*batch_size} images")
    print(f"size of eval loader  ->{len(val_loader)} images")
    print(f"Running for epochs ->{epochs}")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        #logging.debug(f"\nEPOCH {epoch}")
        print(f"\nEPOCH {epoch}")

        # Training step
        for i, (images, positions) in enumerate(train_loader):

            images = images.to(device)
            positions = positions.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, positions) # compare labels with predictions
            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            
            if((i+1) % (iprnt * batch_size) == 0):
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with training after epoch ->{epoch}")
        #print(f"Start validations in epoch ->{epoch}")
        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            
            for i, (images, positions) in enumerate(val_loader):

                images = images.to(device)
                positions = positions.to(device)

                outputs = model(images)
                loss = criterion(outputs, positions)

                val_losses_epoch.append(loss.data.item())
                if((i+1) % (iprnt * batch_size) == 0):
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with validation after epoch ->{epoch}")
        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    #logging.info(f"Out of loop after epoch ->{epoch}")
    return train_losses, val_losses


def train_cnnx(train_loader, val_loader, model, optimizer, device, criterion, 
              batch_size, epochs=10, iprnt=100):
    """
    train the CNN applying a sum_pool2d to images (e.g, 16 x 16 --> 8 x 8, summing blocks of 4 pixels)
    """

    print(f"Training with  ->{len(train_loader)*batch_size} images")
    print(f"size of train loader  ->{len(train_loader)} images")
    print(f"Evaluating with  ->{len(val_loader)*batch_size} images")
    print(f"size of eval loader  ->{len(val_loader)} images")
    print(f"Running for epochs ->{epochs}")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        #logging.debug(f"\nEPOCH {epoch}")
        print(f"\nEPOCH {epoch}")

        # Training step
        for i, (images, positions) in enumerate(train_loader):
            imgsr = sum_pool2d(images, 2)

            if i < 3:
                print(f"images after conv = {imgsr.shape}")
            images = imgsr.to(device)
            positions = positions.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, positions) # compare labels with predictions
            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            if((i+1) % (iprnt) == 0):
                #logging.debug(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with training after epoch ->{epoch}")
        #print(f"Start validations in epoch ->{epoch}")
        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            
            for i, (images, positions) in enumerate(val_loader):
                imgsr = sum_pool2d(images, 2)
                images = imgsr.to(device)
                positions = positions.to(device)

                outputs = model(images)
                loss = criterion(outputs, positions)

                val_losses_epoch.append(loss.data.item())
                if((i+1) % (iprnt) == 0):
                    #logging.debug(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with validation after epoch ->{epoch}")
        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    #logging.info(f"Out of loop after epoch ->{epoch}")
    return train_losses, val_losses


def evaluate_cnn(test_loader, model, device, classical=False, pixel_size = 6, energy=False):
    """
    valuate the CNN returning the difference between true and predicted for the three coordinates

    """
    true_x, true_y, true_z = [],[],[]
    if energy:
        true_e = []

    mean_x, mean_y = [],[]
    sigma_x, sigma_y = [],[]
    predicted_x, predicted_y, predicted_z = [],[],[]
    if energy:
        predicted_e = []
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)
            if energy:
                for e in positions[:,3]: true_e.append(e)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)
            if energy:
                for e in outputs[:,3]: predicted_e.append(e)

            if classical:
                for img in images.cpu().squeeze().numpy():
                    mu_x, mu_y, sd_x, sd_y = weighted_mean_and_sigma(img)
                    mean_x.append(mu_x); mean_y.append(mu_y)
                    sigma_x.append(sd_x); sigma_y.append(sd_y)

    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    if energy:
        true_e = np.array(true_e)

    predicted_x = np.array(predicted_x) 
    predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    if energy:
       predicted_e = np.array(predicted_e)

    mean_x = np.array(mean_x); mean_y = np.array(mean_y)
    sigma_x = np.array(sigma_x); sigma_y = np.array(sigma_y)

    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    if energy:
        delta_e_NN = true_e - predicted_e

    # Compute deltas for the classical method
    if classical: 
        delta_x_classical = true_x - pixel_size*mean_x
        delta_y_classical = true_y - pixel_size*mean_y
    else:
        delta_x_classical = 0.0
        delta_y_classical = 0.0

    if energy:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_classical, delta_y_classical')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_classical, delta_y_classical)
    else:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical)
    
def evaluate_2c_cnn(test_loader, model, device):
    """
    valuate the CNN returning the difference between true and predicted for the 6 coordinates

    """
    true_x1, true_y1, true_z1 = [],[],[]
    true_x2, true_y2, true_z2 = [],[],[]
    

    predicted_x1, predicted_y1, predicted_z1 = [],[],[]
    predicted_x2, predicted_y2, predicted_z2 = [],[],[]
    with torch.no_grad():
        model.eval()
        for i, (images, positions) in enumerate(test_loader):
    
            images = images.to(device)
            outputs = model(images).cpu()
    
            for x in positions[:,0]: true_x1.append(x)
            for y in positions[:,1]: true_y1.append(y)
            for z in positions[:,2]: true_z1.append(z)
            for x in positions[:,3]: true_x2.append(x)
            for y in positions[:,4]: true_y2.append(y)
            for z in positions[:,5]: true_z2.append(z)
                
    
            for x in outputs[:,0]: predicted_x1.append(x)
            for y in outputs[:,1]: predicted_y1.append(y)
            for z in outputs[:,2]: predicted_z1.append(z)
            for x in outputs[:,3]: predicted_x2.append(x)
            for y in outputs[:,4]: predicted_y2.append(y)
            for z in outputs[:,5]: predicted_z2.append(z)
               
        # Convert to numpy arrays
        true_x1 = np.array(true_x1); true_y1 = np.array(true_y1); true_z1 = np.array(true_z1)
        true_x2 = np.array(true_x2); true_y2 = np.array(true_y2); true_z2 = np.array(true_z2)
        
        predicted_x1 = np.array(predicted_x1) 
        predicted_y1 = np.array(predicted_y1); predicted_z1 = np.array(predicted_z1)
        predicted_x2 = np.array(predicted_x2) 
        predicted_y2 = np.array(predicted_y2); predicted_z2 = np.array(predicted_z2)
        
    
    tdeltas12 = namedtuple('tdeltas12',
            'true_x1, true_y1, true_z1, true_x2, true_y2, true_z2, pred_x1, pred_y1, pred_z1, pred_x2, pred_y2, pred_z2')
    return tdeltas12(true_x1, true_y1, true_z1, true_x2, true_y2, true_z2, 
                     predicted_x1, predicted_y1, predicted_z1, predicted_x2, predicted_y2, predicted_z2)
    


def evaluate_cnnx(test_loader, model, device):
    """
    Evaluate the CNN returning the difference between true and predicted for the three coordinates
    applies a sum_pool2d to images (e.g, 16 x 16 --> 8 x 8, summing blocks of 4 pixels)

    """
    true_x, true_y, true_z = [],[],[]
    predicted_x, predicted_y, predicted_z = [],[],[]
    
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):
            imgsr = sum_pool2d(images, 2)
            images = imgsr.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)
            
            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)
            
    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    
    predicted_x = np.array(predicted_x) 
    predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    
    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    tdeltas = namedtuple('tdeltas','delta_x_NN, delta_y_NN, delta_z_NN')
    return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN)

def sum_pool2d(input, kernel_size, stride=None, padding=0):
    """
    Applies a 2D sum pooling over an input signal composed of several input planes.

    Args:
        input (Tensor): The input tensor containing the planes to be sum pooled.
        kernel_size (int or tuple): The size of the window to take a sum over.
        stride (int or tuple, optional): The stride of the window. Default value is kernel_size.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides.

    Returns:
        Tensor: The resulting tensor after applying the sum pooling.
    """

    # If stride is not set, use the kernel_size
    if stride is None:
        stride = kernel_size
    
    # Creating a kernel of ones. The number of ones is equal to the area of the kernel window.
    # The shape of the kernel is designed to match the input tensor's shape for the convolution operation.
    # The kernel is of the shape (C, 1, H, W) where C is the number of channels in the input tensor,
    # H and W are the height and width of the kernel respectively.
    n_channels = input.shape[1]
    one_kernel = torch.ones((n_channels, 1, kernel_size, kernel_size), device=input.device, dtype=input.dtype)

    # Conv2d with groups=n_channels makes sure each input channel is convolved with its corresponding kernel of ones,
    # essentially summing the elements in each window.
    output = F.conv2d(input, one_kernel, stride=stride, padding=padding, groups=n_channels)

    return output


def numpy_to_torch(input_array):
    """
    Converts a numpy array of shape (H, W) to a torch tensor of shape (1, 1, H, W).

    Args:
        input_array (numpy.ndarray): The input numpy array with shape (H, W).

    Returns:
        torch.Tensor: The converted torch tensor with shape (1, 1, H, W).
    """
    # Convert the numpy array to a torch tensor.
    tensor = torch.from_numpy(input_array)
    
    # Add two dimensions to get a shape of (1, 1, H, W).
    # The first unsqueeze adds a dimension at the beginning, making it (1, H, W),
    # and the second unsqueeze adds another dimension at the beginning, making it (1, 1, H, W).
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor


def image_shape_after_conv(I, P, K, S):
    """
    I: shape (1d) of input image
    P: Padding
    K: Kernel size
    S: stride
    """
    return ((I + 2*P - (K-1) -1)/S +1)


def cnn_evaluation(image, CNNT):
    """
    image : input image
    CNNT  : The convolutional network
    """

    def doconv(i, conv, pool, nconv):
        i1 = conv(i)
        print(f"shape of image after convolution {nconv:} = {i1.shape}")
        i2 = pool(i1)
        print(f"shape of image after pool {nconv:}= {i2.shape}")
        return i2
        
    print(f"shape of input image = {image.shape}")
    
    for i, cnnt in enumerate(CNNT):
        image = doconv(image, cnnt.conv, cnnt.pool, i) 

    print(f"shape of out image = {image.shape}")
    m = nn.Flatten()
    flat = m(image)
    print(f"shape of flattened image = {flat.shape}")

    
def cnn_xeval(val_loader, model, device, prnt=1000):

    print(f"Validation step: size of sample {len(val_loader)}")
    
    with torch.no_grad():  #gradients do not change
        model.eval()       # Sets the module in evaluation mode.
        correct = 0
        c1c = 0
        c2c = 0
        total = 0
        t1c = 0
        t2c = 0
            
        for i, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_0 = torch.where(labels == 0)[0]
            pred_0 = torch.where(predicted == 0)[0]
            labels_1 = torch.where(labels == 1)[0]
            pred_1 = torch.where(predicted == 1)[0]
            result0 = (labels_0[:, None] == pred_0).any(dim=1).sum()
            result1 = (labels_1[:, None] == pred_1).any(dim=1).sum()
            c1c += result0
            c2c += result1
            t1c += len(labels_0)
            t2c += len(labels_1)

            if i%prnt==0:
                print(f"i = {i}")
                print(f"labels = {labels}")
                print(f"predicted = {predicted}")
                print(f"correct = {(predicted == labels).sum().item()}")
                                
                print(f"labels 0 = {labels_0}")
                print(f"predicted 0 = {pred_0}")
                print(f"labels 1 = {labels_1}")
                print(f"predicted 1 = {pred_1}")

                print(f"result0 = {result0}")
                print(f"fraction of 1c == 1c => {result0/len(labels_0)}")
                print(f"result1 = {result1}")
                print(f"fraction of 2c == 2c =Z {result1/len(labels_1)}")
                

    return total, t1c, t2c, correct/total, c1c/t1c, c2c/t2c       
        
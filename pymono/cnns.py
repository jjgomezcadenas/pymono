import torch 
import torch.nn as nn
import numpy as np

from typing import NamedTuple, List

class Conv2dPars(NamedTuple):
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int


class MaxPool2dPars(NamedTuple):
    kernel_size: int
    stride: int


class LinealLayerPars(NamedTuple):
    in_dim: int
    out_dim: int


def build_conv_layer(conv2d : Conv2dPars, mxpl2d : MaxPool2dPars, relu='standard', slope=0.01) -> nn.Sequential:
    """
    Build a convolutiona layer with the following elements:
    1. nn.Conv2d(in_channels, out_channels, kernel_size, padding)
    2. nn.BatchNorm2d(out_channels)
    3. nn.ReLU (or nn.LeakyRelu)
    4. MaxPool2d(kernel_size, stride)

    """
    if relu == 'leaky':
        rl = nn.LeakyReLU(slope)
    else:
        rl = nn.ReLU()

    return nn.Sequential(
            nn.Conv2d(conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, padding=conv2d.padding),
            nn.BatchNorm2d(conv2d.out_channels),
            rl,
            nn.MaxPool2d(mxpl2d.kernel_size, mxpl2d.stride))


def build_conv_layers(layers :List[nn.Sequential])-> nn.Sequential:
    """
    Combine a list of Sequential into one Sequential.

    """
    return nn.Sequential(*layers)  


def build_linear_layers(layers : List[nn.Sequential], df=0.25) -> nn.Sequential:
    """
    Build a sequential of lineal layers, including Flatten and Dropout
    
    """
    LL=[]
    LL.append(nn.Flatten(start_dim=1))
    LL.append(nn.Dropout(p=df)) 
    
    for lyr in layers[0:-1]:
        LL.append(nn.Linear(lyr.in_dim, lyr.out_dim))
        LL.append(nn.ReLU())
    LL.append(nn.Linear(layers[-1].in_dim, layers[-1].out_dim))

    return nn.Sequential(*LL)   



class CNN(nn.Module):
    """
    Define a CNN with 
    - A set of convolutional layers (convlyrs)
    - A set of linear layers (llyrs), which must include Flattening and Dropout
    """
    def __init__(self,convlyrs : nn.Sequential, llyrs : nn.Sequential):
        super().__init__()
        self.conv_layer = convlyrs
        self.fc_layer = llyrs

    def forward(self, x):
        x = self.conv_layer(x)
        #x = x.view(x.size(0), -1)  # Flatten the output for the Dense layer
        x = self.fc_layer(x)
        return x



class ResBlock(nn.Module):
    """
    Implements a residual block consisting in [Conv2d->BatchNorm2d->ReLU] + 
    [Conv2d->BatchNorm2d]. This residual is added to the input (then a second activation ReLU applied)
    
    If downsample = None (e.g, default first pass), then we obtain f(x) + x where 
    f(x) -> [Conv2d->BatchNorm2d->ReLU ->Conv2d->BatchNorm2d]. Otherwise the block is skipped. 
    
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                                  stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                                  stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x # This is the residual (in the case of no downsample)
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:  # this is the residual in the case of downsample
            residual = self.downsample(x)
            
        out += residual # This is it! f(x) + x 
        out = self.relu(out)

        return out
    

class ResNet10(nn.Module):
    """
    Implements the Residual Network with 34 layers:
    The architecture is like this:
    1. Image passes through a convolution (kernel 3x3) 
    with stride = 1 and padding = 1 which increases the features from 3 to 64 and 
    preserves spatian dimensions, then batch normalization and activation. 
    # (W,H,1) => (W,H,64)
    
    2. The layer architecture is as follows (with a skip connection between each pair of layers) 
        6 layers of convolution 3x3 with 64 features
        8 layers of convolution 3x3 with 128 features (max pool 56 -> 28)
        12 layers of convolution 3x3 with 256 features (max pool 28 -> 24)
        6 layers of convolution 3x3 with 512 features (max pool 14 -> 7)
    3. Then avgpool and fc.
    
    """
    def __init__(self, block, in_channels=3, out_channels=32, num_classes = 3, dropout=False, dropout_fraction=0.2, debug=False):
        super().__init__()
        self.debug = debug
        self.inplanes = out_channels

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer0 = self._make_layer(block, out_channels, 1, stride = 1, nlyr = 1)
        self.layer1 = self._make_layer(block, out_channels*2, 1, stride = 2, nlyr = 2)
        self.layer2 = self._make_layer(block, out_channels*4, 1, stride = 2, nlyr = 3)
        self.layer3 = self._make_layer(block, out_channels*8, 1, stride = 2, nlyr = 4)
        self.layer4 = self._make_layer(block, out_channels*16, 1, stride = 2, nlyr = 5)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels*16, num_classes)
        self.dropout = dropout
        self.drop1 = nn.Dropout(p=dropout_fraction)

           
    def _make_layer(self, block, planes, blocks, stride, nlyr):
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        if self.debug:
            print(f" ## make_layer {nlyr}: planes = {planes},  blocks = {blocks}, stride = {stride}")
            print(f" ## make_layer: in_planes={self.inplanes}")
            print(f" ## make_layer: downsample = {downsample}")
            #print(f"layer block = 0: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = {stride}, downsample = {downsample}")
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            if self.debug:
                print(f" layer block = {i}: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = 1, downsample = None")

        return nn.Sequential(*layers)
    
    def forward(self, x):
        if(self.debug): print(f" ResNet10: input data shape =>{x.shape}")
            
        x = self.conv1(x)
        if(self.debug): print(f" ResNet10: after conv1 =>{x.shape}")
            
        #x = self.maxpool(x)
        #if(self.debug): print(f" ResNet: after maxpool =>{x.shape}")
            
        x = self.layer0(x)
        if(self.debug): print(f" ResNet10: after layer0 =>{x.shape}")
        
        x = self.layer1(x)
        if(self.debug): print(f" ResNet10: after layer1 =>{x.shape}")
            
        x = self.layer2(x)
        if(self.debug): print(f" ResNet10: after layer2 =>{x.shape}")
            
        x = self.layer3(x)
        if(self.debug): print(f" ResNet10: after layer3 =>{x.shape}")

        x = self.layer4(x)
        if(self.debug): print(f" ResNet10: after layer4 =>{x.shape}")
            
        x = self.pool(x)
        if(self.debug): print(f" ResNet10: after avgpool =>{x.shape}")

        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)
        if(self.debug): print(f" ResNet10: after flatten =>{x.shape}")
        
        if self.dropout: x = self.drop1(x)  # drop
        x = self.fc(x)
        if(self.debug): print(f" ResNet10: after fc =>{x.shape}")

        self.debug = False
        return x
  


class ResNet8(nn.Module):
    """
    Implements the Residual Network with 34 layers:
    The architecture is like this:
    1. Image passes through a convolution (kernel 3x3) 
    with stride = 1 and padding = 1 which increases the features from 3 to 64 and 
    preserves spatian dimensions, then batch normalization and activation. 
    # (W,H,1) => (W,H,64)
    
    2. The layer architecture is as follows (with a skip connection between each pair of layers) 
        6 layers of convolution 3x3 with 64 features
        8 layers of convolution 3x3 with 128 features (max pool 56 -> 28)
        12 layers of convolution 3x3 with 256 features (max pool 28 -> 24)
        6 layers of convolution 3x3 with 512 features (max pool 14 -> 7)
    3. Then avgpool and fc.
    
    """
    def __init__(self, block, in_channels=3, out_channels=32, num_classes = 3, dropout=False, dropout_fraction=0.2, debug=False):
        super().__init__()
        self.debug = debug
        self.inplanes = out_channels

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        
        self.layer0 = self._make_layer(block, out_channels, 1, stride = 1, nlyr = 1)
        self.layer1 = self._make_layer(block, out_channels*2, 1, stride = 2, nlyr = 2)
        self.layer2 = self._make_layer(block, out_channels*4, 1, stride = 2, nlyr = 3)
        self.layer3 = self._make_layer(block, out_channels*8, 1, stride = 2, nlyr = 4)
        #self.layer4 = self._make_layer(block, out_channels*16, 1, stride = 2, nlyr = 5)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels*8, num_classes)
        self.dropout = dropout
        self.drop1 = nn.Dropout(p=dropout_fraction)

           
    def _make_layer(self, block, planes, blocks, stride, nlyr):
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        if self.debug:
            print(f" ## make_layer {nlyr}: planes = {planes},  blocks = {blocks}, stride = {stride}")
            print(f" ## make_layer: in_planes={self.inplanes}")
            print(f" ## make_layer: downsample = {downsample}")
            #print(f"layer block = 0: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = {stride}, downsample = {downsample}")
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            if self.debug:
                print(f" layer block = {i}: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = 1, downsample = None")

        return nn.Sequential(*layers)
    
    def forward(self, x):
        if(self.debug): print(f"  ResNet8: input data shape =>{x.shape}")
            
        x = self.conv1(x)
        if(self.debug): print(f"  ResNet8: after conv1 =>{x.shape}")
            
        #x = self.maxpool(x)
        #if(self.debug): print(f" ResNet: after maxpool =>{x.shape}")
            
        x = self.layer0(x)
        if(self.debug): print(f"  ResNet8: after layer0 =>{x.shape}")
        
        x = self.layer1(x)
        if(self.debug): print(f"  ResNet8: after layer1 =>{x.shape}")
            
        x = self.layer2(x)
        if(self.debug): print(f"  ResNet8: after layer2 =>{x.shape}")
            
        x = self.layer3(x)
        if(self.debug): print(f"  ResNet8: after layer3 =>{x.shape}")

        #x = self.layer4(x)
        #if(self.debug): print(f" ResNet10: after layer4 =>{x.shape}")
            
        x = self.avgpool(x)
        if(self.debug): print(f"  ResNet8: after avgpool =>{x.shape}")

        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)
        if(self.debug): print(f"  ResNet8: after flatten =>{x.shape}")
        
        if self.dropout: x = self.drop1(x)  # drop
        x = self.fc(x)
        if(self.debug): print(f"  ResNet8: after fc =>{x.shape}")

        self.debug = False
        return x
  


def cnn_evaluation(image : torch.tensor, CL : List[nn.Sequential]):
    """
    Computes the shape of the output tensor after passing through
    each convolutional layer defined by CL

    image : input image
    CNNT  : The convolutional network

    """        
    print(f"shape of input image = {image.shape}")
    
    for i, cl in enumerate(CL):
        image = cl(image) 

        print(f" after cl = {i+1}, shape of out image = {image.shape}")
    m = nn.Flatten()
    flat = m(image)
    print(f"shape of flattened image = {flat.shape}")


def x_single_run(train_loader, device, model, optimizer, criterion, xc=True):
    """
    Classification (x) single run

    note use of torch.max
    a = torch.randn(4, 4)
    a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
    torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

    """
    print(f"** Run for 1 event**")

    for epoch in range(1):
        print(f"epoch = {epoch}")
    
        for i, (images, labels) in enumerate(train_loader):  
            if i>1: break
            print(f"i = {i}")
            print(f"labels shape = {labels.shape}")
            print(f"imgs shape = {images.shape}")
            print(f"labels = {labels}")
            images = images.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)

            print(f"outputs = {outputs.data.shape}")

            if xc:
                _, predicted = torch.max(outputs.data, 1)
                print(f"predicted label = {predicted}")
                correct = (predicted == labels).sum().item()
                print(f"correct = {correct}")

            loss = criterion(outputs, labels)
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
    
            print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

                
def x_train_cnn(train_loader, val_loader, model, optimizer, device, criterion, batch_size, xc=True,
               iprnt=100, epochs=10):
    """
    train the a CNN. If xc = True, this is a classification CNN and we compute also accuracy

    """
    print(f"Training with  ->{len(train_loader)*batch_size} images")
    print(f"size of train loader  ->{len(train_loader)} images")
    print(f"Evaluating with  ->{len(val_loader)*batch_size} images")
    print(f"size of eval loader  ->{len(val_loader)} images")
    print(f"Classification  ->{xc}")
    print(f"Running for epochs ->{epochs}")

    train_losses, val_losses = [], []
    if xc:
        acc = []
    
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        print(f"\nEPOCH {epoch}")
        print(f"training step: size of sample {len(train_loader)}")
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, labels) # compare labels with predictions
            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            if((i+1) % (iprnt * batch_size) == 0):
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")
                
        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            correct = 0
            total = 0

            print(f"Validation step: size of sample {len(val_loader)}")
            for i, (images, labels) in enumerate(val_loader):

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses_epoch.append(loss.data.item())
                
                if((i+1) % (iprnt * batch_size) == 0):
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

                if xc:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))

        if xc:
            acc.append(100 * correct / total)
        
        print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")

        if xc:
            print(f'Accuracy on the {len(val_loader)} validation images: {100 * correct / total} %') 
    
    if xc:
        return train_losses, val_losses, acc
    else:
        return train_losses, val_losses

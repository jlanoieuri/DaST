from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack
from sklearn.externals import joblib
#! from utils import load_data
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from net import Net_s, Net_m, Net_l
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34
cudnn.benchmark = True
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128

# Object to output the log into a file
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger('imitation_network_model.log', sys.stdout)

# The different arguments that can be passed when running the code, as described by their "help" strings
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='azure')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='alpha') # If beta is 0, loss is only about labels
parser.add_argument('--G_type', type=int, default=1, help='iteration limitation')
parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')

opt = parser.parse_args()
print(opt)

# Warn the user if they are not using a GPU but they have one available
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# If 'azure' is passes as an argument:
if opt.dataset == 'azure':

    # Use the MNIST dataset for testing
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                #! transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                #! transforms.RandomCrop(32, 4),
                                                #! normalize,
                                        ]))
    
    # Use a large synthetic model
    netD = Net_l().cuda()
    netD = nn.DataParallel(netD)

    # Load the target model
    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')

    # Set up the adversarial attack (I believe this is a different name for IFGSM)
    adversary_ghost = LinfBasicIterativeAttack(
        netD, # The sythetic model against which the adversarial examples are generated  
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), # The loss function to optimize
        eps=0.25, # The perturbation limit
        nb_iter=100, # The number of steps
        eps_iter=0.01, # The step size for each attack iteration
        clip_min=0.0, clip_max=1.0, # The bounds for the pixel values of the adversarial examples
        targeted=False) # Whether the attack is targeted or untargeted
    nc=1

# If 'mnist' is passed as an argument:
elif opt.dataset == 'mnist':

    # Use the MNIST dataset for testing
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                #! transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                #! transforms.RandomCrop(32, 4),
                                                #! normalize,
                                        ]))
    
    # Use a large synthetic model
    netD = Net_l().cuda()

    # Use parallelization
    netD = nn.DataParallel(netD)

    # Load the target model
    original_net = Net_m().cuda()
    state_dict = torch.load(
        'pretrained/net_m.pth')
    original_net.load_state_dict(state_dict)

    # Use parallelization
    original_net = nn.DataParallel(original_net)
    original_net.eval()

    # Set up the adversarial attack (I believe this is a different name for IFGSM)
    adversary_ghost = LinfBasicIterativeAttack(
        netD, # The synthetic model against which the adversarial examples are generated
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), # The loss function to maximize
        eps=0.25, # The perturbation limit
        nb_iter=200, # The number of steps
        eps_iter=0.02, # The step size for each attack iteration
        clip_min=0.0, clip_max=1.0, # The bounds for the pixel values of the adversarial examples
        targeted=False) # Whether the attack is targeted or untargeted
    nc=1

data_list = [i for i in range(6000, 8000)]

# Use batching to speed up training
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)
#! nc=1

# Use the GPU if 'cuda' is passed as an argument
device = torch.device("cuda:0" if opt.cuda else "cpu")

# Function used to initialize the weights of the training data generation model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Function used to obtain the target Azure model's output label for a given input
def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data) # Get label from target model
    output = torch.from_numpy(output).cuda().long()
    return output

# Function used to obtain the target Azure model's output probabilities for a given input
def cal_azure_proba(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict_proba(data) # Get probabilities from target model
    output = torch.from_numpy(output).cuda().float()
    return output

# Class for calculating the loss for the training data generation model
class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()
        return

    # Calculate the loss for the training data generation model G
    def forward(self, pred, truth, proba):
        criterion_1 = nn.MSELoss() # Mean squared error loss
        criterion = nn.CrossEntropyLoss() # Cross entropy loss
        pred_prob = F.softmax(pred, dim=1) # Get the probabilities of the synthetic model's output using softmax
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta # MSE + CE * weight
        # loss = criterion(pred, truth)
        final_loss = torch.exp(loss * -1) # L_G = e^{−d(T,D)}
        return final_loss

# Class for the deconvolution blocks used to learn the features of each class in the dataset within the training data generation model G
class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64

        # Configure the model layers based on the passed argument
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 2),# Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True) # Activation function
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.ReLU(True),  # Activation function

                #! nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                #! nn.BatchNorm2d(self.nf * 8),
                #! nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.ReLU(True),  # Activation function

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 4), # Normalization
                nn.ReLU(True), # Activation function

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.ReLU(True), # Activation function

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf), # Normalization
                nn.ReLU(True), # Activation function

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.shape[0]), # Normalization
                nn.ReLU(True), # Activation function

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False), # Convolutional layer
                #! if self.shape[0] == 3:
                #!     nn.Tanh()
                #! else:
                nn.Sigmoid() # Activation function, no need to normalize because Sigmoid constrains the outputs between 0 and 1
            )

    # Function to perform a forward pass through the model
    def forward(self, input):
        output = self.pre_conv(input)
        return output

# Initialize deconvolution blocks for each class in the dataset
pre_conv_block = []
for i in range (10):
    pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))

# The network that generates training data for the synthetic model
class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class  # The number of classes in the target model

        # Configure the model layers based on the passed argument
        if opt.G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 4), # Normalization  
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                #! nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                #! nn.BatchNorm2d(self.nf * 4),
                #! nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                #! nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                #! nn.BatchNorm2d(self.nf * 8),
                #! nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 4), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(nc), # Normalization
                nn.LeakyReLU(0.2, inplace=True), # Activation function

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False), # Convolutional layer
                nn.Sigmoid() # Activation function, no need to normalize because Sigmoid constrains the outputs between 0 and 1
            )
        elif opt.G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.ReLU(True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 2), # Normalization
                nn.ReLU(True), # Activation function

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 4), # Normalization
                nn.ReLU(True), # Activation function

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 4), # Normalization
                nn.ReLU(True), # Activation function

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.ReLU(True), # Activation function

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False), # Upsampling layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.ReLU(True), # Activation function

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False), # Convolutional layer
                nn.BatchNorm2d(self.nf * 8), # Normalization
                nn.ReLU(True)
            )

    # Function to perform a forward pass through the model
    def forward(self, input):
        output = self.main(input)
        return output

# Function used to split an array into m equal parts for batching
def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]

# Initialize the training data generation model
netG = Generator(10).cuda()

# Initialize the weights of the training data generation model
netG.apply(weights_init)

# Use parallelization when training the model
netG = nn.DataParallel(netG)

# Define a loss function to be cross entropy loss
criterion = nn.CrossEntropyLoss()

# Define a loss function for the training data generation model (L_G = e^{−d(T,D)})
criterion_max = Loss_max()

# Set up Adam optimizers for the training data generation model G and the synthetic model D
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#! optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#! optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Set up Adam optimizers for the deconvolution blocks for each class in the dataset
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

# Looks like it should estimate the accuracy of the synthetic model D before training
# In actuality this is finding the accuracy of the target model on the test data
with torch.no_grad(): # With no gradient calculation:
    correct_netD = 0.0 # Initilize the number of correct predictions to 0
    total = 0.0 # Initialize the total number of predictions to 0
    netD.eval()
    for data in testloader:
        inputs, labels = data # Split the data into inputs and labels
        inputs = inputs.cuda() # Move the inputs to the GPU
        labels = labels.cuda() # Move the labels to the GPU

        # The following line was commented out in the original code:
        # outputs = netD(inputs)

        # Pass the target model the inputs and get the predicted labels
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, inputs)
        else:
            outputs = original_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
        #! _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # Get the total number of predictions
        
        # Get the number of correct predictions
        correct_netD += (predicted == labels).sum()

    # Print the accuracy of the model's predictions compared to the test data labels
    print('Accuracy of the network on netD: %.2f %%' %
            (100. * correct_netD.float() / total))

# Estimate the attack success rate of the adversarial examples generated against the untrained synthetic model D
correct_ghost = 0.0 # Initialize the number of correct predictions by the target model on the adversarial examples
total = 0.0 # Initialize the total number of predictions by the target model on the adversarial examples
netD.eval()
for data in testloader:
    inputs, labels = data # Split the data into inputs and labels
    inputs = inputs.cuda() # Move the inputs to the GPU
    labels = labels.cuda() # Move the labels to the GPU

    # Generate adversarial examples using the IFGSM attack on the untrained synthetic model D
    adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)

    with torch.no_grad(): # With no gradient calculation:
        # Pass the target model the adversarial examples and get the predicted labels
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, adv_inputs_ghost)
        else:
            outputs = original_net(adv_inputs_ghost)
            _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0) # Get the total number of predictions
    correct_ghost += (predicted == labels).sum() # Get the number of correct predictions

# Print the attack success rate of the adversarial examples generated against the untrained synthetic model D
print('Attack success rate: %.2f %%' %
        (100 - 100. * correct_ghost.float() / total))

# Clean up memory
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

batch_num = 1000 # The number of generated training data batches
best_accuracy = 0.0 # The best accuracy of the synthetic model D's predictions matching the target model's predictions
best_att = 0.0 # The best attack success rate of the adversarial examples, generated against D, on the target model

# Train the training data generation model G and the synthetic model D
# for the number of epochs specified by the passed argument:
for epoch in range(opt.niter): # For each epoch:
    netD.train()

    for ii in range(batch_num): # For each batch:
        netD.zero_grad() # Zero the gradients for the synthetic model D

        # Update the synthetic model D
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device).cuda() # Generate random noise to be used 
        noise_chunk = chunks(noise, 10) # Split the noise into 10 equal parts for each class in the dataset
        for i in range(len(noise_chunk)):

            # Pass the ith chunk of noise through the ith deconvolution block learn features of the ith class
            tmp_data = pre_conv_block[i](noise_chunk[i])

            # Pass the deconvolved noise through the training data generation model G to get synthetic training data
            gene_data = netG(tmp_data)

            #! gene_data = netG(noise_chunk[i], i)

            # Label the generated training data with the ith class label
            label = torch.full((noise_chunk[i].size(0),), i).cuda()

            # Assign the intended label for the generated training data
            if i == 0:
                data = gene_data # Initialize the generated training data with the first batch of generated data
                set_label = label # Assign the intended label for the generated training data
            else:
                data = torch.cat((data, gene_data), 0) # Concatenate each batch of generated training data together
                set_label = torch.cat((set_label, label), 0) # Concatenate the intended labels for the generated training data together

        # Shuffle the generated training data and their corresponding intended labels (maintaining their pairing)
        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]

        # Get the target model's output for the generated training data
        with torch.no_grad():
            #! outputs = original_net(data)

            # If the model is Azure
            if opt.dataset == 'azure':
                outputs = cal_azure_proba(clf, data) # Get probabilities from target Azure model
                label = cal_azure(clf, data) # Get labels from target Azure model

            # If the model is not Azure
            else:
                outputs = original_net(data) # Get raw outputs from target model
                _, label = torch.max(outputs.data, 1) # Get the predicted labels from the target model
                outputs = F.softmax(outputs, dim=1) # Get the softmax probabilities from the target model
            #! _, label = torch.max(outputs.data, 1)
        #! print(label)
        
        # Get the synthetic model D's output for the generated training data
        output = netD(data.detach())

        # Get the softmax probabilities of the synthetic model D's output
        prob = F.softmax(output, dim=1)

        #! print(torch.sum(outputs) / 500.)
        # Get MSE loss between the probabilities of the synthetic model D  and the target model
        errD_prob = mse_loss(prob, outputs, reduction='mean')

        # Get the cross entropy loss between the predicted labels of the synthetic model D and the target model
        errD_fake = criterion(output, label) + errD_prob * opt.beta

        # Get the mean of the loss for logging purposes
        D_G_z1 = errD_fake.mean().item()

        # Backpropagate the loss for the synthetic model D
        errD_fake.backward()

        # Variable for logging the loss
        errD = errD_fake

        # Update the synthetic model D's parameters
        optimizerD.step()

        # Clean up memory
        del output, errD_fake

        # Update the training data generation model G
        netG.zero_grad() # Zero the gradients for the training data generation model G
        for i in range(10):
            pre_conv_block[i].zero_grad() # Zero the gradients for the deconvolution blocks for each class in the dataset
        output = netD(data) # Get the synthetic model D's output for the generated training data

        # Get the loss of the synthetic model D's output compared to the target model's output
        loss_imitate = criterion_max(pred=output, truth=label, proba=outputs)

        # Get the cross entropy loss between the predicted labels of the synthetic model D and the intended labels for the generated training data
        loss_diversity = criterion(output, set_label.squeeze().long())

        # Get the loss for G as a weighted sum of the loss_imitate and the loss_diversity
        errG = opt.alpha * loss_diversity + loss_imitate
        if loss_diversity.item() <= 0.1: # If the loss diversity is low:
            # Decrease the weight of the loss diversity to focus more on making the synthetic model D's output close to the target model's output
            opt.alpha = loss_diversity.item()
        
        # Backpropagate the loss for the training data generation model G
        errG.backward()

        # Get the mean of the loss for logging purposes
        D_G_z2 = errG.mean().item()

        # Update the training data generation model G's parameters
        optimizerG.step()

        # Update the deconvolution block for each class in the dataset's parameters
        for i in range(10):
            optimizer_block[i].step()

        # Log the losses every 40 batches
        if (ii % 40) == 0:
            print('[%d/%d][%d/%d] D: %.4f D_prob: %.4f G: %.4f D(G(z)): %.4f / %.4f loss_imitate: %.4f loss_diversity: %.4f'
                % (epoch, opt.niter, ii, batch_num,
                    errD.item(), errD_prob.item(), errG.item(), D_G_z1, D_G_z2, loss_imitate.item(), loss_diversity.item()))


    # Estimate the attack success rate of the adversarial examples generated against the trained synthetic model D
    correct_ghost = 0.0 # Initialize the number of correct predictions by the target model on the adversarial examples
    total = 0.0 # Initialize the total number of predictions by the target model on the adversarial examples
    netD.eval()
    for data in testloader:
        inputs, labels = data # Split the data into inputs and labels
        inputs = inputs.cuda() # Move the inputs to the GPU
        labels = labels.cuda() # Move the labels to the GPU

        # Generate adversarial examples using the IFGSM attack on the trained synthetic model D
        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)

        with torch.no_grad(): # With no gradient calculation:
            #! outputs = original_net(adv_inputs_ghost)

            # Pass the target model the adversarial examples and get the predicted labels
            if opt.dataset == 'azure':
                predicted = cal_azure(clf, adv_inputs_ghost)
            else:
                outputs = original_net(adv_inputs_ghost)
                _, predicted = torch.max(outputs.data, 1)
            #! _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) # Get the total number of predictions
            correct_ghost += (predicted == labels).sum() # Get the number of correct predictions
    
    # Print the attack success rate of the adversarial examples generated against the trained synthetic model D
    print('Attack success rate: %.2f %%' %
            (100 - 100. * correct_ghost.float() / total))
    
    # If this is the best model so far in terms of attack success rate save both models' parameters
    if best_att < (total - correct_ghost):
        torch.save(netD.state_dict(),
                    opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
        torch.save(netG.state_dict(),
                    opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
        best_att = (total - correct_ghost) # Update the best attack success rate
        print('This is the best model')
    worksheet.write(epoch, 0, (correct_ghost.float() / total).item())

    # Clean up memory
    del inputs, labels, adv_inputs_ghost
    torch.cuda.empty_cache()
    gc.collect()

    # Get the accuracy of the trained synthetic model D's predictions compared against the target model's predictions
    with torch.no_grad(): # With no gradient calculation:
        correct_netD = 0.0 # Initialize the number of correct predictions of the synthetic model D to 0
        total = 0.0 # Initialize the total number of predictions to 0
        netD.eval()
        for data in testloader:
            inputs, labels = data # Split the data into inputs and labels
            inputs = inputs.cuda() # Move the inputs to the GPU
            labels = labels.cuda() # Move the labels to the GPU
            outputs = netD(inputs) # Get the synthetic model D's output for the test data
            _, predicted = torch.max(outputs.data, 1) # Get the predicted class labels
            total += labels.size(0) # Update the total number of predictions
            correct_netD += (predicted == labels).sum() # Update the number of correct predictions

        # Print the accuracy of the trained synthetic model D's predictions compared against the target model's predictions
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))
        
        # If this is the best model so far in terms of accuracy, save both models' parameters
        if best_accuracy < correct_netD:
            torch.save(netD.state_dict(),
                       opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD # Update the best accuracy
            print('This is the best model')
    worksheet.write(epoch, 1, (correct_netD.float() / total).item())
workbook.save('imitation_network_saved_azure.xls')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Visualization
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split


np.set_printoptions(threshold=np.nan)
#load data
data = np.load('finalAB.npy')
labels = np.load('final_labelAB.npy')
# new_arr=[]
# for i in range(data.size):
#     img = data[i]
#     res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
#     new_arr.append(res)
# data=np.array(new_arr)
# final_data=[]
# final_label=[]
# for i in range(labels.size):
#     if(labels[i]==1 or labels[i]==2):
#         final_data.append(data[i])
#         final_label.append(labels[i])

# data = np.array(final_data)
# X = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
# print(X.shape)
# y = np.array(final_label)
X = data
# print(X.shape)
y = labels
# print(y.shape)
# fig = plt.figure(figsize=(24,4))
# fig.subplots_adjust(hspace=0.5)
# for i,index in enumerate(np.random.randint(0,100,8)):
#     ax = fig.add_subplot(2,5,i+1)
#     ax.imshow(X[index].reshape(50,50), cmap='gray')
#     ax.set_title("Label= {}".format(y[index]), fontsize = 20)
#     ax.axis('off')
# plt.show()

# plt.imshow(X[100].reshape(50,50))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Numpy to Tensor Conversion 
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Numpy to Tensor Conversion 
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)



# Make torch datasets from train and validation sets
train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

# Create train and test data loaders using oytorch
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)


class CNN(nn.Module):
    def __init__(self, input_dim = 2500, output_dim = 8):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 2500)
        self.layer2 = nn.Linear(2500, 128)
        self.layer3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.2)
    
    # Feed Forward Function
    def forward(self, modelVal):
        modelVal = modelVal.view(-1, 50 * 50)
        modelVal = F.relu(self.layer1(modelVal))
        modelVal = self.dropout(modelVal)
        modelVal = F.relu(self.layer2(modelVal))
        modelVal = F.relu(self.layer3(modelVal))
        modelVal = self.dropout(modelVal)
        modelVal = self.output_layer(modelVal)
        return modelVal

model = CNN(input_dim = 2500, output_dim = 8)
# visualize the architecture
# print(model)

lossFunction = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9,nesterov = False)

epochs = 10

epochval = []
listTrainingLoss = []
lossy = []
listTrainingacc = []
acc = []

model.train() 

for epoch in range(epochs):
    Losstrain = 0.0
    val_loss = 0.0       
    correct = 0
    total = 0
    
    # Load Train Images with Labels(Targets)
    for data, target in train_loader:
        # Convert our images and labels to Variables to accumulate Gradients
        data = Variable(data).float()
        target = Variable(target).type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        # find Training Accuracy 
        predicted = torch.max(output.data, 1)[1]        
        total += len(target)
        # Total true postives
        loss = lossFunction(output, target)
        correct += (predicted == target).sum()
       
        # backward pass
        loss.backward()
        optimizer.step()
        Losstrain = Losstrain + loss.item()*data.size(0)
    
    # calculate average training loss over an epoch
    Losstrain = Losstrain/len(train_loader.dataset)

    accuracy = 100 * correct / float(total)
    
    listTrainingacc.append(accuracy)
    listTrainingLoss.append(Losstrain)

 #cross validation     
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data).float()
            output = model(data)
            target = Variable(target).type(torch.LongTensor)
            loss = lossFunction(output, target)
            val_loss = val_loss +  loss.item()*data.size(0)
            predicted = torch.max(output.data, 1)[1]
            total = total + len(target)
            correct = correct + (predicted == target).sum()
    
    val_loss = val_loss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    
    acc.append(accuracy)
    lossy.append(val_loss)
     
    # print('In Epoch:{}, the training loss is {:.4f} and accuracy: {:.3f}%'.format(
    #     epoch+1, 
    #     Losstrain,
    #     accuracy
    #     ))
    epochval.append(epoch + 1)


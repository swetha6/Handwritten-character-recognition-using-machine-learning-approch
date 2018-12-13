import numpy as np
import sys
from CNN import model
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below 
    # to use command line, call: python hw07.py K.jpg output

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
        sys.exit(0)
    
    test_data = sys.argv[1]
    data = np.load(test_data)
    out = sys.argv[2]
    str = ".npy"
    outfile = out + str

    if len(sys.argv) == 4:
        debug = sys.argv[3] == '--debug'
    else:
        debug = False
    
new_arr=[]
for i in range(data.size):
    img = data[i]
    res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    new_arr.append(res)
data=np.array(new_arr)

data = np.array(data)
data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
x_test = Variable(torch.from_numpy(data)).float()
results = []
model.eval() 

with torch.no_grad():
    for image in x_test:
        output = model(image)
        predictions = torch.max(output.data, 1)[1]
        results.append(predictions[0].numpy())

# actualY = np.load("testYAB.npy")
results = np.array(results)
# print(accuracy_score(actualY,results))
np.save(outfile,results)


######################## UNCOMMENT TO RUN MLP #######################################
# from run import net
# import mlp
# outputs = net.output_file(data, 2)
# # print(accuracy_score(actualY,outputs))
# np.save("out_MLP.npy", outputs)


######################## UNCOMMENT TO RUN KNN #######################################
# from knn import clf
# res = clf.predict(data)
# # print(accuracy_score(res,actualY))
# np.save("out_KNN.npy", res)
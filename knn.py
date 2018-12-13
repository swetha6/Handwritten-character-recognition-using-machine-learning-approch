import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import cv2
from sklearn import preprocessing
from skimage.transform import resize
import numpy as np

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
# labels = np.array(final_label)

# data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
# data = preprocessing.normalize(data)
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.25,random_state=99)

clf=KNeighborsClassifier(n_neighbors=5).fit(x_train,y_train)
out = clf.predict(x_test)
print("accuracy found is")
print(accuracy_score(y_test,clf.predict(x_test)) * 100)




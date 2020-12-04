import pandas as pd
import numpy as np

animals=pd.read_csv("C:/Users/USER/Desktop/KNN-TECHNIQUE/knn-assignment/Zoo.csv")
animals1=animals.drop(["animal name"],axis='columns')

from sklearn.model_selection import train_test_split
train,test = train_test_split(animals1,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier as KNC
neigh = KNC(n_neighbors= 2)
animals1.shape

###model 

neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
####train accuracy and test accuracy
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])

###checking differnt accuracy in i values
acc=[]

for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:4],train.iloc[:,4])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])
    acc.append([train_acc,test_acc])




####plots for train and test acc
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])

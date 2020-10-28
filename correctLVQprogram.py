import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00476/buddymove_holidayiq.csv')
print(df)
df.shape
features=['Picnic','Religious','Nature','Theatre','Shopping']
X=df[features]
Y=df.Sports
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)
import numpy as np
dataset=np.array(dataset)
X=dataset[:,2:] #data
Y=dataset[:,1] #target
w1=np.random.ranf(5,) #initialising random weights 
w2=np.random.ranf(5,)
def distance(x,w):
    return(np.sqrt(np.sum((x-w)**2)))

def update_w1(x,w,lr):    # if Target=0
    return (w+lr*(x-w))

def update_w2(x,w,lr):    #if Target=1
    return (w-lr*(x-w))
    
 for i in range(1):
  print("Iteration: ",i+1)
  for j in range(249):
    print("Input:", X[j])
    print("Target", Y[j])
    d1=distance(X[j],w1)
    d2=distance(X[j],w2)
    if d1<=d2:
      if Y[j]==0:
        w1=update_w1(X[j],w1,lr)
      else:
        w1=update_w2(X[j],w1,lr)
    else:
      if Y[j]==1:
        w2=update_w1(X[j],w2,lr)
      else:
        w2=update_w2(X[j],w2,lr)   
  lr=lr*0.8
print("Updated lr:",lr)


#Checking the classification
Sample = [ 35, 99, 201, 190, 195] 
dis1=distance(Sample,w1)
dis2=distance(Sample,w2)
if(dis1>dis2):
  print("Sample belongs to Class 1")
else:
  print("Sample belongs to Class 2")      
    
    

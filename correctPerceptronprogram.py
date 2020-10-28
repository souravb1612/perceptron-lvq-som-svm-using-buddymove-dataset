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
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

clf= Perceptron()
clf.fit(X_train, Y_train)
clf.score(X,Y)

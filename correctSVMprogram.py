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
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
clf = svm.SVC(kernel='linear')
clf = clf.fit(X_train, Y_train)
o1=clf.predict(X_test)
accuracy_score(o1,Y_test)
o2=clf.predict(X_train)
accuracy_score(o2,Y_train)
support_vector_indices = clf.support_
print(support_vector_indices)

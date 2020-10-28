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

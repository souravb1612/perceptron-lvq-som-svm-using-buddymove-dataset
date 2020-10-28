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
Z=dataset[:,2:]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
Z = sc.fit_transform(Z)
from minisom import MiniSom
som = MiniSom( x = 10, y = 10, input_len = 5, sigma = 1.0, learning_rate = 0.6)
som.random_weights_init(Z)
som.train_random(data = Z, num_iteration = 100)
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
for i, x in enumerate(Z):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

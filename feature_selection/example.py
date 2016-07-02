from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = [[115.], [140.], [175.]]

weights = np.array(data)
scaler = MinMaxScaler()
rescaled_weights = scaler.fit_transform(weights)

print(rescaled_weights)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
from util import *
#numpy seed
np.random.seed(666)

#path parameters
dataset_path = '../data/amp_data.mat'

#load amp data
amp_file = scipy.io.loadmat(dataset_path)
amp_dataseq = amp_file['amp_data']
amp_dataset = []
#plt.plot(amp_dataseq)
#plt.hist(amp_dataseq, 101, density = 1, alpha = 0.75)
#plt.show()

#preprocess
for i in range(len(amp_dataseq) // 21):
    amp_dataset.append(amp_dataseq[i*21 : (i+1)*21])
amp_mat = np.array(amp_dataset).reshape(-1, 21)
#print(amp_mat.shape)
np.random.shuffle(amp_mat)

X_shuf_train = amp_mat[:math.ceil(0.7*amp_mat.shape[0]), :20]
X_shuf_val = amp_mat[math.ceil(0.7*amp_mat.shape[0]):math.ceil(0.85*amp_mat.shape[0]), :20]
X_shuf_test = amp_mat[math.ceil(0.85*amp_mat.shape[0]):, :20]

y_shuf_train = amp_mat[:math.ceil(0.7*amp_mat.shape[0]), 20]
y_shuf_val = amp_mat[math.ceil(0.7*amp_mat.shape[0]):math.ceil(0.85*amp_mat.shape[0]), 20]
y_shuf_test = amp_mat[math.ceil(0.85*amp_mat.shape[0]):, 20]

#print(X_shuf_train.shape)
time_steps = [i * 0.05 for i in range(20)]
linear_transform = linear_transform_1d(time_steps)
polynomial_transform = polynomial_transform_1d(time_steps, order = 4)

plt.plot([1.0], [y_shuf_train[0]], 'r.')
fit_plot(polynomial_transform, time_steps, X_shuf_train[0], polynomial_transform_1d, order = 4)

#fit_plot(linear_transform, time_steps, X_shuf_train[0], linear_transform_1d)
plt.show()

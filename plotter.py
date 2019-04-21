import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

A0 = np.loadtxt('./serial0/test.txt', delimiter=',')
A1 = np.loadtxt('./serial1/test.txt', delimiter=',')
A2 = np.loadtxt('./serial2/test.txt', delimiter=',')
B0 = np.loadtxt('./dist0/test.txt', delimiter=',')
B1 = np.loadtxt('./dist1/test.txt', delimiter=',')
B2 = np.loadtxt('./dist2/test.txt', delimiter=',')

#B=np.loadtxt('./save2/test.txt',delimiter=',')
#plt.ylim(ymin=-50)
scores_serial = np.zeros((A0.shape[0]//2,), dtype = np.float64) 
scores_dist = np.zeros((A0.shape[0]//2,), dtype = np.float64) 
for i in range(A0.shape[0]//2):
    scores_serial[i] = (A0[i, 2] + A1[i, 2] + A2[i, 2]) / 3.0  
    scores_dist[i] = (B0[i, 2] + B1[i, 2] + B2[i, 2]) / 3.0  
plt.plot(A0[:A0.shape[0]//2, 0], scores_serial)
plt.plot(A0[:A0.shape[0]//2, 0], scores_dist)
#plt.plot(A2[:,0], A2[:,2], 'r')
#plt.plot(B0[:,0], B0[:,2], 'g')
#plt.plot(B1[:,0], B1[:,2], 'g')
#plt.plot(B2[:,0], B2[:,2], 'g')
#plt.plot(B[:,0],B[:,1])
plt.xlabel('Training steps')
plt.ylabel('Average reward')
plt.legend(['1 workstation', '2 workstations'])
plt.title('Convergence of A2C algorithm')
plt.grid()
axes = plt.gca()
axes.ticklabel_format(style='plain')
#axes.xaxis.set_major_locator(MaxNLocator(integer=True))
#axes.set_ylim([-60,60])
plt.savefig('./plot.png')

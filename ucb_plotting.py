#%%
import matplotlib.pyplot as plt
from pickle import load
import numpy as np

#%%
a = load(open("./ucb/acc_pres.data", 'rb'))
ucb = load(open("./ucb/ucb.data", 'rb'))

# %%
colormap = np.array(['r', 'g', 'b'])
cat =  np.array(ucb[0])#ucb o has selected method for each itr


plt.figure(figsize=(30,10))
plt.grid(fillstyle = 'full')

scatter = plt.scatter(range(len(a[2][:90])),a[2][:90], s=500, c = colormap[cat])
plt.text(0, a[2][:90][0], 'margin', fontsize = 16)
plt.text(1, a[2][:90][1], 'entropy', fontsize = 16)
plt.text(2, a[2][:90][2], 'random', fontsize = 16)
plt.legend(['margin','entropy','random'])
plt.title("UCB selected techniques and the respective unlabeled data accuracy {C= 0.1,alpha =0.1 }")
plt.savefig('ubc.png')

'''
scatter = plt.scatter(range(len(ucb[3])),ucb[3], s=500, c = colormap[cat])
plt.text(0, ucb[3][0], 'margin', fontsize = 16)
plt.text(1, ucb[3][1], 'entropy', fontsize = 16)
plt.text(2, ucb[3][2], 'random', fontsize = 16)
plt.title("UCB selected techniques and the respective delta values {C= 0.1,alpha =0.1 }")


'''

# %%

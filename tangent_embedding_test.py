# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 20:19:30 2022

@author: Han Kheng
"""

import os
import glob 
import numpy as np
import torch as th
import tqdm
import matplotlib.pyplot as plt
import pickle


def dbhat(x1, x2, reduction='mean', dev='cuda', debug=False, chunks=1000):
    "from jialin's repo"
    # x1, x2 shape (num_points, num_samples, num_classes)
    np1, ns, _ = x1.size()
    np2, ns, _ = x2.size()
   
    x1, x2 = x1.transpose(0, 1), x2.transpose(0, 1)
    w = np.zeros([np1, np2])

    chunks = chunks or 1;
    for aa in (th.chunk(th.arange(ns), chunks)):
        xx1 = x1[aa, :].to(dev)
        xx2 = x2[aa, :].to(dev)
        aa = th.sqrt(aa)
        w_ = -th.log(th.bmm(th.sqrt(xx1), th.sqrt(xx2).transpose(1, 2)))
        w_[w_ > 1e12] = 10e6
        w_[w_ < 0] = 0
        w += w_.sum(0).cpu().numpy()
    if reduction == 'mean':
        return w/ns
    else:
        return w


def numerical_deriv(y,i,winlen):
    derivative_ = 0
    for j in range(1,winlen):
        derivative_ +=(y[i+j,:,:]-y[i-j,:,:])/(2*j)
    return derivative_/(winlen-1)

        

def linear_interp_ignorance_to_truth(lambda_,truth_,num_class=10):
    # lambda_ is a tuning param along the linearly interpolated path
    # grounnd truth is an array that contains the ground truth label of images nx1 array
    # randsample is an array of the indices sampled from N images
    # num_class is the number of labels 
    
    num_sample = truth_.shape[0]
    # create ignorance distributions
    ignorance_ = np.ones((num_sample,num_class))/num_class
    # compute dG
    dG = 2*np.arccos(np.sum(np.sqrt(ignorance_)*np.sqrt(truth_),1)).reshape(-1,1)
    # compute linear interpolation
    interp_ = np.sin((1-lambda_)*dG/2)*np.sqrt(ignorance_)/np.sin(dG/2) + np.sin(lambda_*dG/2)*np.sqrt(truth_)/np.sin(dG/2)
    return (interp_)**2

# save path 
save_path = r'D:\inpca_data\results\bootstrap_avg_models\allcnn'
# specifiy the directory of data stored
data_path = r'D:\inpca_data\results\models\allcnn_new'

num_chkpoints = 2042
n_seed = 30
t_jialin = np.array([    0,     1,    63,   125,   187,   249,   251,   313,   375,
      437,   499,   501,   563,   625,   687,   749,   751,   813,
      875,   937,   999,  1001,  1063,  1125,  1187,  1249,  1500,
     1750,  2000,  2250,  2500,  2750,  3000,  3250,  3500,  3750,
     4000,  4250,  4500,  4750,  5000,  5250,  5500,  5750,  6000,
     6250,  6500,  7250,  8250,  9250, 10250, 11250, 12250, 13250,
    14250, 15250, 16250, 19000, 22750, 26500, 30250, 34000, 37750,
    41500, 45250, 49000, 50000])
t_save = set(list(np.arange(1000))+list(t_jialin) + list(np.arange(5000,6000)))
t_save = list(t_save)
t_save.sort()
 
jialin_ind = [np.where(t_save==k)[0][0] for k in t_jialin]
## average model over 100:
t_extract = t_save[:50]+list(t_jialin)+list(range(5000,5050))
t_extract = np.unique(t_extract)

extract_ind = [np.where(t_save==k)[0][0] for k in t_extract]
plist = np.zeros((len(t_extract)*n_seed,50000,10))

late_ind = [np.where(t_save==k)[0][0] for k in t_extract]

truth_label= pickle.load(open(os.path.join(save_path,'truth_label.p'), 'rb')) 
truth_ = np.zeros((50000, 10))
truth_[np.arange(50000), truth_label] = 1

count = 0
for ns in tqdm.tqdm(range(n_seed)):
    tmp = th.load(os.path.join(data_path,'seed_'+str(ns),'allcnn_seed_'+str(ns)+'.p'))
    for m,n in enumerate(extract_ind):
        plist[count,:,:]=(np.exp(tmp[n]['yh'].numpy()))
        count +=1

plot_ind_init = np.array([np.arange(50)+s for s in np.arange(n_seed)*len(t_extract)])
plot_ind_final = np.array([np.arange(86,89)+s for s in np.arange(n_seed)*len(t_extract)])
plot_ind = np.array([np.arange(164)+s for s in np.arange(n_seed)*len(t_extract)])

'''
Compute average ignorance tangent vector
'''        
avg_tangent_init = np.array([numerical_deriv(plist[plot_ind[ii,:]],5,5) for ii in range(n_seed)])
avg_tangent_init_ = avg_tangent_init.mean(axis=0)

'''
Check plot for initial gradient estimation
'''
ns= 10
for c in range(10):
    for ii in range(n_seed):
        plt.subplot(3,5,c+1)
        plt.scatter(np.arange(50),plist[plot_ind[ii,:50],ns,c],s=.1,c='k')
        
    ys = [avg_tangent_init_[ns,c]*(t_extract[0]-t_extract[5])+plist[plot_ind[:,5],ns,c].mean(axis=0), avg_tangent_init_[ns,c]*(t_extract[10]-t_extract[5])+plist[plot_ind[:,5],ns,c].mean(axis=0)]
    plt.plot([0,10],ys ,'r')


'''
Compute probabilities on ignorance ray 
'''

ignorance = np.ones((50000,10))*0.1

init_ray_list = np.array([ignorance +avg_tangent_init_*i for i in np.linspace(0,50,50)])
init_ray_list /= np.sum(init_ray_list,2,keepdims = True)


np.sum(init_ray_list[49,:,:]<0)  # check for negative probabilities


'''
Compute average truth tangent vector
 -- very small gradient
 -- using a rough estimate for now -- slope for average probability at t_final - delta andtruth  

'''

#avg_tangent_final = np.array([numerical_deriv(plist[plot_ind[ii,:]],113,25) for ii in range(n_seed)])
#avg_tangent_final_ = avg_tangent_final.mean(axis=0)

avg_tangent_pos_final = np.array([plist[plot_ind[ii,113],:,:] for ii in range(n_seed)]).mean(axis=0)
grad_est = truth_-avg_tangent_pos_final # gonna be sloppy here and ignore the time step and take that into account when generating ray

'''
Compute probabilities on truth ray 
'''
final_ray_list = np.array([ truth_-grad_est*i for i in np.linspace(0,1.,50)])
final_ray_list /= np.sum(final_ray_list,2,keepdims = True)


np.sum(final_ray_list[49,:,:]<0) # check for negative probabilities


'''
Compute probabilities on ig2truth ray 
'''

ig2truth_rays = np.array([ ignorance+(truth_-ignorance)*i for i in np.linspace(0,1,50)])
ig2truth_rays /= np.sum(ig2truth_rays,2,keepdims = True)


ig2truth_ray = np.array([linear_interp_ignorance_to_truth(lambd_,truth_=truth_) for lambd_ in np.linspace(0,1,100)])  


pickle.dump(init_ray_list,open('init_rays.p','wb'))
pickle.dump(final_ray_list,open('final_rays.p','wb'))
pickle.dump(ig2truth_rays,open('ig2truth_rays.p','wb'))

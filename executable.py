import numpy as np
import os
from os import makedirs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rpy2.robjects as robjects

from numpy.random import randint
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import randn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import sys


#### Possible choices for this code:
GAN_version="SpatialCycleGAN"

#Physical variable?
var_phys="tas"
#var_phys="pr"

#### Period in the year?
season="winter_79_16"
#season="summer_79_16"
#season="annual_79_16"

#### Rank version or minmax version?
rank_version=False
#rank_version=True

Ref="SAFRAN"
Mod="IPSLMRbili"

lr_gen=0.0003
lr_disc=5e-06

#### Weights for valid, reconstruct and identity
lambda_val=5
lambda_rec=10
lambda_id=1

##################################################################
##### Automatized below according to the choices
#
sys.path.insert(1,'/gpfswork/rech/eal/urq13cl/CycleGAN/Script/')
if GAN_version=="SpatialCycleGAN":
    import CycleGAN
else:
    import SimpleGAN

### Downscaling (just for rmse plots)
if "SAFRANdet" in Mod:
    is_DS=True
else:
    is_DS=False

if var_phys=="tas":
    PR_version=False
else:
    PR_version=True

# Load temporal indices
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/")
load_index_temp = robjects.r.load('Temporal_indices_79_16.RData')
if season != "Annual_79_16":
    Ind_season=robjects.r['Ind_'+season]
    Ind_season=np.array(Ind_season)-1
else:
    Ind_season=np.array(range(13870))

#### Load Model 
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/" + Mod + "/")
if(rank_version==False):
    datasetA, LON_Paris, LAT_Paris, XminA_, XmaxA_, IND_Paris, point_max, OriginalA = CycleGAN.load_RData_minmax("tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",Ind_season)
else:
    datasetA, LON_Paris, LAT_Paris, XminA_, XmaxA_, IND_Paris, point_max, OriginalA = CycleGAN.load_RData_rank("tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",Ind_season)

#### Load Ref
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/" + Ref + "/")
if(rank_version==False):
    datasetB, LON_Paris, LAT_Paris, XminB_, XmaxB_, IND_Paris, point_max, OriginalB = CycleGAN.load_RData_minmax("tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",Ind_season)
else:
    datasetB, LON_Paris, LAT_Paris, XminB_, XmaxB_, IND_Paris, point_max, OriginalB = CycleGAN.load_RData_rank("tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",Ind_season)

#### Load QQ
if var_phys=="pr" and Mod != "SAFRANdetbili":
    BC1d="1dCDFt"
else:
    BC1d="1dQQ"

os.chdir("/gpfswork/rech/eal/commun/CycleGAN/MBC/" + Ref + "_" + Mod + "/")
if(rank_version==False):
    datasetQQ,_,_,_,_,_,_,OriginalQQ=CycleGAN.load_RData_minmax("tas_pr_day_PC0_"+BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys + "_day_PC0_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",Ind_season)
else:
    datasetQQ,_,_,_,_,_,_,OriginalQQ=CycleGAN.load_RData_rank("tas_pr_day_PC0_"+BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys+"_day_PC0_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",Ind_season)





########### Generation of Bias Correction #######
if rank_version==False:
    name_version="minmax"
else:
    name_version="rank"

savepath="/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/"+ Ref + "_" + Mod + "/SpatialCycleGAN/" + var_phys + "/winter_79_16/" + var_phys + '_' + name_version + '_lrgen'+str(lr_gen)+'_lrdisc'+str(lr_disc) +"_Relu_lval" + str(lambda_val) + "_lrec" + str(lambda_rec) + "_lid" + str(lambda_id) + '_new_archlocal_energy' #+"_new_arch"
os.chdir(savepath)


genA2B = load_model( savepath + '/models/genA2B_model_1361.h5')
datasetA2B = genA2B.predict(datasetA)



#### Functions
def from_rank_to_varphy(rank_version,PR_version,dataset_rank,original, Xmin_, Xmax_):
    res_dataset=np.copy(dataset_rank)
    if rank_version==True:
        for k in range(28):
            for l in range(28):
                quant_to_take=np.array(dataset_rank[:,k,l,0])
                res_dataset[:,k,l,0] = np.quantile(original[:,k,l,0],quant_to_take)
    else:
        #Rescale climatic variables wrt Xmin and Xmax
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                res_dataset[:,k,l,:] = res_dataset[:,k,l,:]*(Xmax_[n] - Xmin_[n])+ Xmin_[n]

    if PR_version==True:
        res_dataset[res_dataset < 1] = 0
    return res_dataset


##### Compute MAE/RMSE ####
def rmse(ref, pred):
    return np.sum((ref.astype("float") - pred.astype("float")) **2)/(ref.shape[1]*ref.shape[2]*ref.shape[0])

def mae(ref, pred):
    return np.sum(abs(ref.astype("float") - pred.astype("float"))/(ref.shape[1]*ref.shape[2]*ref.shape[0]))


#### Compute WD ####
from SBCK.metrics import wasserstein
from SBCK.tools.__OT import OTNetworkSimplex
from SBCK.tools.__tools_cpp import SparseHist
from SBCK.tools import bin_width_estimator

def compute_WD(sample_A, sample_B, ind,point_grid, bin_size= None):
    reversed_datasetA=np.transpose(sample_A[:,:,:,0],(2,1,0))
    reversed_datasetB=np.transpose(sample_B[:,:,:,0],(2,1,0))

    tmp_A=np.transpose(CycleGAN.transform_array_in_matrix(reversed_datasetA, ind, point_grid))
    tmp_B=np.transpose(CycleGAN.transform_array_in_matrix(reversed_datasetB, ind, point_grid))

    mu_tmp_A = SparseHist(tmp_A,bin_size)
    mu_tmp_B = SparseHist(tmp_B, bin_size)
    res=wasserstein(mu_tmp_B, mu_tmp_A)
    return res



### On real rank
from scipy.stats import *
def compute_matrix_rank(data):
    res=np.copy(data)
    for k in range(28):
        for l in range(28):
            res[:,k,l,0]=(rankdata(data[:,k,l,0],method="min")/len(data[:,k,l,0]))
    return res

#### On small samplesi
def compute_localwd_array_new(data_A,data_B, bin_width_size=None):
    def compute_small_WD(sample_A, sample_B,  bin_size= None):#input: nb_time*nb_var 
        mu_tmp_A = SparseHist(sample_A,bin_size)
        mu_tmp_B = SparseHist(sample_B, bin_size)
        res=wasserstein(mu_tmp_B, mu_tmp_A)
        return res

    def flat_array(data_array): #data_array of dim TIME x LAT x LON, output: nb_var*nb_time  
        nb_lon=data_array.shape[2]
        nb_lat=data_array.shape[1]
        nb_var=nb_lon*nb_lat
        nb_time=data_array.shape[0]
        res=np.zeros((nb_var,nb_time))
        k=-1
        for i in range(nb_lat):
            for j in range(nb_lon):
                k=k+1
                res[k,:]=data_array[:,i,j]
        return res

    nb_images=data_A.shape[0]
    res_localwd=np.reshape([None]*28*28,(1,28,28))
    smallA=np.zeros((9,3420))
    smallB= np.zeros((9,3420))
    for k in range(28):
        print('localWD' + str(k))
        for l in range(28):
            if k>0 and k<27:
                if l>0 and l<27:
                    smallA=flat_array(data_A[:,(k-1):(k+2),(l-1):(l+2),0])
                    smallB=flat_array(data_B[:,(k-1):(k+2),(l-1):(l+2),0])
                    res_localwd[:,k,l] = compute_small_WD(np.transpose(smallA),np.transpose(smallB),bin_size=bin_width_size)
    res_localwd = res_localwd.astype(float)
    return res_localwd



import dcor
def compute_localenergy_array_new(data_A,data_B,window_size=3):
    def flat_array(data_array): #data_array of dim TIME x LAT x LON, output: nb_var*nb_time 
        nb_lon=data_array.shape[2]
        nb_lat=data_array.shape[1]
        nb_var=nb_lon*nb_lat
        nb_time=data_array.shape[0]
        res=np.zeros((nb_var,nb_time))
        k=-1
        for i in range(nb_lat):
            for j in range(nb_lon):
                k=k+1
                res[k,:]=data_array[:,i,j]
        return res

    nb_images=data_A.shape[0]
    res_localenergy=np.reshape([None]*28*28,(1,28,28))
    smallA=np.zeros((9,nb_images))
    smallB= np.zeros((9,nb_images))
    if window_size==3:
        add_grid=2
    if window_size==2:
        add_grid=1
    for k in range(28):
        print('localenergy' + str(k))
        for l in range(28):
            if k>0 and k<27:
                if l>0 and l<27:
                    smallA=flat_array(data_A[:,(k-1):(k+add_grid),(l-1):(l+add_grid),0])
                    smallB=flat_array(data_B[:,(k-1):(k+add_grid),(l-1):(l+add_grid),0])
                    res_localenergy[:,k,l] = dcor.homogeneity.energy_test_statistic(np.transpose(smallA), np.transpose(smallB), exponent=1)/2
    res_localenergy = res_localenergy.astype(float)
    return res_localenergy




def plot_maps_localWD(epoch, PR_version, mat_A, mat_QQ, mat_A2B, title, path_plot, lon=np.array(range(27)), lat=np.array(range(27))):
    mat_A = mat_A.astype(float)
    mat_QQ = mat_QQ.astype(float)
    mat_A2B = mat_A2B.astype(float)
    mat_A = mat_A[:,1:27,1:27]
    mat_QQ = mat_QQ[:, 1:27, 1:27]
    mat_A2B = mat_A2B[:, 1:27, 1:27]
    #### On inverse LON_LAT pour plotter correctement
    ####fliplr for (1,28,28), else flipud
    mat_A=np.fliplr(mat_A)
    mat_QQ=np.fliplr(mat_QQ)
    mat_A2B=np.fliplr(mat_A2B)
    #### Mean and sd / MAE ####
    if PR_version==False:
        examples = vstack((mat_A, mat_QQ, mat_A2B, mat_A-mat_A, mat_QQ-mat_A, mat_A2B-mat_A))
        names_=("A","QQ","A2B","A-A","QQ-A","A2B-A")
    else:
        examples = vstack((mat_A, mat_QQ, mat_A2B, (mat_A-mat_A)/mat_A, (mat_QQ-mat_A)/mat_A, (mat_A2B-mat_A)/mat_A))
        names_=("A","QQ","A2B","(A-A)/A","(QQ-A)/A","(A2B-A)/A")
    nchecks=3
    fig, axs = pyplot.subplots(2,nchecks, figsize=(10,10))
    cm = ['YlOrRd','RdBu']
    fig.subplots_adjust(right=0.925) # making some room for cbar
    quant_10=0
    quant_90=np.quantile(mat_A,0.9)
    for row in range(2):
        for col in range(nchecks):
            i=3*row+col
            ax = axs[row,col]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if (row < 1):
                vmin = quant_10
                vmax = quant_90
                pcm = ax.imshow(examples[i,:,:], cmap =cm[row],vmin=vmin, vmax=vmax)
                ax.set_title(str(names_[i]) + ' mean: ' +str(round(np.nanmean(examples[i, :, :]),3)) + ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),3))  ,fontsize=10)

            else:
                vmin=-0.3
                vmax=0.3
                pcm = ax.imshow(examples[3*row + col, :,:], cmap = cm[row], vmin=vmin, vmax=vmax)
                ax.set_title(str(names_[i]) + ' mae: ' +str(round(np.nanmean(abs(examples[i, :, :])),3)) + ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),3)))
        fig.colorbar(pcm, ax = axs[row,:],shrink=0.5)
    pyplot.show()










dataset_varphy_A2B=from_rank_to_varphy(rank_version,PR_version,datasetA2B, OriginalB, XminB_, XmaxB_)

if PR_version==True:
    OriginalA[OriginalA < 1] = 0
    OriginalB[OriginalB < 1] = 0
    OriginalQQ[OriginalQQ <1] =0


if is_DS==True:
    rmse_rank_A = rmse(datasetB, datasetA)
    rmse_rank_A2B = rmse(datasetB, datasetA2B)
    rmse_rank_QQ = rmse(datasetB, datasetQQ)
    print("RMSE rank")
    print("A: " + str(rmse_rank_A))
    print("A2B: " + str(rmse_rank_A2B))
    print("QQ: " + str(rmse_rank_QQ))

    mae_rank_A = mae(datasetB, datasetA)
    mae_rank_A2B = mae(datasetB, datasetA2B)
    mae_rank_QQ = mae(datasetB, datasetQQ)
    print("MAE rank")
    print("A" + str(mae_rank_A))
    print("A2B" + str(mae_rank_A2B))
    print("QQ" + str(mae_rank_QQ))

    rmse_varphy_A = rmse(OriginalA, OriginalB)
    rmse_varphy_A2B = rmse(dataset_varphy_A2B, OriginalB)
    rmse_varphy_QQ = rmse(OriginalQQ, OriginalB)
    print("RMSE varphy")
    print("A: " + str(rmse_varphy_A))
    print("A2B: " + str(rmse_varphy_A2B))
    print("QQ: " + str(rmse_varphy_QQ))

    mae_varphy_A = mae(OriginalA, OriginalB)
    mae_varphy_A2B = mae(dataset_varphy_A2B, OriginalB)
    mae_varphy_QQ = mae(OriginalQQ, OriginalB)
    print("MAE varphy")
    print("A: " + str(mae_varphy_A))
    print("A2B: " + str(mae_varphy_A2B))
    print("QQ: " + str(mae_varphy_QQ))



realrankA=compute_matrix_rank(OriginalA)
realrankB=compute_matrix_rank(OriginalB)
realrankA2B=compute_matrix_rank(dataset_varphy_A2B)
realrankQQ=compute_matrix_rank(OriginalQQ)

#### Energy distance
essA=compute_localenergy_array_new(realrankA, realrankB, window_size=3)
essQQ=compute_localenergy_array_new(realrankQQ, realrankB, window_size=3)
essA2B=compute_localenergy_array_new(realrankA2B, realrankB, window_size=3)

print("ok")

import matplotlib.pyplot as pyplot
matplotlib.use('TkAgg')
plot_maps_localWD(4,False,essA,essQQ,essA2B,"ok","ok")
print("fin ok")
#





#bin_width_realrank = CycleGAN.compute_bin_width_wasserstein(realrankB, IND_Paris, point_max)
#wd_rank_A=compute_WD(realrankA, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#wd_rank_A2B=compute_WD( realrankA2B, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#wd_rank_QQ=compute_WD(realrankQQ, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#print("Wasserstein on real rank bin_width: " + str(round(bin_width_realrank[1],3)))
#print("A: " + str(wd_rank_A))
#print("A2B: " + str(wd_rank_A2B))
#print("QQ: " + str(wd_rank_QQ))




essA=compute_localwd_array_new(realrankA, realrankB, [0.05]*9)
essQQ=compute_localwd_array_new(realrankQQ, realrankB, [0.05]*9)
essA2B=compute_localwd_array_new(realrankA2B, realrankB, [0.05]*9)

print("ok")

import matplotlib.pyplot as pyplot
matplotlib.use('TkAgg')
plot_maps_localWD(4,False,essA,essQQ,essA2B,"ok","ok")
print("fin ok")
#
#def compute_small_WD(sample_A, sample_B,  bin_size= None):
#    mu_tmp_A = SparseHist(sample_A,bin_size)
#    mu_tmp_B = SparseHist(sample_B, bin_size)
#    res=wasserstein(mu_tmp_B, mu_tmp_A)
#    return res
#
#def flat_array(data_array): #data_array of dim TIME x LAT x LON
#    nb_lon=data_array.shape[2]
#    nb_lat=data_array.shape[1]
#    nb_var=nb_lon*nb_lat
#    nb_time=data_array.shape[0]
#    res=np.zeros((nb_var,nb_time))
#    k=-1
#    for i in range(nb_lat):
#        for j in range(nb_lon):
#            k=k+1
#            res[k,:]=data_array[:,i,j]
#    return res
#
#maps_WD_localA = np.empty((28,28))
#maps_WD_localA[:] = np.NaN
#maps_WD_localA2B = np.empty((28,28))
#maps_WD_localA2B[:] = np.NaN
#maps_WD_localQQ = np.empty((28,28))
#maps_WD_localQQ[:] = np.NaN
#
#
#
#smallA=np.zeros((9,3420))
#smallB= np.zeros((9,3420))
#smallA2B=np.zeros((9,3420))
#smallQQ=np.zeros((9,3420))
#k=-1
#bin_width_small=[0.05]*9
#for i in range(28):
#    print(i)
#    for j in range(28):
#        if i>0 and i<27:
#            if j>0 and j<27:
#                smallA=flat_array(realrankA[:,(i-1):(i+2),(j-1):(j+2),0])
#                #print(smallA.shape)
#                smallB=flat_array(realrankB[:,(i-1):(i+2),(j-1):(j+2),0])
#                smallA2B=flat_array(realrankA2B[:,(i-1):(i+2),(j-1):(j+2),0])
#                smallQQ=flat_array(realrankQQ[:,(i-1):(i+2),(j-1):(j+2),0])
#                maps_WD_localA[j,i] = compute_small_WD(np.transpose(smallA),np.transpose(smallB),bin_size=bin_width_small)
#                maps_WD_localA2B[j,i] = compute_small_WD(np.transpose(smallA2B),np.transpose(smallB),bin_size=bin_width_small)
#                maps_WD_localQQ[j,i] = compute_small_WD(np.transpose(smallQQ), np.transpose(smallB),bin_size=bin_width_small)

import matplotlib.pyplot as pyplot
matplotlib.use('TkAgg')



print(np.nanmean(maps_WD_localA))
print(np.nanmean(maps_WD_localA2B))
print(np.nanmean(maps_WD_localQQ))
pyplot.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = pyplot.subplots(3,1)


# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(maps_WD_localA)
axarr[1].imshow(maps_WD_localA2B)
axarr[2].imshow(maps_WD_localQQ)
pyplot.show()



bin_width_=bin_width_estimator(np.transpose(smallB)) #bcse bin_width_estimator needs data of type TIME x nb_var
bin_width_small=[0.05]*len(bin_width_)


wd_rank_A=compute_small_WD(np.transpose(smallA), np.transpose(smallB),bin_size = bin_width_small)
wd_rank_A2B=compute_small_WD(np.transpose(smallA2B), np.transpose(smallB),bin_size = bin_width_small)
wd_rank_QQ=compute_small_WD(np.transpose(smallQQ), np.transpose(smallB),bin_size = bin_width_small)
print("Small Wasserstein on real rank bin_width: " + str(round(bin_width_small[1],3)))
print("A: " + str(wd_rank_A))
print("A2B: " + str(wd_rank_A2B))
print("QQ: " + str(wd_rank_QQ))







bin_size_rank=CycleGAN.compute_bin_width_wasserstein(datasetB, IND_Paris, point_max)
wd_rank_A=compute_WD(datasetA,datasetB,IND_Paris, point_max, bin_size = bin_size_rank)
wd_rank_A2B=compute_WD(datasetA2B,datasetB,IND_Paris, point_max, bin_size = bin_size_rank)
wd_rank_QQ=compute_WD(datasetQQ, datasetB, IND_Paris, point_max, bin_size = bin_size_rank)
print("Wasserstein rank")
print("A: " + str(wd_rank_A))
print("A2B: " + str(wd_rank_A2B))
print("QQ: " + str(wd_rank_QQ))


bin_size_varphy = CycleGAN.compute_bin_width_wasserstein(OriginalB, IND_Paris, point_max)
wd_varphy_A = compute_WD(OriginalA, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
wd_varphy_A2B = compute_WD(dataset_varphy_A2B, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
wd_varphy_QQ = compute_WD( OriginalQQ, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
print("Wasserstein varphy")
print("A: " + str(wd_varphy_A))
print("A2B: " + str(wd_varphy_A2B))
print("QQ: " + str(wd_varphy_QQ))


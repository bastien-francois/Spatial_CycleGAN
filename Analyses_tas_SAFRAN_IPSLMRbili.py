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
from math import *
import dcor
from scipy.stats import *

### Possible choices for this code:
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

#### QQ data en input? 
#QQ2B_version=False 
QQ2B_version=True

#### CV_version?
#CV_version="PC0"
CV_version="CVunif"
#CV_version="CVchrono"

#### Hyperparameters?: learning rate of disc and gen?
lr_gen=1e-4
lr_disc=5e-5

### L-norm
L_norm="L1norm"
#L_norm = "L2norm"

Ref="SAFRAN"
#Mod="IPSLMRbili"
Mod="SAFRANdetbili"

#### Wasserstein distances?
computation_WD=False
computation_localWD=False

computation_energy=False
computation_localenergy=True

#### Weights for valid, reconstruct and identity
lambda_val=1
lambda_rec=10
lambda_id=1

nb_filters_disc=[64,128]
nb_filters_gen=[64,128,256]

ite_to_take= 4001

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

### For QQ loading
if var_phys=="pr" and Mod is not "SAFRANdetbili":
    BC1d="1dCDFt"
else:
    BC1d="1dQQ"


#### Load Model
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/" + Mod + "/")
CalibA, ProjA,LON_Paris,LAT_Paris,minCalibA, maxCalibA, IND_Paris,point_max, OriginalCalibA, OriginalProjA = CycleGAN.load_calib_proj_minmaxrank(CV_version, rank_version,"tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",season)

####Load Ref
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/" + Ref + "/")
CalibB, ProjB, LON_Paris, LAT_Paris, minCalibB, maxCalibB,IND_Paris, point_max, OriginalCalibB, OriginalProjB = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version,"tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",season)


#### Load QQ
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/MBC/" + Ref + "_" + Mod + "/" +  CV_version)
CalibQQ, ProjQQ,_,_,minCalibQQ, maxCalibQQ,_,_,OriginalCalibQQ, OriginalProjQQ = CycleGAN.load_calib_proj_minmaxrank(CV_version,rank_version, "tas_pr_day_" + CV_version + "_" + BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys + "_day_" + CV_version + "_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",season)

if QQ2B_version==True:
    minCalibX=minCalibQQ
    maxCalibX=maxCalibQQ
else:
    minCalibX=minCalibA
    maxCalibX=maxCalibA




########### Generation of Bias Correction #######
if rank_version==False:
    name_version="minmax"
else:
    name_version="rank"

if QQ2B_version==True:
    name_QQ2B="QQ2B"
else:
    name_QQ2B="A2B"
if nb_filters_disc == [64,128]:
    name_new_arch="_new_arch"
else:
    name_new_arch=""
if computation_localenergy==True:
    name_local_energy="_local_energy"
else:
    name_local_energy= ""

savepath="/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/"+ Ref + "_" + Mod + "/SpatialCycleGAN/" + var_phys + "/" + CV_version + "/" + name_QQ2B + "/winter_79_16/" + var_phys + '_' + name_version + "_" + CV_version  +  '_lrgen'+str(lr_gen)+'_lrdisc'+str(lr_disc) +"_Relu_lval" + str(lambda_val) + "_lrec" + str(lambda_rec) + "_lid" + str(lambda_id) + "_" + name_QQ2B +"_" + L_norm +  name_new_arch + name_local_energy

os.chdir(savepath)


genX2B = load_model( savepath + '/models/genX2B_model_' + str(ite_to_take) + '.h5')

if QQ2B_version==True:
    sample_CalibX = np.copy(CalibQQ)
    OriginalCalibX = np.copy(OriginalCalibQQ)
    sample_ProjX = np.copy(ProjQQ)
    OriginalProjX = np.copy(OriginalProjQQ)
else:
    sample_CalibX = np.copy(CalibA)
    OriginalCalibX = np.copy(OriginalCalibA)
    sample_ProjX = np.copy(ProjA)
    OriginalProjX = np.copy(OriginalProjA)


sample_CalibX2B = genX2B.predict(sample_CalibX)
sample_ProjX2B = genX2B.predict(sample_ProjX)


def denormalize_minmax(data, minX, maxX):
    res = np.copy(data)
    n=-1
    for k in range(28):
        for l in range(28):
            n=n+1
            res[:,k,l,:] = data[:,k,l,:]*(maxX[n] - minX[n])+ minX[n]
    return res

if rank_version==False:
    #Rescale climatic variables wrt Xmin and Xmax
    sample_varphy_CalibX2B=denormalize_minmax(sample_CalibX2B, minCalibB, maxCalibB)
    sample_varphy_ProjX2B=denormalize_minmax(sample_ProjX2B, minCalibB, maxCalibB)

#####################################################################################################################
#### VARPHY_REORDERED ######## alla Cannon
#####################################################################################################################
sample_varphy_Calib_mQQsX2B= np.copy(sample_varphy_CalibX2B)
sample_varphy_Proj_mQQsX2B= np.copy(sample_varphy_ProjX2B)

sample_varphy_Calib_mQQsX2B= CycleGAN.alla_Cannon(OriginalCalibQQ, sample_varphy_CalibX2B)
sample_varphy_Proj_mQQsX2B= CycleGAN.alla_Cannon(OriginalProjQQ, sample_varphy_ProjX2B)

### mBsX2B
sample_varphy_Calib_mBsX2B = np.copy(sample_varphy_CalibX2B)
sample_varphy_Proj_mBsX2B = np.copy(sample_varphy_ProjX2B)

sample_varphy_Calib_mBsX2B= CycleGAN.alla_Cannon(OriginalCalibB, sample_varphy_CalibX2B)
sample_varphy_Proj_mBsX2B= CycleGAN.alla_Cannon(OriginalProjB, sample_varphy_ProjX2B)


print("ok alla Cannon")

###!!!! Preprocess for PR !!!
OriginalCalibA_preproc=np.copy(OriginalCalibA)
OriginalCalibB_preproc = np.copy(OriginalCalibB)
OriginalCalibQQ_preproc = np.copy(OriginalCalibQQ)

OriginalProjA_preproc=np.copy(OriginalProjA)
OriginalProjB_preproc = np.copy(OriginalProjB)
OriginalProjQQ_preproc = np.copy(OriginalProjQQ)

if PR_version==True:
    th = np.min(np.concatenate((OriginalCalibB[OriginalCalibB>0], OriginalProjB[OriginalProjB>0])))
    print("Seuil: " + str(th))
    OriginalCalibA_preproc[OriginalCalibA_preproc < th]=0
    OriginalCalibB_preproc[OriginalCalibB_preproc < th]=0
    OriginalCalibQQ_preproc[OriginalCalibQQ_preproc < th]=0
    sample_varphy_CalibX2B[sample_varphy_CalibX2B < th] = 0
    sample_varphy_Calib_mQQsX2B[sample_varphy_Calib_mQQsX2B < th] = 0
    sample_varphy_Calib_mBsX2B[sample_varphy_Calib_mBsX2B < th] = 0

    OriginalProjA_preproc[OriginalProjA_preproc < th]=0
    OriginalProjB_preproc[OriginalProjB_preproc < th]=0
    OriginalProjQQ_preproc[OriginalProjQQ_preproc < th]=0
    sample_varphy_ProjX2B[sample_varphy_ProjX2B < th] = 0
    sample_varphy_Proj_mQQsX2B[sample_varphy_Proj_mQQsX2B < th] = 0
    sample_varphy_Proj_mBsX2B[sample_varphy_Proj_mBsX2B < th] = 0



#################################################################################
#### Compute realrank ####
################################################################################
sample_realrank_CalibA = CycleGAN.compute_matrix_real_rank(OriginalCalibA_preproc)
sample_realrank_CalibB = CycleGAN.compute_matrix_real_rank(OriginalCalibB_preproc)
sample_realrank_CalibQQ = CycleGAN.compute_matrix_real_rank(OriginalCalibQQ_preproc)
sample_realrank_CalibX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_CalibX2B)
sample_realrank_Calib_mQQsX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_Calib_mQQsX2B)
sample_realrank_Calib_mBsX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_Calib_mBsX2B)


sample_realrank_ProjA = CycleGAN.compute_matrix_real_rank(OriginalProjA_preproc)
sample_realrank_ProjB = CycleGAN.compute_matrix_real_rank(OriginalProjB_preproc)
sample_realrank_ProjQQ = CycleGAN.compute_matrix_real_rank(OriginalProjQQ_preproc)
sample_realrank_ProjX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_ProjX2B)
sample_realrank_Proj_mQQsX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_Proj_mQQsX2B)
sample_realrank_Proj_mBsX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_Proj_mBsX2B)




path_plot = "/gpfswork/rech/eal/urq13cl/CycleGAN/Results"
os.chdir(path_plot)

#################################################################################################
##### Compute local-energy ####
#################################################################################################
#path_plot = "/gpfswork/rech/eal/urq13cl/CycleGAN/Results"

bool_compute_local_energy=False

if bool_compute_local_energy==True:
    ####### sur varphy ####
    ###
    ###### Calib
    print("local energy Calib")
    maps_localenergy_CalibA=CycleGAN.compute_localenergy_array_new(OriginalCalibA_preproc, OriginalCalibB_preproc)
    maps_localenergy_CalibQQ=CycleGAN.compute_localenergy_array_new(OriginalCalibQQ_preproc, OriginalCalibB_preproc)
    maps_localenergy_varphy_CalibX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_CalibX2B, OriginalCalibB_preproc)
    maps_localenergy_varphy_Calib_mQQsX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_Calib_mQQsX2B, OriginalCalibB_preproc)
    maps_localenergy_varphy_Calib_mBsX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_Calib_mBsX2B, OriginalCalibB_preproc)
    ###
    ###CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_CalibA, maps_localenergy_CalibQQ, maps_localenergy_varphy_CalibX2B, maps_localenergy_varphy_Calib_mQQsX2B, "Global_Calib_localenergy_varphy",path_plot=path_plot, subfolder = "local_energy")
    ####
    print("CalibA: " + str(round(np.nanmean(abs(maps_localenergy_CalibA)),9)))
    print("CalibQQ: " + str(round(np.nanmean(abs(maps_localenergy_CalibQQ)),9)))
    print("CalibX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_CalibX2B)),9)))
    print("Calib_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mQQsX2B)),9)))
    print("Calib_mBsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mBsX2B)),9)))
    ####
    ######## sur realrank ####
    print("local energy realrank Calib")
    ####### Calib
    ####
    ####
    maps_localenergy_CalibA=CycleGAN.compute_localenergy_array_new(sample_realrank_CalibA, sample_realrank_CalibB)
    maps_localenergy_CalibQQ=CycleGAN.compute_localenergy_array_new(sample_realrank_CalibQQ, sample_realrank_CalibB)
    maps_localenergy_varphy_CalibX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_CalibX2B, sample_realrank_CalibB)
    maps_localenergy_varphy_Calib_mQQsX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_Calib_mQQsX2B, sample_realrank_CalibB)
    maps_localenergy_varphy_Calib_mBsX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_Calib_mBsX2B, sample_realrank_CalibB)
    ####
    ####CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_CalibA, maps_localenergy_CalibQQ, maps_localenergy_varphy_CalibX2B, maps_localenergy_varphy_reordered_CalibX2B, "Global_Calib_localenergy_realrank",path_plot=path_plot, subfolder = "local_energy")
    ####
    print("CalibA: " + str(round(np.nanmean(abs(maps_localenergy_CalibA)),9)))
    print("CalibQQ: " + str(round(np.nanmean(abs(maps_localenergy_CalibQQ)),9)))
    print("CalibX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_CalibX2B)),9)))
    print("Calib_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mQQsX2B)),9)))
    print("Calib_mBsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mBsX2B)),9)))
    ###
    ###
    ##
    ###
    ###
    ####### Proj
    print("local_energy_Proj")
    maps_localenergy_ProjA=CycleGAN.compute_localenergy_array_new(OriginalProjA_preproc, OriginalProjB_preproc)
    maps_localenergy_ProjQQ=CycleGAN.compute_localenergy_array_new(OriginalProjQQ_preproc, OriginalProjB_preproc)
    maps_localenergy_varphy_ProjX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_ProjX2B, OriginalProjB_preproc)
    maps_localenergy_varphy_Proj_mQQsX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_Proj_mQQsX2B, OriginalProjB_preproc)
    maps_localenergy_varphy_Proj_mBsX2B=CycleGAN.compute_localenergy_array_new(sample_varphy_Proj_mBsX2B, OriginalProjB_preproc)
    ####
    ####CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_ProjA, maps_localenergy_ProjQQ, maps_localenergy_varphy_ProjX2B, maps_localenergy_varphy_Proj_mQQsX2B, "Global_Proj_localenergy_varphy",path_plot=path_plot, subfolder = "local_energy")
    ####
    print("ProjA: " + str(round(np.nanmean(abs(maps_localenergy_ProjA)),9)))
    print("ProjQQ: " + str(round(np.nanmean(abs(maps_localenergy_ProjQQ)),9)))
    print("ProjX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_ProjX2B)),9)))
    print("Proj_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Proj_mQQsX2B)),9)))
    print("Proj_mBsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Proj_mBsX2B)),9)))
    ###
    print("local energy real rank Proj")
    ###
    ###
    maps_localenergy_ProjA=CycleGAN.compute_localenergy_array_new(sample_realrank_ProjA, sample_realrank_ProjB)
    maps_localenergy_ProjQQ=CycleGAN.compute_localenergy_array_new(sample_realrank_ProjQQ, sample_realrank_ProjB)
    maps_localenergy_varphy_ProjX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_ProjX2B, sample_realrank_ProjB)
    maps_localenergy_varphy_Proj_mQQsX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_Proj_mQQsX2B, sample_realrank_ProjB)
    maps_localenergy_varphy_Proj_mBsX2B=CycleGAN.compute_localenergy_array_new(sample_realrank_Proj_mBsX2B, sample_realrank_ProjB)
    ####
    ####CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_ProjA, maps_localenergy_ProjQQ, maps_localenergy_varphy_ProjX2B, maps_localenergy_varphy_reordered_ProjX2B, "Global_Proj_localenergy_realrank",path_plot=path_plot, subfolder = "local_energy")
    ####
    ####
    ###
    ###
    print("ProjA: " + str(round(np.nanmean(abs(maps_localenergy_ProjA)),9)))
    print("ProjQQ: " + str(round(np.nanmean(abs(maps_localenergy_ProjQQ)),9)))
    print("ProjX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_ProjX2B)),9)))
    print("Proj_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Proj_mQQsX2B)),9)))
    print("Proj_mBsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Proj_mBsX2B)),9)))
    ##
    ##
    #
    #

##### Compute Global energy ####

def compute_globalenergy_array_new(data_A,data_B):
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
    smallA=np.zeros((784,nb_images))
    smallB= np.zeros((784,nb_images))
    smallA=flat_array(data_A[:,:,:,0])
    smallB=flat_array(data_B[:,:,:,0])
#    print(smallA.shape)
    res_localenergy = sqrt(dcor.homogeneity.energy_test_statistic(np.transpose(smallA), np.transpose(smallB), exponent=1)*(2*nb_images)/(nb_images * nb_images))
    #res_localenergy = res_localenergy.astype(float)
    return res_localenergy


print("compute global energy for varphy")
print("Calib")
globalenergy_CalibA=compute_globalenergy_array_new(OriginalCalibA_preproc, OriginalCalibB_preproc)
print("A: " + str(globalenergy_CalibA))
globalenergy_CalibQQ=compute_globalenergy_array_new(OriginalCalibQQ_preproc, OriginalCalibB_preproc)
print("QQ: " + str(globalenergy_CalibQQ))
globalenergy_CalibX2B=compute_globalenergy_array_new(sample_varphy_CalibX2B, OriginalCalibB_preproc)
print("X2B: " + str(globalenergy_CalibX2B))
globalenergy_Calib_mQQsX2B=compute_globalenergy_array_new(sample_varphy_Calib_mQQsX2B, OriginalCalibB_preproc)
print("mQQsX2B: " + str(globalenergy_Calib_mQQsX2B))
globalenergy_Calib_mBsX2B=compute_globalenergy_array_new(sample_varphy_Calib_mBsX2B, OriginalCalibB_preproc)
print("mBsX2B: " + str(globalenergy_Calib_mBsX2B))


print("compute global energy for rank")
print(sample_realrank_CalibA.shape)
globalenergy_CalibA=compute_globalenergy_array_new(sample_realrank_CalibA, sample_realrank_CalibB)
print("A: " + str(globalenergy_CalibA))
globalenergy_CalibQQ=compute_globalenergy_array_new(sample_realrank_CalibQQ, sample_realrank_CalibB)
print("QQ: " + str(globalenergy_CalibQQ))
globalenergy_CalibX2B=compute_globalenergy_array_new(sample_realrank_CalibX2B, sample_realrank_CalibB)
print("X2B: " + str(globalenergy_CalibX2B))
globalenergy_Calib_mQQsX2B=compute_globalenergy_array_new(sample_realrank_Calib_mQQsX2B, sample_realrank_CalibB)
print("mQQsX2B: " + str(globalenergy_Calib_mQQsX2B))
globalenergy_Calib_mBsX2B=compute_globalenergy_array_new(sample_realrank_Calib_mBsX2B, sample_realrank_CalibB)
print("mBsX2B: " + str(globalenergy_Calib_mBsX2B))


#
print("compute global energy for varphy")
print("Proj")
globalenergy_ProjA=compute_globalenergy_array_new(OriginalProjA_preproc, OriginalProjB_preproc)
print("A: " + str(globalenergy_ProjA))
globalenergy_ProjQQ=compute_globalenergy_array_new(OriginalProjQQ_preproc, OriginalProjB_preproc)
print("QQ: " + str(globalenergy_ProjQQ))
globalenergy_ProjX2B=compute_globalenergy_array_new(sample_varphy_ProjX2B, OriginalProjB_preproc)
print("X2B: " + str(globalenergy_ProjX2B))
globalenergy_Proj_mQQsX2B=compute_globalenergy_array_new(sample_varphy_Proj_mQQsX2B, OriginalProjB_preproc)
print("mQQsX2B: " + str(globalenergy_Proj_mQQsX2B))
globalenergy_Proj_mBsX2B=compute_globalenergy_array_new(sample_varphy_Proj_mBsX2B, OriginalProjB_preproc)
print("mBsX2B: " + str(globalenergy_Proj_mBsX2B))
##
##
print("compute global energy for rank")
globalenergy_ProjA=compute_globalenergy_array_new(sample_realrank_ProjA, sample_realrank_ProjB)
print("A: " + str(globalenergy_ProjA))
globalenergy_ProjQQ=compute_globalenergy_array_new(sample_realrank_ProjQQ, sample_realrank_ProjB)
print("QQ: " + str(globalenergy_ProjQQ))
globalenergy_ProjX2B=compute_globalenergy_array_new(sample_realrank_ProjX2B, sample_realrank_ProjB)
print("X2B: " + str(globalenergy_ProjX2B))
globalenergy_Proj_mQQsX2B=compute_globalenergy_array_new(sample_realrank_Proj_mQQsX2B, sample_realrank_ProjB)
print("mQQsX2B: " + str(globalenergy_Proj_mQQsX2B))
globalenergy_Proj_mBsX2B=compute_globalenergy_array_new(sample_realrank_Proj_mBsX2B, sample_realrank_ProjB)
print("mBsX2B: " + str(globalenergy_Proj_mBsX2B))
##
##
print("end compute global energy")
#
#
#
#
#



print("intermediate energy")
##### Compute Intermediate local energy 5x5; 7x7; 9x9; 11x11...  
def compute_intermediate_localenergy_array_new(data_A,data_B, width_window = 5):
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
    smallA=np.zeros((width_window*width_window,nb_images))
    smallB= np.zeros((width_window*width_window,nb_images))
    for k in range(28):
       for l in range(28):
            if k> (width_window-3)/2 and k<(27 + (3-width_window)/2):
                if l>(width_window-3)/2  and l<(27 + (3-width_window)/2):
                    print(smallA.shape)
                    print('localenergy' + str(k))
                    smallA=flat_array(data_A[:,(k-1-int((width_window-3)/2)):(k+2+int((width_window-3)/2)),(l-1-int((width_window-3)/2)):(l+2+int((width_window-3)/2)),0])
                    smallB=flat_array(data_B[:,(k-1-int((width_window-3)/2)):(k+2+int((width_window-3)/2)),(l-1-int((width_window-3)/2)):(l+2+int((width_window-3)/2)),0])
                    res_localenergy[:,k,l] = sqrt(dcor.homogeneity.energy_test_statistic(np.transpose(smallA), np.transpose(smallB), exponent=1)*(2*nb_images)/(nb_images * nb_images)) #### formula, see Rizza and Szekely 2015
    res_localenergy = res_localenergy.astype(float)
    return res_localenergy



###### sur varphy ####
width_nb = 11
##### Calib
print("local energy Calib of width" + str(width_nb))
maps_localenergy_CalibA=compute_intermediate_localenergy_array_new(OriginalCalibA, OriginalCalibB, width_window = width_nb)
maps_localenergy_CalibQQ=compute_intermediate_localenergy_array_new(OriginalCalibQQ, OriginalCalibB, width_window = width_nb)
maps_localenergy_varphy_CalibX2B=compute_intermediate_localenergy_array_new(sample_varphy_CalibX2B, OriginalCalibB, width_window = width_nb)
maps_localenergy_varphy_Calib_mQQsX2B=compute_intermediate_localenergy_array_new(sample_varphy_Calib_mQQsX2B, OriginalCalibB, width_window = width_nb)

#
#CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_CalibA, maps_localenergy_CalibQQ, maps_localenergy_varphy_CalibX2B, maps_localenergy_varphy_Calib_mQQsX2B, "Global_Calib_
#localenergy_varphy",path_plot=path_plot, subfolder = "local_energy")
##
print("CalibA: " + str(round(np.nanmean(abs(maps_localenergy_CalibA)),9)))
print("CalibQQ: " + str(round(np.nanmean(abs(maps_localenergy_CalibQQ)),9)))
print("CalibX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_CalibX2B)),9)))
print("Calib_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mQQsX2B)),9)))
#
###### sur realrank ####
print("local energy realrank Calib")
##### Calib
sample_realrank_CalibA = CycleGAN.compute_matrix_real_rank(OriginalCalibA)
sample_realrank_CalibB = CycleGAN.compute_matrix_real_rank(OriginalCalibB)
sample_realrank_CalibQQ = CycleGAN.compute_matrix_real_rank(OriginalCalibQQ)
sample_realrank_CalibX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_CalibX2B)
sample_realrank_Calib_mQQsX2B = CycleGAN.compute_matrix_real_rank(sample_varphy_Calib_mQQsX2B)
##
##
maps_localenergy_CalibA=compute_intermediate_localenergy_array_new(sample_realrank_CalibA, sample_realrank_CalibB, width_window = width_nb)
maps_localenergy_CalibQQ=compute_intermediate_localenergy_array_new(sample_realrank_CalibQQ, sample_realrank_CalibB, width_window = width_nb)
maps_localenergy_varphy_CalibX2B=compute_intermediate_localenergy_array_new(sample_realrank_CalibX2B, sample_realrank_CalibB, width_window = width_nb)
maps_localenergy_varphy_Calib_mQQsX2B=compute_intermediate_localenergy_array_new(sample_realrank_Calib_mQQsX2B, sample_realrank_CalibB, width_window = width_nb)

##
##CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, maps_localenergy_CalibA, maps_localenergy_CalibQQ, maps_localenergy_varphy_CalibX2B, maps_localenergy_varphy_reordered_CalibX2B, "Global_
#Calib_localenergy_realrank",path_plot=path_plot, subfolder = "local_energy")
##
print("CalibA: " + str(round(np.nanmean(abs(maps_localenergy_CalibA)),9)))
print("CalibQQ: " + str(round(np.nanmean(abs(maps_localenergy_CalibQQ)),9)))
print("CalibX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_CalibX2B)),9)))
print("Calib_mQQsX2B: " + str(round(np.nanmean(abs(maps_localenergy_varphy_Calib_mQQsX2B)),9)))





###################################################################################################
##### Accuracy of discB ####
###################################################################################################
#
#discB = load_model( savepath + '/models/discB_model_' + str(ite_to_take) + '.h5')
#
#def eval_on_disc(data, disc, title_):
#    res = disc.predict(data)
#    print('Score ' +title_)
#    print(res.mean())
#    print(res.std())
#    print(res.min())
#    print(res.max())
#
#def eval_accuracy_on_disc(data, disc, title_):
#    n_samples = data.shape[0]
#    y = ones((n_samples, 1))
#    _, res = disc.evaluate(data, y, verbose=0)
#    print('Accu' + title_)
#    print(res)
#
#def normalize_minmax(data, minX, maxX):
#    res= np.copy(data)
#    n=-1
#    for k in range(28):
#        for l in range(28):
#            n=n+1
#            res[:,k,l,:] = (data[:,k,l,:]- minX[n])/(maxX[n] - minX[n])
#    return res
#
#
#sample_reordered_CalibX2B = normalize_minmax(sample_varphy_reordered_CalibX2B, minCalibX, maxCalibX)
#
#
##eval_on_disc(CalibA, discB, "CalibA")
##eval_on_disc(CalibB, discB, "CalibB")
##eval_on_disc(CalibQQ, discB, "CalibQQ")
##eval_on_disc(sample_CalibX2B, discB, "CalibX2B")
##eval_on_disc(sample_reordered_CalibX2B, discB, "reordered_CalibX2B")
##
#eval_accuracy_on_disc(CalibA, discB, "CalibA")
#eval_accuracy_on_disc(CalibB, discB, "CalibB")
#eval_accuracy_on_disc(CalibQQ, discB, "CalibQQ")
#eval_accuracy_on_disc(sample_CalibX2B, discB, "CalibX2B")
#eval_accuracy_on_disc(sample_reordered_CalibX2B, discB, "reordered_CalibX2B")
#
#
#sample_CalibX2B[:, 10:15, 10:15,:] = CalibQQ[: , 10:15, 10:15, :]
#eval_accuracy_on_disc(sample_CalibX2B, discB, "CalibX2B_modified")
#




#
#
########################################################################################
###### Compute Nuage de points ####
########################################################################################
#
#def compute_correlation_matrix(remove_spat_mean,data,ind,lon,lat,point_grid, method="spearman"):
#    ### needed in python
#    data=np.transpose(data[:,:,:,0],(2,1,0))
#    ### end in python
#    MatData=CycleGAN.transform_array_in_matrix(data,ind,point_grid)
#    tmp_daily_spat_mean=np.mean(MatData, axis=0)
#    means_expanded = np.outer(tmp_daily_spat_mean, np.ones(784))
#    if remove_spat_mean==True:
#        Mat_daily_mean_removed=np.transpose(MatData)-means_expanded
#    else:
#        Mat_daily_mean_removed=np.transpose(MatData)
#    if method== "spearman":
#        Cspearman,_ =spearmanr(Mat_daily_mean_removed)
#    if method =="pearson":
#        Cspearman =np.corrcoef(Mat_daily_mean_removed.T)
#    return Cspearman
#
#print("begin nuage de points")
#CspearmanA = compute_correlation_matrix(False,OriginalCalibA, IND_Paris, LON_Paris, LAT_Paris, point_max)
#CspearmanB = compute_correlation_matrix(False,OriginalCalibB, IND_Paris, LON_Paris, LAT_Paris, point_max)
#CspearmanQQ = compute_correlation_matrix(False,OriginalCalibQQ, IND_Paris, LON_Paris, LAT_Paris, point_max)
#CspearmanX2B= compute_correlation_matrix(False,sample_varphy_CalibX2B, IND_Paris, LON_Paris, LAT_Paris, point_max)
#Cspearman_mQQsX2B = compute_correlation_matrix(False,sample_varphy_Calib_mQQsX2B, IND_Paris, LON_Paris, LAT_Paris, point_max)
#print("end nuage")
#
##plt.scatter(CspearmanB[np.triu_indices(784)].flatten(),CspearmanA[np.triu_indices(784)].flatten() , c = 'red', s=20, alpha= 0.5, label = "A", marker = '+', linewidth = 0.5,edgecolors = None)
#plt.scatter(CspearmanB.flatten(),CspearmanA.flatten() , c = 'red', s=20, alpha= 0.5, label = "A", marker = '+', linewidth = 0.5,edgecolors = None)
##plt.scatter(CspearmanB.flatten(),CspearmanQQ.flatten(), c = 'orange', s=20, alpha= 1, label = "QQ", marker = '+', linewidth = 0.5,edgecolors = None)
##plt.scatter(CspearmanB.flatten(),CspearmanX2B.flatten(), c = 'blue',  s=20, alpha= 1, label = "QQ2B", marker = '+', linewidth = 0.5,edgecolors = None)
##plt.scatter(CspearmanB.flatten(), Cspearman_mQQsX2B.flatten(), c = 'green', s=20,  alpha= 1, label = "mQQsQQ2B", marker = '+', linewidth = 0.5,edgecolors = None)
##zlim_min = np.min(np.concatenate((CspearmanB, CspearmanA,CspearmanQQ, CspearmanX2B, Cspearman_mQQsX2B)))
##zlim_max = np.max(np.concatenate((CspearmanB, CspearmanA,CspearmanQQ, CspearmanX2B, Cspearman_mQQsX2B)))
##plt.plot([zlim_min, zlim_max ], [zlim_min, zlim_max], color = 'black', linewidth = 0.5, linestyle='dashed')
#plt.legend()
#plt.ylabel("Model Spatial Correlation")
#plt.xlabel("Observed Spatial Correlation")
#plt.savefig('Cspearman_bis.png')
#plt.close()
#print("end_print")
#
#
#
##### MSE of correlation matrix by grid cells ####
#print(CspearmanA.shape)
#def mse(ref, pred):
#    res = []
#    for k in range(ref.shape[0]):
#        res.append(np.sum((ref[k,:].astype("float") - pred[k,:].astype("float")) **2)/(ref.shape[0]))
#    res = np.asarray(res)
#    print(res.shape)
#    return res
#
#mse_CspearmanA = mse(CspearmanB, CspearmanA)
#mse_CspearmanQQ = mse(CspearmanB, CspearmanQQ)
#mse_CspearmanX2B = mse(CspearmanB, CspearmanX2B)
#mse_Cspearman_mQQsX2B = mse(CspearmanB, Cspearman_mQQsX2B)
#
#plt.scatter(mse_CspearmanA.flatten(),mse_CspearmanQQ.flatten() , c = 'orange', s=20, alpha= 1, label = "QQ", marker = '+', linewidth = 0.5,edgecolors = None)
#plt.scatter(mse_CspearmanA.flatten(),mse_CspearmanX2B.flatten(), c = 'blue', s=20, alpha= 1, label = "QQ2B", marker = '+', linewidth = 0.5,edgecolors = None)
#plt.scatter(mse_CspearmanA.flatten(),mse_Cspearman_mQQsX2B.flatten(), c = 'green', s=20,  alpha= 1, label = "mQQsQQ2B", marker = '+', linewidth = 0.5,edgecolors = None)
#zlim_min = np.min(np.concatenate((mse_CspearmanA,mse_CspearmanQQ, mse_CspearmanX2B, mse_Cspearman_mQQsX2B)))
#zlim_max = np.max(np.concatenate((mse_CspearmanA,mse_CspearmanQQ, mse_CspearmanX2B, mse_Cspearman_mQQsX2B)))
#plt.plot([zlim_min, zlim_max ], [zlim_min, zlim_max], color = 'black', linewidth = 0.5, linestyle='dashed')
#plt.legend()
#plt.ylabel("MSE (Bias corrected)")
#plt.xlabel("MSE (A)")
#plt.savefig('mse_Cspearman_IPSL.png')
#print("end_print")
#
#
#
#
#


######### WD #########
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



bin_width_fixed=[0.01]*784


print("compute WD on varphy")
wd_A=compute_WD(OriginalCalibA, OriginalCalibB,IND_Paris, point_max,bin_size = bin_width_fixed)
print("A: " + str(wd_A))
wd_QQ=compute_WD( OriginalCalibQQ, OriginalCalibB,IND_Paris, point_max,bin_size = bin_width_fixed)
print("QQ: " + str(wd_QQ))
wd_CalibX2B=compute_WD(sample_varphy_CalibX2B, OriginalCalibB, IND_Paris, point_max, bin_size = bin_width_fixed)
print("X2B: " + str(wd_CalibX2B))
wd_Calib_mQQsX2B=compute_WD(sample_varphy_Calib_mQQsX2B, OriginalCalibB, IND_Paris, point_max, bin_size = bin_width_fixed)
print("mQQsX2B: " + str(wd_Calib_mQQsX2B))
#
print("compute WD on real rank")
wd_A=compute_WD(sample_realrank_CalibA, sample_realrank_CalibB,IND_Paris, point_max,bin_size = bin_width_fixed)
print("A: " + str(wd_A))
wd_QQ=compute_WD( sample_realrank_CalibQQ, sample_realrank_CalibB,IND_Paris, point_max,bin_size = bin_width_fixed)
print("QQ: " + str(wd_QQ))
wd_CalibX2B=compute_WD(sample_realrank_CalibX2B, sample_realrank_CalibB, IND_Paris, point_max, bin_size = bin_width_fixed)
print("X2B: " + str(wd_CalibX2B))
wd_Calib_mQQsX2B=compute_WD(sample_realrank_Calib_mQQsX2B, sample_realrank_CalibB, IND_Paris, point_max, bin_size = bin_width_fixed)
print("mQQsX2B: " + str(wd_Calib_mQQsX2B))
#globalenergy_Calib_mQQsX2B=compute_globalenergy_array_new(sample_realrank_Calib_mQQsX2B, sample_realrank_CalibB)
#print("mQQsX2B: " + str(globalenergy_Calib_mQQsX2B))
#






#
#
#
#
#
#################################################################################################
##### Compute 1d-energy ####
#################################################################################################
#import dcor
#def compute_1denergy_array_new(data_A,data_B):
#    nb_images=data_A.shape[0]
#    res_1denergy=np.reshape([None]*28*28,(1,28,28))
#    smallA=np.zeros((1,nb_images))
#    smallB= np.zeros((1,nb_images))
#    #### dcor.homogeneity need data of dim nb_time*nb_var
#    for k in range(28):
#        print('1denergy' + str(k))
#        for l in range(28):
#            smallA=data_A[:,k,l,0]
#            smallB=data_B[:,k,l,0]
#            res_1denergy[:,k,l] = dcor.homogeneity.energy_test_statistic(np.transpose(smallA), np.transpose(smallB), exponent=1)
#    res_1denergy = res_1denergy.astype(float)
#    return res_1denergy
#
#
#
##### Calib ####
#nb_timestep=OriginalCalibA.shape[0]
#nb_pas=1
#
#coord_localenergy=range(0,nb_timestep,nb_pas)
#
#energy1d_OriginalCalibA = compute_1denergy_array_new(OriginalCalibA[coord_localenergy,:,:], OriginalCalibB[coord_localenergy,:,:])
#print(energy1d_OriginalCalibA.max())
#
#energy1d_OriginalCalibQQ = compute_1denergy_array_new(OriginalCalibQQ[coord_localenergy,:,:], OriginalCalibB[coord_localenergy,:,:])
#print(energy1d_OriginalCalibA.max())
#
#energy1d_varphy_CalibX2B = compute_1denergy_array_new(sample_varphy_CalibX2B[coord_localenergy,:,:], OriginalCalibB[coord_localenergy,:,:])
#energy1d_varphy_reordered_CalibX2B = compute_1denergy_array_new(sample_varphy_reordered_CalibX2B[coord_localenergy,:,:], OriginalCalibB[coord_localenergy,:,:])
#print(energy1d_varphy_reordered_CalibX2B.max())
#
#
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#path_plot = "/gpfswork/rech/eal/urq13cl/CycleGAN/Results"
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#CycleGAN.plot_maps_localWD(ite_to_take,QQ2B_version, False, energy1d_OriginalCalibA, energy1d_OriginalCalibQQ, energy1d_varphy_CalibX2B, energy1d_varphy_reordered_CalibX2B, "plot_calib_energy1d",path_plot=path_plot, subfolder="1d_energy", nb_lon = 28, nb_lat = 28)
#print("ok 1denergy varphy")
###
#
#
#
#
#

















#### On real rank
#from scipy.stats import *
#
#
#
#def plot_maps_1denergy(epoch, PR_version, mat_A, mat_QQ, mat_A2B, title, path_plot, lon=np.array(range(27)), lat=np.array(range(27))):
#    mat_A = mat_A.astype(float)
#    mat_QQ = mat_QQ.astype(float)
#    mat_A2B = mat_A2B.astype(float)
#    #### On inverse LON_LAT pour plotter correctement
#    ####fliplr for (1,28,28), else flipud
#    mat_A=np.fliplr(mat_A)
#    mat_QQ=np.fliplr(mat_QQ)
#    mat_A2B=np.fliplr(mat_A2B)
#    #### Mean and sd / MAE ####
#    if PR_version==False:
#        examples = vstack((mat_A, mat_QQ, mat_A2B, mat_A-mat_A, mat_QQ-mat_A, mat_A2B-mat_A))
#        names_=("A","QQ","A2B","A-A","QQ-A","A2B-A")
#    else:
#        examples = vstack((mat_A, mat_QQ, mat_A2B, (mat_A-mat_A)/mat_A, (mat_QQ-mat_A)/mat_A, (mat_A2B-mat_A)/mat_A))
#        names_=("A","QQ","A2B","(A-A)/A","(QQ-A)/A","(A2B-A)/A")
#    nchecks=3
#    fig, axs = pyplot.subplots(2,nchecks, figsize=(10,10))
#    cm = ['YlOrRd','RdBu']
#    fig.subplots_adjust(right=0.925) # making some room for cbar
#    quant_10=0
#    quant_90=np.quantile(mat_A,0.9)
#    for row in range(2):
#        for col in range(nchecks):
#            i=3*row+col
#            ax = axs[row,col]
#            ax.spines['top'].set_visible(False)
#            ax.spines['right'].set_visible(False)
#            ax.spines['bottom'].set_visible(False)
#            ax.spines['left'].set_visible(False)
#            ax.set_xticks([])
#            ax.set_yticks([])
#            if (row < 1):
#                vmin = quant_10
#                vmax = quant_90
#                pcm = ax.imshow(examples[i,:,:], cmap =cm[row],vmin=vmin, vmax=vmax)
#                ax.set_title(str(names_[i]) + ' mean: ' +str(round(np.nanmean(examples[i, :, :]),3)) + ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),3))  ,fontsize=10)
#
#            else:
#                vmin=-0.2
#                vmax=0.2
#                pcm = ax.imshow(examples[3*row + col, :,:], cmap = cm[row], vmin=vmin, vmax=vmax)
#                ax.set_title(str(names_[i]) + ' mae: ' +str(round(np.nanmean(abs(examples[i, :, :])),3)) + ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),3)))
#        fig.colorbar(pcm, ax = axs[row,:],shrink=0.5)
#    filename = "essai" + title + "_1denergy1661.png"
#    fig.savefig(filename, dpi=150)
#    pyplot.close()
#
#
#
#
#
#dataset_varphy_A2B=from_rank_to_varphy(rank_version,PR_version,datasetA2B, OriginalB, XminB_, XmaxB_)
#
#if PR_version==True:
#    OriginalA[OriginalA < 1] = 0
#    OriginalB[OriginalB < 1] = 0
#    OriginalQQ[OriginalQQ <1] =0
#
#
#if is_DS==True:
#    rmse_rank_A = rmse(datasetB, datasetA)
#    rmse_rank_A2B = rmse(datasetB, datasetA2B)
#    rmse_rank_QQ = rmse(datasetB, datasetQQ)
#    print("RMSE rank")
#    print("A: " + str(rmse_rank_A))
#    print("A2B: " + str(rmse_rank_A2B))
#    print("QQ: " + str(rmse_rank_QQ))
#
#    mae_rank_A = mae(datasetB, datasetA)
#    mae_rank_A2B = mae(datasetB, datasetA2B)
#    mae_rank_QQ = mae(datasetB, datasetQQ)
#    print("MAE rank")
#    print("A" + str(mae_rank_A))
#    print("A2B" + str(mae_rank_A2B))
#    print("QQ" + str(mae_rank_QQ))
#
#    rmse_varphy_A = rmse(OriginalA, OriginalB)
#    rmse_varphy_A2B = rmse(dataset_varphy_A2B, OriginalB)
#    rmse_varphy_QQ = rmse(OriginalQQ, OriginalB)
#    print("RMSE varphy")
#    print("A: " + str(rmse_varphy_A))
#    print("A2B: " + str(rmse_varphy_A2B))
#    print("QQ: " + str(rmse_varphy_QQ))
#
#    mae_varphy_A = mae(OriginalA, OriginalB)
#    mae_varphy_A2B = mae(dataset_varphy_A2B, OriginalB)
#    mae_varphy_QQ = mae(OriginalQQ, OriginalB)
#    print("MAE varphy")
#    print("A: " + str(mae_varphy_A))
#    print("A2B: " + str(mae_varphy_A2B))
#    print("QQ: " + str(mae_varphy_QQ))
#
#
#
#realrankA=compute_matrix_rank(OriginalA)
#realrankB=compute_matrix_rank(OriginalB)
#realrankA2B=compute_matrix_rank(dataset_varphy_A2B)
#realrankQQ=compute_matrix_rank(OriginalQQ)
#
#
#nb_subsample=1
#
##### A la Cannon
#dataset_varphy_reordered_A2B= np.copy(dataset_varphy_A2B)
##Reorder rank data with OriginalData
###datasetA_eval = np.copy(OriginalA)
#datasetB_eval = np.copy(OriginalB)
#for k in range(28):
#    for l in range(28):
#        sorted_OriginalB=np.sort(datasetB_eval[:,k,l,0])
#        idx=rankdata(dataset_varphy_A2B[:,k,l,0],method="min")
#        idx=idx.astype(int)-1
#        dataset_varphy_reordered_A2B[:,k,l,0] = sorted_OriginalB[idx]
#
#
#
#print("ok")
#print(np.array(range(1,29,1)))
#print(dcor.homogeneity.energy_test_statistic(np.array(range(1,1001,1)), np.array(range(1001,2001,1)), exponent=1))
#
#


#
##### Energy distance on realrank
#essrealrankA=compute_localenergy_array_new(realrankA[range(0,3420,nb_subsample),:,:], realrankB[range(0,3420,nb_subsample),:,:], window_size=3)
#essrealrankQQ=compute_localenergy_array_new(realrankQQ[range(0,3420,nb_subsample),:,:], realrankB[range(0,3420,nb_subsample),:,:], window_size=3)
#essrealrankA2B=compute_localenergy_array_new(realrankA2B[range(0,3420,nb_subsample),:,:], realrankB[range(0,3420,nb_subsample),:,:], window_size=3)
#
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#plot_maps_localWD(4,False,essrealrankA,essrealrankQQ,essrealrankA2B,"realrank","ok")
#print("ok realrank")
##
#
##### Energy distance on varphy
#essOriginalA=compute_localenergy_array_new(OriginalA[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#essOriginalQQ=compute_localenergy_array_new(OriginalQQ[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#essvarphyA2B=compute_localenergy_array_new(dataset_varphy_A2B[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#plot_maps_localWD(4,False,essOriginalA,essOriginalQQ,essvarphyA2B,"varphy","ok")
#print("ok varphy")
##
#
#
#
#### Energy distance on reordered 
###### Compute MAE/RMSE ####
#def rmse(ref, pred):
#    return np.sum((ref.astype("float") - pred.astype("float")) **2)/(ref.shape[1]*ref.shape[2]*ref.shape[0])
#
#def mae(ref, pred):
#    return np.sum(abs(ref.astype("float") - pred.astype("float"))/(ref.shape[1]*ref.shape[2]*ref.shape[0]))
#


#essOriginalA=compute_localenergy_array_new(OriginalA[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#essOriginalQQ=compute_localenergy_array_new(OriginalQQ[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#essvarphy_reordered_A2B=compute_localenergy_array_new(dataset_varphy_reordered_A2B[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:], window_size=3)
#
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#plot_maps_localWD(4,False,essOriginalA,essOriginalQQ,essvarphy_reordered_A2B,"varphy_reordered","ok")
#print("ok varphy_reordered")
##
#

##### Compute WD ####
#from SBCK.metrics import wasserstein
#from SBCK.tools.__OT import OTNetworkSimplex
#from SBCK.tools.__tools_cpp import SparseHist
#from SBCK.tools import bin_width_estimator
#
#def compute_WD(sample_A, sample_B, ind,point_grid, bin_size= None):
#    reversed_datasetA=np.transpose(sample_A[:,:,:,0],(2,1,0))
#    reversed_datasetB=np.transpose(sample_B[:,:,:,0],(2,1,0))
#
#    tmp_A=np.transpose(CycleGAN.transform_array_in_matrix(reversed_datasetA, ind, point_grid))
#    tmp_B=np.transpose(CycleGAN.transform_array_in_matrix(reversed_datasetB, ind, point_grid))
#
#    mu_tmp_A = SparseHist(tmp_A,bin_size)
#    mu_tmp_B = SparseHist(tmp_B, bin_size)
#    res=wasserstein(mu_tmp_B, mu_tmp_A)
#    return res
#
#
#
#
#
##### Functions
#def from_rank_to_varphy(rank_version,PR_version,dataset_rank,original, Xmin_, Xmax_):
#    res_dataset=np.copy(dataset_rank)
#    if rank_version==True:
#        for k in range(28):
#            for l in range(28):
#                quant_to_take=np.array(dataset_rank[:,k,l,0])
#                res_dataset[:,k,l,0] = np.quantile(original[:,k,l,0],quant_to_take)
#    else:
#        #Rescale climatic variables wrt Xmin and Xmax
#        n=-1
#        for k in range(28):
#            for l in range(28):
#                n=n+1
#                res_dataset[:,k,l,:] = res_dataset[:,k,l,:]*(Xmax_[n] - Xmin_[n])+ Xmin_[n]
#
#    if PR_version==True:
#        res_dataset[res_dataset < 1] = 0
#    return res_dataset
#
##### On small samplesi
#def compute_localwd_array_new(data_A,data_B, bin_width_size=None):
#    def compute_small_WD(sample_A, sample_B,  bin_size= None):#input: nb_time*nb_var 
#        mu_tmp_A = SparseHist(sample_A,bin_size)
#        mu_tmp_B = SparseHist(sample_B, bin_size)
#        res=wasserstein(mu_tmp_B, mu_tmp_A)
#        return res
#
#    def flat_array(data_array): #data_array of dim TIME x LAT x LON, output: nb_var*nb_time  
#        nb_lon=data_array.shape[2]
#        nb_lat=data_array.shape[1]
#        nb_var=nb_lon*nb_lat
#        nb_time=data_array.shape[0]
#        res=np.zeros((nb_var,nb_time))
#        k=-1
#        for i in range(nb_lat):
#            for j in range(nb_lon):
#                k=k+1
#                res[k,:]=data_array[:,i,j]
#        return res
#
#    nb_images=data_A.shape[0]
#    res_localwd=np.reshape([None]*28*28,(1,28,28))
#    smallA=np.zeros((9,3420))
#    smallB= np.zeros((9,3420))
#    for k in range(28):
#        print('localWD' + str(k))
#        for l in range(28):
#            if k>0 and k<27:
#                if l>0 and l<27:
#                    smallA=flat_array(data_A[:,(k-1):(k+2),(l-1):(l+2),0])
#                    smallB=flat_array(data_B[:,(k-1):(k+2),(l-1):(l+2),0])
#                    res_localwd[:,k,l] = compute_small_WD(np.transpose(smallA),np.transpose(smallB),bin_size=bin_width_size)
#    res_localwd = res_localwd.astype(float)
#    return res_localwd
#
#



###### Energy distance 1d
#energy1d_OriginalA = compute_1denergy_array_new(OriginalA[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#energy1d_OriginalQQ = compute_1denergy_array_new(OriginalQQ[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#energy1d_varphyA2B = compute_1denergy_array_new(dataset_varphy_A2B[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#plot_maps_1denergy(4,False,energy1d_OriginalA,energy1d_OriginalQQ,energy1d_varphyA2B,"varphy","ok", lon=np.array(range(28)), lat=np.array(range(28)))
#print("ok 1denergy varphy")
##




#energy1d_OriginalA = compute_1denergy_array_new(OriginalA[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#energy1d_OriginalQQ = compute_1denergy_array_new(OriginalQQ[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#energy1d_varphy_reordered_A2B = compute_1denergy_array_new(dataset_varphy_reordered_A2B[range(0,3420,nb_subsample),:,:], OriginalB[range(0,3420,nb_subsample),:,:])
#os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Results")
#import matplotlib.pyplot as pyplot
#matplotlib.use('pdf')
#plot_maps_1denergy(4,False,energy1d_OriginalA,energy1d_OriginalQQ,energy1d_varphy_reordered_A2B,"varphy_reordered","ok", lon=np.array(range(28)), lat=np.array(range(28)))
#print("ok 1denergy varphy_reordered")
#





#############################################################################################################
##### Brouillon
#bin_width_realrank = CycleGAN.compute_bin_width_wasserstein(realrankB, IND_Paris, point_max)
#wd_rank_A=compute_WD(realrankA, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#wd_rank_A2B=compute_WD( realrankA2B, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#wd_rank_QQ=compute_WD(realrankQQ, realrankB,IND_Paris, point_max,bin_size = bin_width_realrank)
#print("Wasserstein on real rank bin_width: " + str(round(bin_width_realrank[1],3)))
#print("A: " + str(wd_rank_A))
#print("A2B: " + str(wd_rank_A2B))
#print("QQ: " + str(wd_rank_QQ))
#essA=compute_localwd_array_new(realrankA, realrankB, [0.05]*9)
#essQQ=compute_localwd_array_new(realrankQQ, realrankB, [0.05]*9)
#essA2B=compute_localwd_array_new(realrankA2B, realrankB, [0.05]*9)
#
#print("ok")
#
#import matplotlib.pyplot as pyplot
#matplotlib.use('TkAgg')
#plot_maps_localWD(4,False,essA,essQQ,essA2B,"ok","ok")
#print("fin ok")
##
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

#import matplotlib.pyplot as pyplot
#matplotlib.use('TkAgg')
#print(np.nanmean(maps_WD_localA))
#print(np.nanmean(maps_WD_localA2B))
#print(np.nanmean(maps_WD_localQQ))
#pyplot.figure()
#
##subplot(r,c) provide the no. of rows and columns
#f, axarr = pyplot.subplots(3,1)
#
#
## use the created array to output your multiple images. In this case I have stacked 4 images vertically
#axarr[0].imshow(maps_WD_localA)
#axarr[1].imshow(maps_WD_localA2B)
#axarr[2].imshow(maps_WD_localQQ)
#pyplot.show()
#
#
#
#bin_width_=bin_width_estimator(np.transpose(smallB)) #bcse bin_width_estimator needs data of type TIME x nb_var
#bin_width_small=[0.05]*len(bin_width_)
#
#
#wd_rank_A=compute_small_WD(np.transpose(smallA), np.transpose(smallB),bin_size = bin_width_small)
#wd_rank_A2B=compute_small_WD(np.transpose(smallA2B), np.transpose(smallB),bin_size = bin_width_small)
#wd_rank_QQ=compute_small_WD(np.transpose(smallQQ), np.transpose(smallB),bin_size = bin_width_small)
#print("Small Wasserstein on real rank bin_width: " + str(round(bin_width_small[1],3)))
#print("A: " + str(wd_rank_A))
#print("A2B: " + str(wd_rank_A2B))
#print("QQ: " + str(wd_rank_QQ))
#
#
#
#
#
#
#
#bin_size_rank=CycleGAN.compute_bin_width_wasserstein(datasetB, IND_Paris, point_max)
#wd_rank_A=compute_WD(datasetA,datasetB,IND_Paris, point_max, bin_size = bin_size_rank)
#wd_rank_A2B=compute_WD(datasetA2B,datasetB,IND_Paris, point_max, bin_size = bin_size_rank)
#wd_rank_QQ=compute_WD(datasetQQ, datasetB, IND_Paris, point_max, bin_size = bin_size_rank)
#print("Wasserstein rank")
#print("A: " + str(wd_rank_A))
#print("A2B: " + str(wd_rank_A2B))
#print("QQ: " + str(wd_rank_QQ))
#
#
#bin_size_varphy = CycleGAN.compute_bin_width_wasserstein(OriginalB, IND_Paris, point_max)
#wd_varphy_A = compute_WD(OriginalA, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
#wd_varphy_A2B = compute_WD(dataset_varphy_A2B, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
#wd_varphy_QQ = compute_WD( OriginalQQ, OriginalB, IND_Paris, point_max, bin_size = bin_size_varphy)
#print("Wasserstein varphy")
#print("A: " + str(wd_varphy_A))
#print("A2B: " + str(wd_varphy_A2B))
#print("QQ: " + str(wd_varphy_QQ))
#
#

import sys
sys.path.insert(1,'/gpfswork/rech/eal/urq13cl/CycleGAN/Script/')
import CycleGAN
import numpy as np
import os
from os import makedirs
import matplotlib
import matplotlib.pyplot as pyplot
matplotlib.use('tkagg')
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
#### Possible choices for this code:
#Physical variable?
#var_phys="tas"
var_phys="pr"

#### Period in the year?
#season="winter_79_16"
#season="summer_79_16"
season="annual_79_16"

#### Rank version or minmax version?
#rank_version=False
rank_version=True

##################################################################
##### Automatized below according to the choices
# Load temporal indices
#os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/")
#load_index_temp = robjects.r.load('Temporal_indices_79_16.RData')
#if season != "annual_79_16":
#    Ind_season=robjects.r['Ind_'+season]
#    Ind_season=np.array(Ind_season)-1
#else:
#    Ind_season=np.array(range(13870))
#

Ind_season = np.array(range(13870))



#### Load tas and pr
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdet/")
_,LON_France,LAT_France,_,_,IND_France,point_max,Original_tas=CycleGAN.load_RData_minmaxrank(rank_version,"tas_pr_day_SAFRANdet_79_16_France","tas_day_SAFRANdet_79_16_France",Ind_season,region="France")


_,LON_France,LAT_France,_,_,IND_France,point_max,Original_pr=CycleGAN.load_RData_minmaxrank(rank_version,"tas_pr_day_SAFRANdet_79_16_France","pr_day_SAFRANdet_79_16_France",Ind_season,region="France")





pr_day_SAFRANdetbili_79_16_Paris=np.zeros((13870,28,28))
tas_day_SAFRANdetbili_79_16_Paris=np.zeros((13870,28,28))

#prendre un peu plus large que 28 par 28 (ou 7x7) pour eviter effet de bord...
OriginalParisLarge_tas= Original_tas[:,84:120,51:87, 0]
OriginalParisLarge_pr= Original_pr[:,84:120,51:87, 0]
LON_ParisLarge = LON_France[51:87,84:120]
LAT_ParisLarge = LAT_France[51:87, 84:120]

x=np.zeros((9,9))
y=np.zeros((9,9))

z_tas=np.zeros((9,9))
z_pr=np.zeros((9,9))


for i in range(9):
    for j in range(9):
        coord_lon_start=i*4
        coord_lon_stop=((i+1)*4)
        coord_lat_start=j*4
        coord_lat_stop=((j+1)*4)
        if i==35:
            coord_lon_start=i*3
            coord_lon_stop=((i+1)*3)
        if j==33:
            coord_lat_start=j*2
            coord_lat_stop=((j+2)*3)
        x[i,j]=np.nanmean(LON_ParisLarge[coord_lon_start:coord_lon_stop,coord_lat_start:coord_lat_stop])
        y[i,j]=np.nanmean(LAT_ParisLarge[coord_lon_start:coord_lon_stop,coord_lat_start:coord_lat_stop])

xflat=x.flatten()
yflat=y.flatten()
xnew=LON_ParisLarge
ynew=LAT_ParisLarge

xnewflat=xnew.flatten()
ynewflat=ynew.flatten()

for tt in range(13870):
    if tt %777 == 0:
        print(tt)
    for i in range(9):
        for j in range(9):
            coord_lon_start=i*4
            coord_lon_stop=((i+1)*4)
            coord_lat_start=j*4
            coord_lat_stop=((j+1)*4)
            if i==35:
                coord_lon_start=i*3
                coord_lon_stop=((i+1)*3)
            if j==33:
                coord_lat_start=j*2
                coord_lat_stop=((j+2)*3)
            z_tas[j,i]=np.nanmean(OriginalParisLarge_tas[tt,coord_lat_start:coord_lat_stop,coord_lon_start:coord_lon_stop])
            z_pr[j,i]=np.nanmean(OriginalParisLarge_pr[tt,coord_lat_start:coord_lat_stop,coord_lon_start:coord_lon_stop])
#### For tas
    zflat_tas=z_tas.flatten()
    znewflat_tas = griddata((xflat, yflat), zflat_tas, (xnewflat, ynewflat), method='linear')
    znewflat_tas_NN = griddata((xflat, yflat), zflat_tas, (xnewflat, ynewflat), method='nearest')
    znew_tas=np.zeros((36,36))
    znew_tas_NN=np.zeros((36,36))
    for i in range(36):
        znew_tas[i,:]=znewflat_tas[i*36:((i+1)*36)]
        znew_tas_NN[i,:]=znewflat_tas_NN[i*36:((i+1)*36)]

    znew_tas[np.isnan(znew_tas)]=znew_tas_NN[np.isnan(znew_tas)]
#### For pr
    zflat_pr=z_pr.flatten()
    znewflat_pr = griddata((xflat, yflat), zflat_pr, (xnewflat, ynewflat), method='linear')
    znewflat_pr_NN = griddata((xflat, yflat), zflat_pr, (xnewflat, ynewflat), method='nearest')
    znew_pr=np.zeros((36,36))
    znew_pr_NN=np.zeros((36,36))
    for i in range(36):
        znew_pr[i,:]=znewflat_pr[i*36:((i+1)*36)]
        znew_pr_NN[i,:]=znewflat_pr_NN[i*36:((i+1)*36)]
    znew_pr[np.isnan(znew_pr)]=znew_pr_NN[np.isnan(znew_pr)]

    tas_day_SAFRANdetbili_79_16_Paris[tt,:,:]=znew_tas[4:32,4:32]
    pr_day_SAFRANdetbili_79_16_Paris[tt,:,:]=znew_pr[4:32,4:32]




#### LON_LAT_point_IND for Paris
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdet/")
_, LON_Paris, LAT_Paris, _, _, IND_Paris, point_max, _ = CycleGAN.load_RData_minmaxrank(rank_version,"tas_pr_day_SAFRANdet_79_16_Paris", "tas_day_SAFRANdet_79_16_Paris", Ind_season, region = "Paris")


#### ATTENTION!!! On inverse pour se calquer sur le format RData: LON_LAT_TIME
tas_day_SAFRANdetbili_79_16_Paris = np.transpose(tas_day_SAFRANdetbili_79_16_Paris, (2,  1, 0))
pr_day_SAFRANdetbili_79_16_Paris = np.transpose(pr_day_SAFRANdetbili_79_16_Paris, (2,  1, 0))


#os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdetbili/")
#np.savez('tas_pr_day_SAFRANdetbili_79_16_Paris.npz', tas_day_SAFRANdetbili_79_16_Paris =  tas_day_SAFRANdetbili_79_16_Paris,pr_day_SAFRANdetbili_79_16_Paris=pr_day_SAFRANdetbili_79_16_Paris, IND_Paris = IND_Paris, LON_Paris = LON_Paris, LAT_Paris = LAT_Paris, point_max = point_max)



#### Check if same
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdetbili/")
data = np.load('tas_pr_day_SAFRANdetbili_79_16_Paris.npz')
print(np.array_equal(data['tas_day_SAFRANdetbili_79_16_Paris'],tas_day_SAFRANdetbili_79_16_Paris))
print(np.array_equal(data['pr_day_SAFRANdetbili_79_16_Paris'],pr_day_SAFRANdetbili_79_16_Paris))









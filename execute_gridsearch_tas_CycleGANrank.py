import CycleGAN
import numpy as np
import os
from os import makedirs
import matplotlib
#matplotlib.use('pdf')
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


#### Possible choices for this code:
#Physical variable?
var_phys="tas"
#var_phys="pr"

#### Period in the year?
season="winter_79_16"
#season="summer_79_16"
#season="annual_79_16"

#### Rank version or minmax version?
#rank_version=False
rank_version=True

#### Hyperparameters?: learning rate of disc and gen?
list_lr_gen=[1e-4,5e-4,9e-4]
#0.0005
list_lr_disc=[1e-5,5e-5,9e-5]
#0.0001

##################################################################
##### Automatized below according to the choices
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

#### Load IPSL
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/IPSL/")
if(rank_version==False):
    datasetA, LON_Paris, LAT_Paris, XminA_, XmaxA_, IND_Paris, point_max, OriginalA = CycleGAN.load_RData_minmax("tas_pr_day_IPSL_79_16_Paris",var_phys + "_day_IPSL_79_16_Paris",Ind_season)
else:
    datasetA, LON_Paris, LAT_Paris, XminA_, XmaxA_, IND_Paris, point_max, OriginalA = CycleGAN.load_RData_rank("tas_pr_day_IPSL_79_16_Paris",var_phys + "_day_IPSL_79_16_Paris",Ind_season)

#### Load SAFRAN
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRAN/")
if(rank_version==False):
    datasetB, LON_Paris, LAT_Paris, XminB_, XmaxB_, IND_Paris, point_max, OriginalB = CycleGAN.load_RData_minmax("tas_pr_day_SAFRAN_79_16_Paris",var_phys + "_day_SAFRAN_79_16_Paris",Ind_season)
else:
    datasetB, LON_Paris, LAT_Paris, XminB_, XmaxB_, IND_Paris, point_max, OriginalB = CycleGAN.load_RData_rank("tas_pr_day_SAFRAN_79_16_Paris",var_phys + "_day_SAFRAN_79_16_Paris",Ind_season)


#################################################################
# create the discriminator
for lr_disc in list_lr_disc:
    for lr_gen in list_lr_gen:
        print('disc lr' + str(lr_disc))
        print('gen lr' + str(lr_gen))
        discA = CycleGAN.define_discriminator(lr_disc=lr_disc)
        discB = CycleGAN.define_discriminator(lr_disc=lr_disc)
        # create the generator
        genA2B = CycleGAN.define_generator()
        genB2A = CycleGAN.define_generator()
        # create the gan
        comb_model = CycleGAN.define_combined(genA2B, genB2A, discA, discB,lr_gen=lr_gen)

        #### Create a new folder
        os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/SAFRAN_IPSL/SpatialCycleGAN/"+ var_phys+"/"+season)
        if rank_version==False:
            name_version="minmax"
        else:
            name_version="rank"

        new_folder = var_phys + '_' + name_version + '_lrgen'+str(lr_gen)+'_lrdisc'+str(lr_disc)
        makedirs(new_folder, exist_ok=True)
        makedirs(new_folder + '/models', exist_ok=True)
        makedirs(new_folder + '/diagnostic', exist_ok=True)
        path_to_save="/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/SAFRAN_IPSL/SpatialCycleGAN/"+var_phys+"/"+season+"/"+new_folder

        #### Train CycleGAN
        CycleGAN.train_combined_new(rank_version, PR_version, genA2B, genB2A, discA, discB, comb_model, datasetA, datasetB, OriginalA, OriginalB,IND_Paris, LON_Paris, LAT_Paris,point_max, path_to_save, XminA_=XminA_, XmaxA_=XmaxA_, XminB_= XminB_, XmaxB_ = XmaxB_, n_epochs=1000) #####attention n_epochs






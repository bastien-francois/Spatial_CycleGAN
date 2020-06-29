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

#### Hyperparameters?: learning rate of disc and gen?
list_lr_gen=[1e-4]
list_lr_disc=[5e-6]

Ref="SAFRAN"
Mod="IPSLMRbili"

#### Wasserstein distances?
computation_WD=False

computation_localWD=True

#### Weights for valid, reconstruct and identity
lambda_val=9
lambda_rec=10
lambda_id=1


nb_filters_disc=[64,64]
nb_filters_gen=[64,64,64]

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
    datasetA,LON_Paris,LAT_Paris,XminA_,XmaxA_,IND_Paris,point_max,OriginalA = CycleGAN.load_RData_minmax("tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",Ind_season)
else:
    datasetA, LON_Paris, LAT_Paris, XminA_, XmaxA_, IND_Paris,point_max, OriginalA = CycleGAN.load_RData_rank("tas_pr_day_" + Mod + "_79_16_Paris",var_phys + "_day_" + Mod + "_79_16_Paris",Ind_season)



#### Load Ref
os.chdir("/gpfswork/rech/eal/commun/CycleGAN/Data/" + Ref + "/")
if(rank_version==False):
    datasetB, LON_Paris, LAT_Paris,XminB_,XmaxB_,IND_Paris, point_max, OriginalB = CycleGAN.load_RData_minmax("tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",Ind_season)
else:
    datasetB, LON_Paris, LAT_Paris,XminB_,XmaxB_, IND_Paris, point_max, OriginalB = CycleGAN.load_RData_rank("tas_pr_day_" + Ref + "_79_16_Paris",var_phys + "_day_" + Ref + "_79_16_Paris",Ind_season)


#### Load QQ
if var_phys=="pr" and Mod is not "SAFRANdetbili":
    BC1d="1dCDFt"
else:
    BC1d="1dQQ"

os.chdir("/gpfswork/rech/eal/commun/CycleGAN/MBC/" + Ref + "_" + Mod + "/")
if(rank_version==False):
    datasetQQ,_,_,_,_,_,_,OriginalQQ=CycleGAN.load_RData_minmax("tas_pr_day_PC0_"+BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys + "_day_PC0_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",Ind_season)
else:
    datasetQQ,_,_,_,_,_,_,OriginalQQ=CycleGAN.load_RData_rank("tas_pr_day_PC0_"+BC1d+"_"+Ref+"_"+Mod+"_79_16_Paris",var_phys+"_day_PC0_" + BC1d + "_" + Ref + "_" + Mod + "_79_16_Paris",Ind_season)









#################################################################
# create the discriminator
for lr_disc in list_lr_disc:
    for lr_gen in list_lr_gen:
        print('gen lr' + str(lr_gen))
        print('disc lr' + str(lr_disc))

        discA = CycleGAN.define_discriminator(lr_disc=lr_disc, nb_filters= nb_filters_disc)
        discB = CycleGAN.define_discriminator(lr_disc=lr_disc, nb_filters = nb_filters_disc)
        # create the generator
        genA2B = CycleGAN.define_generator(nb_filters=nb_filters_gen)
        genB2A = CycleGAN.define_generator(nb_filters=nb_filters_gen)
        # create the gan
        comb_model = CycleGAN.define_combined(genA2B, genB2A, discA, discB,lr_gen=lr_gen, lambda_valid= lambda_val, lambda_reconstruct = lambda_rec, lambda_identity = lambda_id)
        # load image data
        #comb_model.summary()
        #### Create a new folder
        os.chdir("/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/" + Ref + "_" + Mod + "/" + GAN_version + "/"+ var_phys+"/"+season)
        if rank_version==False:
            name_version="minmax"
        else:
            name_version="rank"
        if nb_filters_disc == [64,128]:
            new_arch="_new_arch"
        else:
            new_arch=""
        new_folder = var_phys + '_' + name_version + '_lrgen'+str(lr_gen)+'_lrdisc'+str(lr_disc) +"_Relu_lval" + str(lambda_val) + "_lrec" + str(lambda_rec) + "_lid" + str(lambda_id) + "_new_avec_WDlocal" + new_arch
        makedirs(new_folder, exist_ok=True)
        makedirs(new_folder + '/models', exist_ok=True)
        makedirs(new_folder + '/diagnostic', exist_ok=True)
        path_to_save="/gpfswork/rech/eal/urq13cl/CycleGAN/Data/MBC/" + Ref + "_" + Mod + "/" + GAN_version + "/"+var_phys+"/"+season+"/"+new_folder
        #### Train CycleGAN
        CycleGAN.train_combined_new(rank_version, PR_version,is_DS,computation_WD , computation_localWD, genA2B, genB2A, discA, discB, comb_model, datasetA, datasetB, datasetQQ, OriginalA, OriginalB, OriginalQQ ,IND_Paris, LON_Paris, LAT_Paris,point_max, path_to_save, XminA_=XminA_, XmaxA_=XmaxA_, XminB_= XminB_, XmaxB_ = XmaxB_, n_epochs=6000) #####attention n_epochs






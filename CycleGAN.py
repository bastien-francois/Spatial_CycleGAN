import h5netcdf.legacyapi as netCDF4
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy import save
from numpy import load
import numpy as np
from numpy.random import randn
from numpy.random import randint
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
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import os
import pyreadr
from os import makedirs
import tensorflow as tf
from scipy.stats import *
import rpy2.robjects as robjects
from math import *
from SBCK.metrics import wasserstein
from SBCK.tools import bin_width_estimator
from SBCK.tools.__OT import OTNetworkSimplex
from SBCK.tools.__tools_cpp import SparseHist
import dcor


##################################################################################
#### Ne pas toucher
def load_RData_minmaxrank(rank_version,RData_file,variable,index_temporal, region='Paris'):
    load_data = robjects.r.load(RData_file + '.RData') #format LON_Lat_Time
    dataset=robjects.r[variable]
    X = np.array(dataset)
    X= np.transpose(X, (2,  1, 0))
    lon =  robjects.r['LON_' + region]
    lon = np.array(lon)
    lat =  robjects.r['LAT_' + region]
    lat = np.array(lat)
    ind = robjects.r['IND_' + region]
    ind = np.array(ind)-1 ### ATTENTION Python specific  from RData
    point_grid = range(784)

    #Sub-selection of array
    if index_temporal is not None:
        X = X[index_temporal,:,:]
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(X, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')

    #Save original data
    OriginalX = np.copy(X)
    if rank_version==False:
        # scale with min/max by grid cells
        max_=[None]*28*28
        min_=[None]*28*28
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                max_[n]=X[:,k,l,:].max()
                min_[n]=X[:,k,l,:].min()
                X[:,k,l,:]=(X[:,k,l,:]-min_[n])/(max_[n]-min_[n])

    if rank_version==True:
        min_=None
        max_=None
        # scale with rank by grid cells
        for k in range(28):
            for l in range(28):
                X[:,k,l,0]=(rankdata(X[:,k,l,0],method="min")/len(X[:,k,l,0]))
    return X, lon, lat, min_, max_, ind, point_grid, OriginalX

def load_calib_proj_minmaxrank(CV_version,rank_version,RData_file,variable,season= "winter_79_16", region='Paris'): #pct_training=0.75
    #### Load temporal indices
    load_index_temp = robjects.r.load('/gpfswork/rech/eal/commun/CycleGAN/Data/Temporal_and_Random_indices_1979_2016.RData')
    if CV_version=="PC0":
        Ind_Calib_season=np.array(robjects.r['Ind_'+season])-1
        Ind_Proj_season= np.copy(Ind_Calib_season)

    if CV_version=="CVunif" or CV_version=="CVchrono":
        Ind_Calib_season=np.array(robjects.r['Ind_' + CV_version +'_Calib_' + season])-1
        Ind_Proj_season=np.array(robjects.r['Ind_' + CV_version + '_Proj_' + season])-1
    load_data = robjects.r.load(RData_file + '.RData') #format LON_Lat_Time
    dataset=robjects.r[variable]
    X = np.array(dataset)
    X= np.transpose(X, (2,  1, 0))
    lon =  robjects.r['LON_' + region]
    lon = np.array(lon)
    lat =  robjects.r['LAT_' + region]
    lat = np.array(lat)
    ind = robjects.r['IND_' + region]
    ind = np.array(ind)-1 ### ATTENTION Python specific  from RData
    point_grid = range(784)

    # expand to 3d, e.g. add channels dimension
    X = expand_dims(X, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')

    #### Save Original Data
    OriginalCalibX = X[Ind_Calib_season,:,:,:]
    OriginalProjX = X[Ind_Proj_season,:,:,:]

    #### Splitting Calib and Proj Data
    CalibX = X[Ind_Calib_season,:,:,:]
    ProjX = X[Ind_Proj_season,:,:,:]

    if rank_version==False:
        # scale with min/max by grid cells
        minCalib=[None]*28*28
        maxCalib=[None]*28*28
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                minCalib[n]=CalibX[:,k,l,:].min()
                maxCalib[n]=CalibX[:,k,l,:].max()
                CalibX[:,k,l,:]=(CalibX[:,k,l,:]-minCalib[n])/(maxCalib[n]-minCalib[n])
                ProjX[:,k,l,:]=(ProjX[:,k,l,:]-minCalib[n])/(maxCalib[n]-minCalib[n])

    if rank_version==True:
        minCalib=None
        maxCalib=None
        # scale with rank by grid cells
        for k in range(28):
            for l in range(28):
                CalibX[:,k,l,0]=(rankdata(CalibX[:,k,l,0],method="min")/len(CalibX[:,k,l,0]))
                ProjX[:,k,l,0]=(rankdata(ProjX[:,k,l,0],method="min")/len(ProjX[:,k,l,0]))
#    print(CalibX.shape)
#    print(ProjX.shape)
#    print(OriginalCalibX.shape)
#    print(OriginalProjX.shape)
    return CalibX, ProjX, lon, lat, minCalib, maxCalib, ind, point_grid, OriginalCalibX, OriginalProjX



#standalone discriminator model
def define_discriminator(in_shape=(28,28,1), lr_disc=0.0002, nb_filters=[64,64]): #same as Soulivanh
    model = Sequential()
    model.add(Conv2D(nb_filters[0], (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(nb_filters[1], (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=lr_disc, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(in_shape=(28,28,1), nb_filters=[64,64,64], rank_version=False):  #same as Soulivanh (except filter size, no impact on results)
    input = Input(shape = in_shape)
    c28x28 = Conv2D(nb_filters[0], (3,3), padding='same')(input)
    c14x14 = Conv2D(nb_filters[1], (3,3), strides=(2, 2), padding='same')(c28x28)
    model = LeakyReLU(alpha=0.2)(c14x14)
    model = Dropout(0.4)(model)
    model = Conv2D(nb_filters[2], (3,3), strides=(2, 2), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.4)(model)
    # upsample to 14x14
    model = Conv2DTranspose(nb_filters[1], (4,4), strides=(2,2), padding='same')(model) #filter changed only
    model = Add()([model, c14x14]) # SKIP Connection
    model = LeakyReLU(alpha=0.2)(model)
    # upsample to 28x28
    model = Conv2DTranspose(nb_filters[0], (4,4), strides=(2,2), padding='same')(model) #filter changed only
    model = Add()([model, c28x28]) # SKIP Connection
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(1, (1,1), padding='same')(model)
    model = Add()([model, input]) # SKIP Connection
    if rank_version==False:
        model = LeakyReLU(alpha=0.2)(model)
    else:
        model = tf.keras.layers.Activation(activation='sigmoid')(model)
    generator = Model(input, model)
    return generator

def define_combined(genA2B, genB2A, discA, discB,lr_gen=0.002, lambda_valid=1, lambda_reconstruct=10, lambda_identity=1, in_shape=(28,28,1), L_norm = "L1norm" ): #same as Soulivanh
    if L_norm == "L1norm":
        name_norm = 'mae'
    if L_norm == "L2norm":
        name_norm = 'mse'
    inputA = Input(shape = in_shape)
    inputB = Input(shape = in_shape)
    gen_imgB = genA2B(inputA)
    gen_imgA = genB2A(inputB)
    #for cycle consistency
    reconstruct_imgA = genB2A(gen_imgB)
    reconstruct_imgB = genA2B(gen_imgA)
    # identity mapping
    gen_orig_imgB = genA2B(inputB)
    gen_orig_imgA = genB2A(inputA)
    discA.trainable = False
    discB.trainable = False
    valid_imgA = discA(gen_imgA)
    valid_imgB = discB(gen_imgB)
    opt = Adam(lr=lr_gen, beta_1=0.5)
    comb_model = Model([inputA, inputB], [valid_imgA, valid_imgB, reconstruct_imgA, reconstruct_imgB, gen_orig_imgA, gen_orig_imgB])
    comb_model.compile(loss=['binary_crossentropy', 'binary_crossentropy', name_norm, name_norm, name_norm, name_norm],loss_weights=[  lambda_valid, lambda_valid, lambda_reconstruct, lambda_reconstruct, lambda_identity, lambda_identity],optimizer=opt) # sum of the losses 
    return comb_model

# select real samples
def generate_real_samples(dataset, n_samples): #same as Soulivanh
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # reitrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

def generate_fake_samples(g_model, dataset): #same as Soulivanh
    # predict outputs
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((np.size(dataset, 0), 1))
    return X, y

### Compute mean and sd of an array (nb_images,28,28,1)
def compute_mean_sd_array_new(data):
    nb_images=data.shape[0]
    res_mean=np.reshape([None]*28*28,(1,28,28))
    res_sd=np.reshape([None]*28*28,(1,28,28))
    #Compute mean and sd for each grid
    for k in range(28):
        for l in range(28):
            res_mean[:,k,l]=np.mean(data[:,k,l,:])
            res_sd[:,k,l]=np.std(data[:,k,l,:])
    res_mean = res_mean.astype(float)
    res_sd = res_sd.astype(float)
    return res_mean, res_sd

def compute_freqdry_array_new(data):
    nb_images=data.shape[0]
    res_freqdry=np.reshape([None]*28*28,(1,28,28))
    #Compute mean and sd for each grid
    for k in range(28):
        for l in range(28):
            res_freqdry[:,k,l]=np.sum(data[:,k,l,:]==0)/nb_images
    res_freqdry = res_freqdry.astype(float)
    return res_freqdry


########## For evaluation
def transform_array_in_matrix(data_array,ind,point_grid): #input: LONxLATxtime #output: nb_var x time
    res_transform=np.empty([ind.shape[0], data_array.shape[2]])
    k=(-1)
    for point in point_grid:
        k=k+1
        i=ind[point,0]
        j=ind[point,1]
        res_transform[k,:]=data_array[i,j,:]
    return res_transform

def transform_back_in_array(data_matrix,ind,point_grid): #input: nb_var x time output: LONxLATxtime 
    res_transform=np.empty([np.max(ind[:,0]+1), np.max(ind[:,1])+1,data_matrix.shape[1]])
    k=(-1)
    for point in point_grid:
        k=k+1
        i=ind[point,0]
        j=ind[point,1]
        res_transform[i,j,:]=data_matrix[k,:]
    return res_transform


def compute_correlo(remove_spat_mean,data,ind,lon,lat,point_grid):
    ### needed in python
    data=np.transpose(data[:,:,:,0],(2,1,0))
    ### end in python
    lon2=np.empty(784)
    lat2=np.empty(784)
    for i in range(784):
        lon2[i]=lon[ind[i,0],ind[i,1]]
        lat2[i]=lat[ind[i,0],ind[i,1]]
    NS = len(lon2)
    PI=3.141593
    longitudeR=lon2*PI/180
    latitudeR=lat2*PI/180
    Distancekm = np.empty([int(NS*(NS-1)/2)])
    Dist2 = np.empty([NS,NS])
    cpt=-1
    for i in range(0,(NS-1)):
        Dist2[i,i]=0
        for j in range((i+1),NS):
            cpt = cpt +1
            Distancekm[cpt]=6371*acos(sin(latitudeR[i])*sin(latitudeR[j])+cos(latitudeR[i])*cos(latitudeR[j])*cos(longitudeR[i]-longitudeR[j]))
            Dist2[i,j]=Distancekm[cpt]
            Dist2[j,i]=Dist2[i,j]

    Dist2[NS-1,NS-1]=0
    size=8 # corresponds to the size of cells!!! 0.5x0.5-> 55km
    varsize=size/2
    Nc=ceil(Dist2.max()/size)
    MatData=transform_array_in_matrix(data,ind,point_grid)
    tmp_daily_spat_mean=np.mean(MatData, axis=0)
    means_expanded = np.outer(tmp_daily_spat_mean, np.ones(784))
    if remove_spat_mean==True:
        Mat_daily_mean_removed=np.transpose(MatData)-means_expanded
    else:
        Mat_daily_mean_removed=np.transpose(MatData)
    Cspearman,_ =spearmanr(Mat_daily_mean_removed)
    Res_Mean_corr=[]
    Res_Med_corr=[]
    Correlo_dist=[]
    for n in range(0,Nc):
        d=n*size
        Correlo_dist.append(d)
        coordinates=(((Dist2>=(d-varsize)) & (Dist2<(d+varsize))))
        Res_Mean_corr.append(Cspearman[coordinates].mean())
        Res_Med_corr.append(np.median(Cspearman[coordinates]))
    Res_Mean_corr=np.array(Res_Mean_corr)
    Res_Med_corr=np.array(Res_Med_corr)
    Correlo_dist=np.array(Correlo_dist)
    return Res_Mean_corr, Res_Med_corr, Correlo_dist

def plot_maps(epoch,QQ2B_version, PR_version, mat_A, mat_B, mat_QQ, mat_X2B, mat_X2B_reordered, title, path_plot, subfolder, lon=np.array(range(28)), lat=np.array(range(28))):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    mat_A = mat_A.astype(float)
    mat_B = mat_B.astype(float)
    mat_QQ = mat_QQ.astype(float)
    mat_X2B = mat_X2B.astype(float)
    mat_X2B_reordered = mat_X2B_reordered.astype(float)
    #### On inverse LON_LAT pour plotter correctement
    ####fliplr for (1,28,28), else flipud
    mat_A=np.fliplr(mat_A)
    mat_B=np.fliplr(mat_B)
    mat_QQ = np.fliplr(mat_QQ)
    mat_X2B=np.fliplr(mat_X2B)
    mat_X2B_reordered = np.fliplr(mat_X2B_reordered)
    ### Plot
    #### Mean and sd / MAE ####
    if PR_version==False:
        examples = vstack((mat_A, mat_B, mat_QQ, mat_X2B, mat_X2B_reordered, mat_A-mat_B, mat_B-mat_B, mat_QQ - mat_B, mat_X2B-mat_B, mat_X2B_reordered - mat_B))
        names_=("A","B","QQ", X_name + "2B", X_name + "2B_reord.","A-B","B-B", "QQ-B",X_name + "2B-B", X_name + "2B_reord.-B")
    else:
        examples = vstack((mat_A, mat_B, mat_QQ, mat_X2B, mat_X2B_reordered, (mat_A-mat_B)/mat_B, (mat_B-mat_B)/mat_B, (mat_QQ-mat_B)/mat_B, (mat_X2B-mat_B)/mat_B,  (mat_X2B_reordered-mat_B)/mat_B))
        names_=("A","B","QQ", X_name + "2B",X_name + "2B_reord.", "(A-B)/B","(B-B)/B","(QQ-B)/B","(" +X_name +"2B-B)/B", "(" +X_name +"2B_reord.-B)/B")
    nchecks=5

    fig, axs = pyplot.subplots(2,nchecks, figsize=(14,8))
    cm = ['YlOrRd','RdBu']
    fig.subplots_adjust(right=0.925) # making some room for cbar
    quant_10=np.quantile(mat_B,0.1)
    quant_90=np.quantile(mat_B,0.9)
    for row in range(2):
        for col in range(nchecks):
            i=nchecks*row+col
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
                ax.set_title(str(names_[i]) + ' mean: ' +str(round(np.nanmean(examples[i, :, :]),2)),fontsize = 10)  #+ ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),2))  ,fontsize=8)

            else:
                vmin=-1.5
                vmax=1.5
                pcm = ax.imshow(examples[nchecks*row + col, :,:], cmap = cm[row], vmin=vmin, vmax=vmax)
                ax.set_title(str(names_[i]) + ' mae: ' +str(round(np.nanmean(abs(examples[i, :, :])),2)) , fontsize = 10)#+ ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),2)), fontsize = 8)
        fig.colorbar(pcm, ax = axs[row,:],shrink=0.5)

    # # save plot to file
    filename = path_plot + '/' + subfolder + '/plot_criteria_'+ title + '_%03d.png' % (epoch+1)
    fig.savefig(filename, dpi=150)
    pyplot.close()

def recap_accuracy_and_save_gendisc(epoch,genX2B, genB2X, discX, discB, datasetX, datasetB, path_plot, n_samples=100):
    #prepare real samples
    xA_real, yA_real = generate_real_samples(datasetX, n_samples)
    xB_real, yB_real = generate_real_samples(datasetB, n_samples)
    # evaluate discriminator on real examples
    _, accA_real = discX.evaluate(xA_real, yA_real, verbose=0)
    _, accB_real = discB.evaluate(xB_real, yB_real, verbose=0)
    # prepare fake examples
    xA_fake, yA_fake = generate_fake_samples(genB2X, xB_real)
    xB_fake, yB_fake = generate_fake_samples(genX2B, xA_real)
    # evaluate discriminator on fake examples
    _, accA_fake = discX.evaluate(xA_fake, yA_fake, verbose=0)
    _, accB_fake = discB.evaluate(xB_fake, yB_fake, verbose=0)
    # summarize discriminator performance
    print('On a sample of '+ str(n_samples) + ':')
    print('>Accuracy discX real: %.0f%%, discX fake: %.0f%%' % (accA_real*100, accA_fake*100))
    print('>Accuracy discB real: %.0f%%, discB fake: %.0f%%' % (accB_real*100, accB_fake*100))
    # save the disc and gen model
    genB2X.save(path_plot+'/models/genB2X_model_%03d.h5' % (epoch+1))
    genX2B.save(path_plot+'/models/genX2B_model_%03d.h5' % (epoch+1))
    discX.save(path_plot+'/models/discX_model_%03d.h5' % (epoch+1))
    discB.save(path_plot+'/models/discB_model_%03d.h5' % (epoch+1))


def plot_history_loss(epoch,QQ2B_version, nb_data_per_epoch,discX_hist, discB_hist, validX_hist, validB_hist, recX_hist, recB_hist, identX_hist, identB_hist, weighted_hist, discX_acc_hist, discB_acc_hist,path_plot, subfolder):
    ####
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    # plot loss
    pyplot.figure(figsize=(9,9))
    pyplot.subplot(4, 1, 1)
    pyplot.title("Number of epochs:" + str(epoch),fontsize=7)
    pyplot.plot(validX_hist, label='loss-valid' + X_name)
    pyplot.plot(validB_hist, label='loss-validB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    pyplot.subplot(4,1,2)
    pyplot.plot(np.array(recX_hist)*10, label='loss-rec' + X_name)
    pyplot.plot(np.array(recB_hist)*10, label='loss-recB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    pyplot.subplot(4, 1, 3)
    pyplot.plot(identX_hist, label='loss-ident' + X_name)
    pyplot.plot(identB_hist, label='loss-identB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    pyplot.subplot(4, 1, 4)
    pyplot.plot(weighted_hist, label='loss-weighted',color="green")
    pyplot.legend()
    pyplot.ylim((0,3))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_gen_loss.png')
    pyplot.close()
    pyplot.figure(figsize=(9,9))
    pyplot.subplot(2, 1, 1)
    pyplot.title("Number of epochs:" + str(epoch),fontsize=7)
    pyplot.plot(discX_hist, label='loss-disc' + X_name)
    pyplot.plot(discB_hist, label='loss-discB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    pyplot.subplot(2, 1, 2)
    pyplot.plot(discX_acc_hist, label='disc' + X_name + '-acc')
    pyplot.plot(discB_acc_hist, label='discB-acc')
    pyplot.legend()
    pyplot.ylim((-0.1,1.1))
    pyplot.xticks(range(0,(epoch+1)*nb_data_per_epoch,nb_data_per_epoch*50), range(0,epoch+1,50))
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_disc_loss.png')
    pyplot.close()



def plot_history_criteria(QQ2B_version, dict_crit,title_crit,ylim1,ylim2,path_plot, subfolder):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    #plot criteria mean
    pyplot.subplot(1, 1, 1)
    pyplot.hlines(dict_crit["mae_A"][0],xmin=0, xmax=len(dict_crit["mae_X2B"]), label='A', color='red')
    pyplot.hlines(dict_crit["mae_QQ"][0],xmin=0, xmax=len(dict_crit["mae_X2B"]), label='QQ', color='orange')
    pyplot.plot(dict_crit["mae_X2B"], label= X_name + "2B", color="blue")
    pyplot.plot(dict_crit["mae_X2B_reordered"], label= X_name + "2B_reord.", color="green")
    pyplot.legend()
    val, idx = min((val, idx) for (idx, val) in enumerate(dict_crit["mae_X2B"]))
    pyplot.ylim((ylim1,ylim2))
    val_X2B, idx_X2B = min((val, idx) for (idx, val) in enumerate(dict_crit["mae_X2B"]))
    val_X2B_reordered, idx_X2B_reordered = min((val, idx) for (idx, val) in enumerate(dict_crit["mae_X2B_reordered"]))
    pyplot.title("A: " + str(round(dict_crit["mae_A"][0],2)) + ", QQ: " + str(round(dict_crit["mae_QQ"][0],2)) + ", best " + X_name + "2B: " +  str(round(val_X2B,2)) + " at epo. "  +  str(idx_X2B*10+1) +  ", best " + X_name + "2B_reord.: " +  str(round(val_X2B_reordered,2)) + " at epo. "  +  str(idx_X2B_reordered*10+1) , fontsize=7)
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_'+ title_crit + '.png')
    pyplot.close()




def rmse(ref, pred):
    return np.sum((ref.astype("float") - pred.astype("float")) **2)/(ref.shape[1]*ref.shape[2]*ref.shape[0])

def compute_matrix_real_rank(data):
    res=np.copy(data)
    for k in range(28):
        for l in range(28):
            res[:,k,l,0]=(rankdata(data[:,k,l,0],method="min")/len(data[:,k,l,0]))
    return res

def compute_some_rmse(QQ2B_version, is_DS,dict_rmse,sample_A, sample_B, sample_QQ, sample_X2B, sample_B2X, sample_X2B2X, sample_B2X2B,sample_B2X_X, sample_X2B_B):
    def rmse(ref, pred):
        return np.sum((ref.astype("float") - pred.astype("float")) **2)/(ref.shape[1]*ref.shape[2]*ref.shape[0])
    if QQ2B_version==True:
        sample_X = np.copy(sample_QQ)
    else:
        sample_X = np.copy(sample_A)

    dict_rmse["rmse_B2X2B"].append(rmse(sample_B, sample_B2X2B))
    dict_rmse["rmse_X2B_B"].append(rmse(sample_B, sample_X2B_B))
    dict_rmse["rmse_X2B2X"].append(rmse(sample_X,sample_X2B2X))
    dict_rmse["rmse_B2X_X"].append(rmse(sample_X, sample_B2X_X))

    if len(dict_rmse["rmse_QQ"])==0:
        dict_rmse["rmse_QQ"].append(rmse(sample_B, sample_QQ))

    if len(dict_rmse["rmse_A"])==0:
        dict_rmse["rmse_A"].append(rmse(sample_B, sample_A))

    if is_DS==True:
        dict_rmse["rmse_X2B"].append(rmse(sample_B, sample_X2B))
        dict_rmse["rmse_B2X"].append(rmse(sample_X, sample_B2X))

    return dict_rmse



def plot_dict_rmse(QQ2B_version, is_DS,dict_norm, dict_varphy, path_plot, subfolder):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"

    np.save(path_plot+'/models/rmse_dict_norm.npy',dict_norm)
    np.save(path_plot+'/models/rmse_dict_varphy.npy',dict_varphy)
    pyplot.figure(figsize=(9,9))
    pyplot.subplot(2, 1, 1)
    pyplot.plot(dict_norm["rmse_B2X2B"], label='B2' + X_name +'2B', color="red")
    pyplot.plot(dict_norm["rmse_X2B_B"], label= X_name + '2B_B', color= "green")
    if is_DS==True:
        pyplot.hlines(dict_norm["rmse_QQ"],xmin=0, xmax=len(dict_norm["rmse_X2B"]), label='QQ', color='orange')
        pyplot.hlines(dict_norm["rmse_A"],xmin=0, xmax=len(dict_norm["rmse_X2B"]), label='A',color='black')
        pyplot.plot(dict_norm["rmse_X2B"], label=X_name + '2B', color="blue")
        val, idx = min((val, idx) for (idx, val) in enumerate(dict_norm["rmse_X2B"]))
        pyplot.title("Best " + X_name + "2B at epoch " +  str(idx*10+1), fontsize=7)

    pyplot.legend()
    pyplot.yscale('log')
    pyplot.ylim((1e-7,1))

    pyplot.subplot(2,1,2)
    pyplot.plot(dict_norm["rmse_X2B2X"], label= X_name + '2B2' + X_name, color="red")
    pyplot.plot(dict_norm["rmse_B2X_X"], label='B2' + X_name +'_' + X_name, color="green")
    if is_DS==True:
        pyplot.plot(dict_norm["rmse_B2X"], label='B2' + X_name, color="blue")
        val, idx = min((val, idx) for (idx, val) in enumerate(dict_norm["rmse_B2X"]))
        pyplot.title("Best B2X at epoch " +  str(idx*10+1), fontsize=7)
    pyplot.legend()
    pyplot.yscale('log')
    pyplot.ylim((1e-7,1))
    #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_rmse_norm.png')
    pyplot.close()

#### RMSE_varphy
    pyplot.figure(figsize=(9,9))
    pyplot.subplot(2, 1, 1)
    pyplot.plot(dict_varphy["rmse_B2X2B"], label='B2' + X_name +'2B', color="red")
    pyplot.plot(dict_varphy["rmse_X2B_B"], label= X_name + '2B_B', color="green")
    if is_DS==True:
        pyplot.hlines(dict_varphy["rmse_QQ"],xmin=0, xmax=len(dict_varphy["rmse_X2B"]), label='QQ', color='orange')
        pyplot.hlines(dict_varphy["rmse_A"],xmin=0, xmax=len(dict_varphy["rmse_X2B"]), label='A', color="black")
        pyplot.plot(dict_varphy["rmse_X2B"], label=X_name + '2B', color="blue")
        val, idx = min((val, idx) for (idx, val) in enumerate(dict_varphy["rmse_X2B"]))
        pyplot.title("Best " + X_name +"2B at epoch " +  str(idx*10+1), fontsize=7)

    pyplot.legend()
    pyplot.ylim((0,1))

    pyplot.subplot(2,1,2)
    pyplot.plot(dict_varphy["rmse_X2B2X"], label= X_name + '2B2' + X_name, color="red")
    pyplot.plot(dict_varphy["rmse_B2X_X"], label='B2' + X_name + '_' +X_name, color="green")
    if is_DS==True:
        pyplot.plot(dict_varphy["rmse_B2X"], label='B2' + X_name, color="blue")
        val, idx = min((val, idx) for (idx, val) in enumerate(dict_varphy["rmse_B2X"]))
        pyplot.title("Best B2" + X_name + " at epoch " +  str(idx*10+1), fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,1))
    #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_rmse_varphy.png')
    pyplot.close()




#### End ne pas toucher
#################################################################################################################################################################################
##################################################################################################################################################################################

def compute_localenergy_array_new(data_A,data_B):
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
    for k in range(28):
        print('localenergy' + str(k))
        for l in range(28):
            if k>0 and k<27:
                if l>0 and l<27:
                    smallA=flat_array(data_A[:,(k-1):(k+2),(l-1):(l+2),0])
                    smallB=flat_array(data_B[:,(k-1):(k+2),(l-1):(l+2),0])
                    res_localenergy[:,k,l] = dcor.homogeneity.energy_test_statistic(np.transpose(smallA), np.transpose(smallB), exponent=1)
    res_localenergy = res_localenergy.astype(float)
    return res_localenergy


def plot_maps_localWD(epoch, QQ2B_version, PR_version, mat_A, mat_QQ, mat_X2B, mat_X2B_reordered, title, path_plot, subfolder, nb_lon=27, nb_lat=27):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    mat_A = mat_A.astype(float)
    mat_QQ = mat_QQ.astype(float)
    mat_X2B = mat_X2B.astype(float)
    mat_X2B_reordered = mat_X2B_reordered.astype(float)
    mat_A = mat_A[:,1:nb_lat,1:nb_lon]
    mat_QQ = mat_QQ[:, 1:nb_lat, 1:nb_lon]
    mat_X2B = mat_X2B[:, 1:nb_lat, 1:nb_lon]
    mat_X2B_reordered = mat_X2B_reordered[:, 1:nb_lat, 1:nb_lon]
    #### On inverse LON_LAT pour plotter correctement
    ####fliplr for (1,28,28), else flipud
    mat_A=np.fliplr(mat_A)
    mat_QQ=np.fliplr(mat_QQ)
    mat_X2B=np.fliplr(mat_X2B)
    mat_X2B_reordered=np.fliplr(mat_X2B_reordered)
    #### Mean and sd / MAE ####
    if PR_version==False:
        examples = vstack((mat_A, mat_QQ, mat_X2B, mat_X2B_reordered, mat_A-mat_QQ, mat_QQ-mat_QQ, mat_X2B-mat_QQ, mat_X2B_reordered-mat_QQ))
        names_=("A","QQ",X_name + "2B", X_name + "2B_reord." ,"A-QQ","QQ-QQ",X_name + "2B-QQ", X_name + "2B_reord.-QQ")
    else:
        examples = vstack((mat_A, mat_QQ, mat_X2B, mat_X2B_reordered, (mat_A-mat_QQ)/mat_QQ, (mat_QQ-mat_QQ)/mat_QQ, (mat_X2B-mat_QQ)/mat_QQ), (mat_X2B_reordered - mat_QQ)/mat_QQ)
        names_=("A","QQ",X_name +"2B", X_name +"2B_reord.","(A-QQ)/QQ","(QQ-QQ)/QQ","(" + X_name + "2B-QQ)/QQ", "(" + X_name + "2B_reord.-QQ)/QQ")
    nchecks=4
    fig, axs = pyplot.subplots(2,nchecks, figsize=(11, 8))
    cm = ['YlOrRd','RdBu']
    fig.subplots_adjust(right=0.925) # making some room for cbar
    quant_10=0
    quant_90=np.quantile(mat_QQ,0.9)
    for row in range(2):
        for col in range(nchecks):
            i=nchecks*row+col
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
                ax.set_title(str(names_[i]) + ' mean: ' +str(round(np.nanmean(examples[i, :, :]),2)), fontsize = 10) #+ ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),2))  ,fontsize=8)
            else:
                vmin=-0.2
                vmax=0.2
                pcm = ax.imshow(examples[nchecks*row + col, :,:], cmap = cm[row], vmin=vmin, vmax=vmax)
                ax.set_title(str(names_[i]) + ' mae: ' +str(round(np.nanmean(abs(examples[i, :, :])),2)), fontsize = 10)# + ' / sd: ' + str(round(np.nanstd(examples[i, :, :]),2)), fontsize = 8)
        fig.colorbar(pcm, ax = axs[row,:],shrink=0.5)
    # # save plot to file
    filename = path_plot + '/' + subfolder + '/plot_criteria_'+ title + '_%03d.png' % (epoch+1)
    fig.savefig(filename, dpi=150)
    pyplot.close()


def plot_some_raw_maps_and_compute_rmse(epoch, is_DS, rank_version,PR_version, QQ2B_version,  computation_localWD, computation_localenergy, genX2B, genB2X, discB, datasetA,datasetB, datasetQQ, OriginalA, OriginalB, OriginalQQ, minX, maxX, minB, maxB, dict_mean, dict_sd_rel, dict_correlogram, dict_correlogram_wt_remove, dict_norm_rmse, dict_varphy_rmse, dict_realrank_localwd, dict_varphy_localenergy, dict_realrank_localenergy, dict_score_discB, ind, lon, lat,  point_grid, path_plot, subfolder, n_samples=8):

    def plot_raw_varphy(name_data, QQ2B_version, ix, epoch, PR_version, sample_A, sample_B, sample_QQ, sample_X2B, sample_B2X, sample_X2B2X, sample_B2X2B, sample_B2X_X, sample_X2B_B,n=n_samples):
        vmin = np.quantile(vstack((sample_A,sample_B)), 0.025)
        vmax = np.quantile(vstack((sample_A,sample_B)), 0.975)
        nchecks=9
        fig, axs = pyplot.subplots(n, nchecks, figsize= (10,10))
        fig.subplots_adjust(right=0.925) # making some room for cbar
        for i in range(n):
            mat_A = sample_A[i,:,:,0].astype(float)
            mat_B = sample_B[i,:,:,0].astype(float)
            mat_QQ = sample_QQ[i,:,:,0].astype(float)

            mat_X2B = sample_X2B[i,:,:,0].astype(float)
            mat_B2X = sample_B2X[i,:,:,0].astype(float)
            #
            mat_X2B2X = sample_X2B2X[i,:,:,0].astype(float)
            mat_B2X2B = sample_B2X2B[i,:,:,0].astype(float)
            #
            mat_B2X_X = sample_B2X_X[i,:,:,0].astype(float)
            mat_X2B_B = sample_X2B_B[i,:,:,0].astype(float)
            #
            #### On inverse LON_LAT pour plotter correctement
            mat_A=np.flipud(mat_A)
            mat_B=np.flipud(mat_B)
            mat_QQ=np.flipud(mat_QQ)
            #
            mat_X2B=np.flipud(mat_X2B)
            mat_B2X = np.flipud(mat_B2X)
            #
            mat_X2B2X=np.flipud(mat_X2B2X)
            mat_B2X2B = np.flipud(mat_B2X2B)
            #
            mat_B2X_X=np.flipud(mat_B2X_X)
            mat_X2B_B = np.flipud(mat_X2B_B)
            #
            ### Plot
            if QQ2B_version==True:
                names_=("A", "B","QQ2B","B2QQ2B","QQ2B_B","QQ", "B2QQ", "QQ2B2QQ", "B2QQ_QQ")
                examples = vstack((mat_A,mat_B, mat_X2B, mat_B2X2B, mat_X2B_B, mat_QQ, mat_B2X, mat_X2B2X, mat_B2X_X))
            else:
                names_=("B","QQ","A2B","B2A2B","A2B_B","A", "B2A", "A2B2A", "B2A_A")
                examples = vstack((mat_B,mat_QQ, mat_X2B, mat_B2X2B, mat_X2B_B, mat_A, mat_B2X, mat_X2B2X, mat_B2X_X))
            # define subplot
            for j in range(nchecks):
                axs[i,j].spines['top'].set_visible(False)
                axs[i,j].spines['right'].set_visible(False)
                axs[i,j].spines['bottom'].set_visible(False)
                axs[i,j].spines['left'].set_visible(False)
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                # plot raw pixel data
                img1 = axs[i,j].imshow(examples[(j*28):((j+1)*28), :], cmap='YlOrRd',vmin=vmin, vmax=vmax)
                if i==0:
                    axs[i,j].set_title(str(names_[j]), fontsize=10)
                if j==0:
                    axs[i,j].set_ylabel("day " + str(ix[i]+1), fontsize=5)
                 # # save plot to file
        cbar_ax = fig.add_axes([0.9375, 0.333, 0.0125, 0.333])
        fig.colorbar(img1,cax=cbar_ax)

        if PR_version==True:
            name_var_phy="pr"
        else:
            name_var_phy="tas"
        filename = path_plot + '/' + subfolder + '/plot_' + name_data + '_' + name_var_phy +  '_maps_%03d.png' % (epoch+1)
        fig.savefig(filename, dpi=150)
        pyplot.close()

    ##### Begin
    if PR_version==False:
        name_var="T2"
    else:
        name_var="PR"

    ###################################################################################################################
    #### NORM ####
    ##################################################################################################################
    if QQ2B_version==True:
        sample_X = np.copy(datasetQQ)
        OriginalX = np.copy(OriginalQQ)
    else:
        sample_X = np.copy(datasetA)
        OriginalX = np.copy(OriginalA)
    #### Generate correction normalized
    sample_X2B = genX2B.predict(sample_X)
    sample_B2X = genB2X.predict(datasetB)
    sample_X2B2X = genB2X.predict(sample_X2B)
    sample_B2X2B = genX2B.predict(sample_B2X)
    sample_B2X_X = genB2X.predict(sample_X)
    sample_X2B_B = genX2B.predict(datasetB)


    #####################################################################################################################
    #### VARPHY ########
    #####################################################################################################################
    #### Generate correction
    sample_varphy_X2B = np.copy(sample_X2B)
    sample_varphy_B2X = np.copy(sample_B2X)
    sample_varphy_X2B2X = np.copy(sample_X2B2X)
    sample_varphy_B2X2B = np.copy(sample_B2X2B)
    sample_varphy_B2X_X = np.copy(sample_B2X_X)
    sample_varphy_X2B_B = np.copy(sample_X2B_B)

    def denormalize_minmax(data, minX, maxX):
        res=np.copy(data)
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                res[:,k,l,:] = data[:,k,l,:]*(maxX[n] - minX[n])+ minX[n]
        return res

    def denormalize_rank(data, Originaldata):
        res = np.copy(data)
        for k in range(28):
            for l in range(28):
                quant_to_take=np.array(data[:,k,l,0])
                res[:,k,l,0] = np.quantile(Originaldata[:,k,l,0],quant_to_take)
        return res

    if rank_version==False:
        #Rescale climatic variables wrt Xmin and Xmax
        sample_varphy_B2X=denormalize_minmax(sample_varphy_B2X, minX, maxX)
        sample_varphy_X2B2X=denormalize_minmax(sample_varphy_X2B2X, minX, maxX)
        sample_varphy_B2X_X=denormalize_minmax(sample_varphy_B2X_X, minX, maxX)

        sample_varphy_X2B=denormalize_minmax(sample_varphy_X2B, minB, maxB)
        sample_varphy_B2X2B=denormalize_minmax(sample_varphy_B2X2B, minB, maxB)
        sample_varphy_X2B_B=denormalize_minmax(sample_varphy_X2B_B, minB, maxB)
    else:
        #Reorder rank data with OriginalData
        sample_varphy_B2X=denormalize_rank(sample_varphy_B2X, OriginalX)
        sample_varphy_X2B2X=denormalize_rank(sample_varphy_X2B2X, OriginalX)
        sample_varphy_B2X_X=denormalize_rank(sample_varphy_B2X_X, OriginalX)

        sample_varphy_X2B=denormalize_rank(sample_varphy_X2B, OriginalB)
        sample_varphy_B2X2B=denormalize_rank(sample_varphy_B2X2B, OriginalB)
        sample_varphy_X2B_B=denormalize_rank(sample_varphy_X2B_B, OriginalB)

    ##!!!! Preprocess for PR !!! 
    OriginalA_preproc=np.copy(OriginalA)
    OriginalB_preproc = np.copy(OriginalB)
    OriginalQQ_preproc = np.copy(OriginalQQ)
    if PR_version==True:
        sample_varphy_X[sample_varphy_X < 1] = 0
        sample_varphy_X2B[sample_varphy_X2B < 1] = 0
        sample_varphy_B2X[sample_varphy_B2X < 1] = 0
        sample_varphy_X2B2X[sample_varphy_X2B2X < 1] = 0
        sample_varphy_B2X2B[sample_varphy_B2X2B < 1] = 0
        sample_varphy_X2B_B[sample_varphy_X2B_B < 1] = 0
        sample_varphy_B2X_X[sample_varphy_B2X_X < 1] = 0
        OriginalA_preproc[OriginalA_preproc < 1]=0
        OriginalB_preproc[OriginalB_preproc < 1]=0
        OriginalQQ_preproc[OriginalQQ_preproc < 1]=0

    #####################################################################################################################
    #### REALRANK ########
    #####################################################################################################################
    ########################
    #### on realrank c'est ok de calculer sur preproc data???
    sample_realrank_A = compute_matrix_real_rank(OriginalA_preproc)
    sample_realrank_B = compute_matrix_real_rank(OriginalB_preproc)
    sample_realrank_QQ = compute_matrix_real_rank(OriginalQQ_preproc)
    sample_realrank_X2B = compute_matrix_real_rank(sample_varphy_X2B)
    sample_realrank_B2X = compute_matrix_real_rank(sample_varphy_B2X)
    sample_realrank_X2B2X = compute_matrix_real_rank(sample_varphy_X2B2X)
    sample_realrank_B2X2B = compute_matrix_real_rank(sample_varphy_B2X2B)
    sample_realrank_X2B_B = compute_matrix_real_rank(sample_varphy_X2B_B)
    sample_realrank_B2X_X = compute_matrix_real_rank(sample_varphy_B2X_X)

    #####################################################################################################################
    #### VARPHY_REORDERED ######## alla Cannon
    #####################################################################################################################
    sample_varphy_reordered_X2B= np.copy(sample_varphy_X2B)

    def alla_Cannon(data_X2B, data_QQ): ### reorder data_QQ with struct. of data_X2B
        res = np.copy(data_X2B)
        for k in range(28):
            for l in range(28):
                sorted_data_QQ=np.sort(data_QQ[:,k,l,0])
                idx=rankdata(data_X2B[:,k,l,0],method="min")
                idx=idx.astype(int)-1
                res[:,k,l,0] = sorted_data_QQ[idx]
        return res

    sample_varphy_reordered_X2B= alla_Cannon(sample_varphy_reordered_X2B, OriginalQQ)

    #### Preprocess!!!
    if PR_version==True:
        sample_varphy_reordered_X2B[sample_varphy_reordered_X2B < 1] = 0
    sample_realrank_varphy_reordered_X2B = compute_matrix_real_rank(sample_varphy_reordered_X2B) #should be the same as sample_realrank_X2B

    ######################################################################################################################
    #### PLOTS ####
    ### Raw plots ####
    ix = np.random.randint(0, sample_X.shape[0], n_samples)
    plot_raw_varphy('norm',QQ2B_version, ix, epoch, PR_version, datasetA[ix], datasetB[ix], datasetQQ[ix], sample_X2B[ix], sample_B2X[ix], sample_X2B2X[ix], sample_B2X2B[ix], sample_B2X_X[ix], sample_X2B_B[ix])
    plot_raw_varphy('varphy',QQ2B_version, ix, epoch, PR_version, OriginalA_preproc[ix], OriginalB_preproc[ix], OriginalQQ_preproc[ix], sample_varphy_X2B[ix], sample_varphy_B2X[ix], sample_varphy_X2B2X[ix], sample_varphy_B2X2B[ix], sample_varphy_B2X_X[ix], sample_varphy_X2B_B[ix])
    plot_raw_varphy('realrank',QQ2B_version, ix, epoch, PR_version, sample_realrank_A[ix], sample_realrank_B[ix], sample_realrank_QQ[ix], sample_realrank_X2B[ix], sample_realrank_B2X[ix], sample_realrank_X2B2X[ix], sample_realrank_B2X2B[ix], sample_realrank_B2X_X[ix], sample_realrank_X2B_B[ix])
    empty_frame=np.zeros((n_samples,28,28,1))
    #plot_raw_varphy('varphy_reordered',QQ2B_version, ix, epoch, PR_version, OriginalA_preproc[ix], OriginalB_preproc[ix], OriginalQQ_preproc[ix], sample_varphy_reordered_X2B[ix],empty_frame, empty_frame, empty_frame, empty_frame, empty_frame)

    #### compute RMSE
    dict_norm_rmse=compute_some_rmse(QQ2B_version,is_DS,dict_norm_rmse,datasetA, datasetB, datasetQQ, sample_X2B, sample_B2X, sample_X2B2X, sample_B2X2B, sample_B2X_X, sample_X2B_B)
    dict_varphy_rmse=compute_some_rmse(QQ2B_version, is_DS,dict_varphy_rmse,OriginalA_preproc, OriginalB_preproc, OriginalQQ_preproc, sample_varphy_X2B, sample_varphy_B2X, sample_varphy_X2B2X, sample_varphy_B2X2B, sample_varphy_B2X_X, sample_varphy_X2B_B)

    #### Local energy ####
    if computation_localenergy==True and (epoch % 50 ==0):
        nb_timestep=OriginalA_preproc.shape[0]
        #if subfolder=="proj":
        #    nb_pas=2
        #else:
        #    nb_pas=4
        nb_pas = 1
        coord_localenergy=range(0,nb_timestep,nb_pas)
        #### For varphy
        if "maps_localenergy_A" not in dict_varphy_localenergy:
            maps_localenergy_A=compute_localenergy_array_new(OriginalA_preproc[coord_localenergy,:,:], OriginalB_preproc[coord_localenergy,:,:])
            dict_varphy_localenergy["maps_localenergy_A"]=maps_localenergy_A
            dict_varphy_localenergy["energy_A"].append(np.nanmean(maps_localenergy_A))
        if "maps_localenergy_QQ" not in dict_varphy_localenergy:
            maps_localenergy_QQ=compute_localenergy_array_new(OriginalQQ_preproc[coord_localenergy,:,:], OriginalB_preproc[coord_localenergy,:,:])
            dict_varphy_localenergy["maps_localenergy_QQ"]=maps_localenergy_QQ
            dict_varphy_localenergy["energy_QQ"].append(np.nanmean(maps_localenergy_QQ))

        maps_localenergy_varphy_X2B =compute_localenergy_array_new(sample_varphy_X2B[coord_localenergy,:,:], OriginalB_preproc[coord_localenergy,:,:])
        dict_varphy_localenergy["energy_X2B"].append(np.nanmean(maps_localenergy_varphy_X2B))
        maps_localenergy_varphy_reordered_X2B =compute_localenergy_array_new(sample_varphy_reordered_X2B[coord_localenergy,:,:], OriginalB_preproc[coord_localenergy,:,:])
        dict_varphy_localenergy["energy_X2B_reordered"].append(np.nanmean(maps_localenergy_varphy_reordered_X2B))
        plot_maps_localWD(epoch,QQ2B_version, False, dict_varphy_localenergy["maps_localenergy_A"], dict_varphy_localenergy["maps_localenergy_QQ"], maps_localenergy_varphy_X2B, maps_localenergy_varphy_reordered_X2B, "localenergy_varphy",path_plot=path_plot, subfolder = subfolder)

        if "maps_localenergy_A" not in dict_realrank_localenergy:
            maps_localenergy_A=compute_localenergy_array_new(sample_realrank_A[coord_localenergy,:,:], sample_realrank_B[coord_localenergy,:,:])
            dict_realrank_localenergy["maps_localenergy_A"]=maps_localenergy_A
            dict_realrank_localenergy["energy_A"].append(np.nanmean(maps_localenergy_A))
        if "maps_localenergy_QQ" not in dict_realrank_localenergy:
            maps_localenergy_QQ=compute_localenergy_array_new(sample_realrank_QQ[coord_localenergy,:,:], sample_realrank_B[coord_localenergy,:,:])
            dict_realrank_localenergy["maps_localenergy_QQ"]=maps_localenergy_QQ
            dict_realrank_localenergy["energy_QQ"].append(np.nanmean(maps_localenergy_QQ))

        maps_localenergy_varphy_X2B =compute_localenergy_array_new(sample_realrank_X2B[coord_localenergy,:,:], sample_realrank_B[coord_localenergy,:,:])
        dict_realrank_localenergy["energy_X2B"].append(np.nanmean(maps_localenergy_varphy_X2B))
        maps_localenergy_varphy_reordered_X2B =compute_localenergy_array_new(sample_realrank_varphy_reordered_X2B[coord_localenergy,:,:], sample_realrank_B[coord_localenergy,:,:])
        dict_realrank_localenergy["energy_X2B_reordered"].append(np.nanmean(maps_localenergy_varphy_reordered_X2B))
        plot_maps_localWD(epoch,QQ2B_version, False, dict_realrank_localenergy["maps_localenergy_A"], dict_realrank_localenergy["maps_localenergy_QQ"], maps_localenergy_varphy_X2B, maps_localenergy_varphy_reordered_X2B, "localenergy_realrank",path_plot=path_plot, subfolder = subfolder)

    #Compute Mean Sd criteria
    if "data_A" not in dict_mean:
        res_mean_datasetA, res_sd_datasetA = compute_mean_sd_array_new(OriginalA_preproc)
        dict_mean["data_A"]= res_mean_datasetA
        dict_sd_rel["data_A"]= res_sd_datasetA

        res_mean_datasetB, res_sd_datasetB = compute_mean_sd_array_new(OriginalB_preproc)
        dict_mean["data_B"]= res_mean_datasetB
        dict_sd_rel["data_B"]= res_sd_datasetB

        res_mean_datasetQQ, res_sd_datasetQQ = compute_mean_sd_array_new(OriginalQQ_preproc)
        dict_mean["data_QQ"]= res_mean_datasetQQ
        dict_sd_rel["data_QQ"]= res_sd_datasetQQ

    res_mean_varphy_X2B, res_sd_rel_varphy_X2B = compute_mean_sd_array_new(sample_varphy_X2B)
    res_mean_varphy_reordered_X2B, res_sd_rel_varphy_reordered_X2B = compute_mean_sd_array_new(sample_varphy_reordered_X2B)

    if PR_version==False:
        dict_mean["mae_X2B"].append(np.mean(abs(res_mean_varphy_X2B-dict_mean["data_B"])))
        dict_mean["mae_X2B_reordered"].append(np.mean(abs(res_mean_varphy_reordered_X2B-dict_mean["data_B"])))
        dict_mean["mae_QQ"].append(np.mean(abs(dict_mean["data_QQ"] -dict_mean["data_B"])))
        dict_mean["mae_A"].append(np.mean(abs(dict_mean["data_A"] -dict_mean["data_B"])))
        title_="mean_tas"
    else:
        dict_mean["mae_X2B"].append(np.mean(abs((res_mean_varphy_X2B-dict_mean["data_B"])/dict_mean["data_B"])))
        dict_mean["mae_X2B_reordered"].append(np.mean(abs((res_mean_varphy_reordered_X2B-dict_mean["data_B"])/dict_mean["data_B"])))
        dict_mean["mae_QQ"].append(np.mean(abs((dict_mean["data_QQ"]-dict_mean["data_B"])/dict_mean["data_B"])))
        dict_mean["mae_A"].append(np.mean(abs((dict_mean["data_A"]-dict_mean["data_B"])/dict_mean["data_B"])))
        title_="mean_pr"

    print("MAE X2B: " + str(round(dict_mean["mae_X2B"][-1],3)))
    plot_maps(epoch,QQ2B_version, PR_version, dict_mean["data_A"], dict_mean["data_B"], dict_mean["data_QQ"] , res_mean_varphy_X2B, res_mean_varphy_reordered_X2B, title_,path_plot=path_plot, subfolder=subfolder)

    #### SD ####
    dict_sd_rel["mae_X2B"].append(np.mean(abs((res_sd_rel_varphy_X2B-dict_sd_rel["data_B"])/dict_sd_rel["data_B"])))
    dict_sd_rel["mae_X2B_reordered"].append(np.mean(abs((res_sd_rel_varphy_reordered_X2B-dict_sd_rel["data_B"])/dict_sd_rel["data_B"])))
    dict_sd_rel["mae_QQ"].append(np.mean(abs((dict_sd_rel["data_QQ"]-dict_sd_rel["data_B"])/dict_sd_rel["data_B"])))
    dict_sd_rel["mae_A"].append(np.mean(abs((dict_sd_rel["data_A"]-dict_sd_rel["data_B"])/dict_sd_rel["data_B"])))
    if PR_version==False:
        title_="sd_tas"
    else:
        title_="sd_pr"

    plot_maps(epoch,QQ2B_version, True, dict_sd_rel["data_A"], dict_sd_rel["data_B"], dict_sd_rel["data_QQ"] , res_sd_rel_varphy_X2B, res_sd_rel_varphy_reordered_X2B, title_,path_plot=path_plot, subfolder=subfolder)


    #### compute_freq_dry
    if PR_version==True:
        res_freqdrydatasetA = compute_freqdry_array_new(OriginalA_preproc)
        res_freqdrydatasetB = compute_freqdry_array_new(OriginalB_preproc)
        res_freqdrydatasetQQ = compute_freqdry_array_new(OriginalQQ_preproc)
        res_freqdry_varphy_X2B = compute_freqdry_array_new(sample_varphy_X2B)
        res_freqdry_varphy_reordered_X2B = compute_freqdry_array_new(sample_varphy_reordered_X2B)
        plot_maps(epoch, QQ2B_version, PR_version, res_freqdrydatasetA, res_freqdrydatasetB, res_freqdrydatasetQQ, res_freqdry_varphy_X2B, res_freqdry_varphy_reordered_X2B,"freqdry_pr", path_plot=path_plot, subfolder=subfolder)

    #Compute correlograms
    if "data_A" not in dict_correlogram:
        res_correlo_datasetA, _, distance = compute_correlo(True,OriginalA_preproc, ind, lon, lat, point_grid)
        res_correlo_wt_remove_datasetA, _, distance = compute_correlo(False,OriginalA_preproc, ind, lon, lat, point_grid)
        dict_correlogram["data_A"]=res_correlo_datasetA
        dict_correlogram_wt_remove["data_A"] = res_correlo_wt_remove_datasetA

        res_correlo_datasetB, _, distance = compute_correlo(True,OriginalB_preproc, ind, lon, lat, point_grid)
        res_correlo_wt_remove_datasetB, _, distance = compute_correlo(False,OriginalB_preproc, ind, lon, lat, point_grid)
        dict_correlogram["data_B"]=res_correlo_datasetB
        dict_correlogram_wt_remove["data_B"] = res_correlo_wt_remove_datasetB

        res_correlo_datasetQQ, _, distance = compute_correlo(True,OriginalQQ_preproc, ind, lon, lat, point_grid)
        res_correlo_wt_remove_datasetQQ, _, distance = compute_correlo(False,OriginalQQ_preproc, ind, lon, lat, point_grid)
        dict_correlogram["data_QQ"]=res_correlo_datasetQQ
        dict_correlogram_wt_remove["data_QQ"] = res_correlo_wt_remove_datasetQQ

    res_correlo_varphy_X2B, _, distance = compute_correlo(True,sample_varphy_X2B, ind, lon, lat, point_grid)
    res_correlo_wt_remove_varphy_X2B, _, distance = compute_correlo(False,sample_varphy_X2B, ind, lon, lat, point_grid)

    res_correlo_varphy_reordered_X2B, _, distance = compute_correlo(True,sample_varphy_reordered_X2B, ind, lon, lat, point_grid)
    res_correlo_wt_remove_varphy_reordered_X2B, _, distance = compute_correlo(False,sample_varphy_reordered_X2B, ind, lon, lat, point_grid)

    dict_correlogram["mae_A"].append(np.mean(abs(dict_correlogram["data_A"] -dict_correlogram["data_B"])))
    dict_correlogram["mae_QQ"].append(np.mean(abs(dict_correlogram["data_QQ"] -dict_correlogram["data_B"])))
    dict_correlogram["mae_X2B"].append(np.mean(abs(res_correlo_varphy_X2B-dict_correlogram["data_B"])))
    dict_correlogram["mae_X2B_reordered"].append(np.mean(abs(res_correlo_varphy_reordered_X2B-dict_correlogram["data_B"])))

    dict_correlogram_wt_remove["mae_A"].append(np.mean(abs(dict_correlogram_wt_remove["data_A"] -dict_correlogram_wt_remove["data_B"])))
    dict_correlogram_wt_remove["mae_QQ"].append(np.mean(abs(dict_correlogram_wt_remove["data_QQ"] -dict_correlogram_wt_remove["data_B"])))
    dict_correlogram_wt_remove["mae_X2B"].append(np.mean(abs(res_correlo_wt_remove_varphy_X2B-dict_correlogram_wt_remove["data_B"])))
    dict_correlogram_wt_remove["mae_X2B_reordered"].append(np.mean(abs(res_correlo_wt_remove_varphy_reordered_X2B-dict_correlogram_wt_remove["data_B"])))

    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    pyplot.figure(figsize=(9,10))
    title_crit="correlograms"
    pyplot.subplot(2, 1, 1)
    pyplot.plot(distance,dict_correlogram["data_A"],color="red")
    pyplot.plot(distance,dict_correlogram["data_B"],color="black")
    pyplot.plot(distance,dict_correlogram["data_QQ"], color= "orange")
    pyplot.plot(distance,res_correlo_varphy_X2B,color="blue")
    pyplot.plot(distance,res_correlo_varphy_reordered_X2B,color="green")
    pyplot.legend(['A', 'B', 'QQ', X_name + '2B', X_name + '2B_reord.'], loc='upper right')
    pyplot.title('MAE A: ' + str(round(dict_correlogram["mae_A"][-1],3)) + ', QQ: ' + str(round(dict_correlogram["mae_QQ"][-1],3))+ ', ' + X_name + '2B: ' +str(round(dict_correlogram["mae_X2B"][-1],3)) + ', ' + X_name + '2B_reord.: ' + str(round(dict_correlogram["mae_X2B_reordered"][-1],3))  ,fontsize=10, y=1)
    pyplot.ylim((-1,1.05))
    pyplot.ylabel(name_var + " Spearman spatial Corr")
    pyplot.xlabel("Distance (km)")

    pyplot.subplot(2, 1, 2)
    pyplot.plot(distance,dict_correlogram_wt_remove["data_A"],color="red")
    pyplot.plot(distance,dict_correlogram_wt_remove["data_B"],color="black")
    pyplot.plot(distance,dict_correlogram_wt_remove["data_QQ"],color="orange")
    pyplot.plot(distance,res_correlo_wt_remove_varphy_X2B,color="blue")
    pyplot.plot(distance,res_correlo_wt_remove_varphy_reordered_X2B,color="green")
    pyplot.legend(['A', 'B', 'QQ',  X_name + '2B', X_name + '2B_reord.'], loc='lower right')
    pyplot.title('MAE A: ' + str(round(dict_correlogram_wt_remove["mae_A"][-1],3)) + ', QQ: ' + str(round(dict_correlogram_wt_remove["mae_QQ"][-1],3)) +', ' + X_name + '2B: ' +str(round(dict_correlogram_wt_remove["mae_X2B"][-1],3)) + ', ' + X_name + '2B_reord.: ' + str(round(dict_correlogram_wt_remove["mae_X2B_reordered"][-1],3)) ,fontsize=10, y=1)
    pyplot.ylim((0.5,1.05))
    pyplot.ylabel(name_var + " Spearman spatial Corr")
    pyplot.xlabel("Distance (km)")

    pyplot.savefig(path_plot + '/' + subfolder + '/plot_'+ title_crit + '_%03d.png' % (epoch+1))
    pyplot.close()

    #### Score_discB ####
    def compute_score_discB(dict_score, disc, data):
        ### proba
        res_proba = disc.predict(data)
        if "mean" not in dict_score:
            dict_score["mean"]=[]
            dict_score["quant0.025"]=[]
            dict_score["quant0.975"]=[]
        dict_score["mean"].append(res_proba.mean())
        dict_score["quant0.025"].append(np.quantile(res_proba, 0.025))
        dict_score["quant0.975"].append(np.quantile(res_proba, 0.975))

        ### accuracy
        n_samples = data.shape[0]
        y = ones((n_samples, 1))
        _, res_acc = disc.evaluate(data, y, verbose=0)
        if "accuracy" not in dict_score:
            dict_score["accuracy"]=[]
        dict_score["accuracy"].append(res_acc)

        return dict_score

    #keys_score_discB = ["score_discB_A","score_discB_X2B", "score_discB_QQ", "score_discB_X2B_reordered"]
    dict_score_discB["score_discB_A"] = compute_score_discB(dict_score_discB["score_discB_A"], discB, datasetA)
    dict_score_discB["score_discB_B"] = compute_score_discB(dict_score_discB["score_discB_B"], discB, datasetB)
    dict_score_discB["score_discB_QQ"] = compute_score_discB(dict_score_discB["score_discB_QQ"], discB, datasetQQ)
    dict_score_discB["score_discB_X2B"] = compute_score_discB(dict_score_discB["score_discB_X2B"], discB, sample_X2B)
    #### Compute sample_reordered_X2B
    def normalize_minmax(data, minX, maxX):
        res = np.copy(data)
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                res[:,k,l,:] = (data[:,k,l,:]- minX[n])/(maxX[n] - minX[n])
        return res

    sample_reordered_X2B = normalize_minmax(sample_varphy_reordered_X2B, minX, maxX)
    dict_score_discB["score_discB_X2B_reordered"] = compute_score_discB(dict_score_discB["score_discB_X2B_reordered"], discB, sample_reordered_X2B)

    plot_dict_score_discB(QQ2B_version, dict_score_discB, path_plot, subfolder)

    return dict_mean, dict_sd_rel, dict_correlogram, dict_correlogram_wt_remove, dict_norm_rmse, dict_varphy_rmse, dict_realrank_localwd, dict_varphy_localenergy, dict_realrank_localenergy, dict_score_discB


def plot_dict_score_discB(QQ2B_version, dict_score, path_plot, subfolder):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    print("plot en cours")
    np.save(path_plot+'/models/score_discB_dict_' + subfolder + '.npy',dict_score)
    pyplot.figure(figsize=(11,5))
    pyplot.subplot(1,1,1)
    pyplot.hlines([0.5],xmin=0, xmax=len(dict_score["score_discB_A"]["mean"]), color= "black", linestyles = "dotted")

    pyplot.plot(dict_score["score_discB_A"]["mean"],label='A', color= "red")
   # pyplot.plot(dict_score["score_discB_A"]["quant0.025"],label='', color= "darksalmon")
   # pyplot.plot(dict_score["score_discB_A"]["quant0.975"],label='', color= "darksalmon")

    pyplot.plot(dict_score["score_discB_B"]["mean"],label='B', color= "black")
   # pyplot.plot(dict_score["score_discB_B"]["quant0.025"],label='', color= "grey")
   # pyplot.plot(dict_score["score_discB_B"]["quant0.975"],label='', color= "grey")

    pyplot.plot(dict_score["score_discB_QQ"]["mean"],label='QQ', color= "orange")
    #pyplot.plot(dict_score["score_discB_QQ"]["quant0.025"],label='', color= "navajowhite")
    #pyplot.plot(dict_score["score_discB_QQ"]["quant0.975"],label='', color= "navajowhite")

    pyplot.plot(dict_score["score_discB_X2B"]["mean"],label= X_name + '2B', color= "blue")
    #pyplot.plot(dict_score["score_discB_X2B"]["quant0.025"],label='', color= "lightsteelblue")
    #pyplot.plot(dict_score["score_discB_X2B"]["quant0.975"],label='', color= "lightsteelblue")

    pyplot.plot(dict_score["score_discB_X2B_reordered"]["mean"],label=X_name + '2B_reord.', color= "green")
    #pyplot.plot(dict_score["score_discB_X2B_reordered"]["quant0.025"],label='', color= "lightgreen")
    #pyplot.plot(dict_score["score_discB_X2B_reordered"]["quant0.975"],label='', color= "lightgreen")

    pyplot.legend()
    pyplot.ylim((0,1))

    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_score_discB.png')
    pyplot.close()

    pyplot.figure(figsize=(11,5))
    pyplot.subplot(1,1,1)
    pyplot.hlines([0.5],xmin=0, xmax=len(dict_score["score_discB_A"]["mean"]), color= "black", linestyles = "dotted")

    pyplot.plot(dict_score["score_discB_A"]["accuracy"],label='A', color= "red")
    pyplot.plot(dict_score["score_discB_B"]["accuracy"],label='B', color= "black")
    pyplot.plot(dict_score["score_discB_QQ"]["accuracy"],label='QQ', color= "orange")
    pyplot.plot(dict_score["score_discB_X2B"]["accuracy"],label= X_name + '2B', color= "blue")
    pyplot.plot(dict_score["score_discB_X2B_reordered"]["accuracy"],label=X_name + '2B_reord.', color= "green")

    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_accuracy_discB.png')
    pyplot.close()







def plot_dict_localenergy(QQ2B_version,dict_varphy_localenergy, dict_realrank_localenergy, path_plot, subfolder):
    if QQ2B_version==True:
        X_name="QQ"
    else:
        X_name="A"
    np.save(path_plot+'/models/energy_dict_varphy_localenergy' + subfolder + '.npy',dict_varphy_localenergy)
    np.save(path_plot+'/models/energy_dict_realrank_localenergy' + subfolder + '.npy',dict_realrank_localenergy)
    pyplot.figure(figsize=(11,11))
    pyplot.subplot(2,1,1)
    pyplot.hlines(dict_varphy_localenergy["energy_A"],xmin=0, xmax=len(dict_varphy_localenergy["energy_X2B"]),label='varphy_A', color= "red")
    pyplot.hlines(dict_varphy_localenergy["energy_QQ"],xmin=0, xmax=len(dict_varphy_localenergy["energy_X2B"]), label='varphy_QQ',color='orange')
    pyplot.plot(dict_varphy_localenergy["energy_X2B"], label='varphy_' + X_name + '2B', color= "blue")
    pyplot.plot(dict_varphy_localenergy["energy_X2B_reordered"], label='varphy_' + X_name + '2B_reord.', color= "green")
    val_X2B, idx_X2B = min((val, idx) for (idx, val) in enumerate(dict_varphy_localenergy["energy_X2B"]))
    val_X2B_reordered, idx_X2B_reordered = min((val, idx) for (idx, val) in enumerate(dict_varphy_localenergy["energy_X2B_reordered"]))
    pyplot.title("A: " + str(round(dict_varphy_localenergy["energy_A"][0],2)) + ", QQ: " + str(round(dict_varphy_localenergy["energy_QQ"][0],2)) + ", best " + X_name + "2B: " + str(round(val_X2B,2)) + " at epo. "+  str(idx_X2B*10+1) + ", best " + X_name + "2B_reord.: " +  str(round(val_X2B_reordered,2)) + " at epo. "+  str(idx_X2B_reordered*10+1), fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,dict_varphy_localenergy["energy_QQ"][0]*2))

    pyplot.subplot(2,1,2)
    pyplot.hlines(dict_realrank_localenergy["energy_A"],xmin=0, xmax=len(dict_realrank_localenergy["energy_X2B"]),label='realrank_A', color= "red")
    pyplot.hlines(dict_realrank_localenergy["energy_QQ"],xmin=0, xmax=len(dict_realrank_localenergy["energy_X2B"]), label='realrank_QQ',color='orange')
    pyplot.plot(dict_realrank_localenergy["energy_X2B"], label='realrank_' + X_name + '2B', color= "blue")
    pyplot.plot(dict_realrank_localenergy["energy_X2B_reordered"], label='realrank_' + X_name + '2B_reord.', color= "green")
    val_X2B, idx_X2B = min((val, idx) for (idx, val) in enumerate(dict_realrank_localenergy["energy_X2B"]))
    val_X2B_reordered, idx_X2B_reordered = min((val, idx) for (idx, val) in enumerate(dict_realrank_localenergy["energy_X2B_reordered"]))
    pyplot.title("A: " + str(round(dict_realrank_localenergy["energy_A"][0],2)) + ", QQ: " + str(round(dict_realrank_localenergy["energy_QQ"][0],2)) + ", best " + X_name + "2B: " +  str(round(val_X2B,2)) + " at epo. "  +  str(idx_X2B*10+1) +  ", best " + X_name + "2B_reord.: " +  str(round(val_X2B_reordered,2)) + " at epo. "  +  str(idx_X2B_reordered*10+1) , fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,dict_realrank_localenergy["energy_QQ"][0]*2))
  #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_localenergy.png')
    pyplot.close()



def train_combined_new(rank_version,PR_version, QQ2B_version, CV_version, is_DS, computation_localWD, computation_localenergy, genX2B, genB2X, discX, discB, comb_model, CalibA, CalibB, CalibQQ, ProjA, ProjB, ProjQQ, OriginalCalibA, OriginalCalibB, OriginalCalibQQ, OriginalProjA, OriginalProjB, OriginalProjQQ, ind, lon, lat,point_grid,path_to_save ,minX=None,maxX=None,minB=None,maxB=None,n_epochs=100, n_batch=32):

    #### Define train and validation set
    trainsetB = np.copy(CalibB)

    if QQ2B_version==True:
        trainsetX = np.copy(CalibQQ)
    else:
        trainsetX = np.copy(CalibA)

    bat_per_epo = int(trainsetX.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # prepare lists for storing stats each iteration
    discX_hist, discB_hist, validX_hist, validB_hist, recX_hist, recB_hist,identX_hist, identB_hist, weighted_hist = list(), list(), list(), list(), list(), list(), list(), list(), list()
    discX_acc_hist, discB_acc_hist = list(), list()

    ##########################################################################
    #### Calib dict ####
    #### Init dict for mae
    keys_mae = ["mae_A", "mae_QQ", "mae_X2B", "mae_X2B_reordered"]
    dict_mean = {key: [] for key in keys_mae}
    dict_sd_rel = {key: [] for key in keys_mae}
    dict_correlogram = {key: [] for key in keys_mae}
    dict_correlogram_wt_remove = {key: [] for key in keys_mae}

    #### Init dict for rmse 
    keys_rmse = ["rmse_A","rmse_B2X2B", "rmse_X2B_B", "rmse_X2B", "rmse_X2B2X", "rmse_B2X_X","rmse_B2X", "rmse_QQ"]
    dict_norm_rmse = {key: [] for key in keys_rmse}
    dict_varphy_rmse = {key: [] for key in keys_rmse}

    #### Init dict for localWD
    keys_wd = ["wd_A","wd_X2B","bin_size", "wd_QQ"]
    dict_realrank_localwd = {key: [] for key in keys_wd}
    if computation_localWD==True:
        dict_realrank_localwd["bin_size"]=[0.05]*9

    #### Init dict for localenergy
    keys_energy = ["energy_A","energy_X2B", "energy_QQ", "energy_X2B_reordered"]
    dict_varphy_localenergy = {key: [] for key in keys_energy}
    dict_realrank_localenergy = {key: [] for key in keys_energy}

    #### Init dict for score_discB
    keys_score_discB = ["score_discB_A", "score_discB_B","score_discB_X2B", "score_discB_QQ", "score_discB_X2B_reordered"]
    dict_score_discB = {key: {} for key in keys_score_discB}


    #########################################################################
    #### Proj dict ####
    #### Init dict for mae
    proj_dict_mean = {key: [] for key in keys_mae}
    proj_dict_sd_rel = {key: [] for key in keys_mae}
    proj_dict_correlogram = {key: [] for key in keys_mae}
    proj_dict_correlogram_wt_remove = {key: [] for key in keys_mae}

    #### Init dict for rmse
    proj_dict_norm_rmse = {key: [] for key in keys_rmse}
    proj_dict_varphy_rmse = {key: [] for key in keys_rmse}

    #### Init dict for localWD
    proj_dict_realrank_localwd = {key: [] for key in keys_wd}
    if computation_localWD==True:
        proj_dict_realrank_localwd["bin_size"]=[0.05]*9

    #### Init dict for localenergy
    proj_dict_varphy_localenergy = {key: [] for key in keys_energy}
    proj_dict_realrank_localenergy = {key: [] for key in keys_energy}

    #### Init dict for score_discB
    proj_dict_score_discB = {key: {} for key in keys_score_discB}


    ########################################################################
    #### Training step ####
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, labX_real = generate_real_samples(trainsetX, half_batch)
            B_real, labB_real = generate_real_samples(trainsetB, half_batch)
            # generate 'fake' examples
            X_fake, labX_fake = generate_fake_samples(genB2X, B_real)
            B_fake, labB_fake = generate_fake_samples(genX2B, X_real)
            # create training set for the discriminator
            X_disc, labX_disc = vstack((X_real, X_fake)), vstack((labX_real, labX_fake))
            B_disc, labB_disc = vstack((B_real, B_fake)), vstack((labB_real, labB_fake))
            # update discriminator model weights
            discX_loss, discX_acc = discX.train_on_batch(X_disc, labX_disc)
            discB_loss, discB_acc = discB.train_on_batch(B_disc, labB_disc)
            # train generator
            all_loss = comb_model.train_on_batch([X_real, B_real], [labB_real, labX_real, X_real, B_real, X_real, B_real])
            #record history
            n_modulo=2
            if (j+1) % n_modulo == 1:
                discX_hist.append(discX_loss)
                discB_hist.append(discB_loss)
                discX_acc_hist.append(discX_acc)
                discB_acc_hist.append(discB_acc)
                validX_hist.append(all_loss[1])
                validB_hist.append(all_loss[2])
                recX_hist.append(all_loss[3])
                recB_hist.append(all_loss[4])
                identX_hist.append(all_loss[5])
                identB_hist.append(all_loss[6])
                weighted_hist.append(all_loss[0])
        print('>%d, %d/%d, discX=%.3f, discB=%.3f, validX=%.3f, validB=%.3f, recX=%.3f, recB =%.3f, identX=%.3f, identB=%.3f' % (i+1, j+1, bat_per_epo, discX_loss, discB_loss, all_loss[1], all_loss[2], np.array(all_loss[3])*10, np.array(all_loss[4])*10, all_loss[5], all_loss[6]))
        print('Weighted sum: ' + str(all_loss[0]))
        nb_data_per_epoch=int(bat_per_epo/n_modulo)
        plot_history_loss(i,QQ2B_version, nb_data_per_epoch,discX_hist, discB_hist, validX_hist, validB_hist, recX_hist, recB_hist, identX_hist, identB_hist, weighted_hist, discX_acc_hist, discB_acc_hist, path_plot=path_to_save, subfolder= "calib")
        if (i+1) % 10 == 1:
            recap_accuracy_and_save_gendisc(i,genX2B, genB2X, discX, discB, trainsetX, trainsetB, path_plot=path_to_save, n_samples=100)

            ######################################
            #### Evaluation of training data #####
            print("eval calib")
            dict_mean, dict_sd_rel, dict_correlogram, dict_correlogram_wt_remove, dict_norm_rmse, dict_varphy_rmse, dict_realrank_localwd, dict_varphy_localenergy, dict_realrank_localenergy, dict_score_discB = plot_some_raw_maps_and_compute_rmse(i,is_DS,rank_version, PR_version, QQ2B_version,  computation_localWD, computation_localenergy, genX2B, genB2X,discB, CalibA, CalibB, CalibQQ, OriginalCalibA, OriginalCalibB, OriginalCalibQQ, minX, maxX, minB, maxB,dict_mean, dict_sd_rel, dict_correlogram, dict_correlogram_wt_remove, dict_norm_rmse, dict_varphy_rmse, dict_realrank_localwd, dict_varphy_localenergy, dict_realrank_localenergy, dict_score_discB, ind, lon, lat, point_grid, path_to_save, subfolder= "calib")

            plot_dict_rmse(QQ2B_version, is_DS,dict_norm_rmse,dict_varphy_rmse,path_to_save, subfolder="calib")
            if computation_localWD==True:
                plot_dict_localwdenergy(dict_realrank_localwd,dict_realrank_localenergy,path_to_save, subfolder= "calib")
            if computation_localenergy==True:
                plot_dict_localenergy(QQ2B_version, dict_varphy_localenergy,dict_realrank_localenergy,path_to_save, subfolder = "calib")
            #plot history of criteria
            plot_history_criteria(QQ2B_version,dict_mean, "mae_mean",-0.01,1.5,path_to_save, subfolder= "calib")
            plot_history_criteria(QQ2B_version, dict_sd_rel, "mae_sd_rel",-0.01,0.2,path_to_save, subfolder= "calib")
            plot_history_criteria(QQ2B_version, dict_correlogram, "mae_correlogram",-0.01,dict_correlogram["mae_QQ"][0]*2,path_to_save, subfolder="calib")
            plot_history_criteria(QQ2B_version, dict_correlogram_wt_remove, "mae_correlogram_wt_remove",-0.01,dict_correlogram_wt_remove["mae_QQ"][0]*2,path_to_save, subfolder= "calib")

            #########################################
            ##### Evaluation of validation data #####
            #### To do
            if CV_version is not "PC0":
                print("eval proj")
                proj_dict_mean, proj_dict_sd_rel, proj_dict_correlogram, proj_dict_correlogram_wt_remove, proj_dict_norm_rmse, proj_dict_varphy_rmse, proj_dict_realrank_localwd, proj_dict_varphy_localenergy, proj_dict_realrank_localenergy, proj_dict_score_discB = plot_some_raw_maps_and_compute_rmse(i,is_DS,rank_version, PR_version, QQ2B_version,  computation_localWD, computation_localenergy, genX2B, genB2X, discB, ProjA, ProjB, ProjQQ, OriginalProjA, OriginalProjB, OriginalProjQQ, minX, maxX, minB, maxB, proj_dict_mean, proj_dict_sd_rel, proj_dict_correlogram, proj_dict_correlogram_wt_remove, proj_dict_norm_rmse, proj_dict_varphy_rmse, proj_dict_realrank_localwd, proj_dict_varphy_localenergy, proj_dict_realrank_localenergy, proj_dict_score_discB, ind, lon, lat, point_grid, path_to_save, subfolder= "proj")

                plot_dict_rmse(QQ2B_version, is_DS, proj_dict_norm_rmse, proj_dict_varphy_rmse,path_to_save, subfolder="proj")
                if computation_localWD==True:
                    plot_dict_localwdenergy(proj_dict_realrank_localwd,proj_dict_realrank_localenergy,path_to_save, subfolder= "proj")
                if computation_localenergy==True:
                    plot_dict_localenergy(QQ2B_version, proj_dict_varphy_localenergy, proj_dict_realrank_localenergy,path_to_save, subfolder = "proj")
                #plot history of criteria
                plot_history_criteria(QQ2B_version, proj_dict_mean, "mae_mean",-0.01,1.5,path_to_save, subfolder= "proj")
                plot_history_criteria(QQ2B_version, proj_dict_sd_rel, "mae_sd_rel",-0.01,0.2,path_to_save, subfolder= "proj")
                plot_history_criteria(QQ2B_version, proj_dict_correlogram, "mae_correlogram",-0.01,proj_dict_correlogram["mae_QQ"][0]*2,path_to_save, subfolder="proj")
                plot_history_criteria(QQ2B_version, proj_dict_correlogram_wt_remove, "mae_correlogram_wt_remove",-0.01,proj_dict_correlogram_wt_remove["mae_QQ"][0]*2,path_to_save, subfolder= "proj")


        #####EARLY STOPPING
        #if i>50 and (np.mean(g_hist[-2000:])>3.5 or np.std(g_hist[-2000:]) >0.26):
        #    print('BREAK! For the 2000 last points: mean_gloss=%.3f, sd_gloss=%.3f ' % (np.mean(g_hist[-2000:] ), np.std(g_hist[-2000:])))
        #    break




#### A la retraite:
def compute_some_wasserstein(dict_wd, sample_A, sample_B, sample_A2B, sample_QQ,ind,point_grid, bin_size= None):
    reversed_datasetA=np.transpose(sample_A[:,:,:,0],(2,1,0))
    reversed_datasetB=np.transpose(sample_B[:,:,:,0],(2,1,0))
    reversed_datasetA2B = np.transpose(sample_A2B[:,:,:,0],(2,1,0))
    reversed_datasetQQ = np.transpose(sample_QQ[:,:,:,0],(2,1,0))

    tmp_A=np.transpose(transform_array_in_matrix(reversed_datasetA, ind, point_grid))
    tmp_B=np.transpose(transform_array_in_matrix(reversed_datasetB, ind, point_grid))
    tmp_A2B=np.transpose(transform_array_in_matrix(reversed_datasetA2B, ind, point_grid))
    tmp_QQ=np.transpose(transform_array_in_matrix(reversed_datasetQQ, ind, point_grid))

    mu_tmp_A = SparseHist(tmp_A,bin_size)
    mu_tmp_B = SparseHist(tmp_B, bin_size)
    mu_tmp_A2B = SparseHist(tmp_A2B,bin_size)
    mu_tmp_QQ = SparseHist(tmp_QQ, bin_size)
    dict_wd["wd_A2B"].append(wasserstein(mu_tmp_B, mu_tmp_A2B))
    if len(dict_wd["wd_A"])==0:
        dict_wd["wd_A"].append(wasserstein(mu_tmp_B, mu_tmp_A))
    if len(dict_wd["wd_QQ"])==0:
        dict_wd["wd_QQ"].append(wasserstein(mu_tmp_B, mu_tmp_QQ))
    return dict_wd

def compute_bin_width_wasserstein(dataset,ind,point_grid): #dataset of type TIME x LAT x LON
    reversed_dataset=np.transpose(dataset[:,:,:,0],(2,1,0))
    tmp_=np.transpose(transform_array_in_matrix(reversed_dataset, ind, point_grid))
    bin_width=bin_width_estimator(tmp_) #bcse bin_width_estimator needs data of type TIME x nb_var
    bin_width_mean=[bin_width.mean()]*len(bin_width)
    return bin_width_mean

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
    smallA=np.zeros((9,nb_images))
    smallB= np.zeros((9,nb_images))
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

def plot_dict_wd( dict_varphy, dict_realrank, path_plot, subfolder):
    np.save(path_plot+'/models/wd_dict_varphy.npy',dict_varphy)
    np.save(path_plot+'/models/wd_dict_realrank.npy',dict_realrank)
    pyplot.figure(figsize=(11,11))
    pyplot.subplot(2,1,1)
    pyplot.plot(dict_varphy["wd_A2B"], label='wd_varphy_A2B', color= "blue")
    pyplot.hlines(dict_varphy["wd_A"],xmin=0, xmax=len(dict_varphy["wd_A2B"]),label='wd_varphy_A', color= "red")
    pyplot.hlines(dict_varphy["wd_QQ"],xmin=0, xmax=len(dict_varphy["wd_A2B"]), label='wd_varphy_QQ',color='orange')
    val, idx = min((val, idx) for (idx, val) in enumerate(dict_varphy["wd_A2B"]))
    pyplot.title("Best A2B at epoch " +  str(idx*10+1) + " with " + str(round(val,3)) + ", bin_size: " + str(round(dict_varphy["bin_size"][1],3)) +", A:" + str(round(dict_varphy["wd_A"][0],3)) + ", QQ: " + str(round(dict_varphy["wd_QQ"][0],3)), fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,50))
    pyplot.subplot(2, 1, 2)
    pyplot.plot(dict_realrank["wd_A2B"], label='wd_realrank_A2B', color="blue")
    pyplot.hlines(dict_realrank["wd_A"],xmin=0, xmax=len(dict_realrank["wd_A2B"]), label='wd_realrank_A', color="red")
    pyplot.hlines(dict_realrank["wd_QQ"],xmin=0, xmax=len(dict_realrank["wd_A2B"]), label='wd_realrank_QQ',color='orange')
    val, idx = min((val, idx) for (idx, val) in enumerate(dict_realrank["wd_A2B"]))
    pyplot.title("Best A2B at epoch " +  str(idx*10+1) + " with " + str(round(val,3)) + ", bin_size: " + str(round(dict_realrank["bin_size"][1],3)) + ", A: "+str(round(dict_realrank["wd_A"][0],3)) +  ", QQ: " + str(round(dict_realrank["wd_QQ"][0],3)), fontsize=7)
    pyplot.legend()
    pyplot.ylim((-0.1,8))
   #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_wd.png')
    pyplot.close()

def plot_dict_localwdenergy(dict_realrank_localwd, dict_realrank_localenergy, path_plot, subfolder):
    np.save(path_plot+'/models/wd_dict_realrank_localwd.npy',dict_realrank_localwd)
    np.save(path_plot+'/models/energy_dict_realrank_localenergy.npy',dict_realrank_localenergy)
    pyplot.figure(figsize=(11,11))
    pyplot.subplot(2,1,1)
    pyplot.plot(dict_realrank_localwd["wd_A2B"], label='localwd_realrank_A2B', color= "blue")
    pyplot.hlines(dict_realrank_localwd["wd_A"],xmin=0, xmax=len(dict_realrank_localwd["wd_A2B"]),label='localwd_realrank_A', color= "red")
    pyplot.hlines(dict_realrank_localwd["wd_QQ"],xmin=0, xmax=len(dict_realrank_localwd["wd_A2B"]), label='localwd_realrank_QQ',color='orange')
    val, idx = min((val, idx) for (idx, val) in enumerate(dict_realrank_localwd["wd_A2B"]))
    pyplot.title("Best A2B at epoch " +  str(idx*10+1) + " with " + str(round(val,3)) + ", bin_size: " + str(round(dict_realrank_localwd["bin_size"][1],3)) +", A:" + str(round(dict_realrank_localwd["wd_A"][0],3)) + ", QQ: " + str(round(dict_realrank_localwd["wd_QQ"][0],3)), fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,dict_realrank_localwd["wd_QQ"][0]*2))

    pyplot.subplot(2,1,2)
    pyplot.plot(dict_realrank_localenergy["energy_A2B"], label='localenergy_realrank_A2B', color= "blue")
    pyplot.hlines(dict_realrank_localenergy["energy_A"],xmin=0, xmax=len(dict_realrank_localenergy["energy_A2B"]),label='localenergy_realrank_A', color= "red")
    pyplot.hlines(dict_realrank_localenergy["energy_QQ"],xmin=0, xmax=len(dict_realrank_localenergy["energy_A2B"]), label='localenergy_realrank_QQ',color='orange')
    val, idx = min((val, idx) for (idx, val) in enumerate(dict_realrank_localenergy["energy_A2B"]))
    pyplot.title("Best A2B at epoch " +  str(idx*10+1) + " with " + str(round(val,3)) + ", A:" + str(round(dict_realrank_localenergy["energy_A"][0],3)) + ", QQ: " + str(round(dict_realrank_localenergy["energy_QQ"][0],3)), fontsize=7)
    pyplot.legend()
    pyplot.ylim((0,dict_realrank_localenergy["energy_QQ"][0]*2))
  #save plot to file
    pyplot.savefig(path_plot + '/' + subfolder + '/plot_history_localwdenergy.png')
    pyplot.close()


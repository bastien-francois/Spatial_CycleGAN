import h5netcdf.legacyapi as netCDF4
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
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

#### Attention, nouvelle organisation
def load_RData_minmax(RData_file,variable,index_temporal):
    load_data = robjects.r.load(RData_file + '.RData')
    dataset=robjects.r[variable]
    X = np.array(dataset)
    X= np.transpose(X, (2,  1, 0))
    #Sub-selection of array
    X = X[index_temporal,:,:]
    lon =  robjects.r['LON_Paris']
    lon = np.array(lon)
    lat =  robjects.r['LAT_Paris']
    lat = np.array(lat)
    ind = robjects.r['IND_Paris']
    ind = np.array(ind)-1 ### ATTENTION Python specific
    point_grid = range(784)
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(X, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    #Save original data
    OriginalX = np.copy(X)
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
    return X, lon, lat, min_, max_, ind, point_grid, OriginalX

def load_RData_rank(RData_file,variable,index_temporal):
    load_data = robjects.r.load(RData_file + '.RData')
    dataset=robjects.r[variable]
    X = np.array(dataset)
    X= np.transpose(X, (2,  1, 0))
    X= X[index_temporal,:,:]
    lon =  robjects.r['LON_Paris']
    lon = np.array(lon)
    lat =  robjects.r['LAT_Paris']
    lat = np.array(lat)
    ind = robjects.r['IND_Paris']
    ind = np.array(ind)-1
    point_grid = range(784)
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(X, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    #Save original data
    OriginalX = np.copy(X)
    min_=None
    max_=None
    # scale with rank by grid cells
    for k in range(28):
        for l in range(28):
            X[:,k,l,0]=(rankdata(X[:,k,l,0],method="ordinal")/len(X[:,k,l,0]))
    return X, lon, lat, min_, max_, ind, point_grid, OriginalX

##A proposer a Mathieu et Soulivanh
#def load_RData_rank_with_gaussianization(RData_file,variable,index_temporal):
#    load_data = robjects.r.load(RData_file + '.RData')
#    dataset=robjects.r[variable]
#    X = np.array(dataset)
#    X= np.transpose(X, (2,  1, 0))
#    X= X[index_temporal,:,:]
#    lon =  robjects.r['LON_Paris']
#    lon = np.array(lon)
#    lat =  robjects.r['LAT_Paris']
#    lat = np.array(lat)
#    ind = robjects.r['IND_Paris']
#    ind = np.array(ind)-1
#    point_grid = range(784)
#    # expand to 3d, e.g. add channels dimension
#    X = expand_dims(X, axis=-1)
#    # convert from unsigned ints to floats
#    X = X.astype('float32')
#    #Save original data
#    OriginalX = np.copy(X)
#
#    # scale with rank by grid cells
#    for k in range(28):
#        for l in range(28):
#            X[:,k,l,0]=(rankdata(X[:,k,l,0],method="ordinal")/len(X[:,k,l,0]))
#    #####ATTENTION ESSAI
#    min_=[None]*28*28
#    max_=[None]*28*28
#    Y = np.copy(X)
#    n=-1
#    for k in range(28):
#        for l in range(28):
#            n=n+1
#            Y[:,k,l,0]=norm.ppf(X[:,k,l,0]-0.0001)
#            max_[n]=Y[:,k,l,:].max()
#            min_[n]=Y[:,k,l,:].min()
#    Z=np.copy(Y)
#    n=-1
#    for k in range(28):
#        for l in range(28):
#            n=n+1
#            Z[:,k,l,:]=(Y[:,k,l,:]-min_[n])/(max_[n]-min_[n])
#    min_=None
#    max_=None
#    return Z, lon, lat, min_, max_, ind, point_grid, OriginalX
##### END ATTENTION

#standalone discriminator model
def define_discriminator(in_shape=(28,28,1), lr_disc=0.0002): #same as Soulivanh
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=lr_disc, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(in_shape=(28,28,1)):  #same as Soulivanh (except filter size, no impact on results)
    input = Input(shape = in_shape)
    c28x28 = Conv2D(64, (3,3), padding='same')(input)
    c14x14 = Conv2D(64, (3,3), strides=(2, 2), padding='same')(c28x28)
    model = LeakyReLU(alpha=0.2)(c14x14)
    model = Dropout(0.4)(model)
    model = Conv2D(64, (3,3), strides=(2, 2), padding='same')(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.4)(model)
    # upsample to 14x14
    model = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(model) #filter changed only
    model = Add()([model, c14x14]) # SKIP Connection
    model = LeakyReLU(alpha=0.2)(model)
    # upsample to 28x28
    model = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(model) #filter changed only
    model = Add()([model, c28x28]) # SKIP Connection
    model = LeakyReLU(alpha=0.2)(model)
    model = Conv2D(1, (1,1), padding='same')(model)
    model = Add()([model, input]) # SKIP Connection
    model = LeakyReLU(alpha=0.2)(model)
    generator = Model(input, model)
    return generator

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

def plot_maps(epoch, PR_version, mat_A, mat_B, mat_A2B, title, lon, lat,path_plot):
    #mat_A, mat_B, mat_A2B results from compute_mean_sd_array to plot
    mat_A = mat_A.astype(float)
    mat_B = mat_B.astype(float)
    mat_A2B = mat_A2B.astype(float)
    #### On inverse LON_LAT pour plotter correctement
    mat_A=np.fliplr(mat_A)
    mat_B=np.fliplr(mat_B)
    mat_A2B=np.fliplr(mat_A2B)
    ### Plot
    #### Mean and sd / MAE ####
    if PR_version==False:
        examples = vstack((mat_A, mat_B, mat_A2B, mat_A-mat_B, mat_B-mat_B, mat_A2B-mat_B))
        names_=("Mod","Ref","GAN","Mod-Ref","Ref-Ref","GAN-Ref")
    else:
        examples = vstack((mat_A, mat_B, mat_A2B, (mat_A-mat_B)/mat_B, (mat_B-mat_B)/mat_B, (mat_A2B-mat_B)/mat_B))
        names_=("Mod","Ref","GAN","(Mod-Ref)/Ref","(Ref-Ref)/Ref","(GAN-Ref)/Ref")
    nchecks=3
    #extent = [lon.min(), lon.max(), lat.min(), lat.max()] #
    #map = Basemap(projection='merc', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='c')
    # find x,y of map projection grid.
    #xx, yy = np.meshgrid(lon, lat)
    #xx, yy = map(xx, yy)

    pyplot.figure(figsize=(9,9))
    for i in range(2 * nchecks):
        # define subplot
        pyplot.subplot(2, nchecks, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        if (i < (1 * nchecks)):
            pyplot.imshow(examples[i, :, :], cmap='YlOrRd')
            pyplot.title(str(names_[i]) + ' mean: ' +str(round(np.mean(examples[i, :, :]),3)) + ' / sd: ' + str(round(np.std(examples[i, :, :]),3))  ,fontsize=10, y=1)
            vmin = np.quantile(mat_B, 0.1)
            vmax = np.quantile(mat_B, 0.9)
            pyplot.clim(vmin,vmax)
        else:
            pyplot.imshow(examples[i, :, :], cmap='RdBu')
            pyplot.title(str(names_[i]) + ' mae: ' +str(round(np.mean(abs(examples[i, :, :])),3)) + ' / sd: ' + str(round(np.std(examples[i, :, :]),3))  ,fontsize=10, y=1)
            vmin = -3
            vmax = 3
            pyplot.clim(vmin,vmax)
        #map.pcolormesh(xx,yy, examples[i, :, :], vmin = vmin, vmax = vmax, cmap=cmap)
        #map.drawcoastlines(linewidth = 0.2)
        if (i + 1) % nchecks == 0:
                pyplot.colorbar()
        # # save plot to file
    filename = path_plot + '/diagnostic/plot_criteria_'+ title + '_%03d.png' % (epoch+1)
    pyplot.savefig(filename, dpi=150)
    pyplot.close()




def compute_and_plot_criteria_for_early_stopping(rank_version,PR_version,epoch, datasetA, datasetB, OriginalA, OriginalB, genA2B,  XminA_, XmaxA_, XminB_, XmaxB_, ind, lon, lat,point_grid,path_plot):
    print("begin criteria ")
    mae_mean, mae_std_rel, mae_correlogram= None, None, None
    if PR_version==False:
        name_var="T2"
    else:
        name_var="PR"

    # Generate bias correction
    fakesetB = genA2B.predict(datasetA)

    #plot some raw maps
    #Init matrices for evalutation of criteria
    datasetA_eval = np.copy(datasetA)
    datasetB_eval = np.copy(datasetB)
    fakesetB_eval = np.copy(fakesetB)
    if rank_version==False:
        #Rescale climatic variables wrt Xmin and Xmax
        n=-1
        for k in range(28):
            for l in range(28):
                n=n+1
                datasetA_eval[:,k,l,:] = datasetA_eval[:,k,l,:]*(XmaxA_[n] - XminA_[n])+ XminA_[n]
                datasetB_eval[:,k,l,:] = datasetB_eval[:,k,l,:]*(XmaxB_[n] - XminB_[n])+ XminB_[n]
                fakesetB_eval[:,k,l,:] = fakesetB_eval[:,k,l,:]*(XmaxB_[n] - XminB_[n])+ XminB_[n]
    else:
        #Reorder rank data with OriginalData
        datasetA_eval = np.copy(OriginalA)
        datasetB_eval = np.copy(OriginalB)
        for k in range(28):
            for l in range(28):
                sorted_OriginalB=np.sort(datasetB_eval[:,k,l,0])
                idx=rankdata(fakesetB_eval[:,k,l,0])
                idx=idx.astype(int)-1
                fakesetB_eval[:,k,l,0] = sorted_OriginalB[idx]

    ##!!!! Preprocess for PR !!! 
    if PR_version==True:
        datasetA_eval[datasetA_eval < 1] = 0
        datasetB_eval[datasetB_eval < 1] = 0
        fakesetB_eval[fakesetB_eval < 1] = 0

    #Compute Mean Sd criteria
    res_mean_datasetA, res_sd_datasetA = compute_mean_sd_array_new(datasetA_eval)
    res_mean_datasetB, res_sd_datasetB = compute_mean_sd_array_new(datasetB_eval)
    res_mean_fakesetB, res_sd_fakesetB = compute_mean_sd_array_new(fakesetB_eval)
    if PR_version==False:
        mae_mean = np.mean(abs(res_mean_fakesetB-res_mean_datasetB))
        title_="mean_tas"
    else:
        mae_mean = np.mean(abs((res_mean_fakesetB-res_mean_datasetB)/res_mean_datasetB))
        title_="mean_pr"

    mae_std_rel = np.mean(abs((res_sd_fakesetB-res_sd_datasetB)/res_sd_datasetB))
    #Indications for early stopping (should be near 0)
    print('> MAE_mean: ', round(mae_mean,3))
    print('> MAE_sd_rel: mean ', round(mae_std_rel,3))
    #Attention Flemme LON/LAT
    plot_maps(epoch,PR_version, res_mean_datasetA, res_mean_datasetB, res_mean_fakesetB, title_,lon=np.array(range(28)), lat=np.array(range(28)),path_plot=path_plot)
    #Compute correlograms
    #Need to reverse the array
    reversed_datasetA=np.transpose(datasetA_eval[:,:,:,0],(2,1,0))
    res_correlo_datasetA, _, distance = compute_correlo(reversed_datasetA, ind, lon, lat, point_grid)

    reversed_datasetB=np.transpose(datasetB_eval[:,:,:,0],(2,1,0))
    res_correlo_datasetB, _, distance = compute_correlo(reversed_datasetB, ind, lon, lat, point_grid)

    reversed_fakesetB=np.transpose(fakesetB_eval[:,:,:,0],(2,1,0))
    res_correlo_fakesetB, _, distance = compute_correlo(reversed_fakesetB, ind, lon, lat, point_grid)
    #
    ##to del
    #print(res_correlo_datasetA)
    #print(res_correlo_datasetB)
    ##end to del
    mae_correlogram = np.mean(abs(res_correlo_fakesetB-res_correlo_datasetB))
    print('> MAE_correlo ', round(mae_correlogram,3))
    #plot correlograms
    title_crit="correlograms"
    pyplot.subplot(1, 1, 1)
    pyplot.plot(distance,res_correlo_datasetA,color="red")
    pyplot.plot(distance,res_correlo_datasetB,color="black")
    pyplot.plot(distance,res_correlo_fakesetB,color="green")
    pyplot.legend(['Mod', 'Ref', 'CycleGAN'], loc='upper right')
    pyplot.title('CycleGAN mae_correlogram: ' +str(round(mae_correlogram,3))  ,fontsize=10, y=1)
    pyplot.ylim((-1,1.05))
    pyplot.ylabel(name_var + " Spearman spatial Corr")
    pyplot.xlabel("Distance (km)")
    pyplot.savefig(path_plot + '/diagnostic/plot_'+ title_crit + '_%03d.png' % (epoch+1))
    pyplot.close()

    return mae_mean, mae_std_rel, mae_correlogram



def plot_history_criteria(crit,title_crit,ylim1,ylim2,path_plot):
    #plot criteria mean
    pyplot.subplot(1, 1, 1)
    pyplot.plot(crit, label=title_crit)
    x = np.linspace(0,100,100)
    pyplot.legend()
    pyplot.ylim((ylim1,ylim2))
    pyplot.savefig(path_plot + '/diagnostic/plot_'+ title_crit + '.png')
    pyplot.close()





########## For evaluation
def transform_array_in_matrix(data_array,ind,point_grid):
    res_transform=np.empty([ind.shape[0], data_array.shape[2]])
    k=(-1)
    for point in point_grid:
        k=k+1
        i=ind[point,0]
        j=ind[point,1]
        res_transform[k,:]=data_array[i,j,:]
    return res_transform


def compute_correlo(data,ind,lon,lat,point_grid):
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
    Mat_daily_mean_removed=np.transpose(MatData)-means_expanded
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


####################################################################################################################################################
###################################################################################################################################################
##### FOR SimpleGAN ####

# define the combined generator and discriminator model, for updating the generator
def define_gan(genA2B, discB, lr_gen=0.0001):
    # make weights in the discriminator not trainable
    discB.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(genA2B)
    # add the discriminator
    model.add(discB)
    # compile model
    opt = Adam(lr=lr_gen, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def train_gan_new(rank_version, PR_version, genA2B, discB, gan, datasetA, datasetB, OriginalA, OriginalB, ind, lon, lat, point_grid, path_plot, XminA_, XminB_, XmaxA_, XmaxB_, n_epochs=100, n_batch=32):
    bat_per_epo = int(datasetA.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    discB_loss_real_hist, discB_loss_fake_hist, gan_loss_hist,  discB_acc_real_hist, discB_acc_fake_hist = list(), list(), list(), list(), list()
    MAE_mean_, MAE_sd_rel_, MAE_correlogram_= list(), list(), list()
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            xA_real, yA_real = generate_real_samples(datasetA, half_batch)
            xB_real, yB_real = generate_real_samples(datasetB, half_batch)

            # update discriminator model weights on realB
            discB_loss_on_realB, discB_acc_on_realB = discB.train_on_batch(xB_real, yB_real)
            # generate 'fake' examples
            xB_fake, yB_fake = generate_fake_samples(genA2B, xA_real)
            # update discriminator model weights on fakeB
            discB_loss_on_fakeB, discB_acc_on_fakeB = discB.train_on_batch(xB_fake, yB_fake)

            # train generator
            x_gan, y_gan = generate_real_samples(datasetA, n_batch)
            g_loss = gan.train_on_batch(x_gan, y_gan)
            # evaluate the model performance, sometimes
            if (j+1) % 2 == 1:
                discB_loss_real_hist.append(discB_loss_on_realB)
                discB_loss_fake_hist.append(discB_loss_on_fakeB)
                gan_loss_hist.append(g_loss)
                discB_acc_real_hist.append(discB_acc_on_realB)
                discB_acc_fake_hist.append(discB_acc_on_fakeB)
        print('>%d, %d/%d, discB_loss_real=%.3f, discB_loss_fake=%.3f, gan_loss=%.3f' % (i+1, j+1, bat_per_epo, discB_loss_on_realB, discB_loss_on_fakeB, g_loss))
        plot_history_gan_loss(discB_loss_real_hist, discB_loss_fake_hist, gan_loss_hist, discB_acc_real_hist, discB_acc_fake_hist, path_plot)
        if (i+1) % 10 == 1:
          summarize_accu_gan(i, genA2B, datasetA, datasetB, discB, path_plot)
          res_mae_mean, res_mae_sd_rel, res_mae_correlogram = compute_and_plot_criteria_for_early_stopping(rank_version,PR_version,i, datasetA, datasetB, OriginalA, OriginalB, genA2B, XminA_, XmaxA_, XminB_, XmaxB_,ind, lon, lat,point_grid,path_plot)
          MAE_mean_.append(res_mae_mean)
          MAE_sd_rel_.append(res_mae_sd_rel)
          MAE_correlogram_.append(res_mae_correlogram)
          #plot some raw maps
          plot_some_raw_maps_gan(genA2B, datasetA, datasetB, i, path_plot)
          #plot history of criteria
          plot_history_criteria(MAE_mean_, "mae_mean",-0.01,0.5,path_plot)
          plot_history_criteria(MAE_sd_rel_, "mae_sd_rel",-0.01,0.2,path_plot)
          plot_history_criteria(MAE_correlogram_, "mae_correlogram",-0.01,0.3,path_plot)


def summarize_accu_gan(epoch, genA2B, datasetA, datasetB, discB, path_plot, n_samples=100):
    # prepare real samples
    xB_real, yB_real = generate_real_samples(datasetB, n_samples)
    # evaluate discriminator on real examples
    _, discB_acc_real = discB.evaluate(xB_real, yB_real, verbose=0)
    # prepare fake examples
    xA_real, yA_real = generate_real_samples(datasetA, n_samples)
    xB_fake, yB_fake = generate_fake_samples(genA2B, xA_real)
    # evaluate discriminator on fake examples
    _, discB_acc_fake = discB.evaluate(xB_fake, yB_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy discB real: %.0f%%, fake: %.0f%%' % (discB_acc_real*100, discB_acc_fake*100))
    # save plot
    # save the generator model tile file
#   filename = savepath + 'generator_model.h5' 
#   g_model.save(filename)
#   filename = savepath + 'discriminator_model.h5'
#   d_model.save(filename)


# create and save a plot of generated images (reversed grayscale)
def plot_some_raw_maps_gan(genA2B, datasetA, datasetB, epoch, path_plot, n=7):
    # plot images
    ix = np.random.randint(0, datasetA.shape[0], n)
    # reitrieve selected images
    some_xA_real = datasetA[ix]
    some_xB_real = datasetB[ix]

    some_xB_fake, some_yB_fake = generate_fake_samples(genA2B, some_xA_real)
    examples = vstack((some_xA_real, some_xB_real, some_xB_fake))
    pyplot.figure(figsize=(20,20))
    #ix_clim=-1
    for i in range(3 * n):
      #ix_clim=ix_clim+1
      #if(ix_clim>=n):
      #  ix_clim=0
      # define subplot
      pyplot.subplot(3, n, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(examples[i, :, :, 0], cmap='YlOrRd')
      vmin = np.quantile(examples[i, :, :, 0], 0.1)
      vmax = np.quantile(examples[i, :, :, 0], 0.9)
      pyplot.clim(vmin,vmax)
      #if i<n:
      pyplot.colorbar(fraction=0.046, pad=0.04).ax.tick_params(labelsize=14)
    pyplot.subplots_adjust(bottom=0.1, right=0.975, top=0.95, left=0.01)
    # save plot to file
    filename = path_plot + '/diagnostic/plot_raw_maps_gan_e%03d.png' % (epoch+1)
    pyplot.savefig(filename, dpi=150)
    pyplot.close()


def plot_history_gan_loss(discB_loss_real, discB_loss_fake, gan_loss, discB_acc_real, discB_acc_fake, path_plot):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(discB_loss_real, label='loss-discB_real')
    pyplot.plot(discB_loss_fake, label='loss-discB_fake')
    pyplot.plot(gan_loss, label='loss-gan', color='green')
    pyplot.legend()
    pyplot.ylim((0,2))
    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.plot(discB_acc_real, label='discB-acc_real')
    pyplot.plot(discB_acc_fake, label='discB-acc_fake')
    pyplot.legend()
    pyplot.ylim((-0.1,1.1))
    # save plot to file
    pyplot.savefig(path_plot + '/diagnostic/plot_history_gan_loss.png')
    pyplot.close()

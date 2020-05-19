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

def define_combined(genA2B, genB2A, discA, discB,lr_gen=0.002, in_shape=(28,28,1)): #same as Soulivanh
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
    comb_model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae','mae'],loss_weights=[  1, 1, 10, 10, 1, 1],optimizer=opt) # sum of the losses 
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


def plot_some_raw_maps(epoch,genA2B, genB2A, datasetA,datasetB,path_plot,n_samples=10):
    # choose random instances
    ix = np.random.randint(0, datasetA.shape[0], n_samples)
    # reitrieve selected images
    sample_datasetA = datasetA[ix]
    sample_datasetB = datasetB[ix]
    sample_fakesetA2B = genA2B.predict(sample_datasetA)
    sample_fakesetB2A = genB2A.predict(sample_datasetB)

    sample_fakesetA2B2A = genB2A.predict(sample_fakesetA2B)
    sample_fakesetB2A2B = genA2B.predict(sample_fakesetB2A)

    sample_fakesetB2A_A = genB2A.predict(sample_datasetA)
    sample_fakesetA2B_B = genA2B.predict(sample_datasetB)
    #
    names_=("A","B","A2B","B2A","A2B2A","B2A2B", "B2A_A", "A2B_B")
    nchecks=8
    pyplot.figure(figsize=(10,10))

    for i in range(10):
        mat_A = sample_datasetA[i,:,:,0].astype(float)
        mat_B = sample_datasetB[i,:,:,0].astype(float)
        mat_A2B = sample_fakesetA2B[i,:,:,0].astype(float)
        mat_B2A = sample_fakesetB2A[i,:,:,0].astype(float)
        #
        mat_A2B2A = sample_fakesetA2B2A[i,:,:,0].astype(float)
        mat_B2A2B = sample_fakesetB2A2B[i,:,:,0].astype(float)
        #
        mat_B2A_A = sample_fakesetB2A_A[i,:,:,0].astype(float)
        mat_A2B_B = sample_fakesetA2B_B[i,:,:,0].astype(float)
        #
        #### On inverse LON_LAT pour plotter correctement
        mat_A=np.fliplr(mat_A)
        mat_B=np.fliplr(mat_B)
        #
        mat_A2B=np.fliplr(mat_A2B)
        mat_B2A = np.fliplr(mat_B2A)
        #
        mat_A2B2A=np.fliplr(mat_A2B2A)
        mat_B2A2B = np.fliplr(mat_B2A2B)
        #
        mat_B2A_A=np.fliplr(mat_B2A_A)
        mat_A2B_B = np.fliplr(mat_A2B_B)
        #
        ### Plot
        examples = vstack((mat_A, mat_B, mat_A2B, mat_B2A, mat_A2B2A, mat_B2A2B, mat_B2A_A, mat_A2B_B))
        # define subplot
        for j in range(8):
            pyplot.subplot(10, nchecks, i*8 + 1 + j)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            print(examples.shape)
            pyplot.imshow(examples[(j*28):((j+1)*28), :], cmap='YlOrRd')
            if i==0:
                pyplot.title(str(names_[j]) ,fontsize=10, y=1)
            vmin = np.quantile(vstack((mat_A,mat_B)), 0.025)
            vmax = np.quantile(vstack((mat_A,mat_B)), 0.975)
            pyplot.clim(vmin,vmax)
            if (j + 1) % nchecks == 0:
                pyplot.colorbar()
        # # save plot to file
    filename = path_plot + '/diagnostic/plot_raw_maps_%03d.png' % (epoch+1)
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

def recap_accuracy_and_save_gendisc(epoch,genA2B, genB2A, discA, discB, datasetA, datasetB, path_plot, n_samples=100):
    #Recap accuracy of discriminators 
    #prepare real samples
    xA_real, yA_real = generate_real_samples(datasetA, n_samples)
    xB_real, yB_real = generate_real_samples(datasetB, n_samples)
    # evaluate discriminator on real examples
    _, accA_real = discA.evaluate(xA_real, yA_real, verbose=0)
    _, accB_real = discB.evaluate(xB_real, yB_real, verbose=0)
    # prepare fake examples
    xA_fake, yA_fake = generate_fake_samples(genB2A, xB_real)
    xB_fake, yB_fake = generate_fake_samples(genA2B, xA_real)
    # evaluate discriminator on fake examples
    _, accA_fake = discA.evaluate(xA_fake, yA_fake, verbose=0)
    _, accB_fake = discB.evaluate(xB_fake, yB_fake, verbose=0)
    # summarize discriminator performance
    print('On a sample of '+ str(n_samples) + ':')
    print('>Accuracy discA real: %.0f%%, discA fake: %.0f%%' % (accA_real*100, accA_fake*100))
    print('>Accuracy discB real: %.0f%%, discB fake: %.0f%%' % (accB_real*100, accB_fake*100))
    # save the disc and gen model
    #ATTENTION A REACTIVER. BUG H5PY???
    #genB2A.save(path_plot+'/models/genB2A_model_%03d.h5' % (epoch+1))
    #genA2B.save(path_plot+'/models/genA2B_model_%03d.h5' % (epoch+1))
    #discA.save(path_plot+'/models/discA_model_%03d.h5' % (epoch+1))
    #discB.save(path_plot+'/models/discB_model_%03d.h5' % (epoch+1))



# create a line plot of loss for the gan and save to file

def plot_history_loss(discA_hist, discB_hist, validA_hist, validB_hist, recA_hist, recB_hist, identA_hist, identB_hist, weighted_hist, discA_acc_hist, discB_acc_hist,path_plot):
    # plot loss
    pyplot.figure(figsize=(9,9))
    pyplot.subplot(4, 1, 1)
    pyplot.plot(validA_hist, label='loss-validA')
    pyplot.plot(validB_hist, label='loss-validB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.subplot(4,1,2)
    pyplot.plot(np.array(recA_hist)*10, label='loss-recA')
    pyplot.plot(np.array(recB_hist)*10, label='loss-recB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.subplot(4, 1, 3)
    pyplot.plot(identA_hist, label='loss-identA')
    pyplot.plot(identB_hist, label='loss-identB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.subplot(4, 1, 4)
    pyplot.plot(weighted_hist, label='loss-weighted',color="green")
    #pyplot.plot(np.array(validA_hist)+np.array(validB_hist)+np.array(recA_hist)*10+np.array(recB_hist)*10+np.array(identA_hist)+np.array(identB_hist),color="red")
    pyplot.legend()
    pyplot.ylim((0,3))
    # plot discriminator accuracy
   # save plot to file
    pyplot.savefig(path_plot + '/diagnostic/plot_history_gen_loss.png')
    pyplot.close()
    pyplot.figure(figsize=(4,4))
    pyplot.subplot(2, 1, 1)
    pyplot.plot(discA_hist, label='loss-discA')
    pyplot.plot(discB_hist, label='loss-discB')
    pyplot.legend()
    pyplot.ylim((0,1))
    pyplot.subplot(2, 1, 2)
    pyplot.plot(discA_acc_hist, label='discA-acc')
    pyplot.plot(discB_acc_hist, label='discB-acc')
    pyplot.legend()
    pyplot.ylim((-0.1,1.1))
    pyplot.savefig(path_plot + '/diagnostic/plot_history_disc_loss.png')
    pyplot.close()



def plot_history_criteria(crit,title_crit,ylim1,ylim2,path_plot):
    #plot criteria mean
    pyplot.subplot(1, 1, 1)
    pyplot.plot(crit, label=title_crit)
    x = np.linspace(0,100,100)
    pyplot.legend()
    pyplot.ylim((ylim1,ylim2))
    pyplot.savefig(path_plot + '/diagnostic/plot_'+ title_crit + '.png')
    pyplot.close()

def train_combined_new(rank_version,PR_version,genA2B, genB2A, discA, discB, comb_model, datasetA, datasetB, OriginalA, OriginalB, ind, lon, lat,point_grid,path_to_save ,XminA_=None,XmaxA_=None,XminB_=None,XmaxB_=None,n_epochs=100, n_batch=32):
    bat_per_epo = int(datasetA.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    discA_hist, discB_hist, validA_hist, validB_hist, recA_hist, recB_hist,identA_hist, identB_hist, weighted_hist = list(), list(), list(), list(), list(), list(), list(), list(), list()
    discA_acc_hist, discB_acc_hist = list(), list()
    MAE_mean_, MAE_sd_rel_, MAE_correlogram_= list(), list(), list()

    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            xA_real, yA_real = generate_real_samples(datasetA, half_batch)
            xB_real, yB_real = generate_real_samples(datasetB, half_batch)
            # generate 'fake' examples
            xA_fake, yA_fake = generate_fake_samples(genB2A, xB_real)
            xB_fake, yB_fake = generate_fake_samples(genA2B, xA_real)
            # create training set for the discriminator
            xA, yA = vstack((xA_real, xA_fake)), vstack((yA_real, yA_fake))
            xB, yB = vstack((xB_real, xB_fake)), vstack((yB_real, yB_fake))
            # update discriminator model weights
            discA_loss, discA_acc = discA.train_on_batch(xA, yA)
            discB_loss, discB_acc = discB.train_on_batch(xB, yB)
            # train generator
            all_loss = comb_model.train_on_batch([xA_real, xB_real], [yB_real, yA_real, xA_real, xB_real, xA_real, xB_real])
            #record history
            if (j+1) % 2 == 1:
                discA_hist.append(discA_loss)
                discB_hist.append(discB_loss)
                discA_acc_hist.append(discA_acc)
                discB_acc_hist.append(discB_acc)
                validA_hist.append(all_loss[1])
                validB_hist.append(all_loss[2])
                recA_hist.append(all_loss[3])
                recB_hist.append(all_loss[4])
                identA_hist.append(all_loss[5])
                identB_hist.append(all_loss[6])
                weighted_hist.append(all_loss[0])
        print('>%d, %d/%d, discA=%.3f, discB=%.3f, validA=%.3f, validB=%.3f, recA=%.3f, recB =%.3f, identA=%.3f, identB=%.3f' % (i+1, j+1, bat_per_epo, discA_loss, discB_loss, all_loss[1], all_loss[2], np.array(all_loss[3])*10, np.array(all_loss[4])*10, all_loss[5], all_loss[6]))
        print('Weighted sum: ' + str(all_loss[0]))
        #Check of g_loss for early stopping
        plot_history_loss(discA_hist, discB_hist, validA_hist, validB_hist, recA_hist, recB_hist, identA_hist, identB_hist, weighted_hist, discA_acc_hist, discB_acc_hist, path_plot=path_to_save)
        if (i+1) % 10 == 1:
            recap_accuracy_and_save_gendisc(i,genA2B, genB2A, discA, discB, datasetA, datasetB, path_plot=path_to_save, n_samples=100)
            res_mae_mean, res_mae_sd_rel, res_mae_correlogram = compute_and_plot_criteria_for_early_stopping(rank_version,PR_version,i, datasetA, datasetB, OriginalA, OriginalB, genA2B,  XminA_, XmaxA_, XminB_, XmaxB_,ind, lon, lat,point_grid,path_plot=path_to_save)
            plot_some_raw_maps(i, genA2B, genB2A, datasetA, datasetB, path_to_save)
            MAE_mean_.append(res_mae_mean)
            MAE_sd_rel_.append(res_mae_sd_rel)
            MAE_correlogram_.append(res_mae_correlogram)
            #plot history of criteria
            plot_history_criteria(MAE_mean_, "mae_mean",-0.01,0.5,path_to_save)
            plot_history_criteria(MAE_sd_rel_, "mae_sd_rel",-0.01,0.2,path_to_save)
            plot_history_criteria(MAE_correlogram_, "mae_correlogram",-0.01,0.3,path_to_save)
        #####EARLY STOPPING
        #if i>50 and (np.mean(g_hist[-2000:])>3.5 or np.std(g_hist[-2000:]) >0.26):
        #    print('BREAK! For the 2000 last points: mean_gloss=%.3f, sd_gloss=%.3f ' % (np.mean(g_hist[-2000:] ), np.std(g_hist[-2000:])))
        #    break






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



import h5netcdf.legacyapi as netCDF4
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
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
from matplotlib import pyplot
from shutil import copyfile
import os
from os import makedirs
from google.colab import drive
import tensorflow as tf
from scipy.stats import spearmanr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt



def load_samples(nc_file,var): #changed wrt Soulivanh
    dataset = netCDF4.Dataset(nc_file, "r")
    lon = dataset.variables["lon"]
    lat = dataset.variables["lat"]
    X = dataset.variables[var]
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(X, axis=-1)
    dataset.close()
    # convert from unsigned ints to floats
    X = X.astype('float32')
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
    return X, lon, lat, min_, max_

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)): #same as Soulivanh
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
    opt = Adam(lr=0.00002, beta_1=0.5)
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

def define_combined(genA2B, genB2A, discA, discB, in_shape=(28,28,1)): #same as Soulivanh
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
    
    opt = Adam(lr=0.0001, beta_1=0.5)
    comb_model = Model([inputA, inputB], [valid_imgA, valid_imgB, reconstruct_imgA, reconstruct_imgB, gen_orig_imgA, gen_orig_imgB])
#     comb_model.summary()
    comb_model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae', 'mae','mae'],loss_weights=[  1, 1, 10, 10, 1, 1],optimizer=opt) # sum of the losses 
    return comb_model

# select real samples
def generate_real_samples(dataset, n_samples): #same as Soulivanh
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
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

#### Compute mean and sd of an array (nb_images,28,28,1)
def compute_mean_sd_array(data, Xmin_, Xmax_):
  nb_images=data.shape[0]
  tmp_data = np.reshape([None]*28*28*nb_images,(nb_images,28,28,1))
  res_mean=np.reshape([None]*28*28,(1,28,28))
  res_sd=np.reshape([None]*28*28,(1,28,28))
  n=-1
  #Rescale climatic variables wrt Xmin and Xmax
  for k in range(28):
      for l in range(28):
          n=n+1
          tmp_data[:,k,l,:] = data[:,k,l,:]*(Xmax_[n] - Xmin_[n])+ Xmin_[n]
  ##!!!! Preprocess for PR !!! to include if PR!!!
  if is_PR==True:
    tmp_data[tmp_data < 1] = 0    

  #Compute mean and sd for each grid
  for k in range(28):
    for l in range(28):
        res_mean[:,k,l]=np.mean(tmp_data[:,k,l,:])
        res_sd[:,k,l]=np.std(tmp_data[:,k,l,:])
  res_mean = res_mean.astype(float)
  res_sd = res_sd.astype(float)
  return res_mean, res_sd

# create and save a plot of generated images 
def plot_res(epoch, mat_A, mat_B, mat_A2B, title):
    #mat_A, mat_B, mat_A2B results from compute_mean_sd_array to plot
    mat_A = mat_A.astype(float)
    mat_B = mat_B.astype(float)
    mat_A2B = mat_A2B.astype(float)
    ### Plot
    ### Extract LON/LAT
    dataset = netCDF4.Dataset("tas_day_CNRM-CM6-1_piControl_r1i1p1f2_gr_18500101-18691231.nc", "r")
    lon = dataset.variables["lon"][:]
    lat = dataset.variables["lat"][:]
    dataset.close()
    #### Mean and sd / MAE ####
    if is_PR==False:
      examples = vstack((mat_A, mat_B, mat_A2B, mat_A-mat_B, mat_B-mat_B, mat_A2B-mat_B))
      names_=("IPSL","CNRM","GAN","IPSL-CNRM","CNRM-CNRM","GAN-CNRM")
    else:
      examples = vstack((mat_A, mat_B, mat_A2B, (mat_A-mat_B)/mat_B, (mat_B-mat_B)/mat_B, (mat_A2B-mat_B)/mat_B))
      names_=("IPSL","CNRM","GAN","(IPSL-CNRM)/CNRM","(CNRM-CNRM)/CNRM","(GAN-CNRM)/CNRM")

    nchecks=3
    extent = [lon.min(), lon.max(), lat.min(), lat.max()] #
    map = Basemap(projection='merc', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='c')
    # find x,y of map projection grid.
    xx, yy = np.meshgrid(lon, lat)
    xx, yy = map(xx, yy)

    pyplot.figure(figsize=(15,15))
    for i in range(2 * nchecks):
        # define subplot
        pyplot.subplot(2, nchecks, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :], cmap='gray_r')
        if (i < (1 * nchecks)):
            pyplot.title(str(names_[i]) + ' mean: ' +str(round(np.mean(examples[i, :, :]),3)) + ' / sd: ' + str(round(np.std(examples[i, :, :]),3))  ,fontsize=10, y=1)
            vmin = np.quantile(mat_B, 0.1)
            vmax = np.quantile(mat_B, 0.9)
            cmap = pyplot.get_cmap("YlOrRd")
        else:
            pyplot.title(str(names_[i]) + ' mae: ' +str(round(np.mean(abs(examples[i, :, :])),3)) + ' / sd: ' + str(round(np.std(examples[i, :, :]),3))  ,fontsize=10, y=1)
            vmin = -3
            vmax = 3
            cmap = pyplot.get_cmap("RdBu")
        map.pcolormesh(xx,yy, examples[i, :, :], vmin = vmin, vmax = vmax, cmap=cmap)
        map.drawcoastlines(linewidth = 0.2)
        if (i + 1) % nchecks == 0:
            map.colorbar()
        # # save plot to file
    filename = save_path_res + '/diagnostic/plot_criteria_'+ title + '_%03d.png' % (epoch+1)
    pyplot.savefig(filename, dpi=150)
    pyplot.close()


# create and save a plot of generated images 
def save_plot_gan(epoch, datasetA, datasetB, genA2B, XminA_, XmaxA_, XminB_, XmaxB_):
    print("begin criteria ")
    # Generate bias bias correction
    fakesetB = genA2B.predict(datasetA)
    
    #Compute criteria
    res_mean_datasetA, res_sd_datasetA = compute_mean_sd_array(datasetA, XminA_, XmaxA_)
    res_mean_datasetB, res_sd_datasetB = compute_mean_sd_array(datasetB, XminB_, XmaxB_)
    res_mean_fakesetB, res_sd_fakesetB = compute_mean_sd_array(fakesetB, XminB_, XmaxB_)

    if is_PR==False:
      mae_mean = np.mean(abs(res_mean_fakesetB-res_mean_datasetB))
      title_="mean_tas"
    else:
      mae_mean = np.mean(abs((res_mean_fakesetB-res_mean_datasetB)/res_mean_datasetB))
      title_="mean_pr"

    
    mae_std_rel = np.mean(abs((res_sd_fakesetB-res_sd_datasetB)/res_sd_datasetB))
    
    #Indications for early stopping (should be near 0)
    print('> MAE_mean: mean ', round(mae_mean,3))#,' sd: ', round(np.std(abs(resmean_A2B[:,:,0,:]-resmean_B[:,:,0,:])),3))
    print('> MAE_sd_rel: mean ', round(mae_std_rel,3))#, ' -sd: ',round(np.std(abs((ressd_A2B[:,:,0,:]-ressd_B[:,:,0,:])/ressd_B[:,:,0,:])),3))

    plot_res(epoch, res_mean_datasetA, res_mean_datasetB, res_mean_fakesetB, title_)
    return mae_mean, mae_std_rel

# evaluate the discriminator, save generator model
def summarize_performance_combined(epoch, genA2B, genB2A, discA, discB, datasetA, datasetB,XminA_, XmaxA_, XminB_, XmaxB_, MAE_mean_, MAE_sd_rel_,  n_samples=100):
    savepath = './'
    # prepare real samples
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
    print('>Accuracy A real: %.0f%%, A fake: %.0f%%' % (accA_real*100, accA_fake*100))
    print('>Accuracy B real: %.0f%%, B fake: %.0f%%' % (accB_real*100, accB_fake*100))
    # save plot
    tmp_mae_mean, tmp_mae_sd_rel = save_plot_gan(epoch,datasetA, datasetB, genA2B, XminA_, XmaxA_, XminB_, XmaxB_)
    MAE_mean_.append(tmp_mae_mean)
    MAE_sd_rel_.append(tmp_mae_sd_rel)
    # save the generator model
    genA2B.save(save_path_res+'/models/genA2B_model_%03d.h5' % (epoch+1))
    return MAE_mean_, MAE_sd_rel_

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist, MAE_mean_, MAE_sd_rel_):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d1_hist, label='loss-d-real')
    pyplot.plot(d2_hist, label='loss-d-fake')
    pyplot.plot(g_hist, label='loss-gen')
    pyplot.legend()
    pyplot.ylim((0,4))
    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.plot(a1_hist, label='d-acc-real')
    pyplot.plot(a2_hist, label='d-acc-fake')
    pyplot.legend()
    pyplot.ylim((-0.1,1.1))
    # save plot to file
    pyplot.savefig(save_path_res + '/diagnostic/plot_line_plot_loss.png')
    pyplot.close()
    #plot criteria mean
    pyplot.subplot(1, 1, 1)
    pyplot.plot(MAE_mean_, label='mae_mean')
    x = np.linspace(0,100,100)
    CDFt = 0*x+0.002
    pyplot.plot(CDFt, label='CDFt')
    pyplot.legend()
    pyplot.ylim((0,5))
    pyplot.savefig(save_path_res + '/diagnostic/plot_mae_mean_criteria.png')
    pyplot.close()
    #plot criteria
    pyplot.subplot(1, 1, 1)
    pyplot.plot(MAE_sd_rel_, label='mae_std_relative')
    x = np.linspace(0,100,100)
    CDFt = 0*x+0.001
    pyplot.plot(CDFt, label='CDFt')
    pyplot.legend()
    pyplot.ylim((0,0.5))
    pyplot.savefig(save_path_res + '/diagnostic/plot_mae_sd_rel_criteria.png')
    pyplot.close()


def train_combined(genA2B, genB2A, discA, discB, comb_model, datasetA, datasetB, XminA_, XmaxA_, XminB_, XmaxB_, n_epochs=100, n_batch=32):
    bat_per_epo = int(datasetA.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    MAE_mean_, MAE_sd_rel_ = list(), list()
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
            dA_loss, d_acc1 = discA.train_on_batch(xA, yA)
            dB_loss, d_acc2 = discB.train_on_batch(xB, yB)
            # train generator
            g_loss = comb_model.train_on_batch([xA_real, xB_real], [yB_real, yA_real, xA_real, xB_real, xA_real, xB_real])
            if (j+1) % 4 == 1:
              #print(j)
              # record history
              d1_hist.append(dA_loss)
              d2_hist.append(dB_loss)
              g_hist.append(sum(g_loss))
              a1_hist.append(d_acc1)
              a2_hist.append(d_acc2)
        # evaluate the model performance, sometimes
        # summarize loss on this batch
        print('>%d, %d/%d, dA=%.3f, dB=%.3f, gB2A=%.3f, gA2B=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, dA_loss, dB_loss, g_loss[0], g_loss[1], sum(g_loss)))
        plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist, MAE_mean_, MAE_sd_rel_)
        if (i+1) % 10 == 1:
            MAE_mean_, MAE_sd_rel_ = summarize_performance_combined(i, genA2B, genB2A, 
                                                                        discA, discB, 
                                                                        datasetA, datasetB, 
                                                                        XminA_, XmaxA_, XminB_, XmaxB_, 
                                                                        MAE_mean_, MAE_sd_rel_)


def main():
    # create the discriminator
    discA = define_discriminator()
    discB = define_discriminator()
    # create the generator
    genA2B = define_generator()
    genB2A = define_generator()
    # create the gan
    comb_model = define_combined(genA2B, genB2A, discA, discB)
    # load image data
    datasetA, lon, lat, XminA_, XmaxA_ = load_samples("pr_day_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_18500101-18691231.nc","pr")
    datasetB, lon, lat, XminB_, XmaxB_ = load_samples("pr_day_CNRM-CM6-1_piControl_r1i1p1f2_gr_18500101-18691231.nc","pr")

    is_PR=True
    # make folder for results
    save_path_res = 'Work_CycleGAN_PR_withrescale'
    makedirs(save_path_res, exist_ok=True)
    makedirs(save_path_res + '/models', exist_ok=True)
    makedirs(save_path_res + '/diagnostic', exist_ok=True)
    train_combined(genA2B, genB2A, discA, discB, comb_model, datasetA, datasetB, XminA_, XmaxA_, XminB_, XmaxB_, n_epochs = 500) #####attention n_epochs

if __name__ == "__main__":
    main()

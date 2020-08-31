rm(list=ls())
gc()
library(ncdf4)
library(fields)
library(devtools)
# devtools::install_github("yrobink/SBCK/R/SBCK")

library(SBCK)



flatten_array <- function(data_array){ #data_array of type LON x LAT x TIME; output: TIME x VARPHY
  res = matrix(NaN, ncol= dim(data_array)[1] * dim(data_array)[2], nrow = dim(data_array)[3])
  k=0
  for(i in 1:dim(data_array)[1]){
    for(j in 1:dim(data_array)[2]){
      k=k+1
      res[,k]<-data_array[i,j,]
    }
  }
  return(res)
}


convert_matrix_to_array <- function(data_mat,nb_lon = sqrt(ncol(data_mat)), nb_lat = sqrt(ncol(data_mat))){ #data_mat of type TIME x VARPHY; output: LON x LAT x TIME
  res = array(NaN, dim = c(nb_lon, nb_lat, nrow(data_mat)))
  k=0
  for(i in 1:nb_lon){
    for(j in 1:nb_lat){
      k=k+1
      res[i,j,]<-data_mat[,k]
    }
  }
  return(res)
}


# # path="/home/starmip/bfran/LSCE_These/" #"/homel/bfran/Bureau/Coronavirus
# path="/homel/bfran/Bureau/Coronavirus/" #"
# source(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Script/function_dOTC_algo.R"))
# #### Load data
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRbili/tas_pr_day_IPSLMRbili_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Temporal_and_Random_indices_1979_2016.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/MBC/SAFRAN_IPSLMRbili/", CV, "/tas_pr_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris.RData"))


#### On JZ
load("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
load("/gpfswork/rech/eal/commun/CycleGAN/Data/IPSLMRbili/tas_pr_day_IPSLMRbili_79_16_Paris.RData")

load("/gpfswork/rech/eal/commun/CycleGAN/Data/Temporal_and_Random_indices_1979_2016.RData")
###################################################################################################################
###################################################################################################################

CVtype = c("CVchrono","CVunif")
for(CV in CVtype){
  print(CV)
  assign(paste0('Ind_', CV, '_Calib'),list("winter_79_16"=get(paste0('Ind_', CV, '_Calib_winter_79_16')),
                                           "summer_79_16"=get(paste0('Ind_', CV, '_Calib_summer_79_16')),
                                           "automn_79_16"=get(paste0('Ind_', CV, '_Calib_automn_79_16')),
                                           "spring_79_16"=get(paste0('Ind_', CV, '_Calib_spring_79_16'))))
  assign(paste0('Ind_', CV, '_Proj'),list("winter_79_16"=get(paste0('Ind_', CV, '_Proj_winter_79_16')),
                                          "summer_79_16"=get(paste0('Ind_', CV, '_Proj_summer_79_16')),
                                          "automn_79_16"=get(paste0('Ind_', CV, '_Proj_automn_79_16')),
                                          "spring_79_16"=get(paste0('Ind_', CV, '_Proj_spring_79_16'))))
  ##### TAS
  print("TAS")
  assign(paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_SAFRAN_79_16_Paris
  Mod=tas_day_IPSLMRbili_79_16_Paris
  
  nb_dim=ncol(Ref)^2
  bin_width = rep(0.1, nb_dim)
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    ## Bias correction
    ## Step 1 : construction of the class dOTC
    dotc = SBCK::dOTC$new(bin_width)
    ## Step 2 : Fit the bias correction model
    dotc$fit( flatRef , flatMod_Calib , flatMod_Proj )
    ## Step 3 : perform the bias correction, Z is a list containing
    ## corrections
    Z = dotc$predict(flatMod_Proj,flatMod_Calib)
    
    flatSpatialdOTC_Calib = Z$Z0 ## Correction in calibration period
    
    flatSpatialdOTC_Proj = Z$Z1 ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialdOTC_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialdOTC_Proj)

    eval(parse(text=paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("tas_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
  

  
  #### PR #### probleme de precision R pour les rangs...
  print("PR")
  assign(paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  Mod=pr_day_IPSLMRbili_79_16_Paris
  
  nb_dim=ncol(Ref)^2
  bin_width = rep(0.1, nb_dim)
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    flatRef = flatten_array(Ref[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Calib = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Calib'))[[season]]])
    flatMod_Proj = flatten_array(Mod[,,get(paste0('Ind_', CV, '_Proj'))[[season]]])
    
    ## Bias correction
    ## Step 1 : construction of the class dOTC
    dotc = SBCK::dOTC$new(bin_width)
    ## Step 2 : Fit the bias correction model
    dotc$fit( flatRef , flatMod_Calib , flatMod_Proj )
    ## Step 3 : perform the bias correction, Z is a list containing
    ## corrections
    Z = dotc$predict(flatMod_Proj,flatMod_Calib)
    
    flatSpatialdOTC_Calib = Z$Z0 ## Correction in calibration period
    
    flatSpatialdOTC_Proj = Z$Z1 ## Correction in projection period
    
    tmp_res_Calib = convert_matrix_to_array(flatSpatialdOTC_Calib)
    tmp_res_Proj = convert_matrix_to_array(flatSpatialdOTC_Proj)
    
    eval(parse(text=paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris[,,","Ind_",CV,"_Calib[[season]]]=tmp_res_Calib")))
    eval(parse(text=paste0("pr_day_", CV, "_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris[,,","Ind_",CV,"_Proj[[season]]]=tmp_res_Proj")))
  }
}
# 
# # 
#### Save CVchrono
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_IPSLMRbili/CVchrono")
save(list=c("tas_day_CVchrono_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris",
            "pr_day_CVchrono_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris.RData")
#
#### Save CVunif
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_IPSLMRbili/CVunif")
save(list=c("tas_day_CVunif_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris",
            "pr_day_CVunif_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_SpatialdOTC_SAFRAN_IPSLMRbili_79_16_Paris.RData")
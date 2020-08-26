rm(list=ls())
gc()
library(ncdf4)
library(fields)
# 
# # path="/home/starmip/bfran/LSCE_These/" #"/homel/bfran/Bureau/Coronavirus
# # path="/homel/bfran/Bureau/Coronavirus/" #"
# source(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Script/function_1dQQ.R"))
# #### Load data
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRbili/tas_pr_day_IPSLMRbili_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Temporal_and_Random_indices_1979_2016.RData"))



#### On JZ
source("/gpfswork/rech/eal/commun/CycleGAN/Script/function_1dQQ.R")
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
  assign(paste0("tas_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=tas_day_SAFRAN_79_16_Paris
  Mod=tas_day_IPSLMRbili_79_16_Paris
  
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
  
  #### PR #### probleme de precision R pour les rangs...
  print("PR")
  assign(paste0("pr_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  Mod=pr_day_IPSLMRbili_79_16_Paris
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
}

#### Save CVchrono
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_IPSLMRbili/CVchrono")
save(list=c("tas_day_CVchrono_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris",
            "pr_day_CVchrono_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris.RData")

#### Save CVunif
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_IPSLMRbili/CVunif")
save(list=c("tas_day_CVunif_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris",
            "pr_day_CVunif_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_1dQQ_SAFRAN_IPSLMRbili_79_16_Paris.RData")









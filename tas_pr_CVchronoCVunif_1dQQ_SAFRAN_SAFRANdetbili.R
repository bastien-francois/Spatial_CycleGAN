rm(list=ls())
gc()
library(ncdf4)
library(fields)
# 
# # path="/home/starmip/bfran/LSCE_These/" #"/homel/bfran/Bureau/Coronavirus
# # path="/homel/bfran/Bureau/Coronavirus/" #"
# source(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Script/function_1dQQ.R"))
# #### Load data
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData"))
# load(paste0(path,"MBC_Project/ML_MBC/GAN/CycleGAN4/Temporal_and_Random_indices_1979_2016.RData"))



#### On JZ
source("/gpfswork/rech/eal/commun/CycleGAN/Script/function_1dQQ.R")
load("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
load("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.RData")
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
  assign(paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))

  Ref=tas_day_SAFRAN_79_16_Paris
  Mod=tas_day_SAFRANdetbili_79_16_Paris


  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("tas_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
  
  #### PR #### probleme de precision R pour les rangs...
  print("PR")
  assign(paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris"),array(NaN,dim=c(28,28,13870)))
  
  Ref=pr_day_SAFRAN_79_16_Paris
  Mod=pr_day_SAFRANdetbili_79_16_Paris
  
  for(season in names(get(paste0('Ind_', CV, '_Calib')))){
    print(season)
    for(i in 1:28){
      print(i)
      for(j in 1:28){
        tmp_res=QQb_new(Ref[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Calib'))[[season]]],Mod[i,j,get(paste0('Ind_', CV, '_Proj'))[[season]]])
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Calib[[season]]]=tmp_res$Mch[,1]")))
        eval(parse(text=paste0("pr_day_", CV, "_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,","Ind_",CV,"_Proj[[season]]]=tmp_res$Mph[,1]")))
      }
    }
  }
}

#### Save CVchrono
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_SAFRANdetbili/CVchrono")
save(list=c("tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris.RData")

#### Save CVunif
setwd("/gpfswork/rech/eal/commun/CycleGAN/MBC/SAFRAN_SAFRANdetbili/CVunif")
save(list=c("tas_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "pr_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris",
            "LON_Paris",
            "LAT_Paris",
            "IND_Paris",
            "point_max"),
     file="tas_pr_day_CVunif_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris.RData")







# 
# ####Check
# Ref=tas_day_SAFRAN_79_16_Paris
# Mod=tas_day_SAFRANdetbili_79_16_Paris
# par(mfrow=c(2,2))
# image.plot(1:28,1:28,apply(Ref[,,Ind_winter_79_16], c(1,2), mean))
# image.plot(1:28,1:28,apply(Mod[,,Ind_winter_79_16], c(1,2), mean))
# image.plot(1:28,1:28,apply(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_winter_79_16], c(1,2), mean))
# 
# mean(apply(Ref[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# mean(apply(Mod[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# mean(apply(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# 
# mean(apply(Ref[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# mean(apply(Mod[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# mean(apply(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# 
# 
# #### copula
# copula_Mc=array(NaN,c(28,28,length(Ind_CVchrono_Calib_winter_79_16)))
# copula_Mch=array(NaN,c(28,28,length(Ind_CVchrono_Calib_winter_79_16)))
# for(i in 1:28){
#   for(j in 1:28){
#     copula_Mc[i,j,]=rank(Mod[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min")
#     copula_Mch[i,j,]=rank(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min")
#   }
# }
# 
# identical(copula_Mc,copula_Mch) #=TRUE ok
# library(energy)
# 
# copula_Mp=array(NaN,c(28,28,length(Ind_CVchrono_Proj_winter_79_16)))
# copula_Mph=array(NaN,c(28,28,length(Ind_CVchrono_Proj_winter_79_16)))
# for(i in 1:28){
#   for(j in 1:28){
#     copula_Mp[i,j,]=rank(Mod[i,j,Ind_CVchrono_Proj_winter_79_16],ties.method="min")
#     copula_Mph[i,j,]=rank(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Proj_winter_79_16],ties.method="min")
#   }
# }
# identical(copula_Mp,copula_Mph)
# 
# 
# check_rank_Mc = array(NaN,c(28,28,1))
# check_rank_Mp = array(NaN,c(28,28,1))
# for(i in 1:28){
#   for(j in 1:28){
#     check_rank_Mc[i,j,1]=identical(rank(Mod[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min"),rank(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min"))
#     check_rank_Mp[i,j,1]=identical(rank(Mod[i,j,Ind_CVchrono_Proj_winter_79_16],ties.method="min"),rank(tas_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Proj_winter_79_16],ties.method="min"))
#   }
# }
# image.plot(1:28,1:28,check_rank_Mc[,,1])
# image.plot(1:28,1:28,check_rank_Mp[,,1])
# 
# 
# #### PR
# Ref=pr_day_SAFRAN_79_16_Paris
# Mod=pr_day_SAFRANdetbili_79_16_Paris
# par(mfrow=c(2,2))
# image.plot(1:28,1:28,apply(Ref[,,Ind_winter_79_16], c(1,2), mean))
# image.plot(1:28,1:28,apply(Mod[,,Ind_winter_79_16], c(1,2), mean))
# image.plot(1:28,1:28,apply(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_winter_79_16], c(1,2), mean))
# 
# mean(apply(Ref[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# mean(apply(Mod[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# mean(apply(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_CVchrono_Calib_winter_79_16], c(1,2), mean))
# 
# mean(apply(Ref[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# mean(apply(Mod[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# mean(apply(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[,,Ind_CVchrono_Proj_winter_79_16], c(1,2), mean))
# 
# for(i in 1:28){
#   for(j in 1:28){
#     copula_Mc[i,j,]=rank(Mod[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min")
#     copula_Mch[i,j,]=rank(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16],ties.method="min")
#   }
# }
# 
# identical(copula_Mc,copula_Mch)
# 
# check_more_wet_Calib = array(NaN,c(28,28,1))
# check_more_wet_Proj = array(NaN,c(28,28,1))
# check_rank_Mc = array(NaN,c(28,28,1))
# check_rank_Mp = array(NaN,c(28,28,1))
# sumdiff_rank_Mc = array(NaN,c(28,28,1))
# sumdiff_rank_Mp = array(NaN,c(28,28,1))
# for(i in 1:28){
#   for(j in 1:28){
#     check_more_wet_Calib[i,j,1] = mean(Ref[i,j,Ind_CVchrono_Calib_winter_79_16]==0)>mean(Mod[i,j,Ind_CVchrono_Calib_winter_79_16]==0)
#     check_more_wet_Proj[i,j,1] = mean(Ref[i,j,Ind_CVchrono_Proj_winter_79_16]==0)>mean(Mod[i,j,Ind_CVchrono_Proj_winter_79_16]==0)
#     ind_wet_Calib = pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16]>0
#     check_rank_Mc[i,j,1]=identical(rank(Mod[i,j,Ind_CVchrono_Calib_winter_79_16[ind_wet_Calib]],ties.method="min"),rank(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16[ind_wet_Calib]],ties.method="min"))
#     sumdiff_rank_Mc[i,j,1]=sum(rank(Mod[i,j,Ind_CVchrono_Calib_winter_79_16[ind_wet_Calib]],ties.method="min")-rank(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Calib_winter_79_16[ind_wet_Calib]],ties.method="min"))
#     
#     ind_wet_Proj = pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Proj_winter_79_16]>0
#     check_rank_Mp[i,j,1]=identical(rank(Mod[i,j,Ind_CVchrono_Proj_winter_79_16[ind_wet_Proj]],ties.method="min"),rank(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Proj_winter_79_16[ind_wet_Proj]],ties.method="min"))
#     sumdiff_rank_Mp[i,j,1]=sum(rank(Mod[i,j,Ind_CVchrono_Proj_winter_79_16[ind_wet_Proj]],ties.method="min")-rank(pr_day_CVchrono_1dQQ_SAFRAN_SAFRANdetbili_79_16_Paris[i,j,Ind_CVchrono_Proj_winter_79_16[ind_wet_Proj]],ties.method="min"))
#     }
# }
# par(mfrow=c(3,2))
# image.plot(1:28,1:28,check_rank_Mc[,,1])
# image.plot(1:28,1:28,check_rank_Mp[,,1])
# 
# image.plot(1:28,1:28,check_more_wet_Calib[,,1])
# image.plot(1:28,1:28,check_more_wet_Proj[,,1])
# image.plot(1:28,1:28,sumdiff_rank_Mc[,,1])
# image.plot(1:28,1:28,sumdiff_rank_Mp[,,1])
# 
# 
# 
# season="winter_79_16"
# Rc = Ref[i,j,Ind_CVchrono_Calib[[season]]]
# Mc = Mod[i,j,Ind_CVchrono_Calib[[season]]]
# Mp = Mod[i,j,Ind_CVchrono_Proj[[season]]]
# i=28
# j=28
# FMc=ecdf(Mc)
# FMC=FMc(Mc)
# FRc=ecdf(Rc)
# FRC=FRc(Rc)
# # Mch[,k]=quantile(Rc[,k],probs=FMc(Mc[,k]),type=7)
# Mch=approx(FRC,Rc, FMC, yleft=min(Rc), yright= max(Rc))$y
# #Save the correction done for highest and lowest quantiles (will be used later to correct Mp in a context of climate change)
# correc_high_quntl=max(Mc)-max(Mch)
# correc_low_quntl=min(Mc)-min(Mch)
# 
# 
# ess_ind=Mch>0
# rank(Mc[ess_ind],ties.method="min")
# rank(Mch[ess_ind],ties.method="min")
# rank(Mc[ess_ind],ties.method="min")-rank(Mch[ess_ind],ties.method="min")
# Mc[ess_ind][913]
# FMC[ess_ind][913]
# sort(FRC)
# Mch[ess_ind][913]
# 
# xA=sort(FRC)[1106]
# xB=sort(FRC)[1109]
# 
# yA=sort(Rc)[1106]
# yB=sort(Rc)[1109]
# 
# x=FMC[ess_ind][913]
# 
# d1= (yA-yB)/(xA-xB)*x
# d11=((xA*yB)-(xB*yA))/(xA-xB)
# ess1 = d1+d11
# 
# ess1 = mpfr(d1+d11,300)
# 
# 
# rank(Mc[ess_ind])
# 
# 
# Mch[ess_ind][271]
# 
# Mc[ess_ind][271]
# FMC[ess_ind][271]
# sort(FRC)
# Mch[ess_ind][913]
# 
# xA=sort(FRC)[1106]
# xB=sort(FRC)[1109]
# 
# yA=sort(Rc)[1106]
# yB=sort(Rc)[1109]
# 
# x=FMC[ess_ind][271]
# 
# d2= (yA-yB)/(xA-xB)*x
# d22=((xA*yB)-(xB*yA))/(xA-xB)
# ess2 = d2+d22
# ess2 = mpfr(d2+d22,300)
# 
# d1==d2
# d11==d22
# ess1==ess2
# 
# 
# 
# 
# falseRc=rep(NaN,1000)
# falseMc=rep(NaN,1000)
# falseMp=rep(NaN,1000)
# 
# ind_dryRc=sample(1:1000,700)
# ind_wetRc = which(!(1:1000 %in% ind_dryRc))
# falseRc[ind_dryRc]=0
# falseRc[ind_wetRc]=runif(length(ind_wetRc),0.2, 10)
# 
# ind_dryMc=sample(1:1000,500)
# ind_wetMc = which(!(1:1000 %in% ind_dryMc))
# falseMc[ind_dryMc]=0
# falseMc[ind_wetMc]=runif(length(ind_wetMc),0.3, 20)
# 
# ind_dryMp=sample(1:1000,500)
# ind_wetMp = which(!(1:1000 %in% ind_dryMp))
# falseMp[ind_dryMp]=0
# falseMp[ind_wetMp]=runif(length(ind_wetMp),0.3, 25)
# 
# tmp_res=QQb_new(falseRc,falseMc,falseMp)
# 
# mean(falseRc)
# mean(tmp_res$Mch[,1])
# mean(falseMc)
# 
# 
# rank(falseMc,ties.method="min")-rank(tmp_res$Mch[,1],ties.method="min")
# 
# ind_wetfalseMc = which(tmp_res$Mch[,1]>0)
# rank(falseMc[ind_wetfalseMc],ties.method="min")-rank(tmp_res$Mch[ind_wetfalseMc,1],ties.method="min")
# 
# ind_wetfalseMp = which(tmp_res$Mph[,1]>0)
# rank(falseMp[ind_wetfalseMp],ties.method="min")-rank(tmp_res$Mph[ind_wetfalseMp,1],ties.method="min")
# 




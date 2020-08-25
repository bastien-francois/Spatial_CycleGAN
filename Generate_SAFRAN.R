###################################### To run from obelix
rm(list=ls())
library(timeDate)
library(fields)

### preproc_SAFRAN
load("/home/starmip/mvrac/SAFRAN/t2m_daily_SAFRAN_1959_2017.RData")
load("/home/starmip/mvrac/SAFRAN/pr_daily_SAFRAN_1959_2017.RData")
#Dropping 29 of February for each year in the reference
DATES_regular_1959_2017 = atoms(timeSequence(from="1959-01-01",to="2017-12-31",by='day'))
head(DATES_regular_1959_2017)

Ind_1959_2017_to_del = which((DATES_regular_1959_2017[,1] %in% c(1959:1978,2017)) | ((DATES_regular_1959_2017[,2]==02) &
                                                                                       (DATES_regular_1959_2017[,3]==29)))
DATES_365_1959_2017 = DATES_regular_1959_2017[-Ind_1959_2017_to_del,] #remove rows with 29 Feb. and 1959:1978,2017


tas_day_SAFRAN_79_16_France<-t2m_daily[,,-Ind_1959_2017_to_del]-273.15
pr_day_SAFRAN_79_16_France<-pr_daily[,,-Ind_1959_2017_to_del]*24*3600

LAT_France<-LAT
LON_France<-LON
IND_France=which(!is.na(pr_day_SAFRAN_79_16_France),arr.ind=TRUE)

rm(LON,LAT,t2m_daily,pr_daily,Nx,Ny)

setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN")
save(tas_day_SAFRAN_79_16_France,
     pr_day_SAFRAN_79_16_France,
     LAT_France,
     LON_France,
     IND_France,
     file="tas_pr_day_SAFRAN_79_16_France.RData")

# image.plot(LON_France,LAT_France,tas_day_SAFRAN_79_16_France[,,1])

#### Paris ####

#### SAFRAN ####
setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN")
load("tas_pr_day_SAFRAN_79_16_France.RData")

lat_paris=48.8534 
lon_paris=2.3488

ess=sqrt((LAT_France-(lat_paris))^2+(LON_France-(lon_paris))^2)
which(ess==min(ess),arr.ind=TRUE)
### row 69 col 102

# (69-13):(69+14),(102-13):(102+14)

LON_Paris=LON_France[(69-13):(69+14),(102-13):(102+14)]
LAT_Paris=LAT_France[(69-13):(69+14),(102-13):(102+14)]

tas_day_SAFRAN_79_16_Paris=tas_day_SAFRAN_79_16_France[(69-13):(69+14),(102-13):(102+14),]
pr_day_SAFRAN_79_16_Paris=pr_day_SAFRAN_79_16_France[(69-13):(69+14),(102-13):(102+14),]

# image.plot(LON_France[(69-13):(69+14),(102-13):(102+14)],LAT_France[(69-13):(69+14),(102-13):(102+14)],tas_day_SAFRAN_79_16_France[(69-13):(69+14),(102-13):(102+14),1])
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris)
IND_Paris=which(!is.na(pr_day_SAFRAN_79_16_Paris[,,1]),arr.ind=TRUE)

save(tas_day_SAFRAN_79_16_Paris,
     pr_day_SAFRAN_79_16_Paris,
     LON_Paris,
     LAT_Paris,
     IND_Paris,
     file="tas_pr_day_SAFRAN_79_16_Paris.RData")















###################################
#### Brouillon to not use ####
# #### For IPSL ####
# setwd(dir="/home/starmip/bfran/LSCE_M2/Work/RData")
# load("T2_PR_FRANCE_IPSL_DSwithSAFRAN_interp_79_16.RData")
# 
# tas_day_IPSL_79_16_Paris=T2_DS_IPSL_interp_79_16_France[(69-13):(69+14),(102-13):(102+14),]
# pr_day_IPSL_79_16_Paris=PR_DS_IPSL_interp_79_16_France[(69-13):(69+14),(102-13):(102+14),]
# 
# # image.plot(LON_France[(69-13):(69+14),(102-13):(102+14)],LAT_France[(69-13):(69+14),(102-13):(102+14)],tas_day_SAFRAN_79_16_France[(69-13):(69+14),(102-13):(102+14),1])
# # image.plot(LON_Paris,LAT_Paris,tas_day_IPSL_79_16_Paris)
# IND_Paris=which(!is.na(pr_day_IPSL_79_16_Paris[,,1]),arr.ind=TRUE)
# 
# dim(tas_day_IPSL_79_16_Paris)
# dim(pr_day_IPSL_79_16_Paris)
# 
# setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSL")
# save(tas_day_IPSL_79_16_Paris,pr_day_IPSL_79_16_Paris,LON_Paris,LAT_Paris,IND_Paris,
#      file="tas_pr_day_IPSL_79_16_Paris.RData")
# 
# load("tas_pr_day_IPSL_79_16_Paris.RData")
# 
# 
# 
# #### Save to NPY format:
# library(RcppCNPy)
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSL")
# load("tas_pr_day_IPSL_79_16_Paris.RData")
# npySave("tas_day_IPSL_79_16_Paris.npy", tas_day_IPSL_79_16_Paris)
# npySave("pr_day_IPSL_79_16_Paris.npy", pr_day_IPSL_79_16_Paris)
# 
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN")
# load("tas_pr_day_SAFRAN_79_16_Paris.RData")
# npySave("tas_day_SAFRAN_79_16_Paris.npy", tas_day_SAFRAN_79_16_Paris)
# npySave("pr_day_SAFRAN_79_16_Paris.npy", pr_day_SAFRAN_79_16_Paris)
# 
# 
# 
# #### Essai with st_threshold
# th=0.00033 #Vrac et al. 2015
# 
# th_O = th
# th_M = th
# 
# ObsRp=pr_day_SAFRAN_79_16_Paris
# DataGp=pr_day_IPSL_79_16_Paris
# ### st for stoch simulations : from 0 to Unif [0,th]
# ObsRp_st = pr_day_SAFRAN_79_16_Paris
# DataGp_st = pr_day_IPSL_79_16_Paris
# 
# WObs = which(ObsRp<=th_O)
# ObsRp_st[WObs] = runif(length(WObs),0,th_O)
# WGp = which(DataGp<=th_M)
# DataGp_st[WGp] = runif(length(WGp),0,th_M)
# 
# pr_st_day_SAFRAN_79_16_Paris = ObsRp_st 
# pr_st_day_IPSL_79_16_Paris = DataGp_st 
# 
# 
# image.plot(LON_Paris,LAT_Paris,apply(DataGp_st,c(1,2),mean))
# image.plot(LON_Paris,LAT_Paris,apply(ObsRp_st,c(1,2),min))
# image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRAN_79_16_Paris,c(1,2),mean))
# 
# npySave("pr_st_day_SAFRAN_79_16_Paris.npy", pr_st_day_SAFRAN_79_16_Paris)
# npySave("pr_st_day_IPSL_79_16_Paris.npy", pr_st_day_IPSL_79_16_Paris)
# 
# 
# 
# 
# # ####
# # image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,2])
# # image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,3])
# # image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,4])
# # image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRAN_79_16_Paris,c(1,2),max))
# # image.plot(LON_Paris,LAT_Paris,apply(pr_day_IPSL_79_16_Paris,c(1,2),mean))
# # min(pr_day_IPSL_79_16_Paris[pr_day_IPSL_79_16_Paris>0.1])
# 
# # 
# # #####################################################################################################################
# # rm(list=ls())
# # setwd(dir="/home/starmip/bfran/LSCE_M2/Work/")
# # 
# # #Load Ref. SAFRAN
# # source("preproc_SAFRAN.R")
# # source("functions_BC.R")
# # 
# # #Load Model. IPSL
# # setwd(dir="/home/starmip/bfran/LSCE_M2/Work/RData")
# # load("T2_PR_FRANCE_IPSL_DSwithSAFRAN_interp_79_16.RData")
# # load("SAFRAN_point_neighbor_3_regions.RData")
# # 
# # library(ARyga)
# # getNamespaceExports("ARyga")
# # 
# # setwd("/home/starmip/bfran/LSCE_M2/Work/dOTC")
# # #CV1: No evaluation and calibration period
# # 
# # #Initially, we have 
# # #T2_safran_79_16_Europe/PR_safran_79_16_Europe and 
# # #T2_DS_IPSL_interp_79_16_Europe/PR_DS_IPSL_interp_79_16_Europe
# # 
# # 
# # temp=13870 
# # IND = which(!is.na(t(PR_DS_IPSL_interp_79_16_France[,,1])), arr.ind = TRUE)
# # dim(IND) #8981    2
# # point_max=1:dim(IND)[1] #number of gridcells
# # 
# # #### dOTC ####
# # 
# # #### Paris ####
# # #### Temperature & Precip ####
# # time_DS_CV1_690d_dOTC=proc.time()
# # DS_CV1_690d_dOTC_Paris_79_16_France=function_MBC_CV1(Ref=list(PR_safran_79_16_France,T2_safran_79_16_France),
# #                                                      Model=list(PR_DS_IPSL_interp_79_16_France,T2_DS_IPSL_interp_79_16_France),
# #                                                      nb_month=12,
# #                                                      point_range=point_neighbor_paris,
# #                                                      list_temp_ind1=list_CV1_Ind_month_79_16,
# #                                                      allvar_bool=TRUE,
# #                                                      sep_intervar_bool=FALSE,
# #                                                      dotc_allvar_bool=TRUE)
# # 
# # PR_DS_CV1_690d_dOTC_Paris_79_16_France<-DS_CV1_690d_dOTC_Paris_79_16_France$res.var1
# # T2_DS_CV1_690d_dOTC_Paris_79_16_France<-DS_CV1_690d_dOTC_Paris_79_16_France$res.var2
# # 
# # # Outputs to save:
# # setwd("/home/starmip/bfran/LSCE_M2/Work/RData/France/CV1")
# # time_DS_CV1_690d_dOTC=proc.time()-time_DS_CV1_690d_dOTC
# # save(time_DS_CV1_690d_dOTC,
# #      PR_DS_CV1_690d_dOTC_Paris_79_16_France,
# #      T2_DS_CV1_690d_dOTC_Paris_79_16_France,
# #      file="DS_CV1_690d_dOTC_Paris_79_16_France.RData")
# # 

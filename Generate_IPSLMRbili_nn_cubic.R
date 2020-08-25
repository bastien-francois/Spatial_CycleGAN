# # 
### A lancer sur Obelix/Ciclad
rm(list=ls())
# Pour recup la grille SAFRAN
#system(paste0("cdo -seltimestep,1/10 /home/starmip/mvrac/T_SAFRAN_1959-2010.nc* /home/starmip/bfran/LSCE_These/COMPROMISE_Project/Data/my_Grid_SAFRAN_France.nc"))

Grid_SAFRAN_France=paste("/home/bfrancois/MBC_project/CycleGAN4/my_Grid_SAFRAN_France.nc")

# 
# ##### IPSLMRbili
# ### TAS
# system(paste0("cdo mergetime ",
#         "-remapbil,", Grid_SAFRAN_France ,
#         " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#         "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#         "-remapbil,", Grid_SAFRAN_France ,
#         " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#         "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#         "-remapbil,", Grid_SAFRAN_France ,
#         " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#         "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#         "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRbili/tas_day_IPSLMRbili_79_16_France.nc"))
# 
### PR
# system(paste0("cdo mergetime ",
#               "-remapbil,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#               "-remapbil,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#               "-remapbil,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#               "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRbili/pr_day_IPSLMRbili_79_16_France.nc"))

# 
# 
# 
# 
# ##### IPSLMRnn
# ### TAS
# system(paste0("cdo mergetime ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#               "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRnn/tas_day_IPSLMRnn_79_16_France.nc"))
# 
### PR
# system(paste0("cdo mergetime ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#               "-remapnn,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#               "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRnn/pr_day_IPSLMRnn_79_16_France.nc"))
# 
# 
# 
# ##### IPSLMRcub
# ### TAS
# system(paste0("cdo mergetime ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -addc,-273.15 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#               "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRcub/tas_day_IPSLMRcub_79_16_France.nc"))
# 
### PR
# system(paste0("cdo mergetime ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231.nc ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/historical/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231.nc ",
#               "-remapbic,", Grid_SAFRAN_France ,
#               " -selyear,1979/2016 -mulc,86400 -del29feb ",
#               "/bdd/CMIP5/output/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/pr/pr_day_IPSL-CM5A-MR_rcp85_r1i1p1_20060101-20551231.nc ",
#               "/home/bfrancois/MBC_project/CycleGAN4/Data/IPSLMRcub/pr_day_IPSLMRcub_79_16_France.nc"))





rm(list=ls())
### For IND_Paris, LON_ LAT_
load("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSL/tas_pr_day_IPSL_79_16_Paris.RData")
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSL/tas_pr_day_IPSL_79_16_Paris.RData")
rm(tas_day_IPSL_79_16_Paris)
rm(pr_day_IPSL_79_16_Paris)


gc()
library(ncdf4)
library(fields)
#### ne pas toucher

convert_netcdf_RData<-function(ncfile,var){
        ncname <- ncfile
        ncfname <- paste(ncname,".nc", sep="")
        ncin <- nc_open(ncfname)
        res=ncvar_get(ncin, var)
        nc_close(ncin) 
        return(res)
}

#### IPSLMRbili
setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRbili")
#setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRbili")

point_max=1:784

#### Load IPSLMRbili France
tas_day_IPSLMRbili_79_16_France=convert_netcdf_RData("tas_day_IPSLMRbili_79_16_France","tas")
pr_day_IPSLMRbili_79_16_France=convert_netcdf_RData("pr_day_IPSLMRbili_79_16_France","pr")

#### Select sub zone for Paris
tas_day_IPSLMRbili_79_16_Paris=tas_day_IPSLMRbili_79_16_France[(69-13):(69+14),(102-13):(102+14),]
pr_day_IPSLMRbili_79_16_Paris=pr_day_IPSLMRbili_79_16_France[(69-13):(69+14),(102-13):(102+14),]


save(tas_day_IPSLMRbili_79_16_Paris,
     pr_day_IPSLMRbili_79_16_Paris,
     LON_Paris,
     LAT_Paris,
     IND_Paris,
     point_max,
     file="tas_pr_day_IPSLMRbili_79_16_Paris.RData")



#### IPSLMRnn
setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRnn")
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRnn")

point_max=1:784

#### Load IPSLMRnn France
tas_day_IPSLMRnn_79_16_France=convert_netcdf_RData("tas_day_IPSLMRnn_79_16_France","tas")
pr_day_IPSLMRnn_79_16_France=convert_netcdf_RData("pr_day_IPSLMRnn_79_16_France","pr")

#### Select sub zone for Paris
tas_day_IPSLMRnn_79_16_Paris=tas_day_IPSLMRnn_79_16_France[(69-13):(69+14),(102-13):(102+14),]
pr_day_IPSLMRnn_79_16_Paris=pr_day_IPSLMRnn_79_16_France[(69-13):(69+14),(102-13):(102+14),]


save(tas_day_IPSLMRnn_79_16_Paris,
     pr_day_IPSLMRnn_79_16_Paris,
     LON_Paris,
     LAT_Paris,
     IND_Paris,
     point_max,
     file="tas_pr_day_IPSLMRnn_79_16_Paris.RData")



#### IPSLMRcub
setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRcub")
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRcub")

point_max=1:784

#### Load IPSLMRcub France
tas_day_IPSLMRcub_79_16_France=convert_netcdf_RData("tas_day_IPSLMRcub_79_16_France","tas")
pr_day_IPSLMRcub_79_16_France=convert_netcdf_RData("pr_day_IPSLMRcub_79_16_France","pr")

#### Select sub zone for Paris
tas_day_IPSLMRcub_79_16_Paris=tas_day_IPSLMRcub_79_16_France[(69-13):(69+14),(102-13):(102+14),]
pr_day_IPSLMRcub_79_16_Paris=pr_day_IPSLMRcub_79_16_France[(69-13):(69+14),(102-13):(102+14),]


save(tas_day_IPSLMRcub_79_16_Paris,
     pr_day_IPSLMRcub_79_16_Paris,
     LON_Paris,
     LAT_Paris,
     IND_Paris,
     point_max,
     file="tas_pr_day_IPSLMRcub_79_16_Paris.RData")





# 
# 
# #### Verif
# library(fields)
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRnn/tas_pr_day_IPSLMRnn_79_16_Paris.RData")
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRcub/tas_pr_day_IPSLMRcub_79_16_Paris.RData")
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLMRbili/tas_pr_day_IPSLMRbili_79_16_Paris.RData")
# 
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSLbili/tas_pr_day_IPSLbili_79_16_Paris.RData")
# 
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Temporal_indices_1979_2016.RData")
# tt=10
# par(mfrow=c(3,3))
# image.plot(LON_Paris, LAT_Paris, tas_day_IPSLMRbili_79_16_Paris[,,tt])
# image.plot(LON_Paris, LAT_Paris, tas_day_IPSLMRnn_79_16_Paris[,,tt])
# image.plot(LON_Paris, LAT_Paris, tas_day_IPSLMRcub_79_16_Paris[,,tt])
# image.plot(LON_Paris, LAT_Paris, tas_day_IPSLbili_79_16_Paris[,,tt])
# 
# 
# image.plot(LON_Paris, LAT_Paris, apply(tas_day_IPSLMRbili_79_16_Paris[,,Ind_winter_79_16],c(1,2),sd))
# image.plot(LON_Paris, LAT_Paris, apply(tas_day_IPSLMRnn_79_16_Paris[,,Ind_winter_79_16],c(1,2),sd))
# image.plot(LON_Paris, LAT_Paris, apply(tas_day_IPSLMRcub_79_16_Paris[,,Ind_winter_79_16],c(1,2),sd))
# image.plot(LON_Paris, LAT_Paris, apply(tas_day_IPSLbili_79_16_Paris[,,Ind_winter_79_16],c(1,2),sd))


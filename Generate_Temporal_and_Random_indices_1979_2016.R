# to run from obelix to get the right seed
#### Period: 20 ans
rm(list=ls())
set.seed(42) #For CV sample
library(timeDate)

#Dropping 29 of February for each year in the reference
DATES_regular_1979_2016 = atoms(timeSequence(from="1979-01-01",to="2016-12-31",by='day'))
head(DATES_regular_1979_2016)

Ind_29Feb_1979_2016 = which((DATES_regular_1979_2016[,2]==02) &
                              (DATES_regular_1979_2016[,3]==29))
DATES_365_1979_2016 = DATES_regular_1979_2016[-Ind_29Feb_1979_2016,] #remove rows with 29 Feb.


#For Seasons (meteorological)
#1985-2014
Ind_spring_79_16 =  which(DATES_365_1979_2016[,2]==03 |
                                DATES_365_1979_2016[,2]==04 |
                                DATES_365_1979_2016[,2]==05)
Ind_summer_79_16 = which(DATES_365_1979_2016[,2]==06|
                               DATES_365_1979_2016[,2]==07|
                               DATES_365_1979_2016[,2]==08)
Ind_automn_79_16 =  which(DATES_365_1979_2016[,2]==09|
                                DATES_365_1979_2016[,2]==10 |
                                DATES_365_1979_2016[,2]==11)
Ind_winter_79_16 =  which(DATES_365_1979_2016[,2]==12|
                                DATES_365_1979_2016[,2]==01 |
                                DATES_365_1979_2016[,2]==02)



#### CVunif: Random Indices for Calibration/projection
#### nb data in calib (pourcentage)
pct_calib=0.7

Ind_CVunif_Calib_winter_79_16 = sample(Ind_winter_79_16, size = floor(pct_calib*length(Ind_winter_79_16)) , replace = FALSE, prob = NULL)
Ind_CVunif_Proj_winter_79_16 = Ind_winter_79_16[!(Ind_winter_79_16 %in% Ind_CVunif_Calib_winter_79_16)]

Ind_CVunif_Calib_summer_79_16 = sample(Ind_summer_79_16, size = floor(pct_calib*length(Ind_summer_79_16)) , replace = FALSE, prob = NULL)
Ind_CVunif_Proj_summer_79_16 = Ind_summer_79_16[!(Ind_summer_79_16 %in% Ind_CVunif_Calib_summer_79_16)]

Ind_CVunif_Calib_spring_79_16 = sample(Ind_spring_79_16, size = floor(pct_calib*length(Ind_spring_79_16)) , replace = FALSE, prob = NULL)
Ind_CVunif_Proj_spring_79_16 = Ind_spring_79_16[!(Ind_spring_79_16 %in% Ind_CVunif_Calib_spring_79_16)]

Ind_CVunif_Calib_automn_79_16 = sample(Ind_automn_79_16, size = floor(pct_calib*length(Ind_automn_79_16)) , replace = FALSE, prob = NULL)
Ind_CVunif_Proj_automn_79_16 = Ind_automn_79_16[!(Ind_automn_79_16 %in% Ind_CVunif_Calib_automn_79_16)]


#### CVchrono: Split Indices for Calibration/projection
Ind_CVchrono_Calib_winter_79_16 = Ind_winter_79_16[1:floor(length(Ind_winter_79_16)*pct_calib)]
Ind_CVchrono_Proj_winter_79_16 = Ind_winter_79_16[!(Ind_winter_79_16 %in% Ind_CVchrono_Calib_winter_79_16)]

Ind_CVchrono_Calib_summer_79_16 = Ind_summer_79_16[1:floor(length(Ind_summer_79_16)*pct_calib)]
Ind_CVchrono_Proj_summer_79_16 = Ind_summer_79_16[!(Ind_summer_79_16 %in% Ind_CVchrono_Calib_summer_79_16)]

Ind_CVchrono_Calib_spring_79_16 = Ind_spring_79_16[1:floor(length(Ind_spring_79_16)*pct_calib)]
Ind_CVchrono_Proj_spring_79_16 = Ind_spring_79_16[!(Ind_spring_79_16 %in% Ind_CVchrono_Calib_spring_79_16)]

Ind_CVchrono_Calib_automn_79_16 = Ind_automn_79_16[1:floor(length(Ind_automn_79_16)*pct_calib)]
Ind_CVchrono_Proj_automn_79_16 = Ind_automn_79_16[!(Ind_automn_79_16 %in% Ind_CVchrono_Calib_automn_79_16)]

# setwd(dir="/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/")
setwd(dir="/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/")
save(Ind_spring_79_16,Ind_summer_79_16,Ind_automn_79_16,Ind_winter_79_16,
     #### CVunif
     Ind_CVunif_Calib_winter_79_16,Ind_CVunif_Proj_winter_79_16,
     Ind_CVunif_Calib_summer_79_16,Ind_CVunif_Proj_summer_79_16,
     Ind_CVunif_Calib_spring_79_16,Ind_CVunif_Proj_spring_79_16,
     Ind_CVunif_Calib_automn_79_16,Ind_CVunif_Proj_automn_79_16,
     #### CVchrono
     Ind_CVchrono_Calib_winter_79_16,Ind_CVchrono_Proj_winter_79_16,
     Ind_CVchrono_Calib_summer_79_16,Ind_CVchrono_Proj_summer_79_16,
     Ind_CVchrono_Calib_spring_79_16,Ind_CVchrono_Proj_spring_79_16,
     Ind_CVchrono_Calib_automn_79_16,Ind_CVchrono_Proj_automn_79_16,
     file="Temporal_and_Random_indices_1979_2016.RData")


#####################################################################################
#### Brouillon to not use

# #DATES_365_1979_2016 is used
# Ind_jan_1979_2016 = which((DATES_365_1979_2016[,2]==01))
# Ind_feb_1979_2016 = which((DATES_365_1979_2016[,2]==02))
# Ind_mar_1979_2016 = which((DATES_365_1979_2016[,2]==03))
# Ind_apr_1979_2016 = which((DATES_365_1979_2016[,2]==04))
# Ind_may_1979_2016 = which((DATES_365_1979_2016[,2]==05))
# Ind_jun_1979_2016 = which((DATES_365_1979_2016[,2]==06))
# Ind_jul_1979_2016 = which((DATES_365_1979_2016[,2]==07))
# Ind_aug_1979_2016 = which((DATES_365_1979_2016[,2]==08))
# Ind_sep_1979_2016 = which((DATES_365_1979_2016[,2]==09))
# Ind_oct_1979_2016 = which((DATES_365_1979_2016[,2]==10))
# Ind_nov_1979_2016 = which((DATES_365_1979_2016[,2]==11))
# Ind_dec_1979_2016 = which((DATES_365_1979_2016[,2]==12))
# 
# list_Ind_month_1979_2016<-list(Ind_jan_1979_2016,
#                                Ind_feb_1979_2016,
#                                Ind_mar_1979_2016,
#                                Ind_apr_1979_2016,
#                                Ind_may_1979_2016,
#                                Ind_jun_1979_2016,
#                                Ind_jul_1979_2016,
#                                Ind_aug_1979_2016,
#                                Ind_sep_1979_2016,
#                                Ind_oct_1979_2016,
#                                Ind_nov_1979_2016,
#                                Ind_dec_1979_2016)

# 
# list_Ind_season_1979_2016=list("Spring"=Ind_spring_1979_2016,
#                                "Summer"=Ind_summer_1979_2016,
#                                "Automn"=Ind_automn_1979_2016,
#                                "Winter"=Ind_winter_1979_2016,
#                                "Annual"=1:dim(DATES_365_1979_2016)[1])
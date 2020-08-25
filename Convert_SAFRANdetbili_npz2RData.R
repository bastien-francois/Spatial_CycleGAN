#### A lancer depuis JZ
rm(list=ls())

library(reticulate)
np <- import("numpy")
### ATTENTION PAS D'INSTALL DE MINICONDA A FAIRE!!! : reply n

data = np$load("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdetbili/tas_pr_day_SAFRANdetbili_79_16_Paris.npz")
data$files


tas_day_SAFRANdetbili_79_16_Paris= data$f[["tas_day_SAFRANdetbili_79_16_Paris"]]
pr_day_SAFRANdetbili_79_16_Paris= data$f[["pr_day_SAFRANdetbili_79_16_Paris"]]
IND_Paris = data$f[["IND_Paris"]]

LON_Paris = data$f[["LON_Paris"]]
LAT_Paris = data$f[["LAT_Paris"]]
point_max = data$f[["point_max"]]

setwd("/gpfswork/rech/eal/commun/CycleGAN/Data/SAFRANdetbili/")

save(tas_day_SAFRANdetbili_79_16_Paris, pr_day_SAFRANdetbili_79_16_Paris,
     LON_Paris,
     LAT_Paris,IND_Paris, point_max,
     file="tas_pr_day_SAFRANdetbili_79_16_Paris.RData"
)


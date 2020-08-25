##### Need of SAFRANdet France to derive SAFRANdetbili Paris
rm(list=ls())
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
library(fields)

#### France ####
load("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_France.RData")

#### SAFRANdet : remap conservatif vite fait
tas_day_SAFRANdet_79_16_France=array(NaN,dim=c(143,134,13870))
pr_day_SAFRANdet_79_16_France=array(NaN,dim=c(143,134,13870))

for(i in 1:13870){
  if(i%%777==0){print(i)}
  for(lon_ in 1:36){
    for(lat_ in 1:34){
      coord_lon=((lon_-1)*4+1):(lon_*4)
      coord_lat=((lat_-1)*4+1):(lat_*4)
      if(lon_==36){
        coord_lon=((lon_-1)*3+1):(lon_*3)
      }
      if(lat_==34){
        coord_lat=((lat_-1)*2+1):(lat_*2)
      }
      tas_day_SAFRANdet_79_16_France[coord_lon,coord_lat,i]<-mean(tas_day_SAFRAN_79_16_France[coord_lon,coord_lat,i],na.rm=TRUE)
      pr_day_SAFRANdet_79_16_France[coord_lon,coord_lat,i]<-mean(pr_day_SAFRAN_79_16_France[coord_lon,coord_lat,i],na.rm=TRUE)
    }
  }
}


setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRANdet/")
save(IND_France,
     LON_France,
     LAT_France,
     pr_day_SAFRANdet_79_16_France,
     tas_day_SAFRANdet_79_16_France,
     file="tas_pr_day_SAFRANdet_79_16_France.RData")

par(mfrow=c(1,2))
# image.plot(LON_France,LAT_France,pr_day_SAFRAN_79_16_France[,,13869],zlim=c(0,1))
# image.plot(LON_France,LAT_France,pr_day_SAFRANdet_79_16_France[,,13869],zlim=c(0,1))
image.plot(LON_France[50:60,50:60],LAT_France[50:60,50:60],pr_day_SAFRAN_79_16_France[50:60,50:60,13869],zlim=c(0,1))
image.plot(LON_France[50:60,50:60],LAT_France[50:60,50:60],pr_day_SAFRANdet_79_16_France[50:60,50:60,13869],zlim=c(0,1))














##### brouillon

#### Paris remapping bilinear with SAFRAN_Francedet ###
# 
# ##### Paris #### 
# #### SAFRANdet : remap conservatif vite fait
# tas_day_SAFRANdet_79_16_Paris=array(NaN,dim=c(28,28,13870))
# pr_day_SAFRANdet_79_16_Paris=array(NaN,dim=c(28,28,13870))
# 
# for(i in 1:13870){
#   if(i%%777==0){print(i)}
#   for(lon_ in 1:7){
#     for(lat_ in 1:7){
#       tas_day_SAFRANdet_79_16_Paris[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i]<-mean(tas_day_SAFRAN_79_16_Paris[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i])
#       pr_day_SAFRANdet_79_16_Paris[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i]<-mean(pr_day_SAFRAN_79_16_Paris[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i])
#     }
#   }
# }
# 
# 
# 
# par(mfrow=c(2,4))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris[,,30],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris[,,300],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris[,,40],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris[,,20],zlim=c(0,10))
# 
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRANdet_79_16_Paris[,,30],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRANdet_79_16_Paris[,,300],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRANdet_79_16_Paris[,,40],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRANdet_79_16_Paris[,,20],zlim=c(0,10))
# 
# par(mfrow=c(1,2))
# image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRANdet_79_16_Paris,c(1,2),mean),zlim=c(5,15))
# image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRAN_79_16_Paris,c(1,2),mean),zlim=c(5,15))
# 
# image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRANdet_79_16_Paris,c(1,2),sd),zlim=c(5,15))
# image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRAN_79_16_Paris,c(1,2),sd),zlim=c(5,15))
# 
# 
# 
# par(mfrow=c(2,4))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,30],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,300],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,40],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,20],zlim=c(0,10))
# 
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,30],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,300],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,40],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,20],zlim=c(0,10))
# 
# par(mfrow=c(1,2))
# image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRANdet_79_16_Paris,c(1,2),mean),zlim=c(0,2))
# image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRAN_79_16_Paris,c(1,2),mean),zlim=c(0,2))
# 
# image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRANdet_79_16_Paris,c(1,2),sd),zlim=c(0,5))
# image.plot(LON_Paris,LAT_Paris,apply(pr_day_SAFRAN_79_16_Paris,c(1,2),sd),zlim=c(0,5))

#setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRANdet/")
# setwd("/home/starmip/bfran/LSCE_These/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRANdet/")
# save(IND_Paris,
#      LON_Paris,
#      LAT_Paris,
#      pr_day_SAFRANdet_79_16_Paris,
#      tas_day_SAFRANdet_79_16_Paris,
#      file="tas_pr_day_SAFRANdet_79_16_Paris.RData")


# ### Verif
# rm(list=ls())
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRAN/tas_pr_day_SAFRAN_79_16_Paris.RData")
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/IPSL/tas_pr_day_IPSL_79_16_Paris.RData")
# 
# library(fields)
# library(RColorBrewer)
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Data/CMIP6_Data/SAFRANdet/tas_pr_day_SAFRANdet_79_16_Paris.RData")
# 
# load("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/Temporal_indices_1979_2016.RData")
# col_=rev(colorRampPalette(brewer.pal(11, "RdBu"))(64))[33:64]
# 
# par(mfrow=c(2,4))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,13869],zlim=c(0,1))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,13869],zlim=c(0,1))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,300],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,40],zlim=c(0,10))
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRANdet_79_16_Paris[,,20],zlim=c(0,10))
# 
# 
# rankSAFRAN=array(NaN,dim=c(28,28,length(Ind_winter_79_16)))
# rankSAFRANdet=array(NaN,dim=c(28,28,length(Ind_winter_79_16)))
# 
# for(i in 1:28){
#   for(j in 1:28){
#     rankSAFRAN[i,j,]=rank(tas_day_SAFRAN_79_16_Paris[i,j,Ind_winter_79_16],ties.method="min")/length(pr_day_SAFRAN_79_16_Paris[i,j,Ind_winter_79_16])
#     rankSAFRANdet[i,j,]=rank(tas_day_SAFRANdet_79_16_Paris[i,j,Ind_winter_79_16],ties.method="min")/length(pr_day_SAFRANdet_79_16_Paris[i,j,Ind_winter_79_16])
#   }
# }
# 
# par(mfrow=c(1,2))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[2370]],col=col_,zlim=c(-2,10))
# image.plot(LON_Paris,LAT_Paris,tas_day_SAFRANdet_79_16_Paris[,,Ind_winter_79_16[2370]],col=col_,zlim=c(0.1,0.95))
# mean(rankSAFRANdet[,,2100])
# 
# image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16],c(1,2),mean),col=col_,zlim=c(3,4.5))
# 
# 
#   image.plot(LON_Paris,LAT_Paris,apply(tas_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16],c(1,2),mean),col=col_)
# 
# 
#   which(apply(pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16],3,sum)<=0)
# 
# par(mfrow=c(2,4))
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,101],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,686],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,3420],zlim=c(0,0.5),col=col_)
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,101],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,686],zlim=c(0,0.5),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,3420],zlim=c(0,0.5),col=col_)
# 
# which(apply(pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16],3,sum)<=0)
# 
# 
# good_rankSAFRANdet=array(NaN,dim=c(28,28,length(Ind_winter_79_16)))
# for(i in 1:length(Ind_winter_79_16)){
#   if(i%%777==0){print(i)}
#   for(lon_ in 1:7){
#     for(lat_ in 1:7){
#       good_rankSAFRANdet[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i]<-mean(rankSAFRAN[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),i])
#     }
#   }
# }
# 
# 
# 
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/")
# pdf("dry_days_SAFRAN_SAFRANdet.pdf",widt=8,height=8)
# par(mfrow=c(4,5),mar=c(3, 3, 3, 1) + 0.1,mgp=c(4,2,0),oma=c(3,1,3,3),xpd=NA)
# #par(mfrow=c(3,5))
# 
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[1]],zlim=c(0,1),col=col_,main="day1",ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="SAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[148]],zlim=c(0,1),col=col_,main="day148",xlab="",ylab="",xaxt="n",yaxt="n")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[961]],zlim=c(0,1),col=col_,main="day961",xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[1737]],zlim=c(0,1),col=col_,main="day1737",xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[3301]],zlim=c(0,1),col=col_,main="day3301",xlab="",xaxt="n",yaxt="n",ylab="")
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="rankSAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,148],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,961],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1737],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,3301],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="rankSAFRANdet", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,148],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,961],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1737],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,3301],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# 
# 
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,1],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="upscale_rankSAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,148],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,961],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,1737],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,3301],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# dev.off()
# 
# 
# 
# which(apply(pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16],3,sum)>100)
# 
# 
# setwd("/homel/bfran/Bureau/Coronavirus/MBC_Project/ML_MBC/GAN/CycleGAN4/")
# pdf("wet_days_SAFRAN_SAFRANdet.pdf",widt=8,height=8)
# par(mfrow=c(4,5),mar=c(3, 3, 3, 1) + 0.1,mgp=c(4,2,0),oma=c(3,1,3,3),xpd=NA)
# #par(mfrow=c(3,5))
# 
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[4]],col=col_,main="day4",ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="SAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[343]],col=col_,main="day343",xlab="",ylab="",xaxt="n",yaxt="n")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[1689]],col=col_,main="day1689",xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[2111]],col=col_,main="day2111",xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[3415]],col=col_,main="day3415",xlab="",xaxt="n",yaxt="n",ylab="")
# 
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,4],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="rankSAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,343],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1689],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,2111],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,3415],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# 
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,4],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="rankSAFRANdet", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,343],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1689],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,2111],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,3415],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# 
# 
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,4],col=col_,ylab="",xlab="",xaxt="n",yaxt="n")
# title(ylab="upscale_rankSAFRAN", line=0, cex.lab=1.2)
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,343],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,1689],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,2111],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# image.plot(LON_Paris,LAT_Paris,good_rankSAFRANdet[,,3415],col=col_,xlab="",xaxt="n",yaxt="n",ylab="")
# dev.off()
# 
# 
# 
# 
# par(mfrow=c(3,5))
# 
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[1]],col=col_,main="day1")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[148]],col=col_,main="day148")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[961]],col=col_,main="day961")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[1737]],col=col_,main="day1737")
# image.plot(LON_Paris,LAT_Paris,pr_day_SAFRAN_79_16_Paris[,,Ind_winter_79_16[3301]],col=col_,main="day3301")
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,148],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,961],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,1737],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,3301],col=col_)
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,148],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,961],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,1737],col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,3301],col=col_)
# 
# 
# 
# new=matrix(NaN,nrow=28,ncol=28)
# for(lon_ in 1:7){
#   for(lat_ in 1:7){
#     new[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4)]<-mean(rankSAFRAN[((lon_-1)*4+1):(lon_*4),((lat_-1)*4+1):(lat_*4),2000])
#   }
# }
# 
# image.plot(LON_Paris,LAT_Paris,rankSAFRAN[,,2000],zlim=c(0,0.7),col=col_)
# image.plot(LON_Paris,LAT_Paris,rankSAFRANdet[,,2000],zlim=c(0,0.7),col=col_)
# image.plot(LON_Paris,LAT_Paris,new,zlim=c(0,0.7),col=col_)
# 
# ess=c()
# for(i in 1:length(Ind_winter_79_16)){
#   ess=c(ess,mean(abs(rankSAFRAN[,,i]-rankSAFRANdet[,,i])))
#   
# }
# 
# plot(ess,type='l')
# 
# 
# 
# 
# a=rnorm(100)
# rank_a=rank(a)/100
# 
# rank_b=runif(100,0,1)
# rank_b
# 
# plot(sort(rank_a))
# 
# plot(sort(rank_b))
# b_gan=unname(quantile(a,probs=rank_b))
# 
# b_gan
# a
# mean(b_gan)
# mean(a)

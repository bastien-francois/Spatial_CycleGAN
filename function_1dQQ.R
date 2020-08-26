
#### ne pas toucher #### 
# QQb<-function(Rc,Mc,Mp){
#   N=ncol(Mc)
#   I_Mc=nrow(Mc)
#   I_Mp=nrow(Mp)
#   if(is.null(N)&is.null(I_Mc)){
#     N=1
#     I_Mc=length(Mc)
#     I_Mp=length(Mp)
#     Rc=matrix(Rc,ncol=1)
#     Mc=matrix(Mc,ncol=1)
#     Mp=matrix(Mp,ncol=1)
#   }
#   Mch=matrix(rep(NA,I_Mc*N),ncol=N)
#   Mph=matrix(rep(NA,I_Mp*N),ncol=N)
#   for(k in 1:N){ #for each column (variable)
#     FMc=ecdf(Mc[,k])
#     Mch[,k]=quantile(Rc[,k],probs=FMc(Mc[,k]),type=7)
#     # Mph[,k]=quantile(Rc[,k],probs=FMc(Mp[,k]),type=7) not good for proj...
#     ### approx probs for Mph to avoid changing rank structure for projection period.
#     FMC=FMc(Mc[,k])
#     probs_FMc_Mp=approx(Mc[,k], FMC, Mp[,k], yleft = min(FMC), yright = max(FMC), 
#                         ties = "mean")$y
#     Mph[,k]=quantile(Rc[,k],probs=probs_FMc_Mp,type=7)
#   }
#   return(list(Mch=Mch,Mph=Mph))
# }

#### end ne pas toucher #### 

#Quantile-Quantile version that allow to correct Mp in CC context (out-of-sample values) (Gudmundsson et al., 2012)
# QQb_new<-function(Rc,Mc,Mp,p=0){ #the parameter p is necessary to implement QQ for MRec (see Pegram and Bardossy, 2012): p=0 classical QQ
#   N=ncol(Mc)
#   I_Mc=nrow(Mc)
#   I_Mp=nrow(Mp)
#   if(is.null(N)&is.null(I_Mc)){ #for 1d vectors
#     N=1
#     I_Mc=length(Mc)
#     I_Mp=length(Mp)
#     Rc=matrix(Rc,ncol=1)
#     Mc=matrix(Mc,ncol=1)
#     Mp=matrix(Mp,ncol=1)
#   }
#   Mch=matrix(rep(NA,I_Mc*N),ncol=N)
#   Mph=matrix(rep(NA,I_Mp*N),ncol=N)
#   for(k in 1:N){ #for each column (variable)
#     #Classic quantile-quantile for Mc
#     FMc=ecdf(Mc[,k])
#     Mch[,k]=quantile(Rc[,k],probs=FMc(Mc[,k])*(1-p)+p,type=4)
#     #Save the correction done for highest and lowest quantiles (will be used later to correct Mp in a context of climate change)
#     correc_high_quntl=max(Mc[,k])-max(Mch[,k])
#     correc_low_quntl=min(Mc[,k])-min(Mch[,k])
#     
#     #Quantile-quantile for Mp
#     # which value in Mp are within [min(Mc),max(Mc)]?
#     in_range=((Mp[,k]<=max(Mc[,k]))&(Mp[,k]>=min(Mc[,k])))
#     #for these values, classic quantile quantile
#     Mph[in_range,k]=quantile(Rc[,k],probs=FMc(Mp[in_range,k])*(1-p)+p,type=4)
#     #for out-of-sample values of Mp, same correction than for Mc
#     out_range_low=(Mp[,k]<min(Mc[,k]))
#     out_range_high=(Mp[,k]>max(Mc[,k]))
#     Mph[out_range_low,k]<-Mp[out_range_low,k]-correc_low_quntl
#     Mph[out_range_high,k]<-Mp[out_range_high,k]-correc_high_quntl
#   }
#   return(list(Mch=Mch,Mph=Mph))
# }

#Quantile-Quantile version that allow to correct Mp in CC context (out-of-sample values) (Gudmundsson et al., 2012)
QQb_new<-function(Rc,Mc,Mp){ 
  N=ncol(Mc)
  I_Mc=nrow(Mc)
  I_Mp=nrow(Mp)
  if(is.null(N)&is.null(I_Mc)){ #for 1d vectors
    N=1
    I_Mc=length(Mc)
    I_Mp=length(Mp)
    Rc=matrix(Rc,ncol=1)
    Mc=matrix(Mc,ncol=1)
    Mp=matrix(Mp,ncol=1)
  }
  Mch=matrix(rep(NA,I_Mc*N),ncol=N)
  Mph=matrix(rep(NA,I_Mp*N),ncol=N)
  for(k in 1:N){ #for each column (variable)
    #Classic quantile-quantile for Mc
    FMc=ecdf(Mc[,k])
    FMC=FMc(Mc[,k])
    FRc=ecdf(Rc[,k])
    FRC=FRc(Rc[,k])
    # Mch[,k]=quantile(Rc[,k],probs=FMc(Mc[,k]),type=7)
    Mch[,k]=approx(FRC,Rc[,k], FMC, yleft=min(Rc[,k]), yright= max(Rc[,k]))$y
    #Save the correction done for highest and lowest quantiles (will be used later to correct Mp in a context of climate change)
    correc_high_quntl=max(Mc[,k])-max(Mch[,k])
    correc_low_quntl=min(Mc[,k])-min(Mch[,k])
    
    #Quantile-quantile for Mp
    # which value in Mp are within [min(Mc),max(Mc)]?
    in_range=((Mp[,k]<=max(Mc[,k]))&(Mp[,k]>=min(Mc[,k])))
    #for these values, classic quantile quantile with approximation for interpolation
    # Mph[in_range,k]=quantile(Rc[,k],probs=FMc(Mp[in_range,k])*(1-p)+p,type=4)
    
    probs_FMc_Mp=approx(Mc[,k], FMC, Mp[in_range,k], yleft = min(FMC), yright = max(FMC), 
                        ties = "mean")$y
    # Mph[in_range,k]=quantile(Rc[,k],probs=probs_FMc_Mp,type=7)
    Mph[in_range,k]=approx(FRC,Rc[,k],probs_FMc_Mp,  yleft=min(Rc[,k]), yright= max(Rc[,k]))$y
    
    #for out-of-sample values of Mp, same correction than for Mc (Boe, Deque 2007)
    out_range_low=(Mp[,k]<min(Mc[,k]))
    out_range_high=(Mp[,k]>max(Mc[,k]))
    Mph[out_range_low,k]<-Mp[out_range_low,k]-correc_low_quntl
    Mph[out_range_high,k]<-Mp[out_range_high,k]-correc_high_quntl
  }
  return(list(Mch=Mch,Mph=Mph))
}
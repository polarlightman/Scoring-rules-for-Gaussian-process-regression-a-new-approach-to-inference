
#####Mean CRPS
m_CRPS <- function(data,mu,sigma){
  stand_term <- (data-mu)/sigma
  crps <- sigma*(stand_term*(2*pnorm(stand_term)-1)+2*dnorm(stand_term)-1/sqrt(pi))
  my_crps <- mean(crps)
  return (my_crps)
}
#####Mean log score
m_Log_score <- function(data,mu,sigma){
  ls=mean(-log(dnorm(data,mu,sigma)))
  return (ls)
}
#####RBF kernel
rbf <- function(X1,X2,l=1,k=1) {  #x1 and x2 can be the same 
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      Sigma[i,j] <- k^2*exp(-0.5*((X1[i]-X2[j])/l)^2) #squared exponential function (l=1)
      }
  } 
  return(Sigma)
}
#####cholesky solve linear system
chole_solve <- function(B,A){
  U <- chol(A)
  res <- solve(U,solve(t(U),B))
  return (res)
}

#####creat data
library(MASS)
num_train=20
noise=0.1
x<- seq(-6,6,length.out=num_train)
sigma_cov <- rbf(x,x,l=1,k=1)
noise_term <- matrix(rnorm(num_train)*noise,nrow=num_train)
y <- mvrnorm(1,rep(0,num_train),sigma_cov)+noise_term
dim(y) <- c(num_train,1)
plot(x,y)

#####Using LOOCV-CRPS
cal_m_crps <- function(i,j){
      k_ff <- rbf(x,x,l=i)
      big_k <- k_ff+diag(num_train)*j^2
      k_ii_diag=diag(chole_solve(diag(num_train),big_k))
      dim(k_ii_diag)=c(num_train,1)
      mean_term=y-chole_solve(y,big_k)/k_ii_diag
      cov_term=1/k_ii_diag
      sigma=sqrt(cov_term)
      mean_crps <-m_CRPS(y,mean_term,sigma) 
      return (mean_crps)
        }
#####calculate wrong method from GP
wrong_cal_m_crps <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  inverse_term <- k_ff+diag(num_train)*j^2
  mean_term=k_ff%*%chole_solve(y,inverse_term)
  cov_term=diag(diag(num_train)*j^2+k_ff-k_ff%*%chole_solve(k_ff,inverse_term))
  dim(cov_term)=c(num_train,1)
  sigma=sqrt(cov_term)
  mean_crps <-m_CRPS(y,mean_term,sigma) 
  return (mean_crps)
}


#####USing NLML
cal_NLML <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  inverse_term <- k_ff+diag(num_train)*j^2
  NLML <- 0.5*t(y)%*%chole_solve(y,inverse_term)+0.5*log(det(inverse_term))+num_train/2*log(2*pi)
  return (NLML)
}
#####USing log score
cal_m_logs <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  big_k <- k_ff+diag(num_train)*j^2
  k_ii_diag=diag(chole_solve(diag(num_train),big_k))
  dim(k_ii_diag)=c(num_train,1)
  mean_term=y-chole_solve(y,big_k)/k_ii_diag
  cov_term=1/k_ii_diag+j^2
  sigma=sqrt(cov_term)
  ls=m_Log_score(y,mean_term,sigma)
  return (ls)
}

#####length range
l_range <- seq(0.01,2,length.out = 50)

{
crps_series <- unlist(lapply(l_range,cal_m_crps))

logs_series <- unlist(lapply(l_range,cal_m_logs))

NLML_series <- unlist(lapply(l_range,cal_NLML,j=0.1))


plot(l_range[1:120],(crps_series)[1:120],type='l')
lines(l_range,(crps_series),type='l',lty=2)
lines(l_range,(logs_series),type='l',lty=3)
lines(l_range,(logs_series),type='l',lty=4)


plot(l_range,logs_series,type='l',col='blue')
plot(l_range,NLML_series,type='l',col='Red')
l_range[which.min(crps_series)]
}

noise_range <- seq(0.01,1,length.out = 50)



NLML_res <- sapply(l_range, function(xx) mapply(cal_NLML,xx,noise_range))
ma_NLML <- matrix(NLML_res,nrow=50)
#contour(noise_range,l_range,ma_NLML,nlevels=2000)
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='NLML',xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)


wrong_crps_res <- sapply(l_range, function(xx) mapply(wrong_cal_m_crps,xx,noise_range))
wrong_ma_crps <- matrix(wrong_crps_res,nrow=50)
contour(noise_range,l_range,wrong_ma_crps,nlevels=100)
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='Wrong Method Using CRPS', xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)


crps_res <- sapply(l_range, function(xx) mapply(cal_m_crps,xx,noise_range))
ma_crps <- matrix(crps_res,nrow=50)
contour(noise_range,l_range,ma_crps,nlevels=100)
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='CRPS', xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)



logs_res <- sapply(l_range, function(xx) mapply(cal_m_logs,xx,noise_range))
ma_logs <- matrix(logs_res,nrow=50)
#contour(noise_range,l_range,ma_logs,nlevels=800)
contour(noise_range,l_range,ma_logs,levels = c(seq(-10,0,by=0.1),seq(0, 1, by = 0.1),seq(1, 10, by = 0.1)))
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='Log_Score', xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)








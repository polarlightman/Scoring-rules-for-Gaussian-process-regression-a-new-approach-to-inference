m_CRPS <- function(data,mu,sigma){
  stand_term <- (data-mu)/sigma
  crps <- sigma*(stand_term*(2*pnorm(stand_term)-1)+2*dnorm(stand_term)-1/sqrt(pi))
  my_crps <- mean(crps)
  return (my_crps)
}

m_Log_score <- function(data,mu,sigma){
  ls=mean(-log(dnorm(data,mu,sigma)))
  return (ls)
}

rbf <- function(X1,X2,l=1,k=1) {  #x1 and x2 can be the same 
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      Sigma[i,j] <- k^2*exp(-0.5*((X1[i]-X2[j])/l)^2) #squared exponential function (l=1)
      }
  } 
  return(Sigma)
}

chole_solve <- function(B,A){
  U <- chol(A)
  res <- solve(U,solve(t(U),B))
  return (res)
}


library(MASS)
num_train=20
noise=0.1
x<- seq(-6,6,length.out=num_train)
sigma_cov <- rbf(x,x,l=1,k=1)
#set.seed(3311233)
noise_term <- matrix(rnorm(num_train)*noise,nrow=num_train)
y <- mvrnorm(1,rep(0,num_train),sigma_cov)+noise_term
dim(y) <- c(num_train,1)
plot(x,y)


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



cal_NLML <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  inverse_term <- k_ff+diag(num_train)*j^2
  NLML <- 0.5*t(y)%*%chole_solve(y,inverse_term)+0.5*log(det(inverse_term))+num_train/2*log(2*pi)
  return (NLML)
}

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
contour(noise_range,l_range,ma_NLML,levels = c(9,seq(10, 20, by = 1),seq(20, 100, by = 1)))
#this function is really strange, the plot will change according to x and y axis
#noise_range is on the x axis
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='NLML',xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)


wrong_crps_res <- sapply(l_range, function(xx) mapply(wrong_cal_m_crps,xx,noise_range))
wrong_ma_crps <- matrix(wrong_crps_res,nrow=50)
contour(noise_range,l_range,wrong_ma_crps,nlevels=100)
#contour(noise_range,l_range,ma_crps,levels = c(seq(0,1e-2,length.out = 10),seq(1e-2, 0.1, by = 0.01),seq(0.1, 0.3, by = 0.02)))
axis(1, at = 0.1);axis(2, at = 0.3)
abline(h=1,col='red',lty=2);abline(v=0.1,col='red',lty=2)
title(main='Wrong Method Using CRPS', xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)


crps_res <- sapply(l_range, function(xx) mapply(cal_m_crps,xx,noise_range))
ma_crps <- matrix(crps_res,nrow=50)
contour(noise_range,l_range,ma_crps,nlevels=100)
#contour(noise_range,l_range,ma_crps,levels = c(seq(0,0.1,length.out = 100),seq(1e-1, 0.2, by = 0.01),seq(0.2, 1, by = 0.015)))
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



#####
calc_es <- function(X1,X2,n,beta) {  
  RES <- matrix(rep(0, n*n), nrow=n)
  for (i in 1:nrow(RES)) {
    for (j in 1:ncol(RES)) {
      RES[i,j] <-sum((abs(X1[i,]-X2[j,]))^beta)
    }
  }
  return(RES)
}
ES <- function(i,j){
  beta=1
  n=300
  k_ff <- rbf(x,x,l=i)
  inverse_term <- k_ff+diag(num_train)*j^2
  mu <- k_ff%*%chole_solve(y,inverse_term)
  sigma <- diag(num_train)*j^2+k_ff-k_ff%*%chole_solve(k_ff,inverse_term)
  x1 <- mvrnorm(n,mu,sigma)
  x2 <- mvrnorm(n,mu,sigma)
  yy<- y
  dim(yy) <- c(1,num_train)
  first_term <- mean(rowSums((abs(sweep(x1,2,yy)))^beta))
  second_term <- mean(calc_es(x1,x2,n,beta))
  Es_value <- first_term-1/2*second_term
  return(Es_value)
}

ES_res <- sapply(l_range, function(xx) mapply(ES,xx,noise_range))
ma_ES <- matrix(ES_res,nrow=50)
contour(noise_range,l_range,ma_ES,nlevels=30)
#contour(noise_range,l_range,ma_ES,levels=c(seq(0,1,by=0.1),seq(1,15,by=0.1)))
axis(1, at = 0.1);axis(2, at = 0.3)
title(main='ES', xlab="Noise s.d", ylab="Length Scale",cex.lab=1.5)



cal_new <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  mean_term <- matrix(0,nrow=num_train)
  cov_term <- diag(diag(num_train)*j^2+k_ff)
  mean_crps <-m_CRPS(y,mean_term,cov_term) 
  return (mean_crps)
}


new_res <- sapply(l_range, function(xx) mapply(cal_m_crps,xx,noise_range))
ma_new <- matrix(new_res,nrow=50)
contour(noise_range,l_range,ma_new,nlevels=100)




library(mvtnorm)
cal_dss <- function(i,j){
  k_ff <- rbf(x,x,l=i)
  inverse_term <- k_ff+diag(num_train)*j^2
  mean_term <- as.vector(k_ff%*%chole_solve(y,inverse_term))
  cov_term <- as.matrix(diag(num_train)*j^2+k_ff-k_ff%*%chole_solve(k_ff,inverse_term))
  dss=-dmvnorm(as.vector(y),mean_term,cov_term)
  return (dss)
}

dss_res <- sapply(l_range, function(xx) mapply(cal_dss,xx,noise_range))
ma_dss <- matrix(dss_res,nrow=50)
contour(noise_range,l_range,ma_logs,nlevels=300)




"x
[1] -6.0000000 -5.3684211 -4.7368421 -4.1052632 -3.4736842 -2.8421053 -2.2105263 -1.5789474
[9] -0.9473684 -0.3157895  0.3157895  0.9473684  1.5789474  2.2105263  2.8421053  3.4736842
[17]  4.1052632  4.7368421  5.3684211  6.0000000

y
[,1]
[1,]  0.96136739
[2,]  0.77644159
[3,] -0.12782728
[4,] -0.54429729
[5,] -0.53178511
[6,] -0.85245527
[7,] -1.64344410
[8,] -1.93483368
[9,] -1.72917909
[10,] -1.18503091
[11,] -1.48936162
[12,] -1.85997702
[13,] -1.39679480
[14,] -1.01274748
[15,] -0.86193156
[16,] -0.09387182
[17,] -0.10930889
[18,] -1.17751555
[19,] -2.41951809
[20,] -3.02563284"



aaa=matrix(c(1,2,3,7,9,4,8,1,5),byrow=TRUE,nrow=3)
bbb=matrix(c(1,7,1,2,9,5,1,3,5),byrow=TRUE,nrow=3)




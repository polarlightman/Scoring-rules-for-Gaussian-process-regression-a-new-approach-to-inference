###################
#   Global parameter#########
###################
true_mu <- 0
true_sigma_sq <- 1
pre_mu <- seq(-5,5,by=0.5)
pre_sigma_sq <- c(seq(0.05,1,by=0.1),seq(1,10,by=1))
rho <- 0.5

normalized_mean_error <- (true_mu-pre_mu)/true_sigma_sq
normalized_var_error <- (true_sigma_sq-pre_sigma_sq)/true_sigma_sq

num_data <- 500
nn <- 2

true_corr <- seq(0.2,0.8,by=0.2)
range_corr <- seq(0,0.9,by=0.1)
###################
###############################################################################################
###################
#  Functions #########
###################
CRPS <- function(data,mu,sigma){
  stand_term <- (data-mu)/sigma
  my_crps <- sigma*(stand_term*(2*pnorm(stand_term)-1)+2*dnorm(stand_term)-1/sqrt(pi))
  return (my_crps)
}
Log_score <- function(data,mu,sigma){
  ls=-log(dnorm(data,mu,sigma))
  return (ls)
}
calc_es <- function(X1,X2,n,beta=1) {  
  RES <- matrix(rep(0, n*n), nrow=n)
  for (i in 1:nrow(RES)) {
    for (j in 1:ncol(RES)) {
      RES[i,j] <-sum((abs(X1[i,]-X2[j,]))^beta)
    }
  }
  return(RES)
}
ES <- function(n=100,n_mu=2,mu=0,sigma,beta=1,k){
  data <- data1[k,]
  x1 <- mvrnorm(n,rep(mu,n_mu),sigma)
  x2 <- mvrnorm(n,rep(mu,n_mu),sigma)
  first_term <- mean(rowSums((abs(x1-data))^beta))
  second_term <- mean(calc_es(x1,x2,n,beta))
  Es_value <- first_term-1/2*second_term
  return(Es_value)
}
replace_diag <- function(k){
  my_cov <- diag(nn)
  for (i in (1:nn)){
    for (j in (1:nn)){
      if (i==j){
        my_cov[i,j] <- 1 *k
      }else{
        my_cov[i,j] <- rho*k
      }
    }
  }
  return(my_cov)
}
replace_corr <- function(k){
  my_cov <- diag(nn)
  for (i in (1:nn)){
    for (j in (1:nn)){
      if (i==j){
        my_cov[i,j] <- true_sigma_sq*1
      }else{
        my_cov[i,j] <- true_sigma_sq*k
      }
    }
  }
  return(my_cov)
}
###################

##################################################################
#####     CRPS       ################
y <- rnorm(10000,0,1)
relative_change_CRPS_mean_error <- sapply(pre_mu,function(i) mean(CRPS(y,i,sqrt(true_sigma_sq))))
plot(normalized_mean_error,relative_change_CRPS_mean_error,type='b',pch=16,
     ylab='Relative change in CRPS',xlab='Normalized mean error',cex=1.5,cex.lab=1.5)

relative_change_CRPS_var_error <- sapply(sqrt(pre_sigma_sq),function(i) mean(CRPS(y,0,i)))
plot(normalized_var_error ,relative_change_CRPS_var_error,type='b',pch=16,
     ylab='Relative change in CRPS',xlab='Normalized variance error',cex=1.5,cex.lab=1.5)

##################################################################
#####     logs      ################
relative_change_logs_mean_error <- sapply(pre_mu,function(i) mean(Log_score(y,i,sqrt(true_sigma_sq))))
plot(normalized_mean_error,relative_change_logs_mean_error,type='b',pch=17,
     ylab='Relative change in Log Score',xlab='Normalized mean error',cex=1.5,cex.lab=1.5)

relative_change_logs_var_error <- sapply(sqrt(pre_sigma_sq),function(i) mean(Log_score(y,0,i)))
plot(normalized_var_error,relative_change_logs_var_error,type='b',pch=17,
     ylab='Relative change in Log Score',xlab='Normalized variance error',cex=1.5,cex.lab=1.5)
##################################
##################################################################
#####     dss      ################
library(mvtnorm)
data1 <- mvrnorm(num_data,rep(0,nn),replace_diag(true_sigma_sq))
#
my_res <- 0
f1 <- function(i){
  for (j in 1:num_data){
    my_res[j]=-log(dmvnorm(data1[j,],rep(i,nn),replace_diag(true_sigma_sq)))
  }
  return (mean(my_res))
}
truth<- f1(0)

a <- sapply(pre_mu, function(i) (f1(i)-truth)/truth)
plot(normalized_mean_error,a,type='b',pch=18,
     ylab='Relative change in DSS',xlab='Normalized mean error',cex=1.5,cex.lab=1.5)

#
f2 <- function(i){
  for (j in 1:num_data){
    my_res[j]=-log(dmvnorm(data1[j,],rep(0,nn),replace_diag(i)))
  }
  return (mean(my_res))
}
truth <- f2(true_sigma_sq)

b <- sapply(pre_sigma_sq, function(i) (f2(i)-truth)/truth)
plot(normalized_var_error,b,type='b',pch=18,
     ylab='Relative change in DSS',xlab='Normalized var error',cex=1.5,cex.lab=1.5)
#

f3 <- function(i){
  for (j in 1:num_data){
    my_res[j]=-log(dmvnorm(data1[j,],rep(0,nn),replace_corr(i)))
  }
  return (mean(my_res))
}
res_corr <- matrix(rep(0,length(true_corr)*length(range_corr)),nrow=length(true_corr),ncol=length(range_corr))

for( w in (true_corr*10)){
  ww <-w/10
  data1 <- mvrnorm(num_data,rep(0,nn),replace_corr(ww))
  truth_dss <- f3(ww)
  res_corr[w/2,] <- sapply(range_corr, function(i) (f3(i)-truth_dss)/truth_dss)
}

plot(range_corr, res_corr[1,]  ,type='b',pch=18,col=1,
     ylab='Relative change in DSS',xlab='Predictive Correlation',cex=1.5,cex.lab=1.5)
lines(range_corr,res_corr[2,],type='b',pch=19,col=2,cex=1.5,cex.lab=1.5)
lines(range_corr,res_corr[3,],type='b',pch=20,col=3,cex=1.5,cex.lab=1.5)
lines(range_corr,res_corr[4,],type='b',pch=21,col=4,cex=1.5,cex.lab=1.5)
legend("topleft",text.width = 1.5*strwidth(expression(paste(rho, " = ", 0.4))),
       legend = c(expression(paste(rho, " = ", 0.2)),
                            expression(paste(rho, " = ", 0.4)),
                            expression(paste(rho, " = ", 0.6)),
                            expression(paste(rho, " = ", 0.8))), pch = 18:21,col=1:4,
                            cex=1.2)
##################################################################
#####     ES      ################



data1 <- mvrnorm(num_data,rep(0,nn),truth)

my_res <- 0
f1 <- function(i){
  for (j in 1:num_data){
    my_res[j]=(ES(n=100,n_mu=2,mu=i,replace_diag(true_sigma_sq),beta=1,j))
  }
  return (mean(my_res))
}

truth <- f1(0)
a <- sapply(pre_mu, function(i) (f1(i)-truth)/truth)
plot(normalized_mean_error,a,type='b',pch=18,
     ylab='Relative change in ES',xlab='Normalized mean error',cex=1.5,cex.lab=1.5)

#
f2 <- function(i){
  for (j in 1:num_data){
    my_res[j]=(ES(n=100,n_mu=2,mu=0,replace_diag(i),beta=1,j))
  }
  return (mean(my_res))
}
truth <- f2(1)

b <- sapply(pre_sigma_sq[6:length(pre_sigma_sq)], function(i) (f2(i)-truth)/truth)
plot(normalized_var_error[6:length(pre_sigma_sq)],b,type='b',pch=18,
     ylab='Relative change in ES',xlab='Normalized var error',cex=1.5,cex.lab=1.5)
#
f3 <- function(i){
  for (j in 1:num_data){
    my_res[j]=(ES(n=100,n_mu=2,mu=0,replace_corr(i),beta=1,j))
  }
  return (mean(my_res))
}
res_corr <- matrix(rep(0,length(true_corr)*length(range_corr)),nrow=length(true_corr),ncol=length(range_corr))

for( w in (true_corr*10)){
  ww <- w/10
  data1 <- mvrnorm(num_data,rep(0,nn),replace_corr(ww))
  truth_es <- f3(ww)
  res_corr[w/2,] <- sapply(range_corr, function(i) (f3(i)-truth_es)/truth_es)
}

plot(range_corr, res_corr[1,]  ,type='b',pch=18,col=1,
     ylab='Relative change in ES',xlab='Predictive Correlation',cex=1.5,cex.lab=1.5)
lines(range_corr,res_corr[2,],type='b',pch=19,col=2)
lines(range_corr,res_corr[3,],type='b',pch=20,col=3)
lines(range_corr,res_corr[4,],type='b',pch=21,col=4)
legend("topright",text.width = 1.5*strwidth(expression(paste(rho, " = ", 0.4))),
       legend = c(expression(paste(rho, " = ", 0.2)),
                             expression(paste(rho, " = ", 0.4)),
                             expression(paste(rho, " = ", 0.6)),
                             expression(paste(rho, " = ", 0.8))), pch = 18:21,col=1:4)








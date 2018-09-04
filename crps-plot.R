

par(mfrow=c(1,2))
k <- rnorm(10000)
plot(ecdf(k),lwd=3,col='red',xlab='',ylab='CDF',main='Probabilistic Forecast')
x <- seq(-5,5,length=100000)
y <- pnorm(x)
x0 <- 0.5
lines(c(-5,x0),c(0,0),type='l',lwd=3)
lines(c(x0,x0),c(0,1),type='l',lwd=3)
lines(c(x0,5),c(1,1),type='l',lwd=3)

polygon(c( x[x<=x0], x0 ), 
        c(y[x<=x0],0 ), col="gray87")
polygon(c( x[x>=x0], x0 ), 
        c(y[x>=x0],1 ), col="gray87")
lines(ecdf(k),lwd=3,col='red')



plot(c(-5,5),c(0,1),xlab='',ylab='CDF',main='Deterministic Forecast',type='n')
abline(h=1,lty=2);abline(h=0,lty=2)
yjitter1 <-0.997
yjitter2 <--0.003 
lines(c(-5,x0),c(yjitter2,yjitter2),type='l',lwd=4)
lines(c(x0,x0),c(0,1),type='l',lwd=4)
lines(c(x0,5),c(yjitter1,yjitter1),type='l',lwd=4)

xp <- -1.4
lines(c(-5,xp),c(0,0),type='l',lwd=4,col='red')
lines(c(xp,xp),c(0,1),type='l',lwd=6,col='red')
lines(c(xp,5),c(1,1),type='l',lwd=4,col='red')

polygon(c( xp,x0, x0,xp ), 
        c(0,0,0.999,0.999), col="gray87")


###############################################






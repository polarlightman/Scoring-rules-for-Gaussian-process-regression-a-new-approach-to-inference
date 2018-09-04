import torch 
import math
import numpy as np
from torch import nn
from matplotlib import pyplot as plt

def ARD(x, xp, a, b):
    b=b.view(1,-1)
    scale_sq = torch.exp(a)#trick to make interesting parameter positive
    length_dim = torch.exp(b)
    x=x/length_dim
    xp=xp/length_dim
    n, d = x.size()
    m, d = xp.size()
    res = 2 * torch.mm(x, xp.transpose(0,1))
    x_sq = torch.bmm(x.view(n, 1, d), x.view(n, d, 1))
    xp_sq = torch.bmm(xp.view(m, 1, d), xp.view(m, d, 1))
    x_sq = x_sq.view(n, 1).expand(n, m)
    xp_sq = xp_sq.view(1, m).expand(n, m)
    res = res - x_sq - xp_sq
    res = (0.5 * res) 
    res = scale_sq.expand_as(res) * torch.exp(res)
    return res 


def exp_ARD(x, xp, a, b):
    b=b.view(1,-1)
    scale_sq = torch.exp(a)#trick to make interesting parameter positive
    length_dim = torch.exp(b).pow(0.5)
    x=x/length_dim
    xp=xp/length_dim
    n, d = x.size()
    m, d = xp.size()
    res = 2 * torch.mm(x, xp.transpose(0,1))
    x_sq = torch.bmm(x.view(n, 1, d), x.view(n, d, 1))
    xp_sq = torch.bmm(xp.view(m, 1, d), xp.view(m, d, 1))
    x_sq = x_sq.view(n, 1).expand(n, m)
    xp_sq = xp_sq.view(1, m).expand(n, m)
    res= -res + x_sq + xp_sq
    if n==m:
        d=(torch.diag(res-(1e-12)))*torch.eye(n)
        res =(res-d).pow(0.5)
    else:
        res =(res).pow(0.5)
    res = (-0.5 * res) 
    res = scale_sq.expand_as(res) * torch.exp(res)
    return res 


def chol_solve(B,A):
     c=torch.potrf(A)
     s1=torch.gesv(B, c.transpose(0,1))[0]
     s2=torch.gesv(s1,c)[0]
     return s2

#a=train_x
#u=inducing_x
#b=train_x
def Q(a,u,b):
    K_au=ARD(a,u,para_k,para_l)
    K_uu=ARD(u,u,para_k,para_l)
    #K_uu_inverse=(K_uu).inverse()
    K_uu_jittered=(K_uu+0.001*torch.eye(K_uu.shape[0]).type(dtype))
    K_ub=ARD(u,b,para_k,para_l)
    Q_ab=K_au.mm(chol_solve(K_ub,K_uu_jittered))
    return Q_ab

def ES(m,c,shape1,data_y,num_sim=300,beta=1):
    mean_term=m              
    cov_term=c
           
    eigen_value=torch.diag(torch.svd(cov_term)[1])
    eigen_vector=torch.svd(cov_term)[0]
    #eigen_vector.mm(eigen_value).mm(eigen_vector.transpose(0,1))
    root_cov=eigen_vector.mm(eigen_value.pow(0.5)).mm(eigen_vector.transpose(0,1))
    #root_cov.mm( root_cov.transpose(0,1))            
    z=torch.randn(num_sim,shape1).mm(root_cov)
    zp=torch.randn(num_sim,shape1).mm(root_cov)
    
    Nz0 = z.size(0)
    Nzp0 = zp.size(0)
    Nz1 = zp.size(1)
    
    zz = z.unsqueeze(1).expand(Nz0, Nzp0, Nz1)
    zzpp= zp.unsqueeze(0).expand(Nz0, Nzp0, Nz1)
    Dist_zzp= torch.pow(zz - zzpp, 2).sum(2).pow(0.5)
    Dist_zzp_beta=Dist_zzp.pow(beta)    
    
    z_minus_zp=(Dist_zzp_beta.sum(0).sum(0))/(num_sim*(num_sim-1))      
    mu_minus_y=(mean_term-data_y).transpose(0,1).expand_as(z)
    Ny0=mu_minus_y.size(0)   
    zz2=z.unsqueeze(1).expand(Nz0, Ny0, Nz1)
    muy = mu_minus_y.unsqueeze(0).expand(Nz0, Ny0, Nz1)
    Dist_zy= torch.pow(zz2 - muy, 2).sum(2).pow(0.5)
    Dist_zy_beta=Dist_zy.pow(beta)/num_sim
    z_minus_y=Dist_zy_beta.sum(0)[0]
    
    ES_res=z_minus_y-0.5*z_minus_zp
    return ES_res



def spgp_cal_mean_and_cov(k1,Q1,Q2,k2,num_test,num_jitter,data_y):
     G=(torch.diag(
               k1-Q1+sigma_noise_sq*torch.eye(num_jitter)
                  )*torch.eye(num_jitter)
                    ).type(dtype)
     mean_term=Q2.mm(chol_solve(data_y,Q1+G))
     cov_term=sigma_noise_sq*torch.eye(num_test).type(dtype)+k2-Q2.mm(chol_solve(Q2.t(),Q1+G))
     return mean_term,cov_term



def logs(m,c,data_y):
    mean_term=m
    cov_term=c         
    first_term=(data_y-mean_term).pow(2)/(2*cov_term)
    logs_ave=(first_term+cov_term.pow(0.5).log()+torch.tensor([0.5*np.log((2*math.pi))]).expand_as(data_y)).mean()
    return logs_ave


def crps(m,c,data_y):
    mean_term=m
    cov_term=c 
    CRPS_sigma=cov_term.pow(0.5)
    stand_term=(data_y-mean_term)/CRPS_sigma     #(y-mu)/sigma
    norm_cdf=0.5*(1+torch.erf(stand_term/math.sqrt(2)))   #cdf of standrd normal
    norm_pdf=(1/math.sqrt(2*math.pi))*torch.exp(-stand_term.pow(2)/2)  #pdf of standrd normal        
    CRPS_ave=(CRPS_sigma*(stand_term*(2*norm_cdf-1)+2*norm_pdf-1/math.sqrt(math.pi))).mean()
    return CRPS_ave


def dss(m,c,shape1,data_y):
    mean_term=m
    cov_term=c 
    half_log_det_dds=torch.potrf(cov_term).diag().log().sum()
    dss_ave=torch.tensor([0.5*shape1*(np.log(2*np.pi))])+half_log_det_dds + 0.5*((data_y-mean_term).transpose(0,1)).mm(cov_term.inverse()).mm((data_y-mean_term))  
    return dss_ave


def trivial_loss(m,c,data_y,data_yp):
    mean_term=m
    cov_term=c 
    mean_yp=data_yp.mean()
    var_yp=data_yp.var()        
    first_term=(data_y-mean_term).pow(2)/(2*cov_term)
    logs_sum=(first_term+cov_term.pow(0.5).log()+torch.tensor([0.5*np.log((2*math.pi))]).expand_as(data_y))
    trivial=torch.tensor([0.5*np.log((2*math.pi*var_yp))])+(data_y-mean_yp).pow(2)/(2*var_yp)
    t_loss=(logs_sum-trivial).mean()
    return t_loss


def SMSE(m,data_y,data_yp):
    mean_term=m
    mean_yp=data_yp.mean()      
    mean_pre_mse=((mean_term-data_y)**2).mean()
    mean_trivial_mse=((mean_yp-data_y)**2).mean()
    mean_smse=mean_pre_mse/mean_trivial_mse
    return mean_smse    
# =============================================================================
# 
# =============================================================================
import pandas as pd
import xlrd
import random
address1='H:\Project\DATA________________\kin40k.xlsx'
address2='G:\!帝国理工\M00000-Summer-Project\DATA________________\kin40k.xlsx'


xls=pd.ExcelFile(address1)


TT=10
mse_crps_series=np.zeros(TT)
smse_crps_series=np.zeros(TT)
logs_crps_series=np.zeros(TT)
crps_crps_series=np.zeros(TT)
MSLL_crps_series=np.zeros(TT)
res_crps_series=np.zeros(TT)

mse_gp_series=np.zeros(TT)
smse_gp_series=np.zeros(TT)
logs_gp_series=np.zeros(TT)
crps_gp_series=np.zeros(TT)
MSLL_gp_series=np.zeros(TT)
res_gp_series=np.zeros(TT)

mse_logs_series=np.zeros(TT)
smse_logs_series=np.zeros(TT)
logs_mse_series=np.zeros(TT)
logs_test_logs_series=np.zeros(TT)
logs_test_crps_series=np.zeros(TT)
MSLL_logs_series=np.zeros(TT)
res_logs_series=np.zeros(TT)

mse_dss_series=np.zeros(TT)
smse_dss_series=np.zeros(TT)
logs_dss_series=np.zeros(TT)
crps_dss_series=np.zeros(TT)
MSLL_dss_series=np.zeros(TT)
res_dss_series=np.zeros(TT)

mse_kc_series=np.zeros(TT)
smse_kc_series=np.zeros(TT)
logs_kc_series=np.zeros(TT)
crps_kc_series=np.zeros(TT)
MSLL_kc_series=np.zeros(TT)
res_kc_series=np.zeros(TT)



for j in range(TT):
#    p=100*j
    random.seed(j*100)
    num_va=300
    sam=np.reshape(random.sample(range(0, 10000),500+num_va) , (500+num_va))
    full_x = pd.read_excel(io=xls,sheetname='trainx',header=None).values[sam,:]
    full_y = pd.read_excel(io=xls,sheetname='trainy',header=None).values[sam,:]
    test_x = pd.read_excel(io=xls,sheetname='testx',header=None).values[:500,:]
    test_y = pd.read_excel(io=xls,sheetname='testy',header=None).values[:500,:]
    #注意这里只用了2000个点,实在是太慢了····
    
    #torch.manual_seed(222)
    va=np.reshape(random.sample(range(0, full_x.shape[0]),num_va) , (num_va))
    
    
    dtype = torch.FloatTensor
    va_x=torch.from_numpy(full_x[va,]).type(dtype)
    va_y=torch.from_numpy(full_y[va,]).type(dtype)
    train_x=torch.from_numpy(np.delete(full_x,va,axis=0)).type(dtype)
    train_y=torch.from_numpy(np.delete(full_y,va,axis=0)).type(dtype)
    test_x=torch.from_numpy(test_x).type(dtype)
    test_y=torch.from_numpy(test_y).type(dtype)
    
    num_train=train_x.shape[0]
    num_test=test_x.shape[0]
    num_inducing=20
    #num_va=va_x.shape[0]
    #num_total=num_train+num_test+num_va
    
    
    #learning_rate=5
    #learning_rate2=5#设置了不同的learning rate！
    itr=2000
    
    CRPS_series=np.zeros(itr)
    CRPS_ave_va_series=np.zeros(itr)
    #sigma_k_series=torch.zeros(record_num)
    
   # torch.manual_seed(p)
    torch.manual_seed(j*100) #run together!
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.],requires_grad=True).type(dtype)
#    para_l=torch.tensor([1.4754,  1.3571,  0.8203,  0.6243,  0.5130,  0.3543,  0.2835,0.7229],requires_grad=True).type(dtype)
    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    original_inducing_x=inducing_x.detach().numpy().copy()
    
    
    #alpha=0.2
    for i in range(itr):  #best 1000
        learning_rate=1
#        if i >100:
#             learning_rate=0.5

        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)
        Q_ff=Q(train_x,inducing_x,train_x)
        G=(torch.diag(
          k_ff-Q_ff+sigma_noise_sq*torch.eye(num_train)
             )*torch.eye(num_train)
               ).type(dtype)
        big_Q=Q_ff+G
#        small_Q=torch.diag(big_Q).view(num_train,1)
#        small_k=torch.diag(k_ff).view(num_train,1)
        Q_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_Q)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_Q)/Q_ii_diag
#        cov_term=1/Q_ii_diag+sigma_noise_sq.expand_as(small_k) -small_Q+small_k
        cov_term=1/Q_ii_diag
               
        CRPS_ave=crps(mean_term,cov_term,train_y)     
    
        CRPS_ave.backward()
#        print('CRPS:%.5f' % CRPS_ave)
#        
    
#        CRPS_series[i]=CRPS_ave.detach().numpy() #store values 
    
        
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            inducing_x -=learning_rate * inducing_x.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
            inducing_x.grad.zero_()
#            print('iteration:%d,sigma_k: %.5f,sigma_noise: %.5f' % (
#            i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#            ))
#        #validation
#        k_ff = ARD(train_x,train_x,para_k,para_l) 
#        Q_ff=Q(train_x,inducing_x,train_x)
#        Q_sf=Q(va_x,inducing_x,train_x)
#        k_ss=ARD(va_x,va_x,para_k,para_l)
#     
#        mean_term_va,cov_term_va=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_va,num_train,train_y)   
#        cov_term_va=cov_term_va.diag().view(-1,1)
#        CRPS_ave_va=crps(mean_term_va,cov_term_va,va_y)   
#        CRPS_ave_va_series[i]=CRPS_ave_va.detach().numpy()
    
  
    # =============================================================================
    # predict
    # =============================================================================
    k_ff = ARD(train_x,train_x,para_k,para_l)
    Q_ff=Q(train_x,inducing_x,train_x)
    k_ss=ARD(test_x,test_x,para_k,para_l)
    Q_sf=Q(test_x,inducing_x,train_x)
    
    
    mean_crps,cov_crps=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_test,num_train,train_y)   
    cov_crps_diag=cov_crps.diag().view(-1,1)
    
    
    
    ################################
    mse_crps=((mean_crps-test_y)**2).mean()
    # ================  
    smse_crps=SMSE(mean_crps,test_y,train_y)
    # ================  
    logs_crps=logs(mean_crps, cov_crps_diag,test_y)
    # ================
    crps_crps=crps(mean_crps, cov_crps_diag,test_y)  
    # ================
    MSLL_crps=trivial_loss(mean_crps,cov_crps_diag,test_y,train_y)

    up=mean_crps+2*cov_crps_diag**(0.5)
    low=mean_crps-2*cov_crps_diag**(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()
    
    
    mse_crps_series[j]=mse_crps.detach().numpy()
    smse_crps_series[j]=smse_crps.detach().numpy()
    logs_crps_series[j]=logs_crps.detach().numpy()
    crps_crps_series[j]=crps_crps.detach().numpy()
    MSLL_crps_series[j]=MSLL_crps.detach().numpy()
    res_crps_series[j]=res
 
###    
#    plt.plot(CRPS_ave_va_series,c='purple')
##    
#    plt.plot(CRPS_series,c='blue')    
    
    
#torch.exp(para_l).pow(0.5)    
#tensor([[2.2085, 1.8465, 1.3557, 1.5456, 1.2826, 1.2533, 1.2271, 1.4172]],
##2000 lr1    
#mse_crps
#Out[68]: tensor(0.3424, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[69]: tensor(-0.4415, grad_fn=<MeanBackward1>)
#
#mse_crps
#Out[70]: tensor(0.3424, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[71]: tensor(0.3234, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[72]: tensor(1.0112, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[73]: tensor(0.3190, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[74]: tensor(-0.4415, grad_fn=<MeanBackward1>)    
    
############################################

    itr=2800
    Neg_logL_series=np.zeros(itr)
    Neg_logL_va_series=np.zeros(itr)
    torch.manual_seed(j*100)
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
      
  
    for i in range(itr):  
       
        learning_rate=0.0001
        learning_rate2=0.001

        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)     
        Q_ff=Q(train_x,inducing_x,train_x)
        G=(torch.eye(train_x.shape[0]).type(dtype))*torch.diag(
                            k_ff-Q_ff+sigma_noise_sq*(torch.eye(train_x.shape[0]).type(dtype))
                                                 )
        inverse_term_ml=Q_ff+G
        
        half_log_det=torch.potrf(inverse_term_ml).diag().log().sum()
    
        Neg_logL=torch.tensor([0.5*train_x.shape[0]*(np.log(2*np.pi))]).type(dtype)+half_log_det + (
                            0.5*(train_y.transpose(0,1)).mm(chol_solve(train_y,inverse_term_ml)))
        
#        Neg_logL_series[i]=Neg_logL.detach().numpy()
    
        Neg_logL.backward()
    
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            inducing_x -=learning_rate2 * inducing_x.grad
            para_l.grad.zero_()  
            para_k.grad.zero_()
            para_noise.grad.zero_()
            inducing_x.grad.zero_()
#            print('iteration:%d,Neg_logL:%5f, sigma_k:%.5f ,sigma_noise: %.5f' % (
#            i,Neg_logL,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#            ))      
#        k_ff = ARD(va_x,va_x,para_k,para_l)
#        Q_ff=Q(va_x,inducing_x,va_x)
#        G=(torch.eye(va_x.shape[0])*torch.diag(k_ff-Q_ff+sigma_noise_sq*torch.eye(va_x.shape[0])))
#        inverse_term_ml=Q_ff+G
#        half_log_det_va=torch.potrf(inverse_term_ml).diag().log().sum()
#    
#        Neg_logL_va=torch.tensor([0.5*va_x.shape[0]*(np.log(2*np.pi))])+half_log_det_va + (
#                            0.5*(va_y.transpose(0,1)).mm(chol_solve(va_y,inverse_term_ml)))
#        Neg_logL_va_series[i]=Neg_logL_va.detach().numpy()
    
    
    # =============================================================================

#torch.exp(para_l).pow(0.5)
#[2.3811, 2.0813, 1.4470, 1.4910, 1.2300, 1.1655, 1.2010, 1.3968]
####Neg_logL_va_series[Neg_logL_va_series>0].detach().numpy().argmin() 
#plt.plot(Neg_logL_va_series,c='purple')
#plt.plot(Neg_logL_series,c='blue')



#````````````````````````````    
    k_ff = ARD(train_x,train_x,para_k,para_l)
    Q_ff=Q(train_x,inducing_x,train_x)
    k_ss=ARD(test_x,test_x,para_k,para_l)
    Q_sf=Q(test_x,inducing_x,train_x)
    
    
    mean_gp,cov_gp=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_test,num_train,train_y)
    cov_gp_diag=cov_gp.diag().view(-1,1)
    
    
    
    # ================  
    mse_gp=((mean_gp-test_y)**2).mean()
    # ================
    smse_gp=SMSE(mean_gp,test_y,train_y)
    # ================  
    logs_gp=logs(mean_gp, cov_gp_diag,test_y)
    # ================
    crps_gp=crps(mean_gp, cov_gp_diag,test_y)  
    # ================
    MSLL_gp=trivial_loss(mean_gp,cov_gp_diag,test_y,train_y)
    # ================    
    up=mean_gp+2*cov_gp_diag**(0.5)
    low=mean_gp-2*cov_gp_diag**(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()

    
    mse_gp_series[j]=mse_gp.detach().numpy()
    smse_gp_series[j]=smse_gp.detach().numpy()
    logs_gp_series[j]=logs_gp.detach().numpy()
    crps_gp_series[j]=crps_gp.detach().numpy()
    MSLL_gp_series[j]=MSLL_gp.detach().numpy()
    res_gp_series[j]=res
    
    
#######################################

    itr=4000
    logs_ave_series=np.zeros(itr)
    logs_ave_va_series=np.zeros(itr)
    
    torch.manual_seed(j*100)
    para_l=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    
    
    for i in range(itr):

        learning_rate=0.2
        learning_rate2=0.2
    
        
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)
        Q_ff=Q(train_x,inducing_x,train_x)
        G=(torch.diag(
          k_ff-Q_ff+sigma_noise_sq*torch.eye(num_train)
             )*torch.eye(num_train)
               ).type(dtype)
        big_Q=Q_ff+G
        small_Q=torch.diag(big_Q).view(num_train,1)
        small_k=torch.diag(k_ff).view(num_train,1)
        Q_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_Q)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_Q)/Q_ii_diag
        cov_term=1/Q_ii_diag+sigma_noise_sq -small_Q+small_k
        logs_ave=logs(mean_term,cov_term,train_y)       
        
#        print('Log-Score:%.5f'  % logs_ave)
    
#        logs_ave_series[i]=logs_ave.detach().numpy() #store values
        logs_ave.backward()
          
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            inducing_x -=learning_rate2 * inducing_x.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
            inducing_x.grad.zero_()
#            print('iteration:%d,  sigma_k_square: %.5f ,sigma_noise: %.5f' % (
#            i,torch.exp(para_k),torch.exp(para_noise).pow(0.5)
#            ))    
#    #        
#        k_ff = ARD(train_x,train_x,para_k,para_l)
#        Q_ff=Q(train_x,inducing_x,train_x)
#        Q_sf=Q(va_x,inducing_x,train_x)
#        k_ss=ARD(va_x,va_x,para_k,para_l)
#       
#        mean_term_va,cov_term_va=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_va,num_train,train_y)   
#        cov_term_va=cov_term_va.diag().view(-1,1)
#           
#        logs_ave_va=logs(mean_term_va,cov_term_va,va_y)   
#        logs_ave_va_series[i]=logs_ave_va.detach().numpy()
# 

#plt.plot(logs_ave_va_series,c='purple')
#plt.plot(logs_ave_series,c='blue')

         
    # =============================================================================
    # predict
    # =============================================================================
    k_ff = ARD(train_x,train_x,para_k,para_l)
    Q_ff=Q(train_x,inducing_x,train_x)
    k_ss=ARD(test_x,test_x,para_k,para_l)
    Q_sf=Q(test_x,inducing_x,train_x)
    
    
    mean_logs,cov_logs=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_test,num_train,train_y)   
    cov_logs_diag=cov_logs.diag().view(-1,1)
    
    
    mse_logs=((mean_logs-test_y)**2).mean()
    # ================
    logs_smse=SMSE(mean_logs,test_y,train_y)
    # ================  
    logs_logs=logs(mean_logs, cov_logs_diag,test_y)
    # ================
    crps_logs=crps(mean_logs, cov_logs_diag,test_y)  
    # ================
    MSLL_logs=trivial_loss(mean_logs,cov_logs_diag,test_y,train_y)
    # ================
    
    up=mean_gp+2*cov_gp_diag**(0.5)
    low=mean_gp-2*cov_gp_diag**(0.5) 
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()    
    
    logs_mse_series[j]=mse_logs.detach().numpy()
    smse_logs_series[j]=logs_smse.detach().numpy()
    logs_test_logs_series[j]=logs_logs.detach().numpy()
    logs_test_crps_series[j]=crps_logs.detach().numpy()
    MSLL_logs_series[j]=MSLL_logs.detach().numpy()    
    res_logs_series[j]=res    
 
# =============================================================================
#        DSS
# =============================================================================
    itr=3000
    dss_series=np.zeros(itr)
    dss_va_series=np.zeros(itr)
    
    torch.manual_seed(j*100)
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.],requires_grad=True).type(dtype)
#    para_l=torch.tensor([1.4754,  1.3571,  0.8203,  0.6243,  0.5130,  0.3543,  0.2835,0.7229],requires_grad=True).type(dtype)
    inducing_x=torch.randn(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    original_inducing_x=inducing_x.detach().numpy().copy()
    
    
    
    for i in range(itr):  
        learning_rate=0.001
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)     
        fold_k=4
        index1=int(num_train/fold_k)
        index2=int(2*num_train/fold_k) 
        index3=int(3*num_train/fold_k) 
        
        
        Q_ff=Q(train_x,inducing_x,train_x)
        G=(torch.diag(
          k_ff-Q_ff+sigma_noise_sq*torch.eye(num_train)
             )*torch.eye(num_train)
               ).type(dtype)
        big_Q=Q_ff+G
            
        Q_inv_i_j=chol_solve(torch.eye(num_train),big_Q)
        Q_1=Q_inv_i_j[:index1,:index1]
        Q_2=Q_inv_i_j[index1:index2,index1:index2]
        Q_3=Q_inv_i_j[(index2):index3,(index2):index3]
        Q_4=Q_inv_i_j[index3:,index3:]             

        y_1=train_y[:index1]
        y_2=train_y[index1:index2]
        y_3=train_y[(index2):index3]     
        y_4=train_y[index3:]  
        
        Q_inv_y=chol_solve(train_y,big_Q)
    
        m_1=y_1-chol_solve(torch.eye(index1),Q_1).mm(Q_inv_y[:index1])
        m_2=y_2-chol_solve(torch.eye(index1),Q_2).mm(Q_inv_y[index1:index2])
        m_3=y_3-chol_solve(torch.eye(index1),Q_3).mm(Q_inv_y[(index2):index3])
        m_4=y_4-chol_solve(torch.eye(index1),Q_4).mm(Q_inv_y[index3:]) 
        

        cov_1=chol_solve(torch.eye(index1),Q_1)
        cov_2=chol_solve(torch.eye(index1),Q_2)
        cov_3=chol_solve(torch.eye(index1),Q_3)        
        cov_4=chol_solve(torch.eye(index1),Q_4)          
         
        dss1=dss(m_1,cov_1,index1,y_1)
        dss2=dss(m_2,cov_2,index1,y_2)
        dss3=dss(m_3,cov_3,index1,y_3)   
        dss4=dss(m_4,cov_4,index1,y_4) 
        
        dss_ave=(dss1+dss2+dss3+dss4).mean()
        
#        dss_series[i]=dss_ave.detach().numpy()
#        print('DSS:%.5f'  % dss_ave)
        
        dss_ave.backward()
          
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            inducing_x -=learning_rate * inducing_x.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
            inducing_x.grad.zero_()
#            print('iteration:%d,sigma_k: %.5f sigma_noise: %.5f' % (
#            i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#            ))
#        
#        k_ff = ARD(train_x,train_x,para_k,para_l)
#        Q_ff=Q(train_x,inducing_x,train_x)
#        Q_sf=Q(va_x,inducing_x,train_x)
#        k_ss=ARD(va_x,va_x,para_k,para_l)
#    
#        mean_term,cov_term=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_va,num_train,train_y)            
#        dss_va=dss(mean_term,cov_term,num_va,va_y)   
#        dss_va_series[i]=dss_va.detach().numpy()
      
    
    
    # =============================================================================
    # predict
    k_ff = ARD(train_x,train_x,para_k,para_l)
    Q_ff=Q(train_x,inducing_x,train_x)
    k_ss=ARD(test_x,test_x,para_k,para_l)
    Q_sf=Q(test_x,inducing_x,train_x)
    
    
    mean_dss,cov_dss=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_test,num_train,train_y)   
    cov_dss_diag =(cov_dss).diag().view(num_test,1)
    
    
    dss_mse=((mean_dss-test_y)**2).mean()
    # ================
    smse_dss=SMSE(mean_dss,test_y,train_y)
    # ================
    dss_test_logs=logs(mean_dss,cov_dss_diag,test_y)
    # ================
    dss_test_crps=crps( mean_dss,cov_dss_diag,test_y)  
    # ================
    MSLL_dss=trivial_loss(mean_dss,cov_dss_diag,test_y,train_y)

    up=mean_dss+2*cov_dss_diag**(0.5)
    low=mean_dss-2*cov_dss_diag **(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()
    
    
    mse_dss_series[j]=dss_mse.detach().numpy()
    smse_dss_series[j]=smse_dss.detach().numpy()
    logs_dss_series[j]=dss_test_logs.detach().numpy()
    crps_dss_series[j]=dss_test_crps.detach().numpy()
    MSLL_dss_series[j]=MSLL_dss.detach().numpy()
    res_dss_series[j]=res
    
#plt.plot(dss_va_series,c='purple')
#plt.plot(dss_series,c='blue')    
    
# =============================================================================
#    kc  
# =============================================================================
    itr=3000
    kc_series=np.zeros(itr)
    kc_va_series=np.zeros(itr)
    
    torch.manual_seed(j*100)    
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.],requires_grad=True).type(dtype)
#    para_l=torch.tensor([1.4754,  1.3571,  0.8203,  0.6243,  0.5130,  0.3543,  0.2835,0.7229],requires_grad=True).type(dtype)
    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    original_inducing_x=inducing_x.detach().numpy().copy()
    
    try:
        for i in range(itr):
             learning_rate=0.1
             sigma_noise_sq=torch.exp(para_noise)
             k_ff = ARD(train_x,train_x,para_k,para_l)     
             fold_k=4
             index1=int(num_train/fold_k)
             index2=int(2*num_train/fold_k) 
             index3=int(3*num_train/fold_k) 
                 
             Q_ff=Q(train_x,inducing_x,train_x)
             G=(torch.diag(
               k_ff-Q_ff+sigma_noise_sq*torch.eye(num_train)
                  )*torch.eye(num_train)
                    ).type(dtype)
             big_Q=Q_ff+G
             
             Q_inv_i_j=chol_solve(torch.eye(num_train),big_Q)
             Q_1=Q_inv_i_j[:index1,:index1]
             Q_2=Q_inv_i_j[index1:index2,index1:index2]
             Q_3=Q_inv_i_j[(index2):index3,(index2):index3]
             Q_4=Q_inv_i_j[index3:,index3:]        
     
             y_1=train_y[:index1]
             y_2=train_y[index1:index2]
             y_3=train_y[(index2):index3]     
             y_4=train_y[index3:]  
     
     
             Q_inv_y=chol_solve(train_y,big_Q)
         
             m_1=y_1-chol_solve(torch.eye(index1),Q_1).mm(Q_inv_y[:index1])
             m_2=y_2-chol_solve(torch.eye(index1),Q_2).mm(Q_inv_y[index1:index2])
             m_3=y_3-chol_solve(torch.eye(index1),Q_3).mm(Q_inv_y[(index2):index3])
             m_4=y_4-chol_solve(torch.eye(index1),Q_4).mm(Q_inv_y[index3:])        
             
     
             cov_1=torch.diag(chol_solve(torch.eye(index1),Q_1)).view(index1,1)
             cov_2=torch.diag(chol_solve(torch.eye(index1),Q_2)).view(index1,1)
             cov_3=torch.diag(chol_solve(torch.eye(index1),Q_3)).view(index1,1)   
             cov_4=torch.diag(chol_solve(torch.eye(index1),Q_4)).view(index1,1)
         
            
             kc1=crps(m_1,cov_1,y_1)  
             kc2=crps(m_2,cov_2,y_2)  
             kc3=crps(m_3,cov_3,y_3)  
             kc4=crps(m_4,cov_4,y_4)  
            
             kc_ave=(kc1+kc2+kc3+kc4).mean()
             
#             kc_series[i]=kc_ave.detach().numpy()
             
#             print('KC:%.5f'  % kc_ave)
            
             kc_ave.backward()
              
             with torch.no_grad():
                 para_l -= learning_rate * para_l.grad
                 para_k -= learning_rate * para_k.grad
                 para_noise -= learning_rate * para_noise.grad
                 inducing_x -=learning_rate * inducing_x.grad
                 para_l.grad.zero_()  #set gradients to zero 
                 para_k.grad.zero_()
                 para_noise.grad.zero_()
                 inducing_x.grad.zero_()
#                 print('iteration:%d,sigma_k: %.5f sigma_noise: %.5f' % (
#                 i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#                 ))
#            
#             k_ff = ARD(train_x,train_x,para_k,para_l)
#             Q_ff=Q(train_x,inducing_x,train_x)
#             Q_sf=Q(va_x,inducing_x,train_x)
#             k_ss=ARD(va_x,va_x,para_k,para_l)
#         
#             mean_term,cov_term=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_va,num_train,train_y)   
#             cov_crps_diag=cov_term.diag().view(-1,1)
#             kc_va=crps(mean_term,cov_crps_diag,va_y)   
#             kc_va_series[i]=kc_va.detach().numpy()
          
        
        
        # =============================================================================
        # predict
        # =============================================================================
        k_ff = ARD(train_x,train_x,para_k,para_l)
        Q_ff=Q(train_x,inducing_x,train_x)
        k_ss=ARD(test_x,test_x,para_k,para_l)
        Q_sf=Q(test_x,inducing_x,train_x)
        
        
        mean_kc,cov_kc=spgp_cal_mean_and_cov(k_ff,Q_ff,Q_sf,k_ss,num_test,num_train,train_y)     
        cov_kc_diag =(cov_kc).diag().view(num_test,1)
        
        
        kc_mse=((mean_kc-test_y)**2).mean()
        # ================
        smse_kc=SMSE(mean_kc,test_y,train_y)
        # ================
        kc_test_logs=logs(mean_kc,cov_kc_diag,test_y)
        # ================
        kc_test_crps=crps( mean_kc,cov_kc_diag,test_y)  
        # ================
        MSLL_kc=trivial_loss(mean_kc,cov_kc_diag,test_y,train_y)

        up=mean_kc+2*cov_kc_diag**(0.5)
        low=mean_kc-2*cov_kc_diag**(0.5)
        a=((up-test_y)>0).numpy()
        b=((test_y-low)>0).numpy()
        res=np.multiply(a,b).mean()

    
        mse_kc_series[j]=kc_mse.detach().numpy()
        smse_kc_series[j]=smse_kc.detach().numpy()
        logs_kc_series[j]=kc_test_logs.detach().numpy()
        crps_kc_series[j]=kc_test_crps.detach().numpy()
        MSLL_kc_series[j]=MSLL_kc.detach().numpy()
        res_kc_series[j]=res
        
    except RuntimeError:
        mse_kc_series[j]=0.
        smse_kc_series[j]=0.
        logs_kc_series[j]=0.
        crps_kc_series[j]=0.
        MSLL_kc_series[j]=0.
        res_kc_series[j]=0.
    
#plt.plot(kc_va_series,c='purple')
#plt.plot(kc_series,c='blue')    
#torch.exp(para_l).pow(0.5)
    print(j)









mse_gp_series
smse_gp_series
logs_gp_series
crps_gp_series
MSLL_gp_series
res_gp_series

mse_crps_series
smse_crps_series
logs_crps_series
crps_crps_series
MSLL_crps_series
res_crps_series


logs_mse_series
smse_logs_series
logs_test_logs_series
logs_test_crps_series
MSLL_logs_series
res_logs_series


mse_dss_series
smse_dss_series
logs_dss_series
crps_dss_series
MSLL_dss_series
res_dss_series

mse_kc_series
smse_kc_series
logs_kc_series
crps_kc_series
MSLL_kc_series
res_kc_series







# =============================================================================
#  NLML
# =============================================================================

mse_gp_series
Out[210]: 
array([0.39215443, 0.39881483, 0.36215177, 0.36160874, 0.42472798,
       0.38863039, 0.37792721, 0.38416997, 0.4510245 , 0.37153566])

smse_gp_series
Out[211]: 
array([0.37049448, 0.37771758, 0.3428573 , 0.34221652, 0.40194413,
       0.36768022, 0.35779461, 0.35998401, 0.4265942 , 0.35186753])

logs_gp_series
Out[212]: 
array([0.88487285, 0.8917228 , 0.87178099, 0.86361665, 0.9184007 ,
       0.90169549, 0.90649319, 0.8974182 , 0.94208562, 0.89179945])

crps_gp_series
Out[213]: 
array([0.33654302, 0.33505049, 0.32929769, 0.32848471, 0.34835345,
       0.3347652 , 0.34261447, 0.3408294 , 0.3500261 , 0.33509496])

MSLL_gp_series
Out[214]: 
array([-0.56779128, -0.55567777, -0.57455182, -0.58400744, -0.52968746,
       -0.54497826, -0.53993756, -0.55753505, -0.50500882, -0.5549103 ])

res_gp_series
Out[215]: 
array([0.952, 0.96 , 0.962, 0.972, 0.94 , 0.952, 0.956, 0.934, 0.934,
       0.942])


# =============================================================================
#   CRPS
# =============================================================================

mse_crps_series
Out[216]: 
array([0.34235013, 0.36366448, 0.31601936, 0.31671342, 0.30106655,
       0.34369352, 0.25168774, 0.30732685, 0.38926011, 0.29924747])

smse_crps_series
Out[217]: 
array([0.32344103, 0.34442669, 0.29918271, 0.29972884, 0.28491631,
       0.32516581, 0.23828006, 0.28797865, 0.36817536, 0.28340608])

logs_crps_series
Out[218]: 
array([1.01121318, 0.97822559, 0.85785669, 0.87696755, 0.83963579,
       0.93567044, 0.74144638, 0.8887583 , 1.02271593, 0.92386335])

crps_crps_series
Out[219]: 
array([0.31895155, 0.32551479, 0.3048768 , 0.31088483, 0.30078849,
       0.32144859, 0.28293645, 0.30837849, 0.33235329, 0.31121561])

MSLL_crps_series
Out[220]: 
array([-0.44145107, -0.46917486, -0.58847612, -0.57065648, -0.6084525 ,
       -0.51100332, -0.70498443, -0.56619447, -0.42437857, -0.52284658])

res_crps_series
Out[221]: 
array([0.892, 0.898, 0.9  , 0.92 , 0.912, 0.91 , 0.92 , 0.862, 0.892,
       0.88 ])
# =============================================================================
#    Logs
# =============================================================================

logs_mse_series
Out[222]: 
array([0.48501688, 0.50044304, 0.46954429, 0.46712348, 0.47406575,
       0.50022471, 0.42714193, 0.41908869, 0.49463159, 0.43382347])

smse_logs_series
Out[223]: 
array([0.45822784, 0.47396967, 0.44452825, 0.44207278, 0.44863525,
       0.47325876, 0.40438762, 0.39270437, 0.46783924, 0.41085798])

logs_test_logs_series
Out[224]: 
array([1.02196443, 1.02429616, 1.02252662, 1.00598991, 1.01626277,
       1.05655801, 0.95540518, 0.95474499, 1.04394352, 0.97245342])

logs_test_crps_series
Out[225]: 
array([0.3744933 , 0.37601379, 0.36399028, 0.36896637, 0.36618006,
       0.375552  , 0.35654759, 0.35165206, 0.36756414, 0.35777894])

MSLL_logs_series
Out[226]: 
array([-0.43069977, -0.42310423, -0.42380625, -0.44163409, -0.43182528,
       -0.39011592, -0.49102548, -0.50020808, -0.40315083, -0.4742566 ])

res_logs_series
Out[227]: 
array([0.952, 0.96 , 0.962, 0.972, 0.94 , 0.952, 0.956, 0.934, 0.934,
       0.942])

# =============================================================================
#   DSS
# =============================================================================

mse_dss_series
Out[228]: 
array([0.38988486, 0.37502113, 0.34004724, 0.33105588, 0.33373815,
       0.37681377, 0.33923399, 0.38930765, 0.36948863, 0.36509669])

smse_dss_series
Out[229]: 
array([0.36835027, 0.35518256, 0.32193044, 0.31330216, 0.3158353 ,
       0.35650063, 0.32116264, 0.36479822, 0.34947482, 0.34576944])

logs_dss_series
Out[230]: 
array([0.94298434, 0.95279878, 0.86699021, 0.83757281, 0.85195816,
       0.93035948, 0.86353064, 1.00874674, 0.87835974, 0.92801094])

crps_dss_series
Out[231]: 
array([0.3404347 , 0.33450985, 0.32301867, 0.31487685, 0.31575152,
       0.33219609, 0.32577398, 0.348809  , 0.32261053, 0.33291724])

MSLL_dss_series
Out[232]: 
array([-0.50967985, -0.49460191, -0.5793426 , -0.61005116, -0.59613025,
       -0.51631433, -0.58290005, -0.44620627, -0.568735  , -0.51869911])

res_dss_series
Out[233]: 
array([0.91 , 0.898, 0.904, 0.922, 0.91 , 0.898, 0.922, 0.852, 0.91 ,
       0.896])

# =============================================================================
#   kc
# =============================================================================

   mse_kc_series
Out[234]: 
array([0.40411514, 0.38370919, 0.33522755, 0.32254818, 0.38380754,
       0.36660364, 0.2979165 , 0.32748505, 0.39744246, 0.32870561])

smse_kc_series
Out[235]: 
array([0.38179457, 0.36341104, 0.31736752, 0.3052507 , 0.36321878,
       0.34684089, 0.28204617, 0.30686778, 0.37591448, 0.31130478])

logs_kc_series
Out[236]: 
array([0.96109271, 1.00285697, 0.88152504, 0.8343789 , 0.96519995,
       0.96690661, 0.83324587, 0.94529474, 0.97709793, 0.92601937])

crps_kc_series
Out[237]: 
array([0.33732226, 0.33661351, 0.31321034, 0.30858254, 0.33993345,
       0.32871678, 0.30661502, 0.32237086, 0.33574435, 0.32398608])

MSLL_kc_series
Out[238]: 
array([-0.49157152, -0.44454345, -0.56480801, -0.61324489, -0.48288798,
       -0.4797675 , -0.61318493, -0.5096584 , -0.46999651, -0.52069068])

res_kc_series
Out[239]: 
array([0.91 , 0.908, 0.928, 0.936, 0.904, 0.91 , 0.93 , 0.872, 0.912,
       0.892])   








































mse_gp_series
Out[80]: 
array([0.34721372, 0.37087244, 0.40893653, 0.37398502, 0.44252646,
       0.35174906, 0.38088647, 0.35841435, 0.38160762, 0.3830907 ])

smse_gp_series
Out[81]: 
array([0.32869253, 0.35118979, 0.38704839, 0.35334077, 0.41779032,
       0.33224195, 0.36073983, 0.33812445, 0.3614206 , 0.36221236])

logs_gp_series
Out[82]: 
array([0.86710465, 0.88357121, 0.90633643, 0.87690258, 0.97083783,
       0.86055791, 0.91487914, 0.86026007, 0.9299019 , 0.90165484])

crps_gp_series
Out[83]: 
array([0.32768047, 0.32923633, 0.33897775, 0.33437404, 0.36055142,
       0.32635286, 0.33886978, 0.32568628, 0.34240794, 0.33895266])

MSLL_gp_series
Out[84]: 
array([-0.58118618, -0.56342977, -0.54012275, -0.57158458, -0.48018846,
       -0.58763295, -0.53156292, -0.58919924, -0.5163151 , -0.54565156])

res_gp_series
Out[85]: 
array([0.882, 0.832, 0.792, 0.826, 0.778, 0.87 , 0.778, 0.796, 0.798,
       0.762])

mse_crps_series
Out[86]: 
array([0.29629898, 0.32599354, 0.34177038, 0.28027752, 0.38040113,
       0.25901616, 0.34272859, 0.26321715, 0.29081771, 0.33135739])

smse_crps_series
Out[87]: 
array([0.28049371, 0.30869266, 0.32347727, 0.264806  , 0.35913765,
       0.24465178, 0.32460028, 0.24831638, 0.27543345, 0.31329849])

logs_crps_series
Out[88]: 
array([0.8232981 , 0.87262648, 0.88104206, 0.79173762, 1.0489974 ,
       0.74425304, 1.04237664, 0.80300689, 0.86571687, 0.96292013])

crps_crps_series
Out[89]: 
array([0.30326542, 0.30495003, 0.30379826, 0.29438812, 0.3377403 ,
       0.28596443, 0.32542354, 0.28804636, 0.29877466, 0.31924513])

MSLL_crps_series
Out[90]: 
array([-0.62499291, -0.57437468, -0.56541711, -0.65674937, -0.40202883,
       -0.70393741, -0.40406576, -0.64645249, -0.58050013, -0.48438656])

res_crps_series
Out[91]: 
array([0.624, 0.588, 0.614, 0.602, 0.598, 0.628, 0.532, 0.522, 0.602,
       0.578])

smse_logs_series
Out[92]: 
array([0.5091058 , 0.48878989, 0.5983128 , 0.48442671, 0.5154739 ,
       0.63403136, 0.51101714, 0.46766701, 0.45522943, 0.50618792])

logs_mse_series
Out[93]: 
array([0.53779292, 0.51618445, 0.63214827, 0.51272976, 0.54599363,
       0.67125762, 0.5395565 , 0.4957304 , 0.48065612, 0.53536516])

logs_test_logs_series
Out[94]: 
array([1.02719486, 1.03904116, 1.08335805, 1.03667915, 1.10082352,
       1.1093663 , 0.99735683, 1.01155508, 0.99732453, 1.0712738 ])

logs_test_crps_series
Out[95]: 
array([0.3952117 , 0.38397437, 0.41332406, 0.39141247, 0.40321842,
       0.43254581, 0.38482502, 0.37747237, 0.37294081, 0.38770744])

MSLL_logs_series
Out[96]: 
array([-0.42109621, -0.40795979, -0.36310086, -0.41180816, -0.35020301,
       -0.33882445, -0.44908544, -0.43790439, -0.44889271, -0.37603292])

res_logs_series
Out[97]: 
array([0.882, 0.832, 0.792, 0.826, 0.778, 0.87 , 0.778, 0.796, 0.798,
       0.762])












































































plt.plot(up[:10].detach().numpy(),'bo')
plt.plot(low[:10].detach().numpy(),'go')
plt.plot(test_y[:10].detach().numpy(),'ro')
plt.plot(mean_crps[:10].detach().numpy(),c='pink')


#itr 2000
#50
#Out[8]: array([0.453 , 0.4584, 0.4695, 0.4717, 0.4553])


#itr1500
#
#res_crps_series
#Out[14]: array([0.4739, 0.4888, 0.4848, 0.4854, 0.4863])



MSLL_crps_series[:10]
MSLL_crps_series[10:]


    
    CRPS_ave_va_series[CRPS_ave_va_series>0].detach().numpy().argmin() 
    
    plt.plot(CRPS_ave_va_series,c='purple')
    
    plt.plot(CRPS_series,c='blue')
    
    


itr 2000
mse_crps
Out[15]: tensor(0.2445, grad_fn=<MeanBackward1>)
smse_crps
Out[16]: tensor(0.2520, grad_fn=<DivBackward1>)

logs_crps
Out[17]: tensor(0.8062, grad_fn=<MeanBackward1>)

crps_crps
Out[18]: tensor(0.2800, grad_fn=<MeanBackward1>)

MSLL_crps
Out[19]: tensor(-0.5981, grad_fn=<MeanBackward1>)

res
Out[21]: 0.477




itr 3000
mse_crps
Out[28]: tensor(0.2426, grad_fn=<MeanBackward1>)

smse_crps
Out[29]: tensor(0.2500, grad_fn=<DivBackward1>)

logs_crps
Out[30]: tensor(0.8133, grad_fn=<MeanBackward1>)

crps_crps
Out[31]: tensor(0.2773, grad_fn=<MeanBackward1>)

MSLL_crps
Out[32]: tensor(-0.5911, grad_fn=<MeanBackward1>)

res
Out[33]: 0.418














# itr=1000  lr=1
#mse_crps
#Out[132]: tensor(0.2839)
#
#smse_crps
#Out[133]: tensor(0.2688)
#
#logs_crps
#Out[134]: tensor(0.8495)
#
#crps_crps
#Out[135]: tensor(0.2949)
#
#MSLL_crps
#Out[137]: tensor(-0.6033)




# itr=1250  lr=1
#mse_crps
#Out[142]: tensor(0.2774)
#
#smse_crps
#Out[143]: tensor(0.2626)
#
#logs_crps
#Out[144]: tensor(0.8502)
#
#crps_crps
#Out[145]: tensor(0.2913)
#
#MSLL_crps
#Out[146]: tensor(-0.6027)



#itr=1250, lr=0.5
#mse_crps
#Out[185]: tensor(0.2880)
#
#smse_crps
#Out[186]: tensor(0.2726)
#
#logs_crps
#Out[187]: tensor(0.8486)
#
#crps_crps
#Out[188]: tensor(0.2970)
#
#MSLL_crps
#Out[189]: tensor(-0.6043)



#
#itr=2000 lr=1
#mse_crps
#Out[220]: tensor(0.2623)
#
#smse_crps
#Out[221]: tensor(0.2483)
#
#logs_crps
#Out[222]: tensor(0.8407)
#
#crps_crps
#Out[223]: tensor(0.2830)
#
#MSLL_crps
#Out[224]: tensor(-0.6121)



    
#==============================================================================
# 2000itr    coverage:0.9627
#==============================================================================




#==============================================================================
# 788 noise 0.07
#==============================================================================
#mse_crps
#Out[6]: tensor(0.1964, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[7]: tensor(0.2000, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[8]: tensor(0.6406, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[9]: tensor(0.2441, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[10]: tensor(-0.7699, grad_fn=<MeanBackward1>)

#788+612
#mse_crps noise 0.04xx
#Out[19]: tensor(0.1791, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[20]: tensor(0.1823, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[21]: tensor(0.6213, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[22]: tensor(0.2329, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[23]: tensor(-0.7891, grad_fn=<MeanBackward1>

#    788+612+200
#mse_crps  noise0.039
#Out[26]: tensor(0.1762, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[27]: tensor(0.1793, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[28]: tensor(0.6187, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[29]: tensor(0.2310, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[30]: tensor(-0.7917, grad_fn=<MeanBackward1>)

#788+612+200+200
#mse_crps
#Out[34]: tensor(0.1736, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[35]: tensor(0.1767, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[36]: tensor(0.6158, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[37]: tensor(0.2293, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[38]: tensor(-0.7946, grad_fn=<MeanBackward1>)

#
#788+612+200+200+200
#mse_crps
#Out[44]: tensor(0.1713, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[45]: tensor(0.1744, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[46]: tensor(0.6121, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[47]: tensor(0.2278, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[48]: tensor(-0.7983, grad_fn=<MeanBackward1>)











#
#1800
#mse_crps_series
#Out[5]: 
#array([0.17811269, 0.19037719, 0.17220218, 0.18027298, 0.20525156,
#       0.18446961, 0.18206926, 0.18551755, 0.19047746, 0.        ])       median0.18446961

#
#smse_crps_series
#Out[6]: 
#array([0.18051004, 0.19343925, 0.17515801, 0.18233432, 0.20874971,
#       0.18751277, 0.18403164, 0.188925  , 0.19387384, 0.        ])
#
#logs_crps_series
#Out[7]: 
#array([0.61272752, 0.60596329, 0.67846113, 0.67402458, 0.66832334,
#       0.65269792, 0.65118802, 0.64848542, 0.62757242, 0.        ])
#
#crps_crps_series
#Out[8]: 
#array([0.23128718, 0.23652588, 0.23045078, 0.23501469, 0.24589914,
#       0.2361751 , 0.23663381, 0.23773848, 0.23913835, 0.        ])
#
#MSLL_crps_series
#Out[9]: 
#array([-0.8000592 , -0.80530131, -0.73196781, -0.74063706, -0.7422142 ,
#       -0.75818616, -0.76359236, -0.76201749, -0.78253025,  0.        ])
















#==============================================================================
# 
#==============================================================================




#mse_crps   fixed 0.1 itr=1200
#Out[6]: tensor(0.3665, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[7]: tensor(0.3726, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[8]: tensor(0.8781, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[9]: tensor(0.3299, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[10]: tensor(-0.5332, grad_fn=<MeanBackward1>)

#also can get 

#mse_crps fixed 0.1 itr=1200
#Out[15]: tensor(0.1936, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[16]: tensor(0.1968, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[17]: tensor(0.6589, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[18]: tensor(0.2417, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[19]: tensor(-0.7525, grad_fn=<MeanBackward1>)



#mse_crps   fixed 0.1 itr=1400
#Out[23]: tensor(0.1845, grad_fn=<MeanBackward1>)
#
#smse_crps
#Out[24]: tensor(0.1875, grad_fn=<DivBackward1>)
#
#logs_crps
#Out[25]: tensor(0.6378, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[26]: tensor(0.2361, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[27]: tensor(-0.7735, grad_fn=<MeanBackward1>)























# ================
#dss_crps=dss(mean_crps,cov_crps,num_test,test_y)
# ================
#ES_crps=ES(mean_crps,cov_crps,num_test,test_y)  


#itr=400. decreasing lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01
#it seems it can still further decreasing crps,but let me paste the results:
#mse_crps  (*)
#Out[7]: tensor(0.2396, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[8]: tensor(0.6989, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[9]: tensor(0.2676, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[10]: tensor(-0.7121, grad_fn=<MeanBackward1>)    
#---------------------------------
#the above choice seems decreasing slowly after 100
#now  change to   lr=5-- if i >1:lr=1--elif i>5:lr=0.5
#the results are worse than above, and get CRPS:0.23347.
#-----------------
# let's go back and check the former one: itr=400. decreasing lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01
#we get CRPS:0.23347, strangely we get worse results than (*)
#let's increase the itr=600 since I can see some improvements
 #we get CRPS:0.23248 and all other terms have only very slightly improvement
 #----------------
 #let's try again use above and 600itr:
#we get:     CRPS:0.23487 and get very similar to above results and validation fluctuate after 380.
#----------------
# let's decrease lr after 380, so we decrease lr after 380
#now  itr=600  lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01 elif i>380:lr=0.001 
 #we get :CRPS:0.23260 and get slightly worse results than (*)
#-------------------------------
#let's use same data but with (*)'s choice and itr=380 to compare them 
#we get CRPS:0.25768. It seems really need to increase itr. (results are worse,obviously)
#--------------------------------
#let's use first choice :lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01
    #but (again) increase itr = 600, we get CRPS:0.22189 (**)
#the results shows there is a reasonable improvement !!!!
#mse_crps
#Out[8]: tensor(0.2112, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6590, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2515, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7534, grad_fn=<MeanBackward1>)    
#----------------------------------
#but I really think the curve is decreasing slowly around 100, so I increase third lr:
#itr=600,lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.3 --elif i>45: lr=0.01
#we get CRPS:0.21928, and we find out there is slightly improvements:
#mse_crps
#Out[8]: tensor(0.2132, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6395, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2527, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7707, grad_fn=<MeanBackward1>)
#---------------------------------------     
#but i am not satisfy the curve after 200, I use new:
##itr=600,lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.3 --elif i>45: lr=0.1 (greddy!)
#we get  CRPS:0.23456 and (much) worse results than above!
#-----------------------------------------
#maybe I am too greddy about last lr. so I use:
###itr=600,lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.3 --elif i>45: lr=0.05
#we get CRPS:0.23608
#``````````well `we get a very bad results from above 
#------------------------------------
#let's try last time:
###itr=600,lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.3  i>100: lr=0.01
 #this give worse results than **
 #-------------------------------------
#so let's try ** again to see if it is stable
#we get crps 0.22215, but results are worse than ** itself,
#----------------------
# let's try again
#still worse than ** the first time``````````````````
#--------------------------------------
####``````one more time```````````
#itr=600,lr: if i >1:lr=1 --elif i>5 lr=0.3  --->i>150: lr=0.01---->i>450:lr=0.005
#get CRPS:0.22844 ```gives a very bad results```````````````````````
#-=------------------------------
#fairenough,`````let's use original one````
#itr=600 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01
#try this again,get CRPS:0.22274
 #well````gives:
#mse_crps
#Out[8]: tensor(0.2205, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6822, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2581, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7309, grad_fn=<MeanBackward1>) 
#-------------------------------    
#let's```increase itr=650`````````````and try again
#get CRPS:0.22647 and 
#mse_crps
#Out[9]: tensor(0.2052, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[10]: tensor(0.6581, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[11]: tensor(0.2516, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[12]: tensor(-0.7522, grad_fn=<MeanBackward1>)
#it seems ````````I can increase itr further
#-----------------------------
#Increase itr=700 but use small lr at the end since it exist some fluctuate after 500
#itr=700 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01 i>450:lr=0.005
#get CRPS:0.21920 . give very good result!
#mse_crps
#Out[8]: tensor(0.2045, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6407, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2493, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7694, grad_fn=<MeanBackward1>)
#==============================================================================
# inducing points 50. itr700
# lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01 i>450:lr=0.005

#fixed 0.1 and itr=1200 works bad 

#==============================================================================


#now, go to <<100>> points
#first use above choice
#get CRPS:0.20106
#there is some improvement, but not large`(well ```some terms become worse)
#mse_crps
#Out[8]: tensor(0.1965, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6546, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2423, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7552, grad_fn=<MeanBackward1>)
#------------------------------------
#try itr=700 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.01 
 #get CRPS:0.19673 but other terms are worse ......
 #---------------------------------------------
 #I want to increase itr but the computer might not handle it ````
 #we change to  itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.02 
 #get CRPS:0.19033
# mse_crps
#Out[8]: tensor(0.1954, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6427, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2411, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7687, grad_fn=<MeanBackward1>)
#this is only slightly better than 200 points case ``
#---------------------------------------
#change last lr to see if can learn faster ?
 #we change to  itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 --elif i>45: lr=0.05
#gives almost the same results as above.
#---------------------------------
#be greedy!  use itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>10: lr=0.1 
#get CRPS:0.19621
#and almost no improvement````...
#-----------------------------------
#I have to be more greedy!
#try itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>1-0-0: lr=0.1 
#get CRPS:0.20203, results do not change much ``
#----------------------------------------
#try itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>300: lr=0.1 
#get CRPS:0.19469
#gives slightly better ```:
#mse_crps
#Out[8]: tensor(0.1936, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6370, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2402, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7742, grad_fn=<MeanBackward1>)    
#and we may want to let second large lr continue more time ``
#----------------------------------------
#try itr=750 lr: if i >1:lr=1 --elif i>5 lr=0.5-- elif i>500: lr=0.1 
#results did not change much``
#-------------------------------------
#try itr=750 lr: if i >1:lr=1 --elif i>20 lr=0.6
#get CRPS:0.20053  similar results as above
#-----------------------------------------
#try  itr=750 lr:    learning_rate=5 if i >1:lr=1 fixed 
#gives accpteable results:
#mse_crps
#Out[8]: tensor(0.1937, grad_fn=<MeanBackward1>)    
#logs_crps
#Out[10]: tensor(0.6518, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[11]: tensor(0.2415, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[12]: tensor(-0.7606, grad_fn=<MeanBackward1>)
#but this is very similar to 50points case ``````
#-----------------------------------------
#we want to try larger lr now : learning_rate=5 if i >1:lr=2 fixed 
#something wrong, since I omit something in the validation computation
#try same again
#AMAZING!!!!!! this work well!!!!CRPS:0.18113
#mse_crps
#Out[8]: tensor(0.1632, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[10]: tensor(0.6026, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[11]: tensor(0.2216, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[12]: tensor(-0.8082, grad_fn=<MeanBackward1>)
#---------------------------------------------
#but seems unstable (itr 900), same choice can also give
#mse_crps
#Out[19]: tensor(0.1760, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[20]: tensor(0.7681, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[21]: tensor(0.2253, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[22]: tensor(-0.6436, grad_fn=<MeanBackward1>)
#let's try again and use itr=750
#yes the results are really not stable

#==============================================================================
# # inducing points 100. itr900
# lr:  learning_rate=5 if i >1: lr=2
#==============================================================================

#let's try <<200>> points now !
#first use above choice
#but get similar (actually worse) results than 100 points above
#-----------------------
#try bigger lr  learning_rate=5 if i >1: lr=3
#this works but after around 230 , validation slightly fluctuate
#gives crps :0.15894 and  better mse but worse log score
#mse_crps
#Out[8]: tensor(0.1471, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6114, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2100, grad_fn=<MeanBackward1>)
#MSLL_crps
#Out[12]: tensor(-0.8037, grad_fn=<MeanBackward1>)
#-------------------------------
#try  lr  learning_rate=5 if i >1: lr=3 if i >230: lr=2
#around 310, validation start to fluctuate , gives CRPS:0.15042
#results are better than 100 in terms of mse and crps but worse in terms of log score
#------------------------------------------
#change to learning_rate=5 if i >1: lr=3 if i >230: lr=1
#again , better mseand crps  but worse log score and worse msll:
#mse_crps
#Out[8]: tensor(0.1476, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6651, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2105, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.7449, grad_fn=<MeanBackward1>)   
#---------------------------------------------
#let's increase itr a little bit and use smaller last lr 
#itr=800  learning_rate=5 if i >1: lr=3 if i >230: lr=1 , if i>320:lr=0.1
#now, no more flutuation
#mse_crps
#Out[9]: tensor(0.1497, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[10]: tensor(0.6598, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[11]: tensor(0.2119, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[12]: tensor(-0.7503, grad_fn=<MeanBackward1>)
#but results are strange
#-------------------------------------------
#try all the way lr=2
#curve looks good ,but results are similar to 100 points``
#---------------------------------
#try #itr=800  learning_rate=5 if i >1: lr=3 if i >230: lr=2 , if i>320:lr=1
#gives better results although fluctuate slightly after 400
#mse_crps
#Out[7]: tensor(0.1446, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[8]: tensor(0.6318, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[9]: tensor(0.2064, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[10]: tensor(-0.7806, grad_fn=<MeanBackward1>)
#-----------------------------------------------
#try #itr=800  learning_rate=5 if i >1: lr=3 if i >230: lr=2 , if i>320:lr=1--if i>380:lr=0.5
#results are ````really bad `````
#-----------------------
#above two means `````we may want to let lr=1 continue more
#try #itr=900 ! learning_rate=5 if i >1: lr=3 if i >230: lr=2 , if i>320:lr=1--if i>480:lr=0.5
#curve looks good but results are similar to above 
#it seems too large itr is not good
#------------------------------
#try itr=750  learning_rate=5 if i >1: lr=2.5
#still not good , very large log score and MSLL
#------------------------
#try a strange one: itr=900  learning_rate=5-- if i >1: lr=2--if i >400: lr=3
#this ```````actually gives a very good results in all terms:
#mse_crps
#Out[25]: tensor(0.1512, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[26]: tensor(0.5564, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[27]: tensor(0.2118, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[28]: tensor(-0.8584, grad_fn=<MeanBackward1>)    
#let's try same again to see if it is stable
#-----------------------------
#seems not stable, try itr=900  learning_rate=5-- if i >1: lr=2--if i >400: lr=1
#sh*t results```
#---------------------------
#try itr=900  learning_rate=5-- if i >1: lr=3--if i >400: lr=2
#still, better mse and crps but much worse log score and msll
#-----------------------
#try itr=900  learning_rate=5-- if i >1: lr=3--if i >320: lr=2
#gives better mse and crps but worse msll and log score.
#---------------------------
#mse_crps
#Out[27]: tensor(0.1478, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[28]: tensor(0.6464, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[29]: tensor(0.2095, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[30]: tensor(-0.7639, grad_fn=<MeanBackward1>)
#----------------------------
##try itr=1000  learning_rate=5-- if i >1: lr=3--if i >320: lr=1
#fuctuate
#-----------------------------
#ai```I go back and check 100points to see if stbale
#yes, 100points' results are not stable
#---------------------------
#go back check 200 points
#and increase itr=1200
#give good results  !!!!!!!!!!!
#mse_crps
#Out[17]: tensor(0.1596, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[18]: tensor(0.7257, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[19]: tensor(0.2178, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[20]: tensor(-0.6858, grad_fn=<MeanBackward1>)
#---------------------------
#try again using larger itr=1400
#gives CRPS:0.14034  sigma_noise: 0.02380
#mse_crps
#Out[28]: tensor(0.1527, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[29]: tensor(0.7289, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[30]: tensor(0.2130, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[31]: tensor(-0.6846, grad_fn=<MeanBackward1>)
#let's use this
#==============================================================================
# inducing points :200  CRPS:0.14034  sigma_noise: 0.02380
#lr=5-->if i >1: lr=2
#==============================================================================

#now use <<<<<<<300>>>>>>>!
#use above 
#all terms except mse is improving 
#------------------------
#try itr= 900 #lr=5-->if i >1: lr=2 -->if i>320, lr=1
#give good results 
#mse_crps
#Out[17]: tensor(0.1444, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[18]: tensor(0.6920, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[19]: tensor(0.2096, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[20]: tensor(-0.7185, grad_fn=<MeanBackward1>)
#---------------------
#let's increase itr=1200
#it indeed improved !!!!!!!!!!!!!!!!!!!!
#mse_crps
#Out[27]: tensor(0.1405, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[28]: tensor(0.5899, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[29]: tensor(0.2052, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[30]: tensor(-0.8218, grad_fn=<MeanBackward1>)
#--------------------
#how about increase to itr=1400
#well, not as good as above, let's try the same again

#mse_crps
#Out[7]: tensor(0.1424, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[8]: tensor(0.6274, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[9]: tensor(0.2072, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[10]: tensor(-0.7830, grad_fn=<MeanBackward1>)
#--------------------------
#==============================================================================
# # inducing points :300 itr=1400  CRPS:  sigma_noise:
#lr=5-->if i >1: lr=2-->if i>320, lr=1 
#==============================================================================

# <<<<<<<<<<<<400>>>>>>>>>>>
#try  lr=5-->if i >1: lr=2-- itr=1400
#decay to slow!!!!!!!!!!!
#------------------------------
#try  lr=5-->if i >1: lr=3---->if i>200: lr=2
#curve might(?) improved , try longer large lr
#---------------------------
#try  lr=5-->if i >1: lr=3---->if i>350: lr=2
#well, mse and crps improve slightly but other terms much worse!
#mse_crps
#Out[17]: tensor(0.1256, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[18]: tensor(0.7003, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[19]: tensor(0.1949, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[20]: tensor(-0.7103, grad_fn=<MeanBackward1>)
#--------------------------
#try  lr=5-->if i >1: lr=3 fixed  and itr 1400
#results are really bad, it seems if lr is too large, then noise will be small,
#then all terms will be bad
#-----------------------------
#try lr=2 fixed
#not as good as 300 points
#-------------------------
#might because itr is too large?????
#decrease to itr=1200
#no better than 300 points
#---------------------
#try lr=3 fixed   and itr=1200
#mse_crps
#Out[8]: tensor(0.1382, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[9]: tensor(0.6093, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[10]: tensor(0.2028, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[11]: tensor(-0.8043, grad_fn=<MeanBackward1>)
#well, accpetable````
#let's use this

#==============================================================================
# inducing points 400 CRPS:0.13557 sigma_noise: 0.02125
#lr=3 fixed   and itr=1200
#==============================================================================

#`````````<<<<<<<<500>>>>>>>>>>>>>
#try above method
#not very good
#----------------------
#try a small lr but longer itr
#try lr=0.5 fixed   and itr=1500
#results shows mse and crps worse but log and msll better
#-------------------------
#try lr=2.5 fixed   and itr=1200
#this time mse worse but all other terms better
#mse_crps
#Out[13]: tensor(0.1424, grad_fn=<MeanBackward1>)
#
#logs_crps
#Out[14]: tensor(0.5802, grad_fn=<MeanBackward1>)
#
#crps_crps
#Out[15]: tensor(0.2047, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[16]: tensor(-0.8337, grad_fn=<MeanBackward1>)


















#k_ff,Q_ff,k_uu,k_uf,k_star_u,k_ss,Q_ss,num_test,num_train,train_y
#k_ff,Q_ff,k_uu,k_uf,k_uf.transpose(0,1),k_ff,Q_ff,num_train,num_train,train_y
k1,Q1,k2,k3,k4,k5,Q2,num_test,num_jitter,data_y=k_ff,Q_ff,k_uu,k_uf,k_uf.transpose(0,1),k_ff,Q_ff,num_train,num_train,train_y
#
     G=(torch.diag(
               k1-Q1+sigma_noise_sq*torch.eye(num_jitter)
                  )*torch.eye(num_jitter)
                    ).type(dtype)
     print('a')
     Lam=k2+k3.mm(chol_solve(k3.transpose(0,1),G))
     #Lam=Lam+0.1*torch.eye(num_inducing)
     #torch.symeig(Lam)[0][0]
     print('b')
     mean_term=k4.mm(chol_solve(k3,Lam)).mm(chol_solve(data_y,G))
     print('c')
     cov_term=sigma_noise_sq*torch.eye(num_test).type(dtype)+k5-Q2+k4.mm(chol_solve(k4.transpose(0,1),Lam))
     return mean_term,cov_term
 

#chol_solve(Q_ff,Q_ff+sigma_noise_sq*num_train)
#torch.symeig(Q_ff+sigma_noise_sq*num_train)[0]
    

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

def dss(m,c,shape1,data_y):
    mean_term=m
    cov_term=c 
    half_log_det_dds=torch.potrf(cov_term).diag().log().sum()
    dss_ave=torch.tensor([0.5*shape1*(np.log(2*np.pi))])+half_log_det_dds + 0.5*((data_y-mean_term).transpose(0,1)).mm(chol_solve(data_y-mean_term,cov_term))  
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

#k1,k2,k3,num,eye_num,data_y= k_ff,k_ff,k_ff,num_train,num_train,train_y
def cal_mean_and_cov(k1,k2,k3,num,eye_num,data_y):
     jittered_kff=k2+sigma_noise_sq*torch.eye(eye_num)
     res_mean=k1.mm(chol_solve(data_y,jittered_kff))
     #
     res_cov=sigma_noise_sq*torch.eye(num)+k3-k1.mm(chol_solve(k1.transpose(0,1),jittered_kff))
     return res_mean,res_cov
 
def SMSE(m,data_y,data_yp):
    mean_term=m
    mean_yp=data_yp.mean()      
    mean_pre_mse=((mean_term-data_y)**2).mean()
    mean_trivial_mse=((mean_yp-data_y)**2).mean()
    mean_smse=mean_pre_mse/mean_trivial_mse
    return mean_smse    
#==============================================================================
# 
#==============================================================================
import pandas as pd
import xlrd
import random
address1='H:\Project\DATA________________\kin40k.xlsx'
address2='G:\!帝国理工\M00000-Summer-Project\DATA________________\kin40k.xlsx'


xls=pd.ExcelFile(address2)



TT=30
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

mse_ES_series=np.zeros(TT)
smse_ES_series=np.zeros(TT)
logs_ES_series=np.zeros(TT)
crps_ES_series=np.zeros(TT)
MSLL_ES_series=np.zeros(TT)
res_ES_series=np.zeros(TT)


import random


for j in range(TT):
#    p=100*j
    #torch.manual_seed(p)
    
    random.seed(j*100)
    num_va=300
    sam=np.reshape(random.sample(range(0, 10000),500+num_va) , (500+num_va))
    full_x = pd.read_excel(io=xls,sheetname='trainx',header=None).values[sam,:]
    full_y = pd.read_excel(io=xls,sheetname='trainy',header=None).values[sam,:]
    test_x = pd.read_excel(io=xls,sheetname='testx',header=None).values[:500,:]
    test_y = pd.read_excel(io=xls,sheetname='testy',header=None).values[:500,:]
    #注意这里只用了2000个点,实在是太慢了····
    
    #torch.manual_seed(111)
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
    #num_inducing=300
    #num_va=va_x.shape[0]
    #num_total=num_train+num_test+num_va
    
## =============================================================================
## CRPS    
## =============================================================================
    itr=400
    CRPS_series=np.zeros(itr)
    CRPS_va_series=np.zeros(itr)
    length_series=np.zeros(itr) 
    k_series=np.zeros(itr) 
    noise_series=np.zeros(itr) 
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
#    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
#    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
#    para_l=torch.tensor([0.5202],requires_grad=True).type(dtype)
#    para_k=torch.tensor([0.4135],requires_grad=True).type(dtype)
#    para_noise=torch.tensor([-3.8598],requires_grad=True).type(dtype)   
#    -3.8598
#    para_l=torch.rand(1,requires_grad=True).type(dtype)
    para_k=torch.rand(1,requires_grad=True).type(dtype)
    para_noise=torch.rand(1,requires_grad=True).type(dtype)   
    
#    para_l=torch.tensor([ 1.4294,  1.2469,  0.5428,  0.7232,  0.4853,  0.3494,  0.3514,0.7236],requires_grad=True).type(dtype)
#    para_k=torch.tensor([0.4288],requires_grad=True).type(dtype)
#    para_noise=torch.tensor([-2.9886],requires_grad=True).type(dtype)   
    
    
    for i in range(itr):  
        learning_rate=1  #fixed 1.5 is ok    
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)   
        big_k=k_ff+sigma_noise_sq*torch.eye(num_train)
        k_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_k)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_k)/k_ii_diag
        cov_term=1/k_ii_diag 
#        mean_term,cov_term=cal_mean_and_cov(k_ff,k_ff,k_ff,num_train,eye_num=num_train,data_y=train_y)        
#        cov_term=cov_term.diag().view(num_train,1)       
        CRPS_ave=crps(mean_term,cov_term,train_y)
        
#        print('CRPS:%.5f' % CRPS_ave)
#        CRPS_series[i]=CRPS_ave.detach().numpy() #store values 
#        length_series[i]=torch.exp(para_l).pow(0.5).detach().numpy()
#        noise_series[i]=torch.exp(para_noise).pow(0.5).detach().numpy()   
#        k_series[i]=torch.exp(para_k).pow(0.5).detach().numpy()          
        CRPS_ave.backward()
        
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
#            print('iteration:%d, k: %.5f, sigma_noise: %.5f' % (
#            i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#            ))
##        validation
#        k_ff =ARD(train_x,train_x,para_k,para_l)
#        k_star_f =ARD(test_x,train_x,para_k,para_l)
#        k_f_star = ARD(train_x,test_x,para_k,para_l)  
#        k_ss =ARD(test_x,test_x,para_k,para_l)
#    
#        mean_term,cov_term=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)        
#        cov_term=cov_term.diag().view(num_test,1)   
#        
#        CRPS_va=crps(mean_term,cov_term,test_y)
#        CRPS_va_series[i]=CRPS_va.detach().numpy()



    k_ff =ARD(train_x,train_x,para_k,para_l)
    k_star_f =ARD(test_x,train_x,para_k,para_l)
    k_ss =ARD(test_x,test_x,para_k,para_l)
    
    
    y_mean_crps,y_cov_crps=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)     
    y_cov_crps_diag =(y_cov_crps).diag().view(num_test,1)
    
    
    crps_mse=((y_mean_crps-test_y)**2).mean()
    # ================
    smse_crps=SMSE(y_mean_crps,test_y,train_y)
    # ================
    crps_test_logs=logs( y_mean_crps,y_cov_crps_diag,test_y)
    # ================
    crps_test_crps=crps( y_mean_crps,y_cov_crps_diag,test_y)  
    # ================
    MSLL_crps=trivial_loss(y_mean_crps,y_cov_crps_diag,test_y,train_y)   



    up=y_mean_crps+2*y_cov_crps_diag**(0.5)
    low=y_mean_crps-2*y_cov_crps_diag**(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()

    mse_crps_series[j]=crps_mse.detach().numpy()
    smse_crps_series[j]=smse_crps.detach().numpy()
    logs_crps_series[j]=crps_test_logs.detach().numpy()
    crps_crps_series[j]=crps_test_crps.detach().numpy()
    MSLL_crps_series[j]=MSLL_crps.detach().numpy()
    res_crps_series[j]=res



#plt.plot(CRPS_va_series,c='purple')
#
#plt.plot(CRPS_series,c='blue')

#torch.exp(para_l).pow(0.5)
#tensor([[1.8286, 1.7127, 1.2076, 1.3934, 1.4514, 1.2180, 1.2158, 1.5701]]
# =============================================================================
#           NLML
# =============================================================================
###############################
    itr=400
    Neg_logL_series=np.zeros(itr)
    Neg_logL_va_series=np.zeros(itr)
    length_series=np.zeros(itr) 
    k_series=np.zeros(itr) 
    noise_series=np.zeros(itr) 
    
    
    
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
#    para_l=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    
    
    for i in range(itr):  #best 1000
        learning_rate=0.0005
    #    learning_rate=0.001
    #    if i>260:
    #        learning_rate=0.0001
    #    elif i>80:
    #        learning_rate=0.00001      
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)
        inverse_term_ml=k_ff+torch.eye(train_x.shape[0])*sigma_noise_sq
        half_log_det=torch.potrf(inverse_term_ml).diag().log().sum()
    
        Neg_logL=torch.tensor([0.5*train_x.shape[0]*(np.log(2*np.pi))])+half_log_det + 0.5*(train_y.transpose(0,1)).mm(chol_solve(train_y,inverse_term_ml))
#        Neg_logL_series[i]=Neg_logL.detach().numpy()
#        length_series[i]=torch.exp(para_l).pow(0.5).detach().numpy()
#        noise_series[i]=torch.exp(para_noise).pow(0.5).detach().numpy()   
#        k_series[i]=torch.exp(para_k).pow(0.5).detach().numpy()   
        Neg_logL.backward()
    
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
#            print('iteration:%d,Neg_logL:%5f,sigma_noise: %.5f,K: %.5f' % (
#            i,Neg_logL,torch.exp(para_noise).pow(0.5),torch.exp(para_k).pow(0.5)
#            ))
#        k_ff = ARD(va_x,va_x,para_k,para_l)
#        inverse_term_ml_va=k_ff+torch.eye(va_x.shape[0])*sigma_noise_sq
#        half_log_det_va=torch.potrf(inverse_term_ml_va).diag().log().sum()
#    
#        Neg_logL_va=torch.tensor([0.5*va_x.shape[0]*(np.log(2*np.pi))])+half_log_det_va + 0.5*(va_y.transpose(0,1)).mm(chol_solve(va_y,inverse_term_ml_va))
#        Neg_logL_va_series[i]=Neg_logL_va.detach().numpy()
    
    
#    #
#    plt.plot(Neg_logL_va_series,c='purple')
#    plt.plot(Neg_logL_series)
    
    
    
    k_ff = ARD(train_x,train_x,para_k,para_l)
    k_f_star = ARD(train_x,test_x,para_k,para_l)  
    k_star_f = ARD(test_x,train_x,para_k,para_l)
    k_ss =ARD(test_x,test_x,para_k,para_l)
    
    mean_gp=k_star_f.mm(chol_solve(train_y,k_ff+torch.eye(train_x.shape[0])*sigma_noise_sq))
    cov_gp =torch.eye(test_x.shape[0])*sigma_noise_sq+k_ss-k_star_f.mm(chol_solve(k_f_star,k_ff+torch.eye(train_x.shape[0])*sigma_noise_sq))
    cov_gp_diag=cov_gp.diag().view(-1,1)
    
    
    
    #(mean_gp-test_y).pow(2).sum()/num_test  #tensor(0.0524,
    
    mse_gp=((mean_gp-test_y)**2).mean()
    # ================
    smse_gp=SMSE(mean_gp,test_y,train_y)
    # ================      
    logs_gp=logs(mean_gp, cov_gp_diag,test_y)
    # ================
    crps_gp=crps(mean_gp, cov_gp_diag,test_y)  
    # ================
    MSLL_gp=trivial_loss(mean_gp,cov_gp_diag,test_y,train_y)
    
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
    
    
#    
#torch.exp(para_l).pow(0.5)
#[[2.0080, 1.7985, 1.2335, 1.3406, 1.3110, 1.2163, 1.1965, 1.4563]]
##########################################
# =============================================================================
#    Logs
# =============================================================================
    itr=500
    logs_series=np.zeros(itr)
    logs_va_series=np.zeros(itr)
    
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    
    
    for i in range(itr): 
        learning_rate=0.05
        sigma_noise_sq=torch.exp(para_noise)
    
        k_ff = ARD(train_x,train_x,para_k,para_l)
        big_k=k_ff+sigma_noise_sq*torch.eye(num_train)
        k_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_k)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_k)/k_ii_diag
        cov_term=1/k_ii_diag
#        mean_term,cov_term=cal_mean_and_cov(k_ff,k_ff,k_ff,num_train,eye_num=num_train,data_y=train_y)        
#        cov_term=cov_term.diag().view(num_train,1) 
        logs_ave=logs(mean_term,cov_term,train_y) 
        
#        print('Log-Score:%.5f'  % logs_ave)
#        logs_series[i]=logs_ave.detach().numpy() #store values
        logs_ave.backward()
          
        with torch.no_grad():
            para_l -= learning_rate * para_l.grad
            para_k -= learning_rate * para_k.grad
            para_noise -= learning_rate * para_noise.grad
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
#            print('iteration:%d,sigma_k_square: %.5f sigma_noise: %.5f' % (
#            i,torch.exp(para_k),torch.exp(para_noise).pow(0.5)
#            ))
#        
#        k_ff =ARD(train_x,train_x,para_k,para_l)
#        k_star_f =ARD(va_x,train_x,para_k,para_l)
#        k_f_star = ARD(train_x,va_x,para_k,para_l)  
#        k_ss =ARD(va_x,va_x,para_k,para_l)
#        
#        m_logs_vector,cov_logs_vector=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_va,eye_num=num_train,data_y=train_y)        
#        cov_logs_vector=cov_logs_vector.diag().view(num_va,1) 
#        logs_ave=logs(m_logs_vector,cov_logs_vector,va_y)
#        logs_va_series[i]=logs_ave.detach().numpy()
        
    k_ff =ARD(train_x,train_x,para_k,para_l)
    k_star_f =ARD(test_x,train_x,para_k,para_l)
    k_ss =ARD(test_x,test_x,para_k,para_l)
    
    
    mean_logs,cov_logs=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)     
    cov_logs_diag =(cov_logs).diag().view(num_test,1)
    
    
    logs_mse=((mean_logs-test_y)**2).mean()
    # ================
    logs_smse=SMSE(mean_logs,test_y,train_y)
    # ================
    logs_test_logs=logs( mean_logs,cov_logs_diag,test_y)
    # ================
    logs_test_crps=crps( mean_logs,cov_logs_diag,test_y)  
    # ================
    MSLL_logs=trivial_loss(mean_logs,cov_logs_diag,test_y,train_y)

    up=mean_logs+2*cov_logs_diag**(0.5)
    low=mean_logs-2*cov_logs_diag**(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()
    
    
    logs_mse_series[j]=logs_mse.detach().numpy()
    smse_logs_series[j]=logs_smse.detach().numpy()
    logs_test_logs_series[j]=logs_test_logs.detach().numpy()
    logs_test_crps_series[j]=logs_test_crps.detach().numpy()
    MSLL_logs_series[j]=MSLL_logs.detach().numpy()
    res_logs_series[j]=res
    
# =============================================================================
#   DSS
# =============================================================================
    itr=150
    dss_series=np.zeros(itr)
    dss_va_series=np.zeros(itr)
    
#    torch.manual_seed(p)    
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    
    
    for i in range(itr):  
        learning_rate=0.001
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)     
        fold_k=4
        index1=int(num_train/fold_k)
        index2=int(2*num_train/fold_k) 
        index3=int(3*num_train/fold_k) 
            
        big_k=k_ff+sigma_noise_sq*torch.eye(num_train)
        k_inv_i_j=chol_solve(torch.eye(num_train),big_k)
        k_1=k_inv_i_j[:index1,:index1]
        k_2=k_inv_i_j[index1:index2,index1:index2]
        k_3=k_inv_i_j[(index2):index3,(index2):index3]
        k_4=k_inv_i_j[index3:,index3:]        

        y_1=train_y[:index1]
        y_2=train_y[index1:index2]
        y_3=train_y[(index2):index3]     
        y_4=train_y[index3:]  


        k_inv_y=chol_solve(train_y,big_k)
    
        m_1=y_1-chol_solve(torch.eye(index1),k_1).mm(k_inv_y[:index1])
        m_2=y_2-chol_solve(torch.eye(index1),k_2).mm(k_inv_y[index1:index2])
        m_3=y_3-chol_solve(torch.eye(index1),k_3).mm(k_inv_y[(index2):index3])
        m_4=y_4-chol_solve(torch.eye(index1),k_4).mm(k_inv_y[index3:])        
        

        cov_1=chol_solve(torch.eye(index1),k_1)
        cov_2=chol_solve(torch.eye(index1),k_2)
        cov_3=chol_solve(torch.eye(index1),k_3)        
        cov_4=chol_solve(torch.eye(index1),k_4)    
        
         
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
            para_l.grad.zero_()  #set gradients to zero 
            para_k.grad.zero_()
            para_noise.grad.zero_()
#            print('iteration:%d,sigma_k: %.5f sigma_noise: %.5f' % (
#            i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5)
#            ))
#        
#        k_ff =ARD(train_x,train_x,para_k,para_l)
#        k_star_f =ARD(va_x,train_x,para_k,para_l)
#        k_f_star = ARD(train_x,va_x,para_k,para_l)  
#        k_ss =ARD(va_x,va_x,para_k,para_l)
#    
#        mean_term,cov_term=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_va,eye_num=num_train,data_y=train_y)            
#        dss_va=dss(mean_term,cov_term,num_va,va_y)   
#        dss_va_series[i]=dss_va.detach().numpy()
      
    
    
    # =============================================================================
    # predict
    # =============================================================================
    k_ff =ARD(train_x,train_x,para_k,para_l)
    k_star_f =ARD(test_x,train_x,para_k,para_l)
    k_ss =ARD(test_x,test_x,para_k,para_l)
    
    
    mean_dss,cov_dss=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)     
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
    low=mean_dss-2*cov_dss_diag**(0.5)
    a=((up-test_y)>0).numpy()
    b=((test_y-low)>0).numpy()
    res=np.multiply(a,b).mean()
    
    
    mse_dss_series[j]=dss_mse.detach().numpy()
    smse_dss_series[j]=smse_dss.detach().numpy()
    logs_dss_series[j]=dss_test_logs.detach().numpy()
    crps_dss_series[j]=dss_test_crps.detach().numpy()
    MSLL_dss_series[j]=MSLL_dss.detach().numpy()
    res_dss_series[j]=res


# =============================================================================
# ES
# =============================================================================
    itr=25
    ES_series=np.zeros(itr)
    ES_va_series=np.zeros(itr)
    
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype)
    
    try:
        for i in range(itr):  
             learning_rate=0.1
             sigma_noise_sq=torch.exp(para_noise)
             k_ff = ARD(train_x,train_x,para_k,para_l)     
             fold_k=4
             index1=int(num_train/fold_k)
             index2=int(2*num_train/fold_k) 
             index3=int(3*num_train/fold_k) 
                 
             big_k=k_ff+sigma_noise_sq*torch.eye(num_train)
             k_inv_i_j=chol_solve(torch.eye(num_train),big_k)
             k_1=k_inv_i_j[:index1,:index1]
             k_2=k_inv_i_j[index1:index2,index1:index2]
             k_3=k_inv_i_j[(index2):index3,(index2):index3]
             k_4=k_inv_i_j[index3:,index3:]        
     
             y_1=train_y[:index1]
             y_2=train_y[index1:index2]
             y_3=train_y[(index2):index3]     
             y_4=train_y[index3:]  
     
     
             k_inv_y=chol_solve(train_y,big_k)
         
             m_1=y_1-chol_solve(torch.eye(index1),k_1).mm(k_inv_y[:index1])
             m_2=y_2-chol_solve(torch.eye(index1),k_2).mm(k_inv_y[index1:index2])
             m_3=y_3-chol_solve(torch.eye(index1),k_3).mm(k_inv_y[(index2):index3])
             m_4=y_4-chol_solve(torch.eye(index1),k_4).mm(k_inv_y[index3:])        
             
     
             cov_1=chol_solve(torch.eye(index1),k_1)
             cov_2=chol_solve(torch.eye(index1),k_2)
             cov_3=chol_solve(torch.eye(index1),k_3)        
             cov_4=chol_solve(torch.eye(index1),k_4)    
         
            
             ES1=ES(m_1,cov_1,index1,y_1,300)
             ES2=ES(m_2,cov_2,index1,y_2,300)
             ES3=ES(m_3,cov_3,index1,y_3,300)
             ES4=ES(m_4,cov_4,index1,y_4,300)
            
             ES_ave=(ES1+ES2+ES3+ES4).mean()
             
#             ES_series[i]=ES_ave.detach().numpy()
             
#             print('ES:%.5f'  % ES_ave)
            
             ES_ave.backward()
              
             with torch.no_grad():
                  
                 para_l -= learning_rate * para_l.grad
                 para_k -= learning_rate * para_k.grad
                 para_noise -= learning_rate * para_noise.grad
                 para_l.grad.zero_()  #set gradients to zero 
                 para_k.grad.zero_()
                 para_noise.grad.zero_()
#                 print('iteration:%d,sigma_k_square: %.5f sigma_noise: %.5f' % (
#                 i,torch.exp(para_k),torch.exp(para_noise).pow(0.5)
#                  ))
#            
#             k_ff =ARD(train_x,train_x,para_k,para_l)
#             k_star_f =ARD(va_x,train_x,para_k,para_l)
#             k_f_star = ARD(train_x,va_x,para_k,para_l)  
#             k_ss =ARD(va_x,va_x,para_k,para_l)
#         
#             mean_term,cov_term=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_va,eye_num=num_train,data_y=train_y)            
#             ES_va=ES(mean_term,cov_term,num_va,va_y)   
#             ES_va_series[i]=ES_va.detach().numpy()
          
#plt.plot(ES_va_series,c='purple')
#plt.plot(ES_series,c='blue')
        
        
        # =============================================================================
        # predict
        # =============================================================================
        k_ff =ARD(train_x,train_x,para_k,para_l)
        k_star_f =ARD(test_x,train_x,para_k,para_l)
        k_ss =ARD(test_x,test_x,para_k,para_l)
        
        
        mean_ES,cov_ES=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)     
        cov_ES_diag =(cov_ES).diag().view(num_test,1)
        
        
        ES_mse=((mean_ES-test_y)**2).mean()
        # ================
        smse_ES=SMSE(mean_ES,test_y,train_y)
        # ================
        ES_test_logs=logs(mean_ES,cov_ES_diag,test_y)
        # ================
        ES_test_crps=crps( mean_ES,cov_ES_diag,test_y)  
        # ================
        MSLL_ES=trivial_loss(mean_ES,cov_ES_diag,test_y,train_y)

        up=mean_ES+2*cov_ES_diag**(0.5)
        low=mean_ES-2*cov_ES_diag**(0.5)
        a=((up-test_y)>0).numpy()
        b=((test_y-low)>0).numpy()
        res=np.multiply(a,b).mean()

    
        mse_ES_series[j]=ES_mse.detach().numpy()
        smse_ES_series[j]=smse_ES.detach().numpy()
        logs_ES_series[j]=ES_test_logs.detach().numpy()
        crps_ES_series[j]=ES_test_crps.detach().numpy()
        MSLL_ES_series[j]=MSLL_ES.detach().numpy()
        res_ES_series[j]=res
        
    except RuntimeError:
        mse_ES_series[j]=0.
        smse_ES_series[j]=0.
        logs_ES_series[j]=0.
        crps_ES_series[j]=0.
        MSLL_ES_series[j]=0.
        res_ES_series[j]=0.



    print(j)








    mse_gp_series.mean()
    smse_gp_series.mean()
    logs_gp_series.mean()
    crps_gp_series.mean()
    MSLL_gp_series.mean()
    res_gp_series.mean()


    mse_crps_series.mean()
    smse_crps_series.mean()
    logs_crps_series.mean()
    crps_crps_series.mean()
    MSLL_crps_series.mean()
    res_crps_series.mean()


    logs_mse_series.mean()
    smse_logs_series.mean()
    logs_test_logs_series.mean()
    logs_test_crps_series.mean()
    MSLL_logs_series.mean()
    res_logs_series.mean()



    mse_dss_series.mean()
    smse_dss_series.mean()
    logs_dss_series.mean()
    crps_dss_series.mean()
    MSLL_dss_series.mean()
    res_dss_series.mean()

    mse_ES_series.mean()
    smse_ES_series.mean()
    logs_ES_series.mean()
    crps_ES_series.mean()
    MSLL_ES_series.mean()
    res_ES_series.mean()

# =============================================================================
# 
# =============================================================================
np.std(   mse_gp_series)
np.std(  smse_gp_series)
np.std(   logs_gp_series)
np.std(  crps_gp_series)
np.std(   MSLL_gp_series)
np.std(   res_gp_series)


np.std(    mse_crps_series)
np.std(    smse_crps_series)
np.std(    logs_crps_series)
np.std(    crps_crps_series)
np.std(    MSLL_crps_series)
np.std(    res_crps_series)


np.std(    logs_mse_series)
np.std(    smse_logs_series)
np.std(    logs_test_logs_series)
np.std(    logs_test_crps_series)
np.std(    MSLL_logs_series)
np.std(    res_logs_series)



np.std(    mse_dss_series)
np.std(    smse_dss_series)
np.std(    logs_dss_series)
np.std(    crps_dss_series)
np.std(    MSLL_dss_series)
np.std(    res_dss_series)

np.std(    mse_ES_series)
np.std(    smse_ES_series)
np.std(    logs_ES_series)
np.std(    crps_ES_series)
np.std(    MSLL_ES_series)
np.std(    res_ES_series)






# =============================================================================
# 
# =============================================================================

np.std(   mse_gp_series)
Out[194]: 0.019331347793634823

np.std(  smse_gp_series)
Out[195]: 0.018259755363716112

np.std(   logs_gp_series)
Out[196]: 0.0597642008245582

np.std(  crps_gp_series)
Out[197]: 0.011400494824306212

np.std(   MSLL_gp_series)
Out[198]: 0.05934881474973117

np.std(   res_gp_series)
Out[199]: 0.016839239887833393


# =============================================================================
# 
# =============================================================================
np.std(    mse_crps_series)
Out[200]: 0.017059731326566258

np.std(    smse_crps_series)
Out[201]: 0.016154467812622277

np.std(    logs_crps_series)
Out[202]: 0.05536075909763593

np.std(    crps_crps_series)
Out[203]: 0.01023672656297403

np.std(    MSLL_crps_series)
Out[204]: 0.05523199326374739

np.std(    res_crps_series)
Out[205]: 0.017258299130820737
# =============================================================================
# 
# =============================================================================

np.std(    logs_mse_series)
Out[206]: 0.019626960207587373

np.std(    smse_logs_series)
Out[207]: 0.01857484923252598

np.std(    logs_test_logs_series)
Out[208]: 0.05526667138836927

np.std(    logs_test_crps_series)
Out[209]: 0.011298357682713287

np.std(    MSLL_logs_series)
Out[210]: 0.055079895958080854

np.std(    res_logs_series)
Out[211]: 0.01616250805619802

# =============================================================================
# 
# =============================================================================

np.std(    mse_dss_series)
Out[212]: 0.01773604051925016

np.std(    smse_dss_series)
Out[213]: 0.016801847967220174

np.std(    logs_dss_series)
Out[214]: 0.05735514229841237

np.std(    crps_dss_series)
Out[215]: 0.010541159756351379

np.std(    MSLL_dss_series)
Out[216]: 0.05720909941209938

np.std(    res_dss_series)
Out[217]: 0.01750796644070601



# =============================================================================
# 
# =============================================================================

np.std(    mse_ES_series)
Out[218]: 0.020883023380594414

np.std(    smse_ES_series)
Out[219]: 0.019712098689868517

np.std(    logs_ES_series)
Out[220]: 0.05345971066473978

np.std(    crps_ES_series)
Out[221]: 0.0117812676576289

np.std(    MSLL_ES_series)
Out[222]: 0.052974746669083084

np.std(    res_ES_series)
Out[223]: 0.015549776704363156


# =============================================================================
# 
# =============================================================================







smse_to_plot = [ mse_gp_series, mse_crps_series,logs_mse_series,mse_dss_series,mse_ES_series]
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(smse_to_plot, patch_artist=True)
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )
## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)
## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
ax.set_xticklabels(['NLML', 'CRPS', 'Logs', 'Dss','ES'])
boxColors = ['darkkhaki', 'royalblue']
plt.figtext(0.4, 0.83, 'MSE',backgroundcolor=boxColors[0],
            color='black', weight='roman',
            size='x-large')
plt.axhline(y=np.median( mse_gp_series), xmin=0, xmax=100, linewidth=3, color = 'red',linestyle='--')








# =============================================================================
# NLML   30
# =============================================================================
mse_gp_series.mean()
Out[21]: 0.17874876111745835

smse_gp_series.mean()
Out[22]: 0.169037564098835

logs_gp_series.mean()
Out[23]: 0.48336944778760277

crps_gp_series.mean()
Out[24]: 0.2267873485883077

MSLL_gp_series.mean()
Out[25]: -0.9654809772968292

res_gp_series.mean()
Out[26]: 0.9478000000000001

# =============================================================================
# CRPS 30
# =============================================================================
mse_crps_series.mean()
Out[27]: 0.17637162208557128

smse_crps_series.mean()
Out[28]: 0.16679306228955587

logs_crps_series.mean()
Out[29]: 0.4762532909711202

crps_crps_series.mean()
Out[30]: 0.22395426481962205

MSLL_crps_series.mean()
Out[31]: -0.9725971341133117

res_crps_series.mean()
Out[32]: 0.9402666666666667

# =============================================================================
# logs 30
# =============================================================================

logs_mse_series.mean()
Out[33]: 0.17682222525278726

smse_logs_series.mean()
Out[34]: 0.16721940239270527

logs_test_logs_series.mean()
Out[35]: 0.4798559715350469

logs_test_crps_series.mean()
Out[36]: 0.22517957439025243

MSLL_logs_series.mean()
Out[37]: -0.9689944545427959

res_logs_series.mean()
Out[38]: 0.9509333333333333

# =============================================================================
#  dss   30
# =============================================================================
mse_dss_series.mean()
Out[39]: 0.17628994931777317

smse_dss_series.mean()
Out[40]: 0.16671679069598516

logs_dss_series.mean()
Out[41]: 0.47374741633733114

crps_dss_series.mean()
Out[42]: 0.22393851627906164

MSLL_dss_series.mean()
Out[43]: -0.9751030206680298

res_dss_series.mean()
Out[44]: 0.9438666666666667

# =============================================================================
#   ES   30
# =============================================================================
mse_ES_series.mean()
Out[45]: 0.18499006231625875

smse_ES_series.mean()
Out[46]: 0.17493749409914017

logs_ES_series.mean()
Out[47]: 0.5082949807246526

crps_ES_series.mean()
Out[48]: 0.23157887061436971

MSLL_ES_series.mean()
Out[49]: -0.9405554473400116

res_ES_series.mean()
Out[50]: 0.9598000000000001


# =============================================================================
# 
# =============================================================================










































































# =============================================================================
#      NLML
# =============================================================================
mse_gp_series[:10]
Out[46]: 
array([0.20494711, 0.18490061, 0.15973766, 0.17020451, 0.15232079,
       0.1679159 , 0.16035412, 0.20431297, 0.20590092, 0.17218563])

smse_gp_series[:10]
Out[47]: 
array([0.19362718, 0.17511936, 0.1512273 , 0.16107684, 0.14414982,
       0.15886398, 0.1518119 , 0.19145013, 0.19474804, 0.16307056])

logs_gp_series[:10]
Out[48]: 
array([0.51927656, 0.48443443, 0.41038477, 0.46625751, 0.42478681,
       0.47704172, 0.45043802, 0.59736335, 0.53439814, 0.44369379])

crps_gp_series[:10]
Out[49]: 
array([0.23299029, 0.22668591, 0.21339507, 0.21994202, 0.20858063,
       0.22356336, 0.21730372, 0.24583818, 0.23931946, 0.2233672 ])

MSLL_gp_series[:10]
Out[50]: 
array([-0.93338758, -0.96296602, -1.03594804, -0.98136646, -1.02330136,
       -0.96963215, -0.99599266, -0.85758972, -0.9126963 , -1.00301611])

res_gp_series[:10]
Out[89]: 
array([0.935, 0.941, 0.941, 0.946, 0.962, 0.961, 0.963, 0.912, 0.936,
       0.941])
# =============================================================================
# CRPS
# =============================================================================
mse_crps_series[:10]
Out[52]: 
array([0.21227118, 0.18285131, 0.16428521, 0.15777689, 0.15759508,
       0.17632897, 0.1624677 , 0.18480138, 0.19616975, 0.18412361])

smse_crps_series[:10]
Out[53]: 
array([0.20054673, 0.17317846, 0.15553255, 0.1493157 , 0.14914119,
       0.16682352, 0.15381289, 0.17316693, 0.18554397, 0.17437656])

logs_crps_series[:10]
Out[54]: 
array([0.55753088, 0.47753665, 0.45554996, 0.41316473, 0.44693455,
       0.52930129, 0.43002155, 0.61447239, 0.55137646, 0.48278207])

crps_crps_series[:10]
Out[55]: 
array([0.23677231, 0.22367655, 0.21702327, 0.20866343, 0.20852944,
       0.22962992, 0.21449646, 0.23480849, 0.23471522, 0.23055942])

MSLL_crps_series[:10]
Out[56]: 
array([-0.89513326, -0.96986383, -0.99078292, -1.03445923, -1.00115359,
       -0.91737258, -1.01640916, -0.84048069, -0.89571798, -0.96392781])

res_crps_series[:10]
Out[90]: 
array([0.918, 0.94 , 0.909, 0.941, 0.943, 0.911, 0.946, 0.871, 0.908,
       0.914])
# =============================================================================
#    Logs
# =============================================================================
logs_mse_series[:10]
Out[58]: 
array([0.21464799, 0.18787506, 0.15829051, 0.15973879, 0.1541741 ,
       0.17252862, 0.15700485, 0.18727463, 0.2034931 , 0.17662635])

smse_logs_series[:10]
Out[59]: 
array([0.20279226, 0.17793646, 0.14985724, 0.15117238, 0.14590374,
       0.16322805, 0.14864105, 0.17548448, 0.19247064, 0.1672762 ])

logs_test_logs_series[:10]
Out[60]: 
array([0.56311136, 0.5048961 , 0.42281324, 0.42395172, 0.42459032,
       0.50165308, 0.42209321, 0.59771663, 0.55935591, 0.47338557])

logs_test_crps_series[:10]
Out[61]: 
array([0.23777373, 0.2268627 , 0.2128863 , 0.21100913, 0.20679989,
       0.22670342, 0.2122933 , 0.23575063, 0.23848383, 0.22736928])

MSLL_logs_series[:10]
Out[62]: 
array([-0.88955277, -0.94250435, -1.02351964, -1.02367222, -1.02349782,
       -0.94502085, -1.02433753, -0.85723644, -0.88773859, -0.9733243 ])

res_logs_series[:10]
Out[91]: 
array([0.92 , 0.931, 0.932, 0.939, 0.95 , 0.928, 0.95 , 0.883, 0.915,
       0.922])
# =============================================================================
#   DSS
# =============================================================================
mse_dss_series
Out[70]: 
array([0.20978953, 0.18770874, 0.16212364, 0.15615094, 0.15501437,
       0.17217603, 0.16234821, 0.18604881, 0.20349658, 0.17775232])

smse_dss_series
Out[71]: 
array([0.19820213, 0.17777893, 0.15348615, 0.14777693, 0.14669892,
       0.16289446, 0.15369976, 0.17433582, 0.19247393, 0.16834255])

logs_dss_series
Out[72]: 
array([0.53170234, 0.4992004 , 0.43462378, 0.40727559, 0.42822856,
       0.48721257, 0.43212155, 0.54197717, 0.53903419, 0.44981447])

crps_dss_series
Out[73]: 
array([0.23351732, 0.2264486 , 0.21639471, 0.20808025, 0.20726426,
       0.22603652, 0.2146122 , 0.23381829, 0.23872347, 0.22597603])

MSLL_dss_series
Out[74]: 
array([-0.9209618 , -0.94820005, -1.01170909, -1.04034841, -1.01985955,
       -0.95946133, -1.01430917, -0.91297585, -0.90806031, -0.99689537])

res_dss_series
Out[92]: 
array([0.927, 0.931, 0.924, 0.951, 0.953, 0.956, 0.95 , 0.915, 0.931,
       0.939])

# =============================================================================
# ES
# =============================================================================
mse_ES_series
Out[76]: 
array([0.21907533, 0.19103699, 0.15788361, 0.1658728 , 0.15965638,
       0.17609306, 0.15582231, 0.18943388, 0.21024935, 0.1780706 ])

smse_ES_series
Out[77]: 
array([0.19754191, 0.19040212, 0.14947203, 0.15697743, 0.15109192,
       0.16660033, 0.1475215 , 0.17624865, 0.19886093, 0.168644  ])

logs_ES_series
Out[78]: 
array([0.54312192, 0.52595747, 0.41866058, 0.47293097, 0.47538561,
       0.50020534, 0.45771474, 0.54891845, 0.55003297, 0.47839785])

crps_ES_series
Out[79]: 
array([0.22179733, 0.23579612, 0.21371187, 0.21958305, 0.21779597,
       0.22950716, 0.21552391, 0.2283056 , 0.24315389, 0.22875834])

MSLL_ES_series
Out[80]: 
array([-0.92954221, -0.92144305, -1.02767229, -0.974693  , -0.97270256,
       -0.94646859, -0.98871601, -0.93603457, -0.89706147, -0.96831203])

res_ES_series
Out[93]: 
array([0.959, 0.951, 0.962, 0.961, 0.98 , 0.963, 0.977, 0.935, 0.944,
       0.968])








# =============================================================================
# 
# =============================================================================



mse_gp_series[:10]
Out[186]: 
array([0.18682963, 0.18776894, 0.15320325, 0.18394996, 0.20413372,
       0.17103751, 0.18179764, 0.1574425 , 0.17325251, 0.15816462])
smse_gp_series[:10]
Out[187]: 
array([0.17694336, 0.17734002, 0.14499979, 0.17391188, 0.19330852,
       0.16041365, 0.17188627, 0.14910987, 0.16408581, 0.14979436])
logs_gp_series[:10]
Out[188]: 
array([0.53252733, 0.51695776, 0.44096884, 0.48380935, 0.54609597,
       0.48458701, 0.50023609, 0.44507319, 0.47430226, 0.4205707 ])
crps_gp_series[:10]
Out[189]: 
array([0.23549446, 0.23360965, 0.21572009, 0.22575206, 0.24072798,
       0.22420466, 0.23193459, 0.21508241, 0.22489095, 0.215396  ])
MSLL_gp_series[:10]
Out[190]: 
array([-0.91372615, -0.93059713, -1.00581419, -0.96360648, -0.90257448,
       -0.97203219, -0.94792908, -1.00434554, -0.97901475, -1.02565038])
res_gp_series[:10]
Out[191]: 
array([0.666, 0.6  , 0.61 , 0.616, 0.58 , 0.576, 0.566, 0.586, 0.56 ,
       0.614])
    
    
    
mse_crps_series[:10]
Out[192]: 
array([0.1858834 , 0.20447122, 0.15292674, 0.18230848, 0.2079508 ,
       0.16645642, 0.17654875, 0.1671944 , 0.20313114, 0.15341856])
smse_crps_series[:10]
Out[193]: 
array([0.17604722, 0.19311462, 0.14473809, 0.17235997, 0.19692318,
       0.15611711, 0.16692354, 0.15834565, 0.19238357, 0.14529945])
logs_crps_series[:10]
Out[194]: 
array([0.51732069, 0.60794574, 0.43704948, 0.48805401, 0.57283014,
       0.49715036, 0.51435202, 0.50998658, 0.59102029, 0.40473738])

crps_crps_series[:10]
Out[195]: 
array([0.23090543, 0.24263169, 0.21451986, 0.22149755, 0.2421606 ,
       0.22149111, 0.22854105, 0.22135432, 0.24450675, 0.2100125 ])
MSLL_crps_series[:10]
Out[196]: 
array([-0.92893243, -0.83960909, -1.00973368, -0.95936209, -0.87584031,
       -0.95946896, -0.9338131 , -0.93943214, -0.86229706, -1.04148388])
res_crps_series[:10]
Out[197]: 
array([0.598, 0.452, 0.526, 0.528, 0.518, 0.512, 0.482, 0.482, 0.472,
       0.512])
    
    
    
logs_mse_series[:10]
Out[198]: 
array([0.1807999 , 0.19677463, 0.15367875, 0.18010382, 0.20425053,
       0.16635253, 0.17031893, 0.15416561, 0.17275038, 0.14977869])
smse_logs_series[:10]
Out[199]: 
array([0.17123272, 0.18584552, 0.14544983, 0.17027563, 0.19341913,
       0.15601967, 0.16103336, 0.14600642, 0.16361025, 0.14185221])
logs_test_logs_series[:10]
Out[200]: 
array([0.51118553, 0.56173903, 0.43870249, 0.47452095, 0.56657076,
       0.48689705, 0.48083279, 0.43707439, 0.48242268, 0.39359462])
logs_test_crps_series[:10]
Out[201]: 
array([0.23013763, 0.2386612 , 0.21508211, 0.2201459 , 0.2404103 ,
       0.22188173, 0.22386737, 0.21186507, 0.22570543, 0.2089376 ])
MSLL_logs_series[:10]
Out[202]: 
array([-0.93506783, -0.88581616, -1.00808048, -0.97289521, -0.88209957,
       -0.96972233, -0.96733236, -1.01234448, -0.97089452, -1.05262661])
res_logs_series[:10]
Out[203]: 
array([0.592, 0.486, 0.518, 0.54 , 0.53 , 0.514, 0.506, 0.532, 0.516,
       0.55 ])























































































torch.exp(para_k).pow(0.5)

para_k
Out[74]: tensor([ 0.4288])

para_l
Out[75]: 
tensor([[ 1.4294,  1.2469,  0.5428,  0.7232,  0.4853,  0.3494,  0.3514,
          0.7236]])

para_noise
Out[76]: tensor([-2.9886])








crps_mse
Out[119]: tensor(0.4023)

crps_test_crps
Out[120]: tensor(0.3539)

#
#
#LOOCV  ARD itr 2000
#crps_mse
#Out[556]: tensor(0.1698)
#
#smse_crps
#Out[557]: tensor(0.1607)
#
#crps_test_logs
#Out[558]: tensor(0.4777)
#
#crps_test_crps
#Out[559]: tensor(0.2209)
#
#MSLL_crps
#Out[560]: tensor(-0.9699)



#LOOCV  ARD itr 1500
crps_mse
Out[540]: tensor(0.1709)

smse_crps
Out[541]: tensor(0.1618)

crps_test_logs
Out[542]: tensor(0.4813)

crps_test_crps
Out[543]: tensor(0.2220)

MSLL_crps
Out[544]: tensor(-0.9663)

res
Out[546]: 0.57



    
#LOOCV  RBF itr 1300
crps_mse
Out[497]: tensor(0.2387)

smse_crps
Out[498]: tensor(0.2259)

crps_test_logs
Out[499]: tensor(0.6460)

crps_test_crps
Out[500]: tensor(0.2625)

MSLL_crps
Out[501]: tensor(-0.8016)

    
res
Out[503]: 0.728

#==============================================================================
# itr400 coverage:0.9606  0.9478, 0.9712, 0.9461, 0.9269, 0.9406, 0.9462, 0.9376, 0.9698, 0.9627, 0.    ]
#==============================================================================


CRPS_va_series[CRPS_va_series>0].argmin() #1088

plt.plot(CRPS_va_series,c='purple')

plt.plot(CRPS_series,c='blue')

plt.plot(length_series,c='blue')
plt.plot(k_series,c='green')
plt.plot(noise_series,c='red')



crps_test_crps
Out[160]: tensor(0.3308)
L: 1.18099, sigma_noise: 0.04961
K tensor([ 1.8759])
#########################################################################################








plt.plot(CRPS_series.log().detach().numpy())

import seaborn as sns
import pandas as pd
sns.heatmap(k_ff.detach().numpy())

# =============================================================================
# predict
# =============================================================================
k_ff =ARD(train_x,train_x,para_k,para_l)
k_star_f =ARD(test_x,train_x,para_k,para_l)
k_ss =ARD(test_x,test_x,para_k,para_l)


y_mean_crps,y_cov_crps=cal_mean_and_cov(k_star_f,k_ff,k_ss,num_test,eye_num=num_train,data_y=train_y)     
y_cov_crps_diag =(y_cov_crps).diag().view(num_test,1)


crps_mse=((y_mean_crps-test_y)**2).mean()
# ================
crps_test_logs=logs( y_mean_crps,y_cov_crps_diag,test_y)
# ================
crps_test_crps=crps( y_mean_crps,y_cov_crps_diag,test_y)  
# ================
MSLL_crps=trivial_loss(y_mean_crps,y_cov_crps_diag,test_y,train_y)

#250  lr=1 fixed. (itr=100 will be worse, itr=500 will be worse, 300will be slightly better)
        #lr=0.1 fixed, very bad!
        #lr=0.6 fixed, better than lr=0.1, but worse than lr=1
#crps_mse
#Out[14]: tensor(0.1108, grad_fn=<MeanBackward1>)
#
#crps_test_logs
#Out[15]: tensor(0.4228, grad_fn=<MeanBackward1>)
#
#crps_test_crps
#Out[16]: tensor(0.1935, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[17]: tensor(-0.9885, grad_fn=<MeanBackward1>)
#----------------------------------------------------
#However, when validation is 800, it shows i>100 will overfitting!
#----so i have to use itr=100( lr=1 fixed)
#crps_mse
#Out[36]: tensor(0.1294, grad_fn=<MeanBackward1>)
#
#crps_test_logs
#Out[37]: tensor(0.5425, grad_fn=<MeanBackward1>)
#
#crps_test_crps
#Out[38]: tensor(0.2127, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[39]: tensor(-0.8682, grad_fn=<MeanBackward1>)

#Now we more increase validation set!
#But I think this results are not very good, so I use decreasing lr now.
#we use decreasing lr:    lr=1 -->if i>50: l4=0.5, and even if itr=250, it pass the validation 'test'!
#However ! crps decrease so slow, the results are really bad than above two !
#so we increase the second lr and increase itr=350 and lr=1 -->if i>50: l4=0.6
#it turns out will over fitting after 150, so we change itr=150
#now itr=150,and lr=1 -->if i>50: l4=0.6: CRPS:0.03626
#results are slightly worse than very first one, but get more reasonable noise:0.10906
#crps_mse
#Out[39]: tensor(0.1173, grad_fn=<MeanBackward1>)
#
#crps_test_logs
#Out[40]: tensor(0.4509, grad_fn=<MeanBackward1>)
#
#crps_test_crps
#Out[41]: tensor(0.1973, grad_fn=<MeanBackward1>)
#
#MSLL_crps
#Out[42]: tensor(-0.9595, grad_fn=<MeanBackward1>)
#---------------------------------------




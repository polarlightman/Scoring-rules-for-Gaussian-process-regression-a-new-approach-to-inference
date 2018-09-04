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
    num_va=300
    sam=np.reshape(random.sample(range(0, 10000),500+num_va) , (500+num_va))
    full_x = pd.read_excel(io=xls,sheetname='trainx',header=None).values[sam,:]
    full_y = pd.read_excel(io=xls,sheetname='trainy',header=None).values[sam,:]
    test_x = pd.read_excel(io=xls,sheetname='testx',header=None).values[:500,:]
    test_y = pd.read_excel(io=xls,sheetname='testy',header=None).values[:500,:]
    
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

    itr=2000
    
    CRPS_series=np.zeros(itr)
    CRPS_ave_va_series=np.zeros(itr)
    para_l=torch.rand(1,train_x.shape[1],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.],requires_grad=True).type(dtype)
#    para_l=torch.tensor([1.4754,  1.3571,  0.8203,  0.6243,  0.5130,  0.3543,  0.2835,0.7229],requires_grad=True).type(dtype)
    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    original_inducing_x=inducing_x.detach().numpy().copy()
    

    for i in range(itr):  
        learning_rate=1

        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)
        Q_ff=Q(train_x,inducing_x,train_x)
        G=(torch.diag(
          k_ff-Q_ff+sigma_noise_sq*torch.eye(num_train)
             )*torch.eye(num_train)
               ).type(dtype)
        big_Q=Q_ff+G
        Q_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_Q)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_Q)/Q_ii_diag
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
    

    
############################################

    itr=3000
    Neg_logL_series=np.zeros(itr)
    Neg_logL_va_series=np.zeros(itr)
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

    itr=3000
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





import torch 
import math
import numpy as np
from torch import nn
from matplotlib import pyplot as plt


def rbf(x, xp, a, b):
    scale = torch.exp(a)
    length = torch.exp(b)
    n, d = x.size()
    m, d = xp.size()
    res = 2 * torch.mm(x, xp.transpose(0,1))
    x_sq = torch.bmm(x.view(n, 1, d), x.view(n, d, 1))#batch matrix matrix product
    xp_sq = torch.bmm(xp.view(m, 1, d), xp.view(m, d, 1))
    x_sq = x_sq.view(n, 1).expand(n, m)
    xp_sq = xp_sq.view(1, m).expand(n, m)
    res = res - x_sq - xp_sq
    res = (0.5 * res) / length.expand_as(res)
    res = scale.expand_as(res) * torch.exp(res)
    return res 


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

#def spgp_cal_mean_and_cov(k1,Q1,k2,k3,k4,k5,Q2,num_test,num_jitter,data_y):
#     G=(torch.diag(
#               k1-Q1+sigma_noise_sq*torch.eye(num_jitter)
#                  )*torch.eye(num_jitter)
#                    ).type(dtype)
#     print('a')
#     Lam=k2+k3.mm(chol_solve(k3.transpose(0,1),G))
#     #Lam=Lam+0.1*torch.eye(num_inducing)
#     print('b')
#     mean_term=k4.mm(chol_solve(k3,Lam)).mm(chol_solve(data_y,G))
#     print('c')
#     cov_term=sigma_noise_sq*torch.eye(num_test).type(dtype)+k5-Q2+k4.mm(chol_solve(k4.transpose(0,1),Lam))
#     return mean_term,cov_term

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

k1,k2,k3,num,eye_num,data_y= k_ff,k_ff,k_ff,num_train,num_train,train_y
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


xls=pd.ExcelFile(address1)



TT=100
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


for j in range(TT):
    p=100*j
    torch.manual_seed(p)
    num_train=120
    num_test=300
    num_va=30
    num_total=num_train+num_test+num_va
    dtype = torch.FloatTensor
    
    full_x=2*torch.randn(num_total)
#    full_x= -6+((12*torch.rand(num_total)))
    true_sigma_noise=0.3
    true_log_l_sq=torch.tensor([1.0],requires_grad=True).type(dtype).log() #log term!
    true_log_k_sq=torch.tensor([1.0],requires_grad=True).type(dtype).log()
    #remember to add noise on diagonal
    k_init =rbf(full_x.view(num_total,1),full_x.view(num_total,1),true_log_k_sq,true_log_l_sq)+torch.eye(num_total)*(true_sigma_noise**2)
    full_y=torch.distributions.MultivariateNormal(torch.from_numpy(np.zeros(num_total)).type(dtype), k_init).sample()
    train_x =(full_x[:num_train,]).view(num_train,1)
    test_x=(full_x[num_train:(num_train+num_test),]).view(num_test,1)
    va_x=(full_x[(num_train+num_test):,]).view(num_va,1)

    train_y =(full_y[:num_train,]).view(num_train,1)
    test_y=(full_y[num_train:(num_train+num_test),]).view(num_test,1)
    va_y=(full_y[(num_train+num_test):,]).view(num_va,1)
    
    #plt.figure(figsize=(12,7))
    #pdot1,=plt.plot(test_x.numpy(),test_y.detach().numpy(),'bo')
    #pdot2,=plt.plot(train_x.numpy(),train_y.detach().numpy(),'ro')
    #pdot3,=plt.plot(va_x.numpy(),va_y.detach().numpy(),'go')
    num_inducing=5

    itr=1000
    
    CRPS_series=np.zeros(itr)
    CRPS_ave_va_series=np.zeros(itr)

    torch.manual_seed(p)
    para_l=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype) 
    #inducing_x=torch.linspace(-4, 4, num_inducing,requires_grad=True).type(dtype).view(num_inducing,1)
#    inducing_x=(torch.randn(num_inducing,train_x.shape[1],requires_grad=True)).type(dtype)
    inducing_x=torch.randint(-3,3,(num_inducing,train_x.shape[1]),requires_grad=True)
    original_inducing_x=inducing_x.detach().numpy().copy()
    
    
    for i in range(itr):  
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
        Q_ii_diag=torch.diag(chol_solve(torch.eye(num_train),big_Q)).view(num_train,1)
        mean_term=train_y-chol_solve(train_y,big_Q)/Q_ii_diag
        cov_term=1/Q_ii_diag
              
        CRPS_ave=crps(mean_term,cov_term,train_y)     
    
        CRPS_ave.backward()
#        print('CRPS:%.5f' % CRPS_ave)
        
    
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
#            print('iteration:%d,sigma_k: %.5f,sigma_noise: %.5f,L: %.5f' % (
#            i,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5),torch.exp(para_l).pow(0.5)
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

    crps_inducing_points=inducing_x.detach().numpy().copy()


    mse_crps_series[j]=mse_crps.detach().numpy()
    smse_crps_series[j]=smse_crps.detach().numpy()
    logs_crps_series[j]=logs_crps.detach().numpy()
    crps_crps_series[j]=crps_crps.detach().numpy()
    MSLL_crps_series[j]=MSLL_crps.detach().numpy()
    res_crps_series[j]=res

# =============================================================================
# 
#CRPS_va_series[CRPS_va_series>0].argmin() #1088
#
#plt.plot(CRPS_va_series,c='purple')
#
#plt.plot(CRPS_series,c='blue')

# =============================================================================
#     GP
# =============================================================================
    itr=1200
    Neg_logL_series=np.zeros(itr)
    Neg_logL_va_series=np.zeros(itr)
    
    torch.manual_seed(p)
    para_l=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype) 
    
#    inducing_x=(torch.randn(num_inducing,train_x.shape[1],requires_grad=True)).type(dtype)
#    inducing_x=torch.rand(num_inducing,train_x.shape[1],requires_grad=True).type(dtype)
    inducing_x=torch.randint(-3,3,(num_inducing,train_x.shape[1]),requires_grad=True)
    original_inducing_x=inducing_x.detach().numpy().copy()
      
  
    for i in range(itr):  
       
    #    learning_rate=0.0002
    #    learning_rate2=0.1
    #    if i >100:
        learning_rate=0.0005
        learning_rate2=0.005
 
        sigma_noise_sq=torch.exp(para_noise)
        k_ff = ARD(train_x,train_x,para_k,para_l)     
        Q_ff=Q(train_x,inducing_x,train_x)
        #千万记住！！！torch.diag是得到对角线的值，而不是创造以对角线的值的diagonal矩阵！！！
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
#            print('iteration:%d,Neg_logL:%5f, sigma_k:%.5f ,sigma_noise: %.5f,L: %.5f' % (
#            i,Neg_logL,torch.exp(para_k).pow(0.5),torch.exp(para_noise).pow(0.5),torch.exp(para_l).pow(0.5)
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


#
#Neg_logL_va_series[Neg_logL_va_series>0].detach().numpy().argmin() 
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
    
    

    gp_inducing_points=inducing_x.detach().numpy().copy()


    
    mse_gp_series[j]=mse_gp.detach().numpy()
    smse_gp_series[j]=smse_gp.detach().numpy()
    logs_gp_series[j]=logs_gp.detach().numpy()
    crps_gp_series[j]=crps_gp.detach().numpy()
    MSLL_gp_series[j]=MSLL_gp.detach().numpy()
    res_gp_series[j]=res




# =============================================================================
# Logs
# =============================================================================
    itr=2500
    logs_ave_series=np.zeros(itr)
    logs_ave_va_series=np.zeros(itr)
    
    
    torch.manual_seed(p)
    para_l=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_k=torch.tensor([1.0],requires_grad=True).type(dtype)
    para_noise=torch.tensor([1.0],requires_grad=True).type(dtype) 
    #inducing_x=torch.linspace(-4, 4, num_inducing,requires_grad=True).type(dtype).view(num_inducing,1)
#    inducing_x=(torch.randn(num_inducing,train_x.shape[1],requires_grad=True)).type(dtype)
    inducing_x=torch.randint(-3,3,(num_inducing,train_x.shape[1]),requires_grad=True)
    original_inducing_x=inducing_x.detach().numpy().copy()
    
    
    for i in range(itr):

        learning_rate=0.005
        learning_rate2=0.005

        
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
    
    logs_inducing_points=inducing_x.detach().numpy().copy()

    
    logs_mse_series[j]=mse_logs.detach().numpy()
    smse_logs_series[j]=logs_smse.detach().numpy()
    logs_test_logs_series[j]=logs_logs.detach().numpy()
    logs_test_crps_series[j]=crps_logs.detach().numpy()
    MSLL_logs_series[j]=MSLL_logs.detach().numpy()    
    res_logs_series[j]=res    


    print(j)    
    
  
     
    
    
mse_crps_series.mean()
smse_crps_series.mean()
logs_crps_series.mean()
crps_crps_series.mean()
MSLL_crps_series.mean()
res_crps_series.mean()

mse_gp_series.mean()
smse_gp_series.mean()
logs_gp_series.mean()
crps_gp_series.mean()
MSLL_gp_series.mean()
res_gp_series.mean()


logs_mse_series.mean()
smse_logs_series.mean()
logs_test_logs_series.mean()
logs_test_crps_series.mean()
MSLL_logs_series.mean()
res_logs_series.mean()
    
 
plt.figure(figsize=(16,10))
plt.axhline(y=0, xmin=0, xmax=100, linewidth=3, color = 'red',linestyle='--')
p1,=plt.plot(mse_crps_series-mse_gp_series,c='green',marker='o')
plt.legend([p1],[ 'MSE of CRPS substract MSE of NLML'],loc=1,prop={'size': 20})

    
plt.figure(figsize=(16,10))
plt.axhline(y=0, xmin=0, xmax=100, linewidth=3, color = 'red',linestyle='--')
p1,=plt.plot(MSLL_crps_series-MSLL_logs_series,c='blue',marker='o')
plt.legend([p1],[ 'MSLL of CRPS substract MSLL of NLML'],loc=1,prop={'size': 20})





plt.plot(MSLL_gp_series,c='red',marker='o')
plt.plot(MSLL_logs_series,c='orange',linestyle='--')






mse_crps_series.mean()
Out[339]: 0.1580240484327078

smse_crps_series.mean()
Out[340]: 0.31513592686504127

logs_crps_series.mean()
Out[341]: 0.44543443754315376

crps_crps_series.mean()
Out[342]: 0.21451152995228767

MSLL_crps_series.mean()
Out[343]: -0.7385387866292149

res_crps_series.mean()
Out[344]: 0.9370333333333334
#
mse_gp_series.mean()
Out[345]: 0.1721345542371273

smse_gp_series.mean()
Out[346]: 0.3434893921390176

logs_gp_series.mean()
Out[347]: 0.4565193697810173

crps_gp_series.mean()
Out[348]: 0.2200839887559414

MSLL_gp_series.mean()
Out[349]: -0.7274538565799594

res_gp_series.mean()
Out[350]: 0.9418333333333334
#
logs_mse_series.mean()
Out[351]: 0.2191786553710699

smse_logs_series.mean()
Out[352]: 0.4149230838194489

logs_test_logs_series.mean()
Out[353]: 0.5498149606585503

logs_test_crps_series.mean()
Out[354]: 0.24845902130007744

MSLL_logs_series.mean()
Out[355]: -0.634158263830468

res_logs_series.mean()
Out[356]: 0.9418333333333334  
# =============================================================================
#     
from scipy import stats
crps_mse_sd=np.std(mse_crps)
mse_gp_sd=np.std(mse_gp)
logs_mse_sd=np.std(logs_mse)

smse_crps_series_sd=np.std(smse_crps_series)
logs_crps_series_sd=np.std(logs_crps_series)
crps_crps_series_sd=np.std(crps_crps_series)
MSLL_crps_series_sd=np.std(MSLL_crps_series)
res_crps_series_sd=np.std(res_crps_series)



smse_gp_series_sd=np.std(smse_gp_series)
logs_gp_series_sd=np.std(logs_gp_series)
crps_gp_series_sd=np.std(crps_gp_series)
MSLL_gp_series_sd=np.std(MSLL_gp_series)
res_gp_series_sd=np.std(res_gp_series)


smse_logs_series_sd=np.std(smse_logs_series)
logs_test_logs_series_sd=np.std(logs_test_logs_series)
logs_test_crps_series_sd=np.std(logs_test_crps_series)
MSLL_logs_series_sd=np.std(MSLL_logs_series)
res_logs_series_sd=np.std(res_logs_series)


plt.plot(MSLL_crps_series-MSLL_gp_series)


# =============================================================================
    mse_crps_series
Out[303]: 
array([0.11681513, 0.09358291, 0.09883157, 0.18415688, 0.17631781,
       0.09619058, 0.14275594, 0.11240911, 0.09914126, 0.24457245,
       0.13415772, 0.23379618, 0.11384623, 0.14693837, 0.09793686,
       0.11976503, 0.18691537, 0.18585989, 0.14609694, 0.19114055,
       0.19729882, 0.10926557, 0.21202797, 0.13953172, 0.15502255,
       0.11677065, 0.21571894, 0.12230513, 0.13293022, 0.13997255,
       0.11665114, 0.32467726, 0.1768368 , 0.27120909, 0.13402335,
       0.10579106, 0.15234183, 0.10114608, 0.11140889, 0.1470284 ,
       0.15832561, 0.12654732, 0.10074662, 0.12608892, 0.11331916,
       0.12165025, 0.11633188, 0.10935331, 0.10316602, 0.16092108,
       0.19930467, 0.16337751, 0.50727463, 0.09916554, 0.15762985,
       0.12334471, 0.14814736, 0.17996173, 0.11988848, 0.13414492,
       0.31140402, 0.24676175, 0.1236468 , 0.11802781, 0.35549197,
       0.15769625, 0.16743754, 0.16207765, 0.09751999, 0.18562409,
       0.14251679, 0.14302196, 0.15343834, 0.18752053, 0.10813974,
       0.15904178, 0.11479458, 0.10114505, 0.12822406, 0.09004932,
       0.23758866, 0.12483885, 0.10235953, 0.32096878, 0.12563473,
       0.1084917 , 0.12131392, 0.24220191, 0.33071071, 0.11658644,
       0.12477119, 0.27144295, 0.21839228, 0.14832719, 0.09579411,
       0.13748199, 0.13589841, 0.15262148, 0.15121493, 0.1103107 ])

smse_crps_series
Out[304]: 
array([0.27575594, 0.2720643 , 0.12097254, 0.20824815, 0.10646771,
       0.03310046, 0.41581979, 0.51257956, 0.1141849 , 0.78301048,
       0.12584701, 0.51271391, 0.16983375, 0.43181309, 0.11315059,
       0.2640439 , 0.32934183, 0.22588743, 0.31892344, 0.18991876,
       0.28352669, 0.14656009, 0.26056194, 0.29936185, 0.53772485,
       0.4663372 , 0.32483456, 0.24814145, 0.75218606, 0.1051918 ,
       0.10452562, 0.9683699 , 0.20574103, 0.86999917, 0.23928221,
       0.10752174, 0.1167466 , 0.34609064, 0.15873128, 0.13234384,
       0.05723732, 0.22096208, 0.41060433, 0.32304883, 0.12579148,
       0.19089651, 0.19854113, 0.19454132, 0.07147764, 0.25332174,
       0.67732215, 0.31013802, 0.52523792, 0.42782459, 0.15379059,
       0.17693147, 0.87981081, 0.65937716, 0.2372759 , 0.10067137,
       0.95722353, 0.89889008, 0.04817452, 0.08437161, 0.51611227,
       0.46035478, 0.18745445, 0.1068976 , 0.24485312, 0.31060761,
       0.35750929, 0.21465763, 0.28898913, 0.33707482, 0.10203858,
       0.30346757, 0.51895243, 0.07643   , 0.14959429, 0.17124538,
       0.18457468, 0.12540765, 0.18990542, 0.77718383, 0.20277439,
       0.112405  , 0.45661467, 0.93210155, 0.97288072, 0.14627898,
       0.0680107 , 0.2982429 , 0.29696205, 0.10208531, 0.05602318,
       0.86135691, 0.33580312, 0.41791621, 0.143517  , 0.10639128])

logs_crps_series
Out[305]: 
array([0.3468273 , 0.32418916, 0.27440402, 0.43970037, 0.4929724 ,
       0.29498026, 0.59032714, 0.34824765, 0.29710081, 0.72800195,
       0.4826647 , 0.51091063, 0.33512267, 0.3892425 , 0.28621268,
       0.35103717, 0.52421266, 0.55259097, 0.49688894, 0.54173529,
       0.55887532, 0.33764803, 0.6250025 , 0.4090867 , 0.46248084,
       0.36393419, 0.52908587, 0.3506799 , 0.47897065, 0.47276059,
       0.37921551, 0.86030728, 0.45238179, 0.77749598, 0.39937329,
       0.26935256, 0.45854855, 0.35225463, 0.27113062, 0.36917165,
       0.34247676, 0.40083069, 0.23070581, 0.3457672 , 0.32731619,
       0.40756822, 0.35507473, 0.36357951, 0.33111057, 0.4022091 ,
       0.60666221, 0.57988173, 0.97253716, 0.2573868 , 0.39753371,
       0.35850075, 0.46643472, 0.5653283 , 0.4368566 , 0.2744571 ,
       0.83035469, 0.72702289, 0.33664855, 0.36812356, 0.76330972,
       0.41839445, 0.49030036, 0.35219863, 0.26702073, 0.48169386,
       0.38240936, 0.4201408 , 0.4198671 , 0.55402017, 0.31328136,
       0.55350524, 0.36859918, 0.31327295, 0.39251104, 0.23930825,
       0.68019259, 0.39450088, 0.31304055, 0.85668474, 0.32084212,
       0.34237778, 0.36956677, 0.72914726, 0.87414157, 0.34653497,
       0.41927248, 0.50465643, 0.59606135, 0.3842355 , 0.22358237,
       0.43216407, 0.39136475, 0.47629717, 0.34826374, 0.34119016])

crps_crps_series
Out[306]: 
array([0.19221221, 0.1776903 , 0.17766175, 0.21883219, 0.22506012,
       0.17900059, 0.21584621, 0.18851919, 0.177044  , 0.28318077,
       0.20360878, 0.23698746, 0.18849406, 0.20688635, 0.17752457,
       0.19251186, 0.23012857, 0.24107096, 0.21422178, 0.24093254,
       0.23986289, 0.18351927, 0.26057461, 0.20895678, 0.20723242,
       0.1922898 , 0.23166852, 0.19495638, 0.19825108, 0.2114213 ,
       0.19264461, 0.32241017, 0.22206137, 0.29780951, 0.20438273,
       0.18213785, 0.21013062, 0.18434468, 0.17938273, 0.20008253,
       0.2028476 , 0.19916335, 0.17216274, 0.19323595, 0.1890551 ,
       0.19652776, 0.19193384, 0.18985386, 0.18243712, 0.20897539,
       0.25218266, 0.21451952, 0.36234221, 0.1737548 , 0.20964469,
       0.19100785, 0.21641241, 0.24046227, 0.19005471, 0.18984112,
       0.3173027 , 0.28287002, 0.19289365, 0.19567777, 0.30967429,
       0.21302454, 0.22023155, 0.20727596, 0.17664167, 0.23276597,
       0.20287965, 0.20518643, 0.21100767, 0.23434862, 0.18400772,
       0.22168384, 0.19350836, 0.18305227, 0.20119567, 0.1714298 ,
       0.26970768, 0.20144731, 0.18329303, 0.3247163 , 0.18663953,
       0.1887279 , 0.19603109, 0.27998012, 0.32785419, 0.18973514,
       0.20017476, 0.24736798, 0.25096208, 0.20385264, 0.17440557,
       0.2084606 , 0.20354533, 0.21880971, 0.19544189, 0.18539293])

MSLL_crps_series
Out[307]: 
array([-0.64377999, -0.5624705 , -1.04607046, -0.91844255, -1.17821908,
       -1.65738034, -0.29976827, -0.31609961, -1.05130672, -0.1096525 ,
       -0.97420323, -0.51696825, -0.8920328 , -0.49189791, -1.06886125,
       -0.67407668, -0.6340003 , -0.8081097 , -0.53176516, -0.88248831,
       -0.68131047, -0.93478441, -0.69484949, -0.62834263, -0.34010291,
       -0.36309975, -0.68642294, -0.74331063, -0.08613541, -1.08903921,
       -1.09603703, -0.01258588, -0.89195102, -0.05879071, -0.74457687,
       -1.15339124, -1.09375513, -0.45162794, -0.97399926, -1.10505164,
       -1.58562541, -0.74000752, -0.52871031, -0.60532892, -1.05964696,
       -0.78769392, -0.79729879, -0.7675606 , -1.27159381, -0.79022294,
       -0.20098796, -0.52209014, -0.42936951, -0.4309206 , -1.03389966,
       -0.88421172, -0.08972856, -0.22349226, -0.64858401, -1.28830743,
       -0.02722707, -0.04554697, -1.55362225, -1.22178829, -0.46923468,
       -0.46501666, -0.87270403, -1.3184303 , -0.6920076 , -0.68897665,
       -0.57669091, -0.79983103, -0.69074106, -0.58757102, -1.13469696,
       -0.54257196, -0.30350474, -1.24583614, -0.94965529, -0.86044043,
       -0.86862451, -1.03641033, -0.79989958, -0.12064819, -0.87729669,
       -1.05887139, -0.38967219, -0.02726047, -0.01106562, -0.96180803,
       -1.30479467, -0.87949878, -0.6718691 , -1.22979534, -1.4638623 ,
       -0.07017292, -0.57731217, -0.48614779, -1.0984658 , -1.10226953])

res_crps_series
Out[308]: 
array([0.96333333, 0.89      , 0.92      , 0.92333333, 0.92333333,
       0.94      , 0.85333333, 0.92333333, 0.90666667, 0.93333333,
       0.89333333, 0.91333333, 0.97333333, 0.94333333, 0.94666667,
       0.94      , 0.89333333, 0.96333333, 0.91333333, 0.99      ,
       0.86333333, 0.90666667, 0.97333333, 0.97333333, 0.91333333,
       0.94666667, 0.91666667, 0.93      , 0.85      , 0.92666667,
       0.90666667, 0.97      , 0.92333333, 0.98      , 0.96333333,
       0.96666667, 0.91666667, 0.92      , 0.95666667, 0.94      ,
       0.95666667, 0.90333333, 0.94666667, 0.96      , 0.93666667,
       0.92333333, 0.93333333, 0.92666667, 0.92666667, 0.96      ,
       0.93666667, 0.87666667, 0.96333333, 0.94666667, 0.96666667,
       0.91666667, 0.96666667, 0.97333333, 0.90333333, 0.96666667,
       0.97333333, 0.94      , 0.95      , 0.93333333, 0.95      ,
       0.95333333, 0.91666667, 0.98333333, 0.96333333, 0.92666667,
       0.91666667, 0.92      , 0.93333333, 0.92333333, 0.93333333,
       0.89333333, 0.92333333, 0.95      , 0.93      , 0.96      ,
       0.94666667, 0.93333333, 0.92333333, 0.96666667, 0.94666667,
       0.93      , 0.96333333, 0.98333333, 0.98      , 0.93333333,
       0.91      , 0.93666667, 0.96333333, 0.94333333, 0.95      ,
       0.97666667, 0.94      , 0.93666667, 0.94666667, 0.94      ])
    
    
    
    
# =============================================================================
#     
# =============================================================================
mse_gp_series
Out[309]: 
array([0.16625316, 0.22153498, 0.09589045, 0.2010459 , 0.16900703,
       0.09499507, 0.14033785, 0.11802797, 0.14852186, 0.18237112,
       0.12643251, 0.22984165, 0.08452392, 0.1549204 , 0.19500095,
       0.13051036, 0.2629047 , 0.28257936, 0.15975744, 0.2029988 ,
       0.12199371, 0.12576973, 0.19704652, 0.10854306, 0.15215722,
       0.12395418, 0.43979618, 0.13299239, 0.2264744 , 0.14189109,
       0.11894547, 0.32188639, 0.10256053, 0.2683281 , 0.15788984,
       0.11927279, 0.15781237, 0.10229782, 0.11961935, 0.16709444,
       0.15617567, 0.13191114, 0.10129523, 0.13618077, 0.11921488,
       0.11840667, 0.11872852, 0.16536362, 0.11015005, 0.17961253,
       0.20013273, 0.17215213, 0.46866447, 0.10612124, 0.16852164,
       0.14323728, 0.14770724, 0.20979065, 0.12786612, 0.12891853,
       0.3102167 , 0.28480932, 0.14576204, 0.11775495, 0.35120243,
       0.16066885, 0.20809399, 0.20546947, 0.09907082, 0.2311765 ,
       0.15009405, 0.15085681, 0.14907014, 0.18489583, 0.11930322,
       0.15212968, 0.18140034, 0.10273162, 0.1234399 , 0.09095814,
       0.23543951, 0.13644566, 0.10527938, 0.32603928, 0.12558056,
       0.10625625, 0.12247892, 0.24437803, 0.33042014, 0.17326418,
       0.23923965, 0.30003476, 0.29053417, 0.15578805, 0.10011123,
       0.13571203, 0.16222139, 0.15906374, 0.15045106, 0.11168044])

smse_gp_series
Out[310]: 
array([0.39246029, 0.6440466 , 0.11737252, 0.22734658, 0.10205317,
       0.03268907, 0.4087764 , 0.53820121, 0.17105849, 0.58386993,
       0.11860036, 0.50404161, 0.12609127, 0.45527017, 0.22529282,
       0.28773394, 0.4632338 , 0.34343681, 0.34874371, 0.20170121,
       0.17531008, 0.16869745, 0.24215119, 0.23287646, 0.5277859 ,
       0.49502549, 0.66225523, 0.26982453, 1.28150618, 0.10663362,
       0.10658146, 0.96004599, 0.11932419, 0.86075735, 0.2818929 ,
       0.12122401, 0.12093893, 0.35003155, 0.17042924, 0.15040578,
       0.05646008, 0.23032776, 0.41284025, 0.34890488, 0.13233611,
       0.1858066 , 0.20263143, 0.2941846 , 0.07631646, 0.28274578,
       0.6801362 , 0.3267948 , 0.48526049, 0.45783323, 0.1644171 ,
       0.2054663 , 0.87719703, 0.7686699 , 0.25306475, 0.09674914,
       0.95357382, 1.03748763, 0.05679093, 0.08417656, 0.5098846 ,
       0.46903256, 0.23297131, 0.13551648, 0.24874693, 0.38683116,
       0.37651715, 0.22641675, 0.28076196, 0.33235681, 0.11257223,
       0.29027858, 0.82005745, 0.07762889, 0.14401279, 0.17297366,
       0.18290508, 0.13706733, 0.19532254, 0.78946137, 0.20268697,
       0.11008891, 0.46099961, 0.94047624, 0.97202593, 0.21739155,
       0.13040553, 0.32965761, 0.39505804, 0.1072202 , 0.05854796,
       0.85026771, 0.40084684, 0.43555677, 0.14279202, 0.10771235])

logs_gp_series
Out[311]: 
array([0.50254226, 0.39752501, 0.27837229, 0.36480996, 0.46332881,
       0.31648836, 0.43979156, 0.36049995, 0.35783163, 0.44001868,
       0.39432329, 0.49483284, 0.3002553 , 0.39018443, 0.47907171,
       0.3928659 , 0.55179214, 0.53719676, 0.47932151, 0.56387907,
       0.41636756, 0.40504834, 0.59266949, 0.39337224, 0.44006109,
       0.37107706, 0.73643231, 0.36343414, 0.48621586, 0.48549634,
       0.35335463, 0.84957433, 0.28013253, 0.7648254 , 0.38805756,
       0.32224482, 0.48069522, 0.34180495, 0.29724181, 0.38029274,
       0.34090093, 0.4249976 , 0.23144558, 0.33646014, 0.33826923,
       0.37036303, 0.39256889, 0.47912276, 0.3702932 , 0.43241036,
       0.60849088, 0.81054115, 0.96893018, 0.297645  , 0.43409666,
       0.43391794, 0.46557626, 0.52072901, 0.37559709, 0.31001946,
       0.82980341, 0.73585069, 0.49012065, 0.39011708, 0.75898534,
       0.44222191, 0.42124769, 0.29981634, 0.26618195, 0.50820571,
       0.39216053, 0.39281484, 0.42093652, 0.56179845, 0.43838647,
       0.46247536, 0.5065859 , 0.31165206, 0.44816592, 0.29861283,
       0.66504252, 0.44859475, 0.29715902, 0.86561418, 0.30171478,
       0.37455481, 0.36858904, 0.71939391, 0.86597002, 0.50050801,
       0.50799352, 0.49436381, 0.54344469, 0.38353533, 0.27173856,
       0.41794276, 0.41380304, 0.48096314, 0.3720434 , 0.38712278])

crps_gp_series
Out[312]: 
array([0.2253439 , 0.2225098 , 0.17780083, 0.21520583, 0.21837874,
       0.18055584, 0.20969456, 0.19246517, 0.1973677 , 0.21993926,
       0.19622847, 0.23307878, 0.17137498, 0.21019585, 0.22895677,
       0.20177047, 0.25234896, 0.26450899, 0.21573226, 0.24517013,
       0.1977365 , 0.2006267 , 0.25093091, 0.192689  , 0.20708063,
       0.19641641, 0.32901391, 0.20108621, 0.2365692 , 0.21518242,
       0.1940905 , 0.32060677, 0.17962387, 0.29560599, 0.21096961,
       0.1926911 , 0.21420978, 0.18443574, 0.18454523, 0.20646024,
       0.19562998, 0.20222281, 0.17233405, 0.19626297, 0.19115627,
       0.19440524, 0.19492416, 0.21905379, 0.18855128, 0.21569183,
       0.25309673, 0.23034245, 0.36267874, 0.18147863, 0.21630064,
       0.20844622, 0.21629991, 0.24322598, 0.19335231, 0.19180711,
       0.31685328, 0.29472017, 0.21505821, 0.19696364, 0.31024683,
       0.21555269, 0.22712971, 0.20484355, 0.17717731, 0.24691333,
       0.2087435 , 0.20747098, 0.21030729, 0.23542926, 0.197621  ,
       0.21303955, 0.23502131, 0.18341911, 0.20055407, 0.17945974,
       0.26729485, 0.20948794, 0.18319483, 0.32820317, 0.18687421,
       0.18875709, 0.19701052, 0.27970397, 0.32729083, 0.22809078,
       0.24820863, 0.25165242, 0.26125818, 0.20660855, 0.18078189,
       0.20489065, 0.21411312, 0.22112341, 0.1993221 , 0.19154808])

MSLL_gp_series
Out[313]: 
array([-0.488065  , -0.48913467, -1.04210222, -0.99333298, -1.20786262,
       -1.63587224, -0.45030385, -0.30384731, -0.99057591, -0.39763576,
       -1.0625447 , -0.53304607, -0.92690015, -0.49095595, -0.87600219,
       -0.63224798, -0.60642081, -0.82350391, -0.54933256, -0.86034453,
       -0.82381821, -0.86738414, -0.72718251, -0.6440571 , -0.36252266,
       -0.35595688, -0.47907647, -0.73055637, -0.0788902 , -1.07630348,
       -1.12189794, -0.02331878, -1.06420028, -0.07146125, -0.75589263,
       -1.10049891, -1.07160842, -0.46207765, -0.94788808, -1.0939306 ,
       -1.58720124, -0.71584058, -0.52797055, -0.61463594, -1.04869401,
       -0.82489908, -0.75980461, -0.65201741, -1.23241127, -0.76002169,
       -0.19915931, -0.29143071, -0.43297645, -0.3906624 , -0.9973368 ,
       -0.80879456, -0.09058701, -0.26809153, -0.70984346, -1.25274503,
       -0.02777839, -0.03671912, -1.40015018, -1.19979477, -0.47355908,
       -0.4411892 , -0.94175673, -1.37081254, -0.69284642, -0.6624648 ,
       -0.56693977, -0.82715702, -0.68967164, -0.57979274, -1.00959182,
       -0.63360184, -0.16551802, -1.24745703, -0.89400035, -0.8011359 ,
       -0.88377464, -0.98231649, -0.81578106, -0.11171876, -0.896424  ,
       -1.0266943 , -0.39064991, -0.03701383, -0.01923721, -0.80783498,
       -1.21607363, -0.88979137, -0.72448575, -1.23049557, -1.41570616,
       -0.08439424, -0.55487388, -0.48148185, -1.07468617, -1.05633688])

res_gp_series
Out[314]: 
array([0.94666667, 0.91666667, 0.94333333, 0.95333333, 0.94      ,
       0.94      , 0.94333333, 0.92666667, 0.93333333, 0.95666667,
       0.92666667, 0.91333333, 0.96333333, 0.95      , 0.94      ,
       0.97      , 0.92666667, 0.93333333, 0.91333333, 0.93      ,
       0.90333333, 0.93      , 0.97      , 0.94666667, 0.91      ,
       0.90333333, 0.97666667, 0.95333333, 0.87333333, 0.93666667,
       0.95666667, 0.96666667, 0.93333333, 0.97      , 0.96      ,
       0.97333333, 0.9       , 0.93333333, 0.96      , 0.92666667,
       0.91333333, 0.91666667, 0.93333333, 0.94666667, 0.90666667,
       0.94      , 0.92333333, 0.90666667, 0.92333333, 0.94      ,
       0.93666667, 0.83333333, 0.96      , 0.97333333, 0.96666667,
       0.94333333, 0.97333333, 0.96333333, 0.95      , 0.94666667,
       0.96666667, 0.94333333, 0.92333333, 0.95      , 0.94333333,
       0.95      , 0.93333333, 0.97666667, 0.95333333, 0.93      ,
       0.95333333, 0.96333333, 0.94333333, 0.92333333, 0.92333333,
       0.89666667, 0.94666667, 0.95666667, 0.91666667, 0.96333333,
       0.95      , 0.92666667, 0.93      , 0.94666667, 0.97333333,
       0.92666667, 0.96      , 0.97666667, 0.98      , 0.94666667,
       0.96666667, 0.92333333, 0.96      , 0.96333333, 0.97333333,
       0.97      , 0.94      , 0.94666667, 0.93333333, 0.98666667])
# =============================================================================
#     
# =============================================================================
logs_mse_series
Out[317]: 
array([0.17863412, 0.20921412, 0.09549018, 0.23854628, 0.20247504,
       0.11503138, 0.16756719, 0.12625317, 0.15820682, 0.24459819,
       0.15136018, 0.18892179, 0.16326049, 0.16492523, 0.20694551,
       0.22300026, 0.3105621 , 0.16545537, 0.18768738, 0.19581489,
       0.26831734, 0.15679239, 0.22416975, 0.20679455, 0.28544617,
       0.16820525, 0.46020949, 0.147154  , 0.17799501, 0.23468013,
       0.51958549, 0.32110286, 0.21930674, 0.27210793, 0.20440391,
       0.12806363, 0.166584  , 0.11067122, 0.15237831, 0.29089257,
       0.16286226, 0.43375143, 0.12173524, 0.25189793, 0.12367773,
       0.16080333, 0.12866512, 0.24134679, 0.14850637, 0.15968223,
       0.20012352, 0.22105658, 0.45480016, 0.18978494, 0.17969097,
       0.30477366, 0.14888638, 0.19114812, 0.16942759, 0.15761666,
       0.31209651, 0.26824227, 0.18913302, 0.12328635, 0.31969845,
       0.15835521, 0.77840006, 0.20290025, 0.19033867, 0.16401994,
       0.16293836, 0.50824761, 0.14159696, 0.17963612, 0.30439523,
       0.40805492, 0.20185344, 0.16284874, 0.15108849, 0.12019868,
       0.24322534, 0.13168339, 0.12381047, 0.32865691, 0.15331708,
       0.14242691, 0.11834451, 0.24545266, 0.33178824, 0.26937076,
       0.28720021, 0.32094932, 0.24509972, 0.18684104, 0.12223545,
       0.13799581, 0.25483802, 0.15124455, 0.17134362, 0.34566477])

smse_logs_series
Out[318]: 
array([0.42168701, 0.60822743, 0.11688258, 0.26975274, 0.12226249,
       0.03958384, 0.48809007, 0.57570773, 0.18221305, 0.78309292,
       0.14198384, 0.41430458, 0.24354906, 0.48467171, 0.23909287,
       0.49164483, 0.54720539, 0.20108852, 0.4097136 , 0.19456322,
       0.38558328, 0.21030875, 0.27548301, 0.4436726 , 0.99012369,
       0.67174727, 0.69299406, 0.29855663, 1.00718534, 0.1763662 ,
       0.46557617, 0.95770901, 0.25515273, 0.87288249, 0.36493802,
       0.13015866, 0.12766103, 0.37868276, 0.217103  , 0.26183948,
       0.05887739, 0.75736588, 0.49614584, 0.64538056, 0.13729015,
       0.25233647, 0.21959001, 0.42935988, 0.10289129, 0.25137153,
       0.68010491, 0.41962969, 0.47090524, 0.81877911, 0.17531438,
       0.43718168, 0.88419968, 0.70036393, 0.33532065, 0.11828615,
       0.95935214, 0.9771381 , 0.07368886, 0.08813066, 0.46414635,
       0.46227843, 0.87145662, 0.13382196, 0.4779022 , 0.27445707,
       0.40873763, 0.76281458, 0.2666868 , 0.32290229, 0.28722149,
       0.77860945, 0.91251987, 0.12305624, 0.1762694 , 0.22857994,
       0.18895362, 0.13228337, 0.22970288, 0.79579961, 0.24745367,
       0.14756425, 0.44543806, 0.94461185, 0.97605062, 0.33797482,
       0.15654804, 0.35263711, 0.33327791, 0.12859224, 0.07148685,
       0.8645761 , 0.62970132, 0.41414586, 0.16262101, 0.33338302])

logs_test_logs_series
Out[319]: 
array([0.53310168, 0.56189162, 0.28999662, 0.37903887, 0.59100533,
       0.32090294, 0.50124305, 0.38043436, 0.37636977, 0.72937131,
       0.39166537, 0.40002596, 0.51733905, 0.52321094, 0.46404448,
       0.4537518 , 0.53443807, 0.56208771, 0.4825162 , 0.63759637,
       0.49157929, 0.50467348, 0.63358051, 0.47503558, 0.77487248,
       0.49394375, 0.81637239, 0.35600734, 0.45481732, 0.72803652,
       1.08866847, 0.85671407, 0.47881293, 0.77572906, 0.48793298,
       0.32335272, 0.46919841, 0.33816093, 0.30559123, 0.58458322,
       0.3447974 , 0.90336984, 0.28453484, 0.61400598, 0.36413041,
       0.37774342, 0.46945491, 0.7383399 , 0.40452254, 0.39685515,
       0.59674031, 0.57498246, 0.96712506, 0.60136527, 0.57622755,
       0.72007769, 0.48019069, 0.55204862, 0.53210104, 0.31646228,
       0.83070707, 0.75234741, 0.38412586, 0.38688648, 0.80715942,
       0.42462081, 1.315925  , 0.36011842, 0.53593105, 0.42613262,
       0.44680119, 1.06161463, 0.41364518, 0.51540154, 0.70211411,
       0.94114405, 0.53208494, 0.47169653, 0.49299398, 0.44490033,
       0.67089939, 0.41484129, 0.340747  , 0.87295938, 0.38343617,
       0.40799117, 0.38816121, 0.72178894, 0.87147582, 0.53011125,
       0.61728299, 0.48617592, 0.63307488, 0.39854354, 0.33163005,
       0.43876114, 0.68280238, 0.46490031, 0.50971168, 0.68708134])

logs_test_crps_series
Out[320]: 
array([0.23379643, 0.24623621, 0.18097018, 0.2250461 , 0.24880424,
       0.18981351, 0.22663586, 0.20000774, 0.20729673, 0.2833797 ,
       0.2070408 , 0.21746489, 0.22812065, 0.22966258, 0.23182045,
       0.23275132, 0.26368487, 0.23523155, 0.23207007, 0.25402623,
       0.25243166, 0.22430959, 0.26495421, 0.23207906, 0.30028367,
       0.22528249, 0.34107098, 0.20568497, 0.22405772, 0.28333005,
       0.41291067, 0.32120165, 0.24056651, 0.29834625, 0.23308763,
       0.19580784, 0.22123958, 0.19216031, 0.19703063, 0.26223657,
       0.20632538, 0.36246321, 0.18245561, 0.26248634, 0.19777313,
       0.20709233, 0.21318527, 0.28571874, 0.21177994, 0.20896055,
       0.25260726, 0.25428087, 0.36606118, 0.24910261, 0.23811665,
       0.29746571, 0.21743232, 0.24240686, 0.2280751 , 0.19843921,
       0.31770867, 0.29233798, 0.21508563, 0.20015927, 0.30726546,
       0.21446496, 0.51566571, 0.2146565 , 0.24273612, 0.21746582,
       0.22040987, 0.40845737, 0.20728813, 0.23114349, 0.29237178,
       0.36288878, 0.24340546, 0.22250423, 0.22172196, 0.20714396,
       0.27009034, 0.20777303, 0.19659837, 0.3300043 , 0.20725393,
       0.21001133, 0.19664262, 0.27989265, 0.32807884, 0.25715852,
       0.27279243, 0.25937176, 0.25982764, 0.2152877 , 0.19193575,
       0.2091779 , 0.27640727, 0.21749981, 0.22802046, 0.29303589])

MSLL_logs_series
Out[321]: 
array([-0.45750555, -0.32476807, -1.03047788, -0.97910404, -1.08018613,
       -1.63145757, -0.38885236, -0.2839129 , -0.97203779, -0.10828316,
       -1.06520259, -0.62785292, -0.7098164 , -0.35792944, -0.89102942,
       -0.57136208, -0.62377489, -0.79861295, -0.54613793, -0.78662723,
       -0.7486065 , -0.76775903, -0.68627149, -0.56239378, -0.02771127,
       -0.23309019, -0.39913642, -0.73798317, -0.11028872, -0.83376324,
       -0.38658401, -0.01617905, -0.86551988, -0.0605576 , -0.65601718,
       -1.0993911 , -1.08310521, -0.46572167, -0.93953866, -0.88964009,
       -1.58330476, -0.23746838, -0.47488129, -0.33709013, -1.02283275,
       -0.81751871, -0.68291861, -0.39280027, -1.19818187, -0.79557687,
       -0.21090986, -0.5269894 , -0.43478161, -0.08694211, -0.85520589,
       -0.5226348 , -0.07597258, -0.23677193, -0.55333954, -1.24630225,
       -0.02687468, -0.02022243, -1.506145  , -1.20302546, -0.425385  ,
       -0.4587903 , -0.04707938, -1.3105104 , -0.42309728, -0.74453789,
       -0.51229912, -0.15835717, -0.69696301, -0.62618971, -0.74586421,
       -0.15493311, -0.14001897, -1.0874126 , -0.84917229, -0.6548484 ,
       -0.87791777, -1.01606989, -0.77219313, -0.10437357, -0.81470263,
       -0.993258  , -0.37107772, -0.03461879, -0.01373136, -0.77823174,
       -1.10678411, -0.89797926, -0.63485557, -1.21548736, -1.3558147 ,
       -0.06357583, -0.28587455, -0.49754468, -0.93701786, -0.75637829])

res_logs_series
Out[322]: 
array([0.94666667, 0.91666667, 0.94333333, 0.95333333, 0.94      ,
       0.94      , 0.94333333, 0.92666667, 0.93333333, 0.95666667,
       0.92666667, 0.91333333, 0.96333333, 0.95      , 0.94      ,
       0.97      , 0.92666667, 0.93333333, 0.91333333, 0.93      ,
       0.90333333, 0.93      , 0.97      , 0.94666667, 0.91      ,
       0.90333333, 0.97666667, 0.95333333, 0.87333333, 0.93666667,
       0.95666667, 0.96666667, 0.93333333, 0.97      , 0.96      ,
       0.97333333, 0.9       , 0.93333333, 0.96      , 0.92666667,
       0.91333333, 0.91666667, 0.93333333, 0.94666667, 0.90666667,
       0.94      , 0.92333333, 0.90666667, 0.92333333, 0.94      ,
       0.93666667, 0.83333333, 0.96      , 0.97333333, 0.96666667,
       0.94333333, 0.97333333, 0.96333333, 0.95      , 0.94666667,
       0.96666667, 0.94333333, 0.92333333, 0.95      , 0.94333333,
       0.95      , 0.93333333, 0.97666667, 0.95333333, 0.93      ,
       0.95333333, 0.96333333, 0.94333333, 0.92333333, 0.92333333,
       0.89666667, 0.94666667, 0.95666667, 0.91666667, 0.96333333,
       0.95      , 0.92666667, 0.93      , 0.94666667, 0.97333333,
       0.92666667, 0.96      , 0.97666667, 0.98      , 0.94666667,
       0.96666667, 0.92333333, 0.96      , 0.96333333, 0.97333333,
       0.97      , 0.94      , 0.94666667, 0.93333333, 0.98666667])





#
#import pandas as pd
#import xlrd
#import random
##address1='H:\Project\DATA________________\kin40k.xlsx'
#address2='G:\!帝国理工\M00000-Summer-Project\SIMPLE-FITC-BOX.xlsx'
#simple_xls=pd.ExcelFile(address2)    
#MSE_BOX=pd.read_excel(io=simple_xls,sheetname='MSE')
#    
#import seaborn as sns
#sns.set_style("whitegrid")
##tips = sns.load_dataset("tips")
##ax = sns.boxplot(x="Method", y="MSE", hue="MSE",              
##data=MSE_BOX, palette="Set3")   
#    
#
#
#plt.figure(figsize=(16,10))
#sns.boxplot( x=MSE_BOX["Method"], y=MSE_BOX["MSE"] )
#plt.axhline(y=0, xmin=0, xmax=100, linewidth=3, color = 'red',linestyle='--')

    
# =============================================================================
#     

diag_value_gp=torch.diag(cov_gp).view(test_x.size()[0],1)
up_gp=mean_gp.view(test_x.size()[0],1)+2*diag_value_gp**(0.5)
low_gp=mean_gp.view(test_x.size()[0],1)-2*diag_value_gp**(0.5)
    
diag_value=torch.diag(cov_crps)
up=mean_crps.view(1,num_test)+2*diag_value**(0.5)
low=mean_crps.view(1,num_test)-2*diag_value**(0.5)

diag_value_logs=torch.diag(cov_logs)
up_logs=mean_logs.view(1,num_test)+2*diag_value_logs**(0.5)
low_logs=mean_logs.view(1,num_test)-2*diag_value_logs**(0.5)

##
up_gp2=up_gp[:,0].detach().numpy()
low_gp2=low_gp[:,0].detach().numpy()
y_mean_gp2=mean_gp[:,0].detach().numpy()

up2=up[0,:].detach().numpy()
low2=low[0,:].detach().numpy()
y_mean2=mean_crps[:,0].detach().numpy()


up_logs2=up_logs[0,:].detach().numpy()
low_logs2=low_logs[0,:].detach().numpy()
y_mean_logs2=mean_logs[:,0].detach().numpy()





test_x_new,test_y_new,y_mean2,up2,low2,y_mean_gp2,up_gp2,low_gp2,y_mean_logs2,up_logs2,low_logs2=zip(*sorted(zip(test_x[:,0].numpy(),test_y[:,0].numpy(),y_mean2,up2,low2,y_mean_gp2,up_gp2,low_gp2,y_mean_logs2,up_logs2,low_logs2)))


plt.figure(figsize=(16,10))
pdot1,=plt.plot(test_x_new,test_y_new,'bo')
pdot2,=plt.plot(train_x.numpy(),train_y.detach().numpy(),'ro')
pgpmean,=plt.plot(test_x_new,y_mean_gp2,linestyle=':',c='purple')
pgpup,=plt.plot(test_x_new,up_gp2,linestyle='--',c='purple')
pgplow,=plt.plot(test_x_new,low_gp2,linestyle='--',c='purple')


pcrpsmean,=plt.plot(test_x_new,y_mean2,linestyle=':',c='green')
pcrpsup,=plt.plot(test_x_new,up2,linestyle='--',c='green')
pcrpslow,=plt.plot(test_x_new,low2,linestyle='--',c='green')

plogsmean,=plt.plot(test_x_new,y_mean_logs2,linestyle=':',c='orange')
plogsup,=plt.plot(test_x_new,up_logs2,linestyle='--',c='orange')
plogslow,=plt.plot(test_x_new,low_logs2,linestyle='--',c='orange')


porigin=plt.scatter( original_inducing_x,np.zeros(num_inducing)+3.5,marker='X', color='lime')
pgp_inducing=plt.scatter(gp_inducing_points,np.zeros(num_inducing)-2.5,marker='X', color='purple')
pcrps_inducing=plt.scatter(crps_inducing_points,np.zeros(num_inducing)-3,marker='X', color='green')
plogs_inducing=plt.scatter(logs_inducing_points,np.zeros(num_inducing)-3.5,marker='X', color='orange')


#plt.xticks(test_x_new)
#py,=plt.plot(test_x[:,0].numpy(),true_y[:,0].detach().numpy(),c='purple')
plt.legend([pdot1,pdot2,pgpup,pcrpsup,plogsup,pgpmean,pcrpsmean,plogsmean,porigin,pgp_inducing,pcrps_inducing,plogs_inducing],
           [ 'test points','train points',"up & low using NLML","up & low using crps",
            "up & low using logs",'predicted mean using NLML','predicted mean using crps',
            'predicted mean using logs','original inducing points','inducing points for NLML',
            'inducing points for crps','inducing points for logs'],loc=1)

    











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

    











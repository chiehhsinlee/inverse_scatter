import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import hankel2
import scipy.constants as const
from numpy.polynomial.legendre import leggauss
from pathlib import Path
from scipy.io import loadmat
def wave_parameter(freq,mu,epsilon):
    vp=1.0/np.sqrt(mu * epsilon)
    wavelength=vp / freq
    k=2.0 * np.pi / wavelength
    omega=2.0 * np.pi * freq
    return vp,wavelength,k,omega
def plot_style():
    mpl.rcParams.update({'figure.figsize': (3.5,3),'figure.dpi': 300,
        'savefig.dpi': 300,'font.size': 12,'font.family': 'serif',
        'font.serif': ['Times New Roman'],'mathtext.fontset': 'stix',
        'axes.labelsize': 9,'axes.titlesize': 9,'xtick.labelsize': 9,
        'ytick.labelsize': 9,'legend.fontsize': 8,'lines.linewidth': 1.5,
        'lines.markersize': 4.5,'axes.linewidth': 0.6,'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,'xtick.minor.width': 0.5,'ytick.minor.width': 0.5,
        'axes.grid': True,'grid.linestyle': ':','grid.linewidth': 0.5,
        'axes.unicode_minus': True,'figure.autolayout': False,'axes.spines.left': True,
        'axes.spines.right': True,'axes.spines.top': True,'axes.spines.bottom': True})
def observation_point(num_obs,obs_radius):
    obs_phi=np.linspace(0,2*np.pi,num_obs,endpoint=False)
    obs_rho=np.full_like(obs_phi,obs_radius)
    obs_x=obs_rho * np.cos(obs_phi)
    obs_y=obs_rho * np.sin(obs_phi)
    return obs_phi,obs_rho,obs_x,obs_y
def Fourier_coeff2_dim(coeff):
    dim=len(coeff)
    K=(dim-1)//2
    return K,dim
def Gielis2shape(coeff):
    a =coeff[0]
    b =coeff[1]
    m =coeff[2]
    n1=coeff[3]
    n2=coeff[4]
    n3=coeff[5]
    def shape(theta):
        phi=m * theta / 4.0
        part1=np.abs(np.cos(phi) / a)**n2
        part2=np.abs(np.sin(phi) / b)**n3
        r=(part1+part2)**(-1.0 / n1)
        return r
    def shape_derivative(theta):
        phi=m * theta / 4.0
        u=np.cos(phi) / a
        v=np.sin(phi) / b
        eps=1e-12 
        abs_u=np.abs(u)+eps
        abs_v=np.abs(v)+eps   
        f=abs_u**n2+abs_v**n3
        du_dtheta=-(m / (4.0 * a)) * np.sin(phi)
        dv_dtheta=(m / (4.0 * b)) * np.cos(phi)    
        df_part1=n2 * (abs_u**(n2 - 1)) * np.sign(u) * du_dtheta
        df_part2=n3 * (abs_v**(n3 - 1)) * np.sign(v) * dv_dtheta
        df_dtheta=df_part1+df_part2      
        r_derivative=-(1.0 / n1) * (f**(-1.0 / n1 - 1.0)) * df_dtheta 
        return r_derivative
    return shape,shape_derivative
def Fourier2shape(coeff,len_Fourier):
    '''
    K harmonics:
    F(theta)=a0+sum a_k cos(k theta)+sum b_k sin(k theta)
    coeff=[a0,a1,a2,...,aK,b1,b2,...,bK]
    '''
    a0=coeff[0]
    a=coeff[1:len_Fourier+1]
    b=coeff[len_Fourier+1:]
    n=np.arange(1,len_Fourier+1)
    derivative_a=-n*a
    derivative_b=n*b
    def shape(theta):
        cos_mat=np.cos(theta[...,None]*n[None,:])
        sin_mat=np.sin(theta[...,None]*n[None,:])
        #cos_mat=np.cos(theta[:,None]*n[None,:])
        #sin_mat=np.sin(theta[:,None]*n[None,:])
        out_flat=a0+cos_mat @ a+sin_mat @ b
        return out_flat
    def shape_derivative(theta):
        sin_mat=np.sin(theta[...,None]*n[None,:])
        cos_mat=np.cos(theta[...,None]*n[None,:])
        #sin_mat=np.sin(theta[:,None]*n[None,:])
        #cos_mat=np.cos(theta[:,None]*n[None,:])
        out_flat=sin_mat @ derivative_a+cos_mat @ derivative_b
        return out_flat
    return shape,shape_derivative
def reconstruct_shape_Fourier(coeff,theta):
    len_Fourier,_=Fourier_coeff2_dim(coeff)
    shape_fun,_=Fourier2shape(coeff,len_Fourier)
    return shape_fun(theta)
def reconstruct_shape_Gielis(coeff,theta):
    shape_fun,_=Gielis2shape(coeff)
    return shape_fun(theta)
def my_nrmse(r_true,r_rec):
    return np.linalg.norm(r_rec-r_true)/np.linalg.norm(r_true)
def shape_nrmse_Fourier(coeff,r_true,theta):
    r_rec=reconstruct_shape_Fourier(coeff,theta)
    return my_nrmse(r_true,r_rec)*100
def shape_nrmse_Gielis(coeff,r_true,theta):
    r_rec=reconstruct_shape_Gielis(coeff,theta)
    return my_nrmse(r_true,r_rec)*100
def radius2xy(theta,radius):
    x=radius*np.cos(theta)
    y=radius*np.sin(theta)
    return x,y
def boundary_section(D):
    phi_sample=2*np.pi/D*(np.arange(D)+0.5)
    phi_low=2*np.pi/D*np.arange(D)
    phi_up =2*np.pi/D*(np.arange(D)+1.0)
    return phi_sample,phi_low,phi_up
def perimeter(t,shape,shape_derivative):
    return np.sqrt(shape(t)**2+shape_derivative(t)**2)
def scatter_field_polar(phi,rho,t_nodes,weights,shape,k,shape_derivative,c_coeff,omega,mu):
    phi_p=phi[None,None,:]          
    rho_p=rho[None,None,:]         
    t=t_nodes[:,:,None]            
    w=weights[:,:,None]             
    r_t=shape(t)
    dist2_obs=rho_p**2-2*rho_p*r_t*np.cos(phi_p-t)+r_t**2
    dist_obs=np.sqrt(dist2_obs)
    integrand_E=hankel2(0,k*dist_obs) *perimeter(t,shape,shape_derivative)   
    E_temp1=np.sum(integrand_E*w,axis=1)            
    E_temp2=E_temp1*c_coeff[:,None]
    return (-1j/4)*(-1j*omega*mu)*np.sum(E_temp2,axis=0)
def build_impedance_matrix(shape,shape_derivative,phi_sample,phi_int_low,
                           phi_int_up,omega,mu,k,D,Ng):
    gamma_e=0.5772156649015329
    xi,wi=leggauss(Ng)
    mid=0.5*(phi_int_low+phi_int_up)
    half=0.5*(phi_int_up-phi_int_low)
    t_nodes=mid[:,None]+half[:,None]*xi[None,:]
    weights=half[:,None]*wi[None,:]
    phi_m=phi_sample[:,None,None]
    t=t_nodes[None,:,:]
    rho_m=shape(phi_sample)[:,None,None]
    r_t=shape(t)
    dist2=rho_m**2-2.0*rho_m*r_t*np.cos(phi_m-t)+r_t**2
    dist2=np.maximum(dist2,0.0)
    dist=np.sqrt(dist2)
    integrand=hankel2(0,k*dist)*perimeter(t,shape,shape_derivative)
    c_coeff1=np.sum(integrand*weights[None,:,:],axis=2)
    dphi=phi_int_up-phi_int_low
    s_prime=perimeter(phi_sample,shape,shape_derivative)
    L=np.maximum(s_prime*dphi,1e-12)
    diag_int=L*(1.0-1j*(2.0/np.pi)*(np.log(k*L/4.0)+gamma_e-1.0))
    for n in range(D):
        c_coeff1[n,n]=diag_int[n]
    c_coeff1 *= (0.25*omega*mu)
    return t_nodes,weights,c_coeff1
def solve_surface_current(c_coeff1,phi_sample,k,E0,shape,inc_ang_rad):
    x_b=shape(phi_sample)*np.cos(phi_sample)
    y_b=shape(phi_sample)*np.sin(phi_sample)
    c_coeff2=E0*np.exp(-1j*k*(x_b*np.sin(inc_ang_rad)+y_b*np.cos(inc_ang_rad)))
    return np.linalg.solve(c_coeff1,c_coeff2)
def expand_surface_current(c_coeff,D,num_plot):
    phi_Js=np.linspace(0,2*np.pi,num_plot)
    Js_MoM=np.zeros_like(phi_Js,dtype=complex)
    for n in range(D):  
        mask=(phi_Js>=(2*np.pi*n/D))&(phi_Js<(2*np.pi*(n+1)/D))
        Js_MoM[mask]=c_coeff[n]
    return phi_Js,Js_MoM
def solve_multiview_scatter(shape,shape_derivative,D,Ng,phi_sample,phi_low,\
                            phi_up,phi_obs,rho_obs,omega,mu,k,E0,inc_angles):
    Es_all=[]
    t_nodes,weights,c_coeff1=build_impedance_matrix(shape,shape_derivative,\
                                    phi_sample,phi_low,phi_up,omega,mu,k,D,Ng)
    for inc_angle in inc_angles:
        c_coeff=solve_surface_current(c_coeff1,phi_sample,k,E0,shape,inc_angle)
        Es=scatter_field_polar(phi_obs,rho_obs,t_nodes,weights,shape,k,\
                               shape_derivative,c_coeff,omega,mu)
        Es_all.append(Es)
    return np.concatenate(Es_all)
def multiview_field(coeff,D,Ng,phi_sample,phi_low,phi_up,phi_obs,rho_obs,omega,\
                    mu,k,E0,inc_angles,len_Fourier):
    shape,shape_derivative=Fourier2shape(coeff,len_Fourier)
    return solve_multiview_scatter(shape,shape_derivative,D,Ng,phi_sample,
                       phi_low,phi_up,phi_obs,rho_obs,omega,mu,k,E0,inc_angles)
def multiview_field_Gielis(coeff,D,Ng,phi_sample,phi_low,phi_up,phi_obs,
                                            rho_obs,omega,mu,k,E0,inc_angles):
    shape,shape_derivative=Gielis2shape(coeff)
    return solve_multiview_scatter(shape,shape_derivative,D,Ng,phi_sample,
                       phi_low,phi_up,phi_obs,rho_obs,omega,mu,k,E0,inc_angles)
def square_sum_abs(x):
    return np.sum(np.abs(x)**2)
def obj_candidate(X,Es_meas,Es_meas_norm,D,Ng,phi_s,phi_l,phi_u,obs_phi,
                  obs_rho,omega,mu,k,E0,inc_use,len_Fourier):
    cost=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        Es_calc=multiview_field(X[i,:],D,Ng,phi_s,phi_l,phi_u,obs_phi,obs_rho,omega
                                ,mu,k,E0,inc_use,len_Fourier)
        cost[i]=np.sum(np.abs(Es_calc-Es_meas)**2)/Es_meas_norm
    return cost

def obj_candidate_Gielis(X,Es_meas,Es_meas_norm,D,Ng,phi_s,phi_l,phi_u,obs_phi,
                          obs_rho,omega,mu,k,E0,inc_use):
    cost=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        Es_calc=multiview_field_Gielis(X[i,:],D,Ng,phi_s,phi_l,phi_u,obs_phi,
                                          obs_rho,omega,mu,k,E0,inc_use)
        cost[i]=np.sum(np.abs(Es_calc - Es_meas)**2) / Es_meas_norm
    return cost
#============20260828 12:30 modify=============================================
def early_convergence(prev_best,best_now,stagnant_count,tol=1e-3,patience=100):
    relative_improvement=(prev_best-best_now)/prev_best
    if(relative_improvement<=tol):
        stagnant_count+=1
    else:
        stagnant_count=0
    stop=stagnant_count>=patience
    return stagnant_count,stop,relative_improvement
#============20260828 12:30 modify=============================================
def pad_nan_history(history_cost,history_coeff,max_iter,dim):
    current_len=len(history_cost)
    if(current_len<max_iter):
        pad_len=max_iter-current_len
        history_cost=np.concatenate((history_cost,np.full(pad_len,np.nan)))
        history_coeff=np.vstack((history_coeff,np.full((pad_len,dim),np.nan)))
    return history_cost,history_coeff
#============20260828 03:00 modify=============================================
def my_pso(obj_func,num_pop,dim,w,c1,c2,max_iter,lb,ub,patience,tol,x0=None):
    t0=time.perf_counter()
    X=lb + (ub - lb) * np.random.rand(num_pop,dim)
    V=0.05 * (ub - lb) * np.random.randn(num_pop,dim)
    Vmax=0.20 * (ub - lb)
    if(x0 is not None):
        x0=np.asarray(x0,dtype=float)
        x0=np.clip(x0,lb,ub)
        X[0,:]=x0
        V[0,:]=0.0
    pbest=X.copy()
    pbest_val=obj_func(X)  
    gbest_idx=np.argmin(pbest_val)
    gbest=pbest[gbest_idx].copy()
    gbest_val=pbest_val[gbest_idx]
    history_cost,history_coeff=[],[]
    #--------early convergence------------
    stagnant_count=0
    prev_gbest_val=gbest_val
    for it in range(max_iter):
        r1=np.random.rand(num_pop,dim)
        r2=np.random.rand(num_pop,dim)
        V=(w*V+ c1*r1*(pbest-X)+ c2*r2*(gbest-X))
        V=np.clip(V,-Vmax,Vmax)
        X=X+V
        X=np.clip(X,lb,ub)
        vals=obj_func(X)  
        improved=vals < pbest_val
        pbest[improved]=X[improved]
        pbest_val[improved]=vals[improved]
        gbest_idx=np.argmin(pbest_val)
        if(pbest_val[gbest_idx]<gbest_val):
            gbest=pbest[gbest_idx].copy()
            gbest_val=pbest_val[gbest_idx]
        history_cost.append(gbest_val)
        history_coeff.append(gbest.copy())
        print(f'Iter {it+1:4d}/{max_iter},best cost={gbest_val:.10e}')
        stagnant_count,stop,relative_improvement=early_convergence(prev_gbest_val
                                        ,gbest_val,stagnant_count,tol,patience)
        prev_gbest_val=gbest_val
        if(stop):
            print(f'Early convergence,stopped at iteration {it+1}.')
            break
    history_cost=np.array(history_cost)
    history_coeff=np.array(history_coeff)
    history_cost,history_coeff=pad_nan_history(history_cost,history_coeff,max_iter,dim)
    actual_iter=it+1
    nfe=num_pop+(actual_iter * num_pop)
    return gbest,gbest_val,history_cost,history_coeff,time.perf_counter()-t0,nfe,actual_iter
#============20260113 17:09 modify=============================================
def calculate_signal_power(sig):
    return np.mean(np.abs(sig)**2)
def calculate_noise_power(sig,snr_dB):
    signal_power=calculate_signal_power(sig)
    return signal_power/dB_to_linear_power(snr_dB)
def complex_Gaussian_noise(noise_power,size_noise):
    return np.sqrt(noise_power/2)*(np.random.normal(size=size_noise.shape)
                                  +1j*np.random.normal(size=size_noise.shape))
#============20260112 22:31 modify=============================================
def dB_to_linear_power(val_dB):
    return 10**(val_dB/10)
def DE(obj,num_pop,dim,mutation,crossover,num_iter,lb,ub,patience,tol):
    t0=time.perf_counter()
    X=np.random.uniform(lb,ub,(num_pop,dim))
    fitness=obj(X)
    best_idx=np.argmin(fitness)
    best_solution=X[best_idx].copy()
    best_fitness=fitness[best_idx]
    history_cost,history_coeff=[],[]
    stagnant_count=0
    prev_best=best_fitness
    for iter_now in range(num_iter):
        random_score=np.random.rand(num_pop,num_pop)
        random_score[np.arange(num_pop),np.arange(num_pop)]=np.inf
        select=np.argpartition(random_score,kth=2,axis=1)[:,:3]
        r1=select[:,0]
        r2=select[:,1]
        r3=select[:,2]
        V=np.clip(X[r1]+mutation*(X[r2]-X[r3]),lb,ub)
        cross_point=np.random.rand(num_pop,dim)<crossover
        rand_dim=np.random.randint(0,dim,num_pop)
        cross_point[np.arange(num_pop),rand_dim]=True
        u=np.where(cross_point,V,X)
        u_fitness=obj(u)
        better_mask=u_fitness<fitness
        X[better_mask]=u[better_mask]
        fitness[better_mask]=u_fitness[better_mask]
        best_idx=np.argmin(fitness)
        best_solution=X[best_idx].copy()
        best_fitness=fitness[best_idx]
        history_cost.append(best_fitness)
        history_coeff.append(best_solution.copy())
        print(f'Iter {iter_now+1:4d}/{num_iter},best cost={best_fitness:.10e}')
        stagnant_count,stop,relative_improvement=early_convergence(prev_best,best_fitness,stagnant_count,tol,patience)
        prev_best=best_fitness
        if(stop):
            print(f'Early convergence,stopped at iteration {iter_now+1}.')
            break
    history_cost=np.array(history_cost)
    history_coeff=np.array(history_coeff)
    history_cost,history_coeff=pad_nan_history(history_cost,history_coeff,num_iter,dim)
    actual_iter=iter_now+1
    nfe=num_pop+(actual_iter*num_pop)
    return best_solution,best_fitness,history_cost,history_coeff,time.perf_counter()-t0,nfe,actual_iter
#================20260730 16:27 modify=========================================
def CMA_ES(obj,num_pop,dim,initial_mean,initial_sigma,num_iter,lb,ub):
    t0=time.perf_counter()
    mean=initial_mean
    mu=num_pop//2
    w=np.log(mu+0.5)-np.log(np.arange(1,mu+1))
    w=w/np.sum(w)
    mu_eff=1.0/np.sum(w**2)
    c_sigma=(mu_eff+2.0)/(dim+mu_eff+5.0)
    d_sigma=1.0+2.0*max(0.0,np.sqrt((mu_eff-1.0)/(dim+1.0))-1.0)+c_sigma
    c_c=(4.0+mu_eff/dim)/(dim+4.0+2.0*mu_eff/dim)
    c_1=2.0/((dim+1.3)**2+mu_eff)
    c_mu=min(1.0-c_1,2.0*(mu_eff-2.0+1.0/mu_eff)/((dim+2.0)**2+mu_eff))
    chi_n=np.sqrt(dim)*(1.0-1.0/(4.0*dim)+1.0/(21.0*dim**2))
    path_normalizer=np.sqrt(1.0-(1.0-c_sigma)**(2.0*(np.arange(1,num_iter+1))))
    path_threshold=(1.4+2.0/(dim+1.0))*chi_n
    cov=np.eye(dim)
    path_sigma=np.zeros(dim)
    path_c=np.zeros(dim)
    best_solution=mean.copy()
    best_fitness=obj(mean[None,:])[0]
    sigma=initial_sigma
    history=np.zeros(num_iter)
    for iter_now in range(num_iter):  
        cov=0.5*(cov+cov.T)
        eigen_val,eigen_vec=np.linalg.eigh(cov)
        eigen_val=np.maximum(eigen_val,1e-30)
        eigen_val_sqrt=np.sqrt(eigen_val)
        standard_sample=np.random.standard_normal(size=(num_pop,dim))
        candidate=mean+sigma*(standard_sample*eigen_val_sqrt)@eigen_vec.T
        candidate=np.clip(candidate,lb,ub)
        fitness=obj(candidate)
        order=np.argsort(fitness)
        candidate=candidate[order]
        fitness=fitness[order]
        old_mean=mean.copy()
        candidate_select=candidate[:mu]
        mean=np.sum(w[:,None]*candidate_select,axis=0)
        weight_step=(mean-old_mean)/sigma
        cov_sqrt_inverse=eigen_vec@ np.diag(1.0/eigen_val_sqrt)@eigen_vec.T
        path_sigma=(1.0-c_sigma)*path_sigma+np.sqrt(c_sigma*(2.0-c_sigma)*mu_eff)*(cov_sqrt_inverse @ weight_step)
        normalized_path_length=(np.linalg.norm(path_sigma)/path_normalizer[iter_now])
        h_sigma=float(normalized_path_length<path_threshold)
        path_c=((1.0-c_c)*path_c+h_sigma*np.sqrt(c_c*(2.0-c_c)*mu_eff)*weight_step)
        step_select=(candidate_select-old_mean)/sigma
        rank_mu_update=np.zeros((dim,dim))
        for weight,step in zip(w,step_select):
            rank_mu_update += weight*np.outer(step,step)
        cov_decay=1.0-c_1-c_mu+c_1*(1.0-h_sigma)*c_c*(2.0-c_c)
        cov=cov_decay*cov+c_1*np.outer(path_c,path_c)+c_mu*rank_mu_update
        sigma*=np.exp((c_sigma/d_sigma)*(np.linalg.norm(path_sigma)/chi_n-1.0))
        if(fitness[0] < best_fitness):
            best_fitness=fitness[0]
            best_solution=candidate[0].copy()
        history[iter_now]=best_fitness
        print(f'Iter {iter_now+1}/{num_iter},best cost={history[iter_now]:.10e}')
    return best_solution,best_fitness,history,time.perf_counter()-t0
#================20260828 16:01 modify=========================================
def L_SHADE_iter(obj,dim,lb,ub,num_iter,num_pop_init,num_pop_min,H,patience,tol):
    t0=time.perf_counter()
    num_pop=num_pop_init
    X=np.random.uniform(lb,ub,size=(num_pop,dim))
    fitness=obj(X)
    nfe=num_pop
    mutation=np.ones(H)*0.5
    crossover=np.ones(H)*0.5
    memory_pos=0
    archive=[]
    history_cost,history_coeff=[],[]
    best_idx=np.argmin(fitness)
    best_solution=X[best_idx].copy()
    best_fitness=fitness[best_idx]
    #----------Early convergence-----------
    stagnant_count=0
    prev_best=best_fitness
    for iter_now in range(num_iter):
        F_arr,CR_arr,delta_f=[],[],[]
        X_new=X.copy()
        fitness_new=fitness.copy()
        best_idx=np.argsort(fitness)
        for i in range(num_pop):
            memory_idx=np.random.randint(H)
            F=-1
            while(F<=0):
                F=np.random.standard_cauchy()*0.1+mutation[memory_idx]
            F=min(F,1)
            CR=np.random.normal(crossover[memory_idx],0.1)
            CR=np.clip(CR,0,1)
            p=np.random.uniform(0.05,0.2)
            p_num=max(2,int(round(p*num_pop)))
            pbest_idx=np.random.choice(best_idx[:p_num])
            if(len(archive)>0):
                A=np.array(archive)
                union=np.vstack([X,A])
            else:
                union=X.copy()
            candidates=list(range(num_pop))
            candidates.remove(i)
            r1=np.random.choice(candidates)
            r2=np.random.randint(len(union))
            while(r2==i):
                r2=np.random.randint(len(union))
            v=X[i]+F*(X[pbest_idx]-X[i])+F*(X[r1]-union[r2])
            v=np.clip(v,lb,ub)
            j_rand=np.random.randint(dim)
            u=X[i].copy()
            for j in range(dim):
                if((np.random.random()<CR)or(j==j_rand)):
                    u[j]=v[j]
            fu=obj(u[None,:])[0]
            nfe+=1
            if(fu<=fitness[i]):
                X_new[i]=u
                fitness_new[i]=fu
                archive.append(X[i].copy())
                F_arr.append(F)
                CR_arr.append(CR)
                delta_f.append(abs(fitness[i]-fu))
        X=X_new
        fitness=fitness_new
        if(len(archive)>num_pop):
            remove_count=len(archive)-num_pop
            remove_idx=np.random.choice(len(archive),remove_count,replace=False)
            archive=[a for k,a in enumerate(archive) if k not in remove_idx]
        if(len(F_arr)>0):
            F_arr=np.array(F_arr)
            CR_arr=np.array(CR_arr)
            delta_f=np.array(delta_f)
            weights=delta_f/np.sum(delta_f)
            mutation[memory_pos]=np.sum(weights*F_arr**2)/np.sum(weights*F_arr)
            crossover[memory_pos]=np.sum(weights*CR_arr)
            memory_pos=(memory_pos+1)%H
        num_pop_new=int(round(num_pop_init-(num_pop_init-num_pop_min)*(iter_now+1)/num_iter))
        num_pop_new=max(num_pop_min,num_pop_new)
        if(num_pop_new<num_pop):
            sorted_idx=np.argsort(fitness)
            X=X[sorted_idx[:num_pop_new]]
            fitness=fitness[sorted_idx[:num_pop_new]]
            num_pop=num_pop_new
        best_idx=np.argmin(fitness)
        best_solution=X[best_idx].copy()
        best_fitness=fitness[best_idx]
        history_cost.append(best_fitness)
        history_coeff.append(best_solution.copy())
        print(f'Iter {iter_now+1:4d}/{num_iter},best cost={best_fitness:.10e}')
        stagnant_count,stop,relative_improvement=early_convergence(
            prev_best,best_fitness,stagnant_count,tol,patience)
        prev_best=best_fitness
        if(stop):
            print(f'Early convergence,stopped at iteration {iter_now+1}.')
            break
    history_cost=np.array(history_cost)
    history_coeff=np.array(history_coeff)
    history_cost,history_coeff=pad_nan_history(history_cost,history_coeff,num_iter,dim)
    actual_iter=iter_now+1
    return best_solution,best_fitness,history_cost,history_coeff,time.perf_counter()-t0,nfe,actual_iter
#================20260828 16:01 modify=========================================
def L_SHADE(obj,dim,lb,ub,num_iter,num_pop_init,num_pop_min,H,patience,tol):
    t0 = time.perf_counter()
    num_pop = num_pop_init
    X = np.random.uniform(lb,ub,size=(num_pop,dim))
    fitness = obj(X)
    nfe = num_pop
    mutation = np.ones(H) * 0.5
    crossover = np.ones(H) * 0.5
    memory_pos = 0
    archive = []
    history_cost,history_coeff = [],[]
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx].copy()
    best_fitness = fitness[best_idx] 
    # ----------Early convergence-----------
    stagnant_count = 0
    prev_best = best_fitness
    max_nfe = num_pop_init * num_iter 
    for iter_now in range(num_iter):
        F_arr,CR_arr,delta_f = [],[],[]
        X_new = X.copy()
        fitness_new = fitness.copy()
        best_idx = np.argsort(fitness)
        for i in range(num_pop):
            memory_idx = np.random.randint(H)
            F = -1
            while(F <= 0):
                F = np.random.standard_cauchy() * 0.1 + mutation[memory_idx]
            F = min(F,1)    
            CR = np.random.normal(crossover[memory_idx],0.1)
            CR = np.clip(CR,0,1)  
            p = np.random.uniform(0.05,0.2)
            p_num = max(2,int(round(p * num_pop)))
            pbest_idx = np.random.choice(best_idx[:p_num]) 
            if(len(archive) > 0):
                A = np.array(archive)
                union = np.vstack([X,A])
            else:
                union = X.copy()           
            candidates = list(range(num_pop))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            r2 = np.random.randint(len(union))
            while(r2 == i):
                r2 = np.random.randint(len(union))        
            v = X[i] + F * (X[pbest_idx] - X[i]) + F * (X[r1] - union[r2])
            v = np.clip(v,lb,ub) 
            j_rand = np.random.randint(dim)
            u = X[i].copy()
            for j in range(dim):
                if((np.random.random() < CR) or (j == j_rand)):
                    u[j] = v[j]         
            fu = obj(u[None,:])[0]
            nfe += 1 
            if(fu <= fitness[i]):
                X_new[i] = u
                fitness_new[i] = fu
                archive.append(X[i].copy())
                F_arr.append(F)
                CR_arr.append(CR)
                delta_f.append(abs(fitness[i] - fu))    
        X = X_new
        fitness = fitness_new
        if(len(F_arr) > 0):
            F_arr = np.array(F_arr)
            CR_arr = np.array(CR_arr)
            delta_f = np.array(delta_f)
            weights = delta_f / np.sum(delta_f)
            mutation[memory_pos] = np.sum(weights * F_arr**2) / np.sum(weights * F_arr)
            crossover[memory_pos] = np.sum(weights * CR_arr)
            memory_pos = (memory_pos + 1) % H
        num_pop_new = int(round(num_pop_init - (num_pop_init - num_pop_min) * (nfe / max_nfe)))
        num_pop_new = max(num_pop_min,num_pop_new)
        if(num_pop_new < num_pop):
            sorted_idx = np.argsort(fitness)
            X = X[sorted_idx[:num_pop_new]]
            fitness = fitness[sorted_idx[:num_pop_new]]
            num_pop = num_pop_new
        if(len(archive) > num_pop):
            remove_count = len(archive) - num_pop
            remove_idx = np.random.choice(len(archive),remove_count,replace=False)
            archive = [a for k,a in enumerate(archive) if k not in remove_idx]
        best_idx = np.argmin(fitness)
        best_solution = X[best_idx].copy()
        best_fitness = fitness[best_idx]
        history_cost.append(best_fitness)
        history_coeff.append(best_solution.copy())
        print(f'Iter {iter_now+1:4d}/{num_iter},best cost={best_fitness:.10e}')
        # ----------Early convergence check-----------
        stagnant_count,stop,relative_improvement=early_convergence(prev_best,
            best_fitness,stagnant_count,tol,patience)
        prev_best = best_fitness
        if(stop):
            print(f'Early convergence,stopped at iteration {iter_now+1}.')
            break
    history_cost = np.array(history_cost)
    history_coeff = np.array(history_coeff)
    history_cost,history_coeff = pad_nan_history(history_cost,history_coeff,num_iter,dim)
    actual_iter=iter_now+1
    return best_solution,best_fitness,history_cost,history_coeff,time.perf_counter()-t0,nfe,actual_iter
def analysis_alg_Fourier(cases,file_map,out_dir,timestamp,xlabel,shape_method):
    plot_style()
    conditions=list(file_map.keys())
    algos=list(next(iter(file_map.values())).keys())
    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    linestyles=["-","--",":","-.","-","--"]
    markers=["o","s","^","D","v","P"]
    algo_styles={alg:{"color":colors[i%len(colors)],"ls":linestyles[i%len(linestyles)],
            "marker":markers[i%len(markers)]} for i,alg in enumerate(algos)}
    print("Loading data and computing shape errors...")
    data_store={cond: {alg: {} for alg in algos} for cond in conditions}
    coeff_true=None
    theta_eval=np.linspace(0,2*np.pi,1000) 
    r_true_eval=None
    for cond in conditions:
        for alg in algos:
            filepath=Path(file_map[cond][alg])
            mat=loadmat(str(filepath))
            history=mat["history_all"]
            final_costs=history[:,-1]
            times=mat["time_all"].squeeze()
            coeffs=mat["coeff_all"]
            if(coeff_true is None):
                coeff_true=mat["coeff_true"].squeeze()
                if(shape_method=='Fourier'):
                    r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
                elif(shape_method=='Gielis'):
                    r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
            if(shape_method=='Fourier'):
                nrmse_list=[shape_nrmse_Fourier(c,r_true_eval,theta_eval) for c in coeffs]    
            elif(shape_method=='Gielis'):
                nrmse_list=[shape_nrmse_Gielis(c,r_true_eval,theta_eval) for c in coeffs]    
            nrmse_array=np.array(nrmse_list)
            best_idx=np.argsort(nrmse_array)[len(nrmse_array) // 2]
            #best_idx=np.argmin(final_costs)  #np.mean(coeffs,axis=0)
            data_store[cond][alg]={"history": history,"final_costs": final_costs,
                "times": times,"best_coeff":coeffs[best_idx] ,"shape_nrmse": nrmse_array, 
                "mean_time": np.mean(times),"std_time": np.std(times)}  #np.mean(coeffs,axis=0)      
    #=======================convergence curve (mean)===========================  
    '''
    all_mean_values=[]
    for cond in conditions:
        for alg in algos:
            hist=np.asarray(data_store[cond][alg]["history"],dtype=float)
            history_mean=np.mean(hist,axis=0)
            valid_values=history_mean[np.isfinite(history_mean) & (history_mean > 0)]
            all_mean_values.extend(valid_values)
    all_mean_values=np.asarray(all_mean_values)
    global_ymin=np.min(all_mean_values)
    global_ymax=np.max(all_mean_values)
    global_ymin *= 0.9
    global_ymax *= 1.1
    for cond in conditions:
        plt.figure()
        for alg in algos:
            hist=np.asarray(data_store[cond][alg]["history"],dtype=float)
            history_mean=np.mean(hist,axis=0)
            max_iter=np.arange(1,len(history_mean)+1)
            plt.plot(max_iter,history_mean,label=alg,color=algo_styles[alg]["color"],
                linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Cost Value")
        #plt.title(f"Average Convergence Curves: {cond}")
        plt.yscale("log")
        plt.ylim(global_ymin,global_ymax)
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.savefig(out_dir / f"converge_{cases}_{cond}_{timestamp}.pdf",
            bbox_inches="tight",pad_inches=0.02)
        plt.show()
        '''
    #=======================convergence curve (Median)=========================       
    all_median_values=[]
    for cond in conditions:
        for alg in algos:
            hist=np.asarray(data_store[cond][alg]["history"],dtype=float)
            history_median=np.median(hist,axis=0)
            valid_values=history_median[np.isfinite(history_median) & (history_median > 0)]
            all_median_values.extend(valid_values)
    all_median_values=np.asarray(all_median_values)
    global_ymin=np.min(all_median_values) * 0.9
    global_ymax=np.max(all_median_values) * 1.1
    for cond in conditions:
        plt.figure()
        for alg in algos:
            hist=np.asarray(data_store[cond][alg]["history"],dtype=float)
            history_median=np.median(hist,axis=0)
            max_iter=np.arange(1,len(history_median)+1)      
            plt.plot(max_iter,history_median,label=alg,color=algo_styles[alg]["color"],
                    linestyle=algo_styles[alg]["ls"])     
        plt.xlabel("Iteration")
        plt.ylabel("Cost Value")
        plt.yscale("log")
        plt.ylim(global_ymin,global_ymax)
        plt.grid(True,which="both",linestyle=":",alpha=0.7)
        plt.legend(loc="best",frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"converge_median_{cases}_{cond}_{timestamp}.pdf",
                    bbox_inches="tight",pad_inches=0.02)
        plt.show()
    #====================Robustness Curve======================================
    x_vals=np.arange(len(conditions))
    plt.figure()
    for alg in algos:
        mean_nrmses=[np.mean(data_store[cond][alg]["shape_nrmse"]) for cond in conditions]
        plt.plot(x_vals,mean_nrmses,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"])
        #print(alg,mean_nrmses)
    plt.xticks(x_vals,conditions)
    #plt.yscale("log")
    plt.ylabel("Shape NRMSE (%)")
    plt.xlabel(xlabel)
    plt.grid(True,which="both",linestyle="--",alpha=0.6)
    plt.legend(loc="best",frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    #====================Computation Time======================================
    plt.figure()
    bar_width=0.2
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        means=[data_store[cond][alg].get("mean_time",0) for cond in conditions]
        #stds=[data_store[cond][alg].get("std_time",0) for cond in conditions]
        plt.bar(x_pos+i*bar_width,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos+bar_width*1.5,conditions)
    plt.ylabel("Execution Time (Seconds)")
    #plt.title("Computational Cost Comparison")
    plt.legend(loc="upper left")
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"Time_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    #====================Shape Reconstruction==================================
    if(shape_method=='Fourier'):
        r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
    elif(shape_method=='Gielis'):
        r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
    x_true,y_true=radius2xy(theta_eval,r_true_eval)
    for cond in conditions:
        plt.figure()
        plt.plot(x_true,y_true,color="black",linewidth=3,label="True Shape")
        for alg in algos:
            if "best_coeff" in data_store[cond][alg]:
                best_c=data_store[cond][alg]["best_coeff"]
                if(shape_method=='Fourier'):
                    r_rec=reconstruct_shape_Fourier(best_c,theta_eval)
                elif(shape_method=='Gielis'):
                    r_rec=reconstruct_shape_Gielis(best_c,theta_eval)
                x_rec,y_rec=radius2xy(theta_eval,r_rec)
                plt.plot(x_rec,y_rec,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"],linewidth=2,label=alg)
        plt.axis("equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True,linestyle="--",alpha=0.6)
        plt.legend(loc="best",frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir/f"Shape_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight")
        plt.show()
def analysis_alg_adaptive_item(cases,file_map,out_dir,timestamp,xlabel,shape_method):
    plot_style()
    conditions=list(file_map.keys())
    algos=list(dict.fromkeys(alg for cond_map in file_map.values() for alg in cond_map))
    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    linestyles=["-","--",":","-.","-","--"]
    markers=["o","s","^","D","v","P"]
    algo_styles={alg:{"color":colors[i%len(colors)],"ls":linestyles[i%len(linestyles)],"marker":markers[i%len(markers)]} for i,alg in enumerate(algos)}
    data_store={cond:{} for cond in conditions}
    coeff_true=None
    theta_eval=np.linspace(0,2*np.pi,1000)
    r_true_eval=None
    for cond in conditions:
        for alg in algos:
            if alg not in file_map[cond]:
                continue
            filepath=Path(file_map[cond][alg])
            mat=loadmat(str(filepath))
            history=np.asarray(mat["history_all"],dtype=float)
            final_costs=history[:,-1]
            times=np.asarray(mat["time_all"]).squeeze()
            coeffs=np.asarray(mat["coeff_all"],dtype=float)
            if coeff_true is None:
                coeff_true=np.asarray(mat["coeff_true"]).squeeze()
                if shape_method=='Fourier':
                    r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
                elif shape_method=='Gielis':
                    r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
                else:
                    raise ValueError("shape_method must be 'Fourier' or 'Gielis'")
            if shape_method=='Fourier':
                nrmse_list=[shape_nrmse_Fourier(c,r_true_eval,theta_eval) for c in coeffs]
            elif shape_method=='Gielis':
                nrmse_list=[shape_nrmse_Gielis(c,r_true_eval,theta_eval) for c in coeffs]
            nrmse_array=np.asarray(nrmse_list,dtype=float)
            order=np.argsort(nrmse_array)
            best_idx=order[len(order)//2]
            data_store[cond][alg]={"history":history,"final_costs":final_costs,"times":times,"best_coeff":coeffs[best_idx],"shape_nrmse":nrmse_array,"mean_time":np.mean(times),"std_time":np.std(times)}
    all_mean_values=[]
    for cond in conditions:
        for alg,data in data_store[cond].items():
            hist=np.asarray(data["history"],dtype=float)
            history_mean=np.mean(hist,axis=0)
            valid_values=history_mean[np.isfinite(history_mean)&(history_mean>0)]
            all_mean_values.extend(valid_values)
    all_mean_values=np.asarray(all_mean_values,dtype=float)
    if len(all_mean_values)>0:
        global_ymin=np.min(all_mean_values)*0.9
        global_ymax=np.max(all_mean_values)*1.1
    else:
        global_ymin=None
        global_ymax=None
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=np.asarray(data_store[cond][alg]["history"],dtype=float)
            history_mean=np.mean(hist,axis=0)
            max_iter=np.arange(1,len(history_mean)+1)
            plt.plot(max_iter,history_mean,label=alg,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Cost Value")
        plt.yscale("log")
        if global_ymin is not None and global_ymax is not None:
            plt.ylim(global_ymin,global_ymax)
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"converge_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight",pad_inches=0.02)
        plt.show()
    x_vals=np.arange(len(conditions))
    plt.figure()
    for alg in algos:
        mean_nrmses=[]
        for cond in conditions:
            if alg in data_store[cond]:
                mean_nrmses.append(np.mean(data_store[cond][alg]["shape_nrmse"]))
            else:
                mean_nrmses.append(np.nan)
        mean_nrmses=np.asarray(mean_nrmses,dtype=float)
        if np.any(np.isfinite(mean_nrmses)):
            plt.plot(x_vals,mean_nrmses,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"])
    plt.xticks(x_vals,conditions)
    plt.ylabel("Shape NRMSE (%)")
    plt.xlabel(xlabel)
    plt.grid(True,which="both",linestyle="--",alpha=0.6)
    plt.legend(loc="best",frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()

    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_time"])
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("Execution Time (Seconds)")
    plt.legend(loc="best")
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"Time_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    if shape_method=='Fourier':
        r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
    elif shape_method=='Gielis':
        r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
    x_true,y_true=radius2xy(theta_eval,r_true_eval)
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        plt.plot(x_true,y_true,color="black",linewidth=3,label="True Shape")
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            best_c=data_store[cond][alg]["best_coeff"]
            if shape_method=='Fourier':
                r_rec=reconstruct_shape_Fourier(best_c,theta_eval)
            elif shape_method=='Gielis':
                r_rec=reconstruct_shape_Gielis(best_c,theta_eval)
            x_rec,y_rec=radius2xy(theta_eval,r_rec)
            plt.plot(x_rec,y_rec,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"],linewidth=2,label=alg)
        plt.axis("equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True,linestyle="--",alpha=0.6)
        plt.legend(loc="best",frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir/f"Shape_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight")
        plt.show()
def analysis_alg_adaptive_item2(cases,file_map,out_dir,timestamp,xlabel,shape_method):
    plot_style()
    conditions=list(file_map.keys())
    algos=list(dict.fromkeys(alg for cond_map in file_map.values() for alg in cond_map))
    colors=["#1f77b4","#ff7f0e","#1b781b","#ff3333","#9467bd","#8c564b"]
    linestyles=["-","--",":","-.",(0,(5,1)),(0,(3,1,1,1,1,1))]
    markers=["o","s","^","D","v","P"]
    algo_styles={alg:{"color":colors[i%len(colors)],"ls":linestyles[i%len(linestyles)],"marker":markers[i%len(markers)]} for i,alg in enumerate(algos)}
    data_store={cond:{} for cond in conditions}
    coeff_true=None
    theta_eval=np.linspace(0,2*np.pi,1000)
    r_true_eval=None
    for cond in conditions:
        for alg in algos:
            if alg not in file_map[cond] or not file_map[cond][alg]:
                continue
            filepath=Path(file_map[cond][alg])
            mat=loadmat(str(filepath))
            history_cost=np.asarray(mat["history_cost_all"],dtype=float)
            history_coeff=np.asarray(mat["history_coeff_all"],dtype=float)
            coeffs=np.asarray(mat["coeff_all"],dtype=float)
            times=np.asarray(mat["time_all"],dtype=float).ravel()
            nfe=np.asarray(mat["nfe"],dtype=float).ravel()
            actual_iter=np.asarray(mat["actual_iter"],dtype=float).ravel()
            if coeff_true is None:
                coeff_true=np.asarray(mat["coeff_true"],dtype=float).squeeze()
                if shape_method=="Fourier":
                    r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
                elif shape_method=="Gielis":
                    r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
                else:
                    raise ValueError("shape_method must be 'Fourier' or 'Gielis'")
            if shape_method=="Fourier":
                nrmse_array=np.asarray([shape_nrmse_Fourier(c,r_true_eval,theta_eval) for c in coeffs],dtype=float)
            else:
                nrmse_array=np.asarray([shape_nrmse_Gielis(c,r_true_eval,theta_eval) for c in coeffs],dtype=float)
            n_seed,n_iter,_=history_coeff.shape
            history_nrmse=np.full((n_seed,n_iter),np.nan)
            for s in range(n_seed):
                for it in range(n_iter):
                    c=history_coeff[s,it]
                    if np.any(~np.isfinite(c)):
                        continue
                    if shape_method=="Fourier":
                        history_nrmse[s,it]=shape_nrmse_Fourier(c,r_true_eval,theta_eval)
                    else:
                        history_nrmse[s,it]=shape_nrmse_Gielis(c,r_true_eval,theta_eval)
            median_nrmse=np.nanmedian(nrmse_array)
            median_idx=np.nanargmin(np.abs(nrmse_array-median_nrmse))
            std_nrmse=np.std(nrmse_array,ddof=1) if len(nrmse_array)>1 else np.nan
            std_time=np.std(times,ddof=1) if len(times)>1 else np.nan
            std_nfe=np.std(nfe,ddof=1) if len(nfe)>1 else np.nan
            data_store[cond][alg]={"history_cost":history_cost,"history_coeff":history_coeff,"history_nrmse":history_nrmse,"coeffs":coeffs,"median_idx":median_idx,"median_coeff":coeffs[median_idx],"median_nrmse":median_nrmse,"shape_nrmse":nrmse_array,"mean_nrmse":np.mean(nrmse_array),"std_nrmse":std_nrmse,"times":times,"mean_time":np.mean(times),"std_time":std_time,"nfe":nfe,"mean_nfe":np.mean(nfe),"std_nfe":std_nfe,"actual_iter":actual_iter}
            print(f"{cond} | {alg} | runs={len(nrmse_array)} | median NRMSE={median_nrmse:.6f}% | selected run={median_idx} | selected NRMSE={nrmse_array[median_idx]:.6f}%")
    #====================Mean Cost Convergence=========================================
    '''
    all_cost_values=[]
    for cond in conditions:
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            history_mean=np.nanmean(hist,axis=0)
            valid=np.isfinite(history_mean)&(history_mean>0)
            all_cost_values.extend(history_mean[valid])
    ymin=np.min(all_cost_values)
    ymax=np.max(all_cost_values)
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            history_mean=np.nanmean(hist,axis=0)
            valid=np.isfinite(history_mean)&(history_mean>0)
            iteration=np.arange(1,len(history_mean)+1)
            plt.plot(iteration[valid],history_mean[valid],label=alg,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Mean Cost Value")
        plt.yscale("log")
        plt.ylim(ymin,ymax)
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"cost_history_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight",pad_inches=0.02)
        plt.show()
    '''
    #====================Median-NRMSE Run Convergence=========================================
    all_cost_values_median=[]
    for cond in conditions:
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            median_idx=data_store[cond][alg]["median_idx"]
            history_median=hist[median_idx]
            valid=np.isfinite(history_median)&(history_median>0)
            all_cost_values_median.extend(history_median[valid])
    ymin_median=np.min(all_cost_values_median)
    ymax_median=np.max(all_cost_values_median)
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            median_idx=data_store[cond][alg]["median_idx"]
            history_median=hist[median_idx]
            valid=np.isfinite(history_median)&(history_median>0)
            iteration=np.arange(1,len(history_median)+1)
            plt.plot(iteration[valid],history_median[valid],label=alg,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Cost Value")
        plt.yscale("log")
        plt.ylim(ymin_median,ymax_median)
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"cost_history_median_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight",pad_inches=0.02)
        plt.show()
    #====================Mean Shape NRMSE=========================================
    '''
    x_vals=np.arange(len(conditions))
    plt.figure()
    for alg in algos:
        means=[]
        for cond in conditions:
            means.append(data_store[cond][alg]["mean_nrmse"] if alg in data_store[cond] else np.nan)
        means=np.asarray(means,dtype=float)
        plt.plot(x_vals,means,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"])
    plt.xticks(x_vals,conditions)
    plt.xlabel(xlabel)
    plt.ylabel("Mean Shape NRMSE (%)")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_mean_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    #====================NFE=========================================
    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_nfe"])
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("NFE")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"nfe_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    #====================Runtime (Minutes)=========================================
    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_time"]/60.0)
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("Execution Time (Minutes)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"time_minutes_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    #====================Reconstructed Shape: Median-NRMSE Run=========================================
    x_true,y_true=radius2xy(theta_eval,r_true_eval)
    all_x=[x_true]
    all_y=[y_true]
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            median_c=data_store[cond][alg]["median_coeff"]
            if shape_method=="Fourier":
                r_rec=reconstruct_shape_Fourier(median_c,theta_eval)
            else:
                r_rec=reconstruct_shape_Gielis(median_c,theta_eval)
            x_rec,y_rec=radius2xy(theta_eval,r_rec)
            all_x.append(x_rec)
            all_y.append(y_rec)
    lim=max(max(np.max(np.abs(x)) for x in all_x),max(np.max(np.abs(y)) for y in all_y))*1.05
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        fig,ax=plt.subplots()
        ax.plot(x_true,y_true,color="black",linewidth=3,label="True Shape")
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            median_c=data_store[cond][alg]["median_coeff"]
            if shape_method=="Fourier":
                r_rec=reconstruct_shape_Fourier(median_c,theta_eval)
            else:
                r_rec=reconstruct_shape_Gielis(median_c,theta_eval)
            x_rec,y_rec=radius2xy(theta_eval,r_rec)
            ax.plot(x_rec,y_rec,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"],linewidth=2,label=alg)
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True,linestyle="--",alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"shape2_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight")
        plt.show()        
def analysis_alg_adaptive_item3(cases,file_map,out_dir,timestamp,xlabel,shape_method):
    plot_style()
    conditions=list(file_map.keys())
    algos=list(dict.fromkeys(alg for cond_map in file_map.values() for alg in cond_map))
    colors=["#1f77b4","#ff7f0e","#1b781b","#ff3333","#9467bd","#8c564b"]
    linestyles=["-","--",":","-.",(0,(5,1)),(0,(3,1,1,1,1,1))]
    markers=["o","s","^","D","v","P"]
    algo_styles={alg:{"color":colors[i%len(colors)],"ls":linestyles[i%len(linestyles)],"marker":markers[i%len(markers)]} for i,alg in enumerate(algos)}
    data_store={cond:{} for cond in conditions}
    coeff_true=None
    theta_eval=np.linspace(0,2*np.pi,1000)
    r_true_eval=None
    for cond in conditions:
        for alg in algos:
            if alg not in file_map[cond] or not file_map[cond][alg]:
                continue
            filepath=Path(file_map[cond][alg])
            mat=loadmat(str(filepath))
            history_cost=np.asarray(mat["history_cost_all"],dtype=float)
            history_coeff=np.asarray(mat["history_coeff_all"],dtype=float)
            coeffs=np.asarray(mat["coeff_all"],dtype=float)
            times=np.asarray(mat["time_all"],dtype=float).ravel()
            nfe=np.asarray(mat["nfe"],dtype=float).ravel()
            actual_iter=np.asarray(mat["actual_iter"],dtype=float).ravel()
            if coeff_true is None:
                coeff_true=np.asarray(mat["coeff_true"],dtype=float).squeeze()
                if shape_method=="Fourier":
                    r_true_eval=reconstruct_shape_Fourier(coeff_true,theta_eval)
                elif shape_method=="Gielis":
                    r_true_eval=reconstruct_shape_Gielis(coeff_true,theta_eval)
                else:
                    raise ValueError("shape_method must be 'Fourier' or 'Gielis'")
            if shape_method=="Fourier":
                nrmse_array=np.asarray([shape_nrmse_Fourier(c,r_true_eval,theta_eval) for c in coeffs],dtype=float)
            else:
                nrmse_array=np.asarray([shape_nrmse_Gielis(c,r_true_eval,theta_eval) for c in coeffs],dtype=float)
            n_seed,n_iter,_=history_coeff.shape
            history_nrmse=np.full((n_seed,n_iter),np.nan)
            for s in range(n_seed):
                for it in range(n_iter):
                    c=history_coeff[s,it]
                    if np.any(~np.isfinite(c)):
                        continue
                    if shape_method=="Fourier":
                        history_nrmse[s,it]=shape_nrmse_Fourier(c,r_true_eval,theta_eval)
                    else:
                        history_nrmse[s,it]=shape_nrmse_Gielis(c,r_true_eval,theta_eval)
            order=np.argsort(nrmse_array)
            best_idx=order[len(order)//2]
            std_nrmse=np.std(nrmse_array,ddof=1) if len(nrmse_array)>1 else np.nan
            std_time=np.std(times,ddof=1) if len(times)>1 else np.nan
            std_nfe=np.std(nfe,ddof=1) if len(nfe)>1 else np.nan
            data_store[cond][alg]={"history_cost":history_cost,"history_coeff":history_coeff,"history_nrmse":history_nrmse,"coeffs":coeffs,"best_coeff":coeffs[best_idx],"shape_nrmse":nrmse_array,"mean_nrmse":np.mean(nrmse_array),"std_nrmse":std_nrmse,"times":times,"mean_time":np.mean(times),"std_time":std_time,"nfe":nfe,"mean_nfe":np.mean(nfe),"std_nfe":std_nfe,"actual_iter":actual_iter}
    all_cost_values=[]
    for cond in conditions:
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            history_mean=np.nanmean(hist,axis=0)
            valid=np.isfinite(history_mean)&(history_mean>0)
            all_cost_values.extend(history_mean[valid])
    ymin=np.min(all_cost_values)
    ymax=np.max(all_cost_values)
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_cost"]
            history_mean=np.nanmean(hist,axis=0)
            valid=np.isfinite(history_mean)&(history_mean>0)
            iteration=np.arange(1,len(history_mean)+1)
            plt.plot(iteration[valid],history_mean[valid],label=alg,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Mean Cost Value")
        plt.yscale("log")
        plt.ylim(ymin,ymax)
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"cost_history_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight",pad_inches=0.02)
        plt.show()
    '''
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        plt.figure()
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            hist=data_store[cond][alg]["history_nrmse"]
            history_mean=np.nanmean(hist,axis=0)
            valid=np.isfinite(history_mean)
            iteration=np.arange(1,len(history_mean)+1)
            plt.plot(iteration[valid],history_mean[valid],label=alg,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"])
        plt.xlabel("Iteration")
        plt.ylabel("Mean Shape NRMSE (%)")
        plt.grid(True,which="both",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"nrmse_history_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight",pad_inches=0.02)
        plt.show()
    '''
    x_vals=np.arange(len(conditions))
    plt.figure()
    for alg in algos:
        means=[]
        for cond in conditions:
            if alg in data_store[cond]:
                means.append(data_store[cond][alg]["mean_nrmse"])
            else:
                means.append(np.nan)
        means=np.asarray(means,dtype=float)
        plt.plot(x_vals,means,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"])
    plt.xticks(x_vals,conditions)
    plt.xlabel(xlabel)
    plt.ylabel("Mean Shape NRMSE (%)")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_mean_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    plt.figure()
    for alg in algos:
        stds=[]
        for cond in conditions:
            if alg in data_store[cond]:
                stds.append(data_store[cond][alg]["std_nrmse"])
            else:
                stds.append(np.nan)
        stds=np.asarray(stds,dtype=float)
        plt.plot(x_vals,stds,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"])
    plt.xticks(x_vals,conditions)
    plt.xlabel(xlabel)
    plt.ylabel("Standard Deviation of Shape NRMSE (%)")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_std_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    '''
    plt.figure()
    for alg in algos:
        means=[]
        stds=[]
        for cond in conditions:
            if alg in data_store[cond]:
                means.append(data_store[cond][alg]["mean_nrmse"])
                stds.append(data_store[cond][alg]["std_nrmse"])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        means=np.asarray(means,dtype=float)
        stds=np.asarray(stds,dtype=float)
        std_plot=np.where(np.isfinite(stds),stds,0)
        plt.errorbar(x_vals,means,yerr=std_plot,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"],capsize=4)
    plt.xticks(x_vals,conditions)
    plt.xlabel(xlabel)
    plt.ylabel("Shape NRMSE (%)")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"nrmse_mean_std_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    '''
    plt.figure()
    for alg in algos:
        means=[]
        stds=[]
        for cond in conditions:
            if alg in data_store[cond]:
                means.append(data_store[cond][alg]["mean_nfe"])
                stds.append(data_store[cond][alg]["std_nfe"])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        means=np.asarray(means,dtype=float)
        stds=np.asarray(stds,dtype=float)
        std_plot=np.where(np.isfinite(stds),stds,0)
        plt.errorbar(x_vals,means,yerr=std_plot,label=alg,color=algo_styles[alg]["color"],marker=algo_styles[alg]["marker"],linestyle=algo_styles[alg]["ls"],capsize=4)
    plt.xticks(x_vals,conditions)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Function Evaluations (NFE)")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"nfe_mean_std_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_nfe"])
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("Number of Function Evaluations (NFE)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"nfe_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_time"])
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("Execution Time (Seconds)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"time_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show()
    '''
    #====================Runtime (Minutes)=========================================
    plt.figure()
    n_alg=max(len(algos),1)
    total_width=0.8
    bar_width=total_width/n_alg
    x_pos=np.arange(len(conditions))
    for i,alg in enumerate(algos):
        positions=[]
        means=[]
        offset=(i-(n_alg-1)/2)*bar_width
        for j,cond in enumerate(conditions):
            if alg not in data_store[cond]:
                continue
            positions.append(x_pos[j]+offset)
            means.append(data_store[cond][alg]["mean_time"]/60.0)
        if len(means)>0:
            plt.bar(positions,means,bar_width,label=alg,
                    color=algo_styles[alg]["color"],alpha=0.8)
    plt.xticks(x_pos,conditions)
    plt.ylabel("Execution Time (Minutes)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir/f"time_minutes_{cases}_{timestamp}.pdf",bbox_inches="tight")
    plt.show() 
    x_true,y_true=radius2xy(theta_eval,r_true_eval)
    all_x=[x_true]
    all_y=[y_true]
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
        for alg in algos:
            if alg not in data_store[cond]:
                continue
            best_c=data_store[cond][alg]["best_coeff"]
            if shape_method=="Fourier":
                r_rec=reconstruct_shape_Fourier(best_c,theta_eval)
            else:
                r_rec=reconstruct_shape_Gielis(best_c,theta_eval)
            x_rec,y_rec=radius2xy(theta_eval,r_rec)
            all_x.append(x_rec)
            all_y.append(y_rec)
    
    lim=max(max(np.max(np.abs(x)) for x in all_x),max(np.max(np.abs(y)) for y in all_y))*1.05
    
    for cond in conditions:
        if len(data_store[cond])==0:
            continue
    
        fig,ax=plt.subplots()
    
        ax.plot(x_true,y_true,color="black",linewidth=3,label="True Shape")
    
        for alg in algos:
            if alg not in data_store[cond]:
                continue
    
            best_c=data_store[cond][alg]["best_coeff"]
    
            if shape_method=="Fourier":
                r_rec=reconstruct_shape_Fourier(best_c,theta_eval)
            else:
                r_rec=reconstruct_shape_Gielis(best_c,theta_eval)
    
            x_rec,y_rec=radius2xy(theta_eval,r_rec)
    
            ax.plot(x_rec,y_rec,color=algo_styles[alg]["color"],linestyle=algo_styles[alg]["ls"],linewidth=2,label=alg)
    
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_aspect("equal",adjustable="box")
    
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True,linestyle="--",alpha=0.6)
        ax.legend()
    
        plt.tight_layout()
        plt.savefig(out_dir/f"shape2_{cases}_{cond}_{timestamp}.pdf",bbox_inches="tight")
        plt.show()       
            
          
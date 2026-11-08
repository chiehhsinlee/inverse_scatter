from pathlib import Path
import time
import numpy as np
import scipy.constants as const
from scipy.io import savemat
#from google.colab import drive
#from joblib import Parallel, delayed
from scatter_utils_new import (wave_parameter, plot_style, observation_point,
    Gielis2shape, boundary_section, multiview_field_Gielis, obj_candidate, complex_Gaussian_noise,
    square_sum_abs, L_SHADE, CMA_ES, DE, my_pso, calculate_noise_power, obj_candidate_Gielis)
plot_style()
timestamp = time.strftime("%Y%m%d_%H%M")
alg ='de'  # 'pso' 'de' 'lshade'
fp = 'full10'
snr_dB = 20
num_seed_noise = 50
num_par=100
seed_noise=np.arange(num_seed_noise)
seed_alg=1000*seed_noise
patience,tol=50,1e-3
#drive.mount("/content/drive")
#outdir = Path("/content/drive/MyDrive/inverse_scattering") / f"{fp}_{alg}_{snr_dB}_par{num_par}_{timestamp}"
outdir=Path(f"{fp}_{alg}_{snr_dB}_par{num_par}_{timestamp}")
outdir.mkdir(parents=True,exist_ok=True)
print(f"Output directory: {outdir}")
t_start = time.perf_counter()
configs={
    'full5':{'D_true':150,'Ng_true':6,'D_inv':75,'Ng_inv':6,'num_obs':6,
        'inc_angle':np.deg2rad([0,90,180,270]),'coeff_true':np.array([1.0,1.0,4,1.0,1.0,1.0]),
        'lb':np.array([0.5,0.5,2.0,0.2,0.2,0.2]),'ub':np.array([2.0,2.0,8.0,3.0,3.0,3.0]),
        'optimizer_set':{
            10:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            20:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            30:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            None:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                  'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                  'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)}}},
    'full8':{'D_true':150,'Ng_true':6,'D_inv':75,'Ng_inv':6,'num_obs':6,
        'inc_angle':np.deg2rad([0,90,180,270]),'coeff_true':np.array([1.0,1.0,5,1.24,1.24,1.24]),
        'lb':np.array([0.5,0.5,1.0,0.5,0.5,0.5]),'ub':np.array([2.0,2.0,8.0,5.0,5.0,5.0]),
        'optimizer_set':{
            10:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            20:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            30:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            None:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                  'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                  'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)}}},
    'full10':{'D_true':200,'Ng_true':6,'D_inv':100,'Ng_inv':6,'num_obs':36,
        'inc_angle':np.deg2rad([0,90,180,270]),'coeff_true':np.array([1.0,1.0,4,0.25,1.3,1.3]),
        'lb':np.array([0.5,0.5,1.0,0.1,0.5,0.5]),'ub':np.array([2.0,2.0,8.0,5.0,5.0,5.0]),
        'optimizer_set':{
            10:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            20:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            30:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)},
            None:{'pso':lambda obj:my_pso(obj,num_par,dim,0.8,2,1,1000,lb,ub,patience,tol,x0=None),
                  'pso2':lambda obj:my_pso(obj,num_par,dim,1,0.5,0.5,1000,lb,ub,patience,tol,x0=None),
                  'de':lambda obj:DE(obj,num_par,dim,0.5,0.8,1000,lb,ub,patience,tol),
                  'lshade':lambda obj:L_SHADE(obj,dim,lb,ub,1000,num_par,20,6,patience,tol)}}}}
c=configs[fp]
D_true,Ng_true,D_inv,Ng_inv=c['D_true'],c['Ng_true'],c['D_inv'],c['Ng_inv']
num_obs=c['num_obs']
inc_angle=c['inc_angle']
coeff_true=c['coeff_true']
lb=c['lb']
ub=c['ub']
dim=len(coeff_true)
E0,freq,mu,epsilon=1.0,1e9,const.mu_0,const.epsilon_0
vp,lambda0,k,omega=wave_parameter(freq,mu,epsilon)
obs_radius=10.0*lambda0
optimizer_set=c['optimizer_set']
#==============================================================================
obs_phi,obs_rho,obs_x,obs_y=observation_point(num_obs,obs_radius)
shape_true_func,shape_true_derivative=Gielis2shape(coeff_true)
phi_t, phi_l_t, phi_u_t=boundary_section(D_true)
phi_inv, phi_l_inv, phi_u_inv=boundary_section(D_inv)
Es_clean=multiview_field_Gielis(coeff_true,D_true,Ng_true,phi_t,phi_l_t,phi_u_t,obs_phi,
     obs_rho,omega,mu,k,E0,inc_angle)
noise_power=0 if snr_dB is None else calculate_noise_power(Es_clean,snr_dB)
optimizer=optimizer_set[snr_dB]
def run_single_seed_old(seed_noise_now,seed_alg_now):
    np.random.seed(seed_noise_now)
    noise = complex_Gaussian_noise(noise_power, Es_clean)
    Es_meas = Es_clean + noise
    Es_meas_norm = square_sum_abs(Es_meas)
    obj = lambda X: obj_candidate_Gielis(X, Es_meas, Es_meas_norm, D_inv, Ng_inv,
           phi_inv, phi_l_inv, phi_u_inv, obs_phi, obs_rho, omega, mu, k, E0, inc_angle)
    np.random.seed(seed_alg_now)
    print(f"== {seed_noise_now} ==== Running {alg.upper()} ===================")
    coeff_opt, cost_opt, history_cost, history_coeff, time_elapse, nfe, actual_iter = optimizer[alg](obj)
    return {"seed_noise_now": seed_noise_now,"coeff_opt": coeff_opt,"time_elapse": time_elapse,
        "history_cost": history_cost,"history_coeff": history_coeff,"nfe": nfe,"actual_iter": actual_iter}
def run_single_seed(seed_noise_now,seed_alg_now):
    np.random.seed(seed_noise_now)
    noise=complex_Gaussian_noise(noise_power,Es_clean)
    Es_meas=Es_clean+noise
    Es_meas_norm=square_sum_abs(Es_meas)
    def obj(X):
        try:
            return obj_candidate_Gielis(X,Es_meas,Es_meas_norm,D_inv,Ng_inv,
                                        phi_inv,phi_l_inv,phi_u_inv,obs_phi,obs_rho,omega,mu,k,E0,inc_angle)
        except np.linalg.LinAlgError:
            return np.full(X.shape[0],1e30)
    np.random.seed(seed_alg_now)
    print(f"== {seed_noise_now} ==== Running {alg.upper()} ===================")
    coeff_opt,cost_opt,history_cost,history_coeff,time_elapse,nfe,actual_iter=optimizer[alg](obj)
    return {"seed_noise_now":seed_noise_now,"coeff_opt":coeff_opt,"time_elapse":time_elapse,"history_cost":history_cost,"history_coeff":history_coeff,"nfe":nfe,"actual_iter":actual_iter}
coeff_all,time_all,history_cost_all,history_coeff_all,seed_noise_run,nfe_all,actual_iter_all=[],[],[],[],[],[],[]
for seed_idx in range(len(seed_noise)):
    r=run_single_seed(seed_noise[seed_idx],seed_alg[seed_idx])
    coeff_all.append(r["coeff_opt"])
    time_all.append(r["time_elapse"])
    history_cost_all.append(r["history_cost"])
    history_coeff_all.append(r["history_coeff"])
    seed_noise_run.append(r["seed_noise_now"])
    nfe_all.append(r["nfe"])
    actual_iter_all.append(r["actual_iter"])
    mat_data={"coeff_true":coeff_true,"coeff_all":np.asarray(coeff_all),
              "history_cost_all":np.asarray(history_cost_all),
              "history_coeff_all":np.asarray(history_coeff_all),
              "time_all":np.asarray(time_all),"nfe":np.asarray(nfe_all),
              "actual_iter":np.asarray(actual_iter_all),
              "seed_noise_run":np.asarray(seed_noise_run)}
    mat_path=outdir/f"{fp}_{alg}_{snr_dB}_par{num_par}_{timestamp}_{seed_noise[seed_idx]}.mat"
    savemat(mat_path,mat_data,do_compression=True,oned_as="row")
print(np.mean(coeff_all, axis=0))
elapsed = time.perf_counter() - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Total script execution time: {int(minutes)} min {seconds:.2f} sec")
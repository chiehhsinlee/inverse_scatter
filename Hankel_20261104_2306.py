import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#============20250917 21:27 modify=============================================
def Hankel_singular(ym):
    N=len(ym)        
    P=(N+1)//2
    L=N-P+1
    idx=np.arange(P)+np.arange(L)[:,np.newaxis]
    H=np.zeros((L,P),dtype=np.complex128)
    H[:,:]=ym[idx]
    U,singular,Vh=np.linalg.svd(H)
    singular_sort=np.sort(singular)[::-1] 
    return U,Vh,singular_sort    
#============20250917 21:27 modify=============================================
def Hankel_recover(ym,D,singular_sort,U,Vh):
    N=len(ym)        
    P=(N+1)//2
    L=N-P+1
    singular_r=np.zeros(len(singular_sort))  #replace
    singular_r[0:D]=singular_sort[0:D]
    singular_r_diag=np.concatenate((np.diag(singular_r),np.zeros((L-P,P))))
    Hr=U@singular_r_diag@Vh
    yr_list=[]
    if(L==P):
        i_arr=np.arange(0,L+P-1)  #square N odd
    else:
        i_arr=np.arange(-1,L+P-2) #rectangular N even
    for i in i_arr:   
        anti_diagonal=np.diagonal(Hr[::-1],offset=i-P+1)
        yr_list.append(np.average(anti_diagonal))      
    yr=np.array(yr_list,dtype=np.complex128)
    return yr
def plot_real(x_axis,ym,yc,yr,x_label,y_label,plot_title):
    plt.plot(x_axis,np.real(ym),'--',label='noisy')
    plt.plot(x_axis,np.real(yc),label='clean')
    plt.plot(x_axis,np.real(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig('Hankel_signal_comparison.pdf', bbox_inches='tight')
    plt.show()
def plot_re_im(x_axis,ym,yc,yr,x_label,plot_title):
    '''
    plt.subplot(2,1,1)
    plt.plot(x_axis,np.real(ym),'--',label='noisy')
    plt.plot(x_axis,np.real(yc),label='clean')
    plt.plot(x_axis,np.real(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Real Part')
    plt.legend()
    plt.grid(True)
    plt.title('Real and Imag Parts: '+plot_title)
    plt.subplot(2,1,2)
    plt.plot(x_axis,np.imag(ym),'--',label='noisy')
    plt.plot(x_axis,np.imag(yc),label='clean')
    plt.plot(x_axis,np.imag(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.grid(True)   
    plt.show()    
    '''
    plt.plot(x_axis,np.real(ym),'--',label='noisy')
    plt.plot(x_axis,np.real(yc),label='clean')
    plt.plot(x_axis,np.real(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Real Part')
    plt.legend()
    plt.grid(True)
    plt.title('Real Part: '+plot_title)
    plt.show() 
    
    plt.plot(x_axis,np.imag(ym),'--',label='noisy')
    plt.plot(x_axis,np.imag(yc),label='clean')
    plt.plot(x_axis,np.imag(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.grid(True)   
    plt.title('Imaginary Part: '+plot_title)
    plt.show()       
    
def plot_mag_phase(x_axis,ym,yc,yr,x_label,plot_title):
    plt.plot(x_axis,np.abs(ym),'--',label='noisy')
    plt.plot(x_axis,np.abs(yc),label='clean')
    plt.plot(x_axis,np.abs(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.title('Magnitude: '+plot_title)
    plt.show()
    plt.plot(x_axis,np.angle(ym),'--',label='noisy')
    plt.plot(x_axis,np.angle(yc),label='clean')
    plt.plot(x_axis,np.angle(yr),label='recovered')
    plt.xlabel(x_label)
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.grid(True)  
    plt.title('Phase: '+plot_title)
    plt.show()
def plot_singular(singular_sort,plot_title):
    plt.plot(singular_sort,'o--')
    plt.xlabel('index of singular values')
    plt.ylabel('singular values')
    plt.grid(True)
    plt.title('Singular Values: '+plot_title)
    plt.savefig('Hankel_singular_values.pdf', bbox_inches='tight')
    plt.show()
def singular_rmse_old(ym,singular_sort,yc,U,Vh,plot_fig):      
    N=len(ym)        
    M=N//2
    D_arr=np.arange(1,M) #1 to M-1
    rmse_h=np.zeros(len(D_arr))
    time_arr=np.arange(N)
    for idx_d in range(len(D_arr)):
        D_now=D_arr[idx_d]
        x_test=Hankel_recover(ym,D_now,singular_sort,U,Vh)
        rmse_h[idx_d]=np.sqrt(np.mean(np.abs(x_test-yc)**2))  
        if(plot_fig==True):
            plt.subplot(2,1,1)
            plt.plot(time_arr,np.abs(ym),'--',label='noisy')
            plt.plot(time_arr,np.abs(yc),'--',label='clean')
            plt.plot(time_arr,np.abs(x_test),'--',label='recovered')
            plt.xlabel('time')
            plt.ylabel('signal')
            plt.title(str(D_now)+' Magnitude')
            plt.legend()
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.plot(time_arr,np.angle(ym),'--',label='noisy')
            plt.plot(time_arr,np.angle(yc),'--',label='clean')
            plt.plot(time_arr,np.angle(x_test),'--',label='recovered')
            #plt.xlim(0,50)
            plt.xlabel('time')
            plt.ylabel('signal')
            plt.title(str(D_now)+' Phase')
            plt.legend()
            plt.grid(True)   
            plt.show()      
    D_min_rmse_idx=np.argmin(rmse_h)
    return D_arr[D_min_rmse_idx],rmse_h
def plot_fft(x,len_t,fs,x_lim,plot_title):
    x_fft=np.fft.fft(x,axis=0)
    f_arr=np.arange(-len_t/2,len_t/2)/len_t*fs
    plt.stem(f_arr,np.fft.fftshift(np.abs(x_fft)))
    plt.xlim(x_lim)
    plt.title('FFT: '+plot_title)
    plt.show()

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
#=========================Case 1===============================================
'''
len_t,fs=99,10
t=np.arange(len_t)/fs
yc=np.cos(2*np.pi*1*t)
#np.random.seed(0)
ym=yc+np.random.normal(0,1,t.shape)
U,Vh,singular_sort=Hankel_singular(ym)
yr=Hankel_recover(ym,2,singular_sort,U,Vh)
plot_real(t,ym,yc,yr,'time (s)','signal','$\cos(2\pi f t)$ for $f=1$ Hz')

plot_singular(singular_sort,'$\cos(2\pi f t)$ for $f=1$ Hz')
plot_fft(yr,len_t,fs,[-2,2],'Recovered $\cos(2\pi f t)$ for $f=1$ Hz')

#==============================================================================
len_t,fs=100,10
t=np.arange(len_t)/fs
yc=np.exp(1j*2*np.pi*1*t)
np.random.seed(0)
r=np.random.rayleigh(scale=1/np.sqrt(2),size=t.shape)     
phi=np.random.uniform(0.0,2*np.pi,size=t.shape)            
noise=r*np.exp(1j*phi)
ym=yc+noise
U,Vh,singular_sort=Hankel_singular(ym)
yr=Hankel_recover(ym,2,singular_sort,U,Vh)
plot_re_im(t,ym,yc,yr,'time (s)','$e^{j 2\pi f t}$ for $f=1$ Hz')
plot_mag_phase(t,ym,yc,yr,'time (s)','$e^{j 2\pi f t}$ for $f=1$ Hz')
plot_singular(singular_sort,'$e^{j 2\pi f t}$ for $f=1$ Hz')
plot_fft(yr,len_t,fs,[-2,2],'Recovered $e^{j 2\pi f t}$ for $f=1$ Hz')
'''
#=========================Case 2===============================================
len_t,fs=100,1000
t=np.arange(len_t)/fs
yc1=0.7*np.cos(2*np.pi*50*t) 
yc2=np.cos(2*np.pi*120*t)
yc=yc1+yc2
ym=yc+np.random.normal(0,1,t.shape)
U,Vh,singular_sort=Hankel_singular(ym)
D_opt=singular_rmse_old(ym,singular_sort,yc,U,Vh,False)
yr=Hankel_recover(ym,4,singular_sort,U,Vh)
'''
plt.plot(t,yc1)
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('$0.7\cos(2\pi f_1 t)$ for $f_1=50$ Hz')
plt.grid(True)
plt.show()

plt.plot(t,yc2)
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('$\cos(2\pi f_2 t)$ for $f_2=120$ Hz')
plt.grid(True)
plt.show()
'''
plot_real(t,ym,yc,yr,'time (s)','signal','$0.7\cos(2\pi f_1 t)+\cos(2\pi f_2 t)$ for $f_1=50$, $f_2=120$ Hz')
plot_singular(singular_sort,'$0.7\cos(2\pi f_1 t)+\cos(2\pi f_2 t)$ for $f_1=50$, $f_2=120$ Hz')

#plot_fft(ym,len_t,fs,[-130,130],'Noisy $0.7\cos(2\pi f_1 t)+\cos(2\pi f_2 t)$ for $f_1=50$, $f_2=120$ Hz')
#plot_fft(yr,len_t,fs,[-130,130],'Recovered $0.7\cos(2\pi f_1 t)+\cos(2\pi f_2 t)$ for $f_1=50$, $f_2=120$ Hz')

'''
#==============================================================================
len_t,fs=100,1000
t=np.arange(len_t)/fs
yc1=0.7*np.exp(1j*2*np.pi*50*t) 
yc2=np.exp(1j*2*np.pi*120*t)
yc=yc1+yc2
np.random.seed(0)
r=np.random.rayleigh(scale=1/np.sqrt(2),size=t.shape)     
phi=np.random.uniform(0.0,2*np.pi,size=t.shape)            
noise=r*np.exp(1j*phi)
ym=yc+noise
U,Vh,singular_sort=Hankel_singular(ym)
D_opt=singular_rmse_old(ym,singular_sort,yc,U,Vh,False)
yr=Hankel_recover(ym,2,singular_sort,U,Vh)
plot_singular(singular_sort,'$0.7e^{j2\pi f_1 t}+e^{j 2\pi f_2 t}$ for $f_1=50$, $f_2=120$ Hz')
plot_re_im(t,ym,yc,yr,'time (s)','$0.7e^{j2\pi f_1 t}+e^{j 2\pi f_2 t}$ for $f_1=50$, $f_2=120$ Hz')
plot_mag_phase(t,ym,yc,yr,'time (s)','$0.7e^{j2\pi f_1 t}+e^{j 2\pi f_2 t}$ for $f_1=50$, $f_2=120$ Hz')
#plot_fft(ym,len_t,fs,[-130,130],'Noisy $0.7e^{j2\pi f_1 t}+e^{j 2\pi f_2 t}$ for $f_1=50$, $f_2=120$ Hz')
#plot_fft(yr,len_t,fs,[-130,130],'Recovered $0.7e^{j2\pi f_1 t}+e^{j 2\pi f_2 t}$ for $f_1=50$, $f_2=120$ Hz')
'''



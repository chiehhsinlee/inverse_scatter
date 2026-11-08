import numpy as np
from scipy.io import loadmat
import time, os
import matplotlib.pyplot as plt
from pathlib import Path
from scatter_utils_new import (analysis_alg_Fourier, analysis_alg_adaptive_item,
                               analysis_alg_adaptive_item2, plot_style)

plot_style()
timestamp = time.strftime("%Y%m%d_%H%M")
t_start = time.perf_counter()

all_file_maps={
'full6_par100':{
    '10dB':{'PSO':'full6_pso_10_20260828_0702_1.mat','DE':'full6_de_10_20260828_1104_1.mat','L-SHADE':'full6_lshade_10_20260908_0451_2.mat'},
    '20dB':{'PSO':'full6_pso_20_20260828_0654_1.mat','DE':'full6_de_20_20260828_1015_1.mat','L-SHADE':'full6_lshade_20_20260830_0048_4.mat'},
    '30dB':{'PSO':'full6_pso_30_20260828_0650_1.mat','DE':'full6_de_30_20260828_1353_1.mat','L-SHADE':'full6_lshade_30_20260830_0222_4.mat'},
    'Noiseless':{'PSO':'full6_pso_None_20260828_0642_0.mat','DE':'full6_de_None_20260828_0959_0.mat','L-SHADE':'full6_lshade_None_20260906_1638_9.mat'}},
'full6_par30':{
    '10dB':{'PSO':'full6_pso_10_20260908_0719_9.mat','DE':'full6_de_10_20260908_0736_9.mat','L-SHADE':'full6_lshade_10_par30_20260908_0805_9.mat'},
    '20dB':{'PSO':'full6_pso_20_par30_20260908_1049_9.mat','DE':'full6_de_20_par30_20260908_1027_9.mat','L-SHADE':'full6_lshade_20_20260908_1511_9.mat'},
    '30dB':{'PSO':'full6_pso_30_par30_20260909_1043_8.mat','DE':'full6_de_30_par30_20260908_1052_9.mat','L-SHADE':'full6_lshade_30_20260907_1428_9.mat'},
    'Noiseless':{'PSO':'full6_pso_None_20260906_0624_9.mat','DE':'full6_de_None_20260906_0617_9.mat','L-SHADE':'full6_lshade_None_20260906_1638_9.mat'}},
'full5_par30':{
    '10dB':{'PSO':'full5_pso_10_par30_20260919_0637_49.mat','DE':'full5_de_10_par30_20260920_1815_49.mat','L-SHADE':'full5_lshade_10_par30_20260920_1820_49.mat'},
    '20dB':{'PSO':'full5_pso_20_par30_20260919_0638_49.mat','DE':'full5_de_20_par30_20260920_1816_49.mat','L-SHADE':'full5_lshade_20_par30_20260920_1821_49.mat'},
    '30dB':{'PSO':'full5_pso_30_par30_20260924_0054_49.mat','DE':'full5_de_30_par30_20260920_1816_49.mat','L-SHADE':'full5_lshade_30_par30_20260919_1420_49.mat'},
    'Noiseless':{'PSO':'full5_pso_None_par30_20260922_0314_49.mat','DE':'full5_de_None_par30_20260920_1825_49.mat','L-SHADE':'full5_lshade_None_par30_20260920_2151_49.mat'}},
'full5_par50':{
    '10dB':{'PSO':'full5_pso_10_par50_20260920_2233_49.mat','DE':'full5_de_10_par50_20260920_1815_49.mat','L-SHADE':'full5_lshade_10_par50_20260919_0600_49.mat'},
    '20dB':{'PSO':'full5_pso_20_par50_20260920_2333_49.mat','DE':'full5_de_20_par50_20260918_1901_49.mat','L-SHADE':'full5_lshade_20_par50_20260919_0600_49.mat'},
    '30dB':{'PSO':'full5_pso_30_par50_20260920_2333_49.mat','DE':'full5_de_30_par50_20260919_0442_49.mat','L-SHADE':'full5_lshade_30_par50_20260919_0559_49.mat'},
    'Noiseless':{'PSO':'full5_pso_None_par50_20260920_2351_49.mat','DE':'full5_de_None_par50_20260920_1827_49.mat','L-SHADE':'full5_lshade_None_par50_20260920_2155_49.mat'}},
'full5_par80':{
    '10dB':{'PSO':'full5_pso_10_par80_20260920_2233_49.mat','DE':'full5_de_10_par80_20260917_0859_49.mat','L-SHADE':'full5_lshade_10_par80_20260926_0038_49.mat'},
    '20dB':{'PSO':'full5_pso_20_par80_20260927_0034_49.mat','DE':'full5_de_20_par80_20260917_1228_49.mat','L-SHADE':'full5_lshade_20_par80_20260920_1821_49.mat'},
    '30dB':{'PSO':'full5_pso_30_par80_20260926_0038_49.mat','DE':'full5_de_30_par80_20260925_0137_49.mat','L-SHADE':'full5_lshade_30_par80_20260920_2158_49.mat'},
    'Noiseless':{'PSO':'full5_pso_None_par80_20260921_0007_49.mat','DE':'full5_de_None_par80_20260920_1819_49.mat','L-SHADE':'full5_lshade_None_par80_20260920_2157_49.mat'}},
'full5_par100':{
    '10dB':{'PSO':'full5_pso_10_par100_20260920_2332_49.mat','DE':'full5_de_10_par100_20260916_1749_49.mat','L-SHADE':'full5_lshade_10_par100_20260920_1821_49.mat'},
    '20dB':{'PSO':'full5_pso_20_par100_20260917_0743_49.mat','DE':'full5_de_20_par100_20260917_0751_49.mat','L-SHADE':'full5_lshade_20_par100_20260917_0855_49.mat'},
    '30dB':{'PSO':'full5_pso_30_par100_20260917_0748_49.mat','DE':'full5_de_30_par100_20260917_0802_49.mat','L-SHADE':'full5_lshade_30_par100_20260917_0852_49.mat'},
    'Noiseless':{'PSO':'full5_pso_None_par100_20260920_2352_49.mat','DE':'full5_de_None_par100_20260920_1819_49.mat','L-SHADE':'full5_lshade_None_par100_20260920_2157_49.mat'}},
'full8_par30':{
    '10dB':{'PSO':'full8_pso_10_par30_20260913_1640_49.mat','DE':'full8_de_10_par30_20260914_1605_49.mat','L-SHADE':'full8_lshade_10_par30_20260913_1646_49.mat'},
    '20dB':{'PSO':'full8_pso_20_par30_20260913_1550_49.mat','DE':'full8_de_20_par30_20260914_1610_49.mat','L-SHADE':'full8_lshade_20_par30_20260913_1451_49.mat'},
    '30dB':{'PSO':'full8_pso_30_par30_20260913_1552_49.mat','DE':'full8_de_30_par30_20260913_1651_49.mat','L-SHADE':'full8_lshade_30_par30_20260913_1301_49.mat'},
    'Noiseless':{'PSO':'full8_pso_None_par30_20260919_1451_49.mat','DE':'full8_de_None_par30_20260913_1241_49.mat','L-SHADE':'full8_lshade_None_par30_20260913_1428_49.mat'}},
'full8_par50':{
    '10dB':{'PSO':'full8_pso_10_par50_20260916_2013_49.mat','DE':'full8_de_10_par50_20260915_1308_49.mat','L-SHADE':'full8_lshade_10_par50_20260915_1409_49.mat'},
    '20dB':{'PSO':'full8_pso_20_par50_20260914_1728_49.mat','DE':'full8_de_20_par50_20260915_1314_49.mat','L-SHADE':'full8_lshade_20_par50_20260914_1712_49.mat'},
    '30dB':{'PSO':'full8_pso_30_par50_20260914_1730_49.mat','DE':'full8_de_30_par50_20260914_1612_49.mat','L-SHADE':'full8_lshade_30_par50_20260914_1718_49.mat'},
    'Noiseless':{'PSO':'full8_pso_None_par50_20260917_2035_49.mat','DE':'full8_de_None_par50_20260915_1326_49.mat','L-SHADE':'full8_lshade_None_par50_20260914_1723_49.mat'}},
'full8_par80':{
    '10dB':{'PSO':'full8_pso_10_par80_20260913_1504_49.mat','DE':'full8_de_10_par80_20260913_0136_49.mat','L-SHADE':'full8_lshade_10_par80_20260912_2337_49.mat'},
    '20dB':{'PSO':'full8_pso_20_par80_20260913_1517_49.mat','DE':'full8_de_20_par80_20260913_0236_49.mat','L-SHADE':'full8_lshade_20_par80_20260914_1716_49.mat'},
    '30dB':{'PSO':'full8_pso_30_par80_20260913_0247_49.mat','DE':'full8_de_30_par80_20260913_1235_49.mat','L-SHADE':'full8_lshade_30_par80_20260913_1727_49.mat'},
    'Noiseless':{'PSO':'full8_pso_None_par80_20260913_1522_49.mat','DE':'full8_de_None_par80_20260915_1405_49.mat','L-SHADE':'full8_lshade_None_par80_20260913_1512_49.mat'}},
'full8_par100':{
    '10dB':{'PSO':'full8_pso_10_par100_20260915_1520_49.mat','DE':'full8_de_10_par100_20260915_1817_49.mat','L-SHADE':'full8_lshade_10_par100_20260915_1504_49.mat'},
    '20dB':{'PSO':'full8_pso_20_par100_20260916_2018_49.mat','DE':'full8_de_20_par100_20260915_1649_49.mat','L-SHADE':'full8_lshade_20_par100_20260915_1652_49.mat'},
    '30dB':{'PSO':'full8_pso_30_par100_20260917_1635_49.mat','DE':'full8_de_30_par100_20260916_1516_49.mat','L-SHADE':'full8_lshade_30_par100_20260916_1952_49.mat'},
    'Noiseless':{'PSO':'full8_pso_None_par100_20260919_1455_49.mat','DE':'full8_de_None_par100_20260917_0152_49.mat','L-SHADE':'full8_lshade_None_par100_20260915_1706_49.mat'}},
'full10_par30':{
    '10dB':{'PSO':'full10_pso_10_par30_20260921_1555_49.mat','DE':'full10_de_10_par30_20260920_0811_49.mat','L-SHADE':'full10_lshade_10_par30_20260920_0718_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par30_20260920_1103_49.mat','DE':'full10_de_20_par30_20260920_0809_49.mat','L-SHADE':'full10_lshade_20_par30_20260920_0731_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par30_20260920_1343_49.mat','DE':'full10_de_30_par30_20260920_0811_49.mat','L-SHADE':'full10_lshade_30_par30_20260920_0650_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par30_20260920_1745_49.mat','DE':'full10_de_None_par30_20260919_1512_49.mat','L-SHADE':'full10_lshade_None_par30_20260919_1626_49.mat'}},
'full10_par50':{
    '10dB':{'PSO':'full10_pso_10_par50_20260922_0347_49.mat','DE':'full10_de_10_par50_20260921_1805_49.mat','L-SHADE':'full10_lshade_10_par50_20260922_0554_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par50_20260921_1811_49.mat','DE':'full10_de_20_par50_20260922_0157_49.mat','L-SHADE':'full10_lshade_20_par50_20260921_1452_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par50_20260922_0157_49.mat','DE':'full10_de_30_par50_20260921_1823_49.mat','L-SHADE':'full10_lshade_30_par50_20260922_1201_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par50_20260920_1749_49.mat','DE':'full10_de_None_par50_20260919_1513_49.mat','L-SHADE':'full10_lshade_None_par50_20260919_1629_49.mat'}},
'full10_par80':{
    '10dB':{'PSO':'full10_pso_10_par80_20260920_1743_49.mat','DE':'full10_de_10_par80_20260922_2245_49.mat','L-SHADE':'full10_lshade_10_par80_20260923_1540_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par80_20260922_1201_49.mat','DE':'full10_de_20_par80_20260922_0327_49.mat','L-SHADE':'full10_lshade_20_par80_20260922_2245_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par80_20260929_2120_49.mat','DE':'full10_de_30_par80_20260923_1540_49.mat','L-SHADE':'full10_lshade_30_par80_20260922_2245_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par80_20260929_1751_49.mat','DE':'full10_de_None_par80_20260926_0038_49.mat','L-SHADE':'full10_lshade_None_par80_20260929_1146_49.mat'}},
'full10_par100':{
    '10dB':{'PSO':'full10_pso_10_par100_20260923_1541_49.mat','DE':'full10_de_10_par100_20260923_1540_49.mat','L-SHADE':'full10_lshade_10_par100_20260922_2245_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par100_20260923_1622_49.mat','DE':'full10_de_20_par100_20260929_1603_49.mat','L-SHADE':'full10_lshade_20_par100_20260921_1459_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par100_20260925_0137_49.mat','DE':'full10_de_30_par100_20260927_0125_49.mat','L-SHADE':'full10_lshade_30_par100_20260922_0550_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par100_20260920_1751_49.mat','DE':'full10_de_None_par100_20260919_1524_49.mat','L-SHADE':'full10_lshade_None_par100_20260920_1549_49.mat'}},
'full10_par120':{
    '10dB':{'PSO':'full10_pso_10_par120_20260925_0137_49.mat','DE':'full10_de_10_par120_20260925_0137_49.mat','L-SHADE':'full10_lshade_10_par120_20260929_0215_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par120_20260925_0137_49.mat','DE':'full10_de_20_par120_20260925_0137_49.mat','L-SHADE':'full10_lshade_20_par120_20260929_0215_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par120_20260925_0148_49.mat','DE':'full10_de_30_par120_20260925_0137_49.mat','L-SHADE':'full10_lshade_30_par120_20260929_0215_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par120_20260929_1301_49.mat','DE':'full10_de_None_par120_20260927_0127_49.mat','L-SHADE':'full10_lshade_None_par120_20260929_1301_49.mat'}},
'full10_par150':{
    '10dB':{'PSO':'full10_pso_10_par150_20260926_0038_49.mat','DE':'full10_de_10_par150_20260929_0214_49.mat','L-SHADE':'full10_lshade_10_par150_20260926_0038_49.mat'},
    '20dB':{'PSO':'full10_pso_20_par150_20260927_0127_49.mat','DE':'full10_de_20_par150_20260929_0236_49.mat','L-SHADE':'full10_lshade_20_par150_20260926_0038_49.mat'},
    '30dB':{'PSO':'full10_pso_30_par150_20260929_1745_49.mat','DE':'full10_de_30_par150_20260927_0127_49.mat','L-SHADE':'full10_lshade_30_par150_20260925_0137_49.mat'},
    'Noiseless':{'PSO':'full10_pso_None_par150_20260929_1720_49.mat','DE':'full10_de_None_par150_20260929_0248_49.mat','L-SHADE':'full10_lshade_None_par150_20260920_1718_49.mat'}}}

case_name = "full5"   
case_populations = {"full5":[30, 50, 80, 100], "full8":[30, 50, 80, 100], "full10":[30, 50, 80, 100, 120, 150]}
populations = case_populations[case_name]
case_maps = {pop: all_file_maps[f"{case_name}_par{pop}"] for pop in populations if f"{case_name}_par{pop}" in all_file_maps}

out_dir = Path(f"fig_{case_name}_pop_snr_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

conditions = ["10dB", "20dB", "30dB", "Noiseless"]
algorithms = ["PSO", "DE", "L-SHADE"]
theta = np.linspace(0, 2*np.pi, 1000)

algo_styles = {
    "PSO": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
    "DE": {"color": "tab:orange", "linestyle": "--", "marker": "s"},
    "L-SHADE": {"color": "tab:green", "linestyle": "-.", "marker": "^"}
}


def Gielis2shape_local(coeff):
    a, b, m, n1, n2, n3 = np.asarray(coeff, dtype=float).squeeze()
    def shape(theta):
        phi = m * theta / 4.0
        return (np.abs(np.cos(phi)/a)**n2 + np.abs(np.sin(phi)/b)**n3)**(-1.0/n1)
    return shape

def radius2xy(theta, r):
    return r * np.cos(theta), r * np.sin(theta)

def load_full_data_from_mat(filepath):
    mat = loadmat(filepath)
    coeff_true = np.asarray(mat["coeff_true"], dtype=float).squeeze()
    coeff_all = np.asarray(mat["coeff_all"], dtype=float)
    if coeff_all.ndim == 1: coeff_all = coeff_all[None, :]

    r_true = Gielis2shape_local(coeff_true)(theta)
    nrmse_all = []
    for coeff in coeff_all:
        r_rec = Gielis2shape_local(coeff)(theta)
        nrmse_all.append(np.linalg.norm(r_rec-r_true)/np.linalg.norm(r_true)*100)
    nrmse_all = np.asarray(nrmse_all)

    median_nrmse = np.nanmedian(nrmse_all)
    median_idx = np.nanargmin(np.abs(nrmse_all-median_nrmse))
    coeff_median = coeff_all[median_idx]

    if "nfe" in mat:
        nfe_all = np.asarray(mat["nfe"], dtype=float).squeeze()
        if nfe_all.ndim == 0: nfe_all = np.array([float(nfe_all)])
        else: nfe_all = nfe_all.ravel()
    else:
        nfe_all = np.full(len(nrmse_all), np.nan)
    nfe_median = nfe_all[median_idx] if median_idx < len(nfe_all) else np.nan

    if "time_all" in mat:
        runtime_all = np.asarray(mat["time_all"], dtype=float).squeeze()
        if runtime_all.ndim == 0: runtime_all = np.array([float(runtime_all)])
        else: runtime_all = runtime_all.ravel()
    else:
        runtime_all = np.full(len(nrmse_all), np.nan)
    runtime_mean = np.nanmean(runtime_all)

    history_cost_all = np.asarray(mat.get("history_cost_all", []))
    if history_cost_all.size > 0:
        if history_cost_all.ndim == 1: history_cost_all = history_cost_all[None, :]
        history_median = np.asarray(history_cost_all[median_idx], dtype=float).squeeze()
        valid_history = np.isfinite(history_median) & (history_median > 0)
        history_median = history_median[valid_history]
    else:
        history_median = np.array([])
        
    return {"coeff_true": coeff_true, "coeff_median": coeff_median, "nrmse_all": nrmse_all, 
            "nrmse_median": nrmse_all[median_idx], "nfe_all": nfe_all, "nfe_median": nfe_median, 
            "history_median": history_median, "runtime_all": runtime_all, "runtime_mean": runtime_mean}

data = {pop: {cond: {} for cond in conditions} for pop in populations}
for pop, file_map in case_maps.items():
    for cond in conditions:
        if cond not in file_map: continue
        for alg in algorithms:
            if alg not in file_map[cond]: continue
            filepath = Path(file_map[cond][alg])
            if not filepath.exists(): continue
            try:
                data[pop][cond][alg] = load_full_data_from_mat(filepath)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
#===================Reconstruction Shape=======================================
fig_legend = plt.figure() 
plt.plot([], [], color="black", linewidth=2.5, label="True Shape")
for alg in algorithms:
    plt.plot([], [], color=algo_styles[alg]["color"], 
             linestyle=algo_styles[alg]["linestyle"], linewidth=1.5, label=alg)
fig_legend.legend(loc='center', ncol=4, frameon=False, fontsize=12)
plt.gca().set_axis_off() 
plt.savefig(out_dir / f"{case_name}_global_legend_{timestamp}.pdf", bbox_inches="tight")
plt.close(fig_legend)

for pop in populations:
    for cond in conditions:
        if not data[pop][cond]: continue
        plt.figure()
        first_alg = next(iter(data[pop][cond]))
        coeff_true = data[pop][cond][first_alg]["coeff_true"]
        r_true = Gielis2shape_local(coeff_true)(theta)
        x_true, y_true = radius2xy(theta, r_true)
        
        plt.plot(x_true, y_true, color="black", linewidth=2.5, label="True Shape")
        
        for alg in algorithms:
            if alg not in data[pop][cond]: continue
            coeff = data[pop][cond][alg]["coeff_median"]
            r_rec = Gielis2shape_local(coeff)(theta)
            x_rec, y_rec = radius2xy(theta, r_rec)
            plt.plot(x_rec, y_rec, color=algo_styles[alg]["color"], 
                     linestyle=algo_styles[alg]["linestyle"], linewidth=1.5, label=alg)
        plt.xlim(-1.6,1.6)
        plt.ylim(-1.6,1.6)
        ticks = np.arange(-1.6, 1.61, 0.8)
        plt.xticks(ticks, fontsize=12)
        plt.yticks(ticks, fontsize=12)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x (m)", fontsize=12)
        plt.ylabel("y (m)", fontsize=12)
        plt.grid(True, linestyle=":")
        #plt.legend() 
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_shape_p{pop}_{cond}_{timestamp}.pdf", bbox_inches="tight")
        plt.show() 
        plt.close()   

for cond in conditions:
    plt.figure()
    has_data = False
    for alg in algorithms:
        pop_available = [pop for pop in populations if alg in data[pop][cond]]
        if not pop_available:
            continue
        has_data = True
        mean_nfe_values = [np.nanmean(data[pop][cond][alg]["nfe_all"]) for pop in pop_available]
        mean_nrmse_values = [np.nanmean(data[pop][cond][alg]["nrmse_all"]) for pop in pop_available]

        plt.plot(mean_nfe_values, mean_nrmse_values,
                 marker=algo_styles[alg]["marker"],
                 linestyle=algo_styles[alg]["linestyle"],
                 color=algo_styles[alg]["color"],
                 linewidth=1.5, markersize=6, label=alg)

    if has_data:
        plt.xlabel(r"Mean NFE ($\times 10^4$)", fontsize=12)
        plt.xlim(0, 70000)

        plt.ylabel("Mean Shape NRMSE (%)", fontsize=12)
        plt.yscale("log")
        plt.ylim(0.5, 50)
        plt.yticks(fontsize=12)

        ticks = np.arange(0, 70001, 10000)
        tick_labels = [str(i) for i in range(8)]
        plt.xticks(ticks, tick_labels, fontsize=12)

        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_nrmse_nfe_mean_{cond}.pdf",
                    bbox_inches="tight")
        plt.show()

    plt.close()
    

for pop in populations:
    for cond in conditions:
        if not data[pop][cond]: continue
        plt.figure()
        has_data = False
        for alg in algorithms:
            if alg not in data[pop][cond]: continue
            history = data[pop][cond][alg]["history_median"]
            if len(history) == 0: continue
            
            has_data = True
            iterations = np.arange(1, len(history) + 1)
            plt.plot(iterations, history, label=alg, color=algo_styles[alg]["color"],
                     linestyle=algo_styles[alg]["linestyle"], linewidth=1.5)            
        if has_data:
            plt.xlabel("Iteration")
            plt.ylabel("Cost Value")
            plt.yscale("log")
            plt.xlim((0, 650))
            plt.ylim((3e-4, 1))
            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{case_name}_converge_p{pop}_{cond}_{timestamp}.pdf", bbox_inches="tight")
            plt.show()  
        plt.close()

for cond in conditions:
    plt.figure()
    x=np.arange(len(populations))
    bar_width=0.8/max(len(algorithms),1)
    has_data=False
    for i,alg in enumerate(algorithms):
        x_alg=[]
        values=[]
        offset=(i-(len(algorithms)-1)/2)*bar_width

        for j,pop in enumerate(populations):
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                x_alg.append(j+offset)
                values.append(data[pop][cond][alg]["runtime_mean"]/60.0)
                has_data=True

        if len(values)>0:
            plt.bar(x_alg,values,bar_width,label=alg,color=algo_styles[alg]["color"],alpha=0.85)

    if has_data:
        plt.xticks(x,[str(p) for p in populations])
        plt.xlabel("Initial Population Size")
        plt.ylabel("Mean Computation Time (min.)")
        plt.ylim(0,40)
        plt.grid(axis="y",linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/f"{case_name}_pop_runtime_{cond}_{timestamp}.pdf",bbox_inches="tight")
        plt.show()
    plt.close()

for cond in conditions:
    plt.figure()
    has_data = False
    
    for alg in algorithms:
        first_point = True
        for pop in populations:
            if pop not in data or cond not in data[pop] or alg not in data[pop][cond]:
                continue
            nfe_all = data[pop][cond][alg]["nfe_all"]
            nrmse_all = data[pop][cond][alg]["nrmse_all"]
            valid = np.isfinite(nfe_all) & np.isfinite(nrmse_all) & (nrmse_all > 0)
            nfe_plot = nfe_all[valid]
            nrmse_plot = nrmse_all[valid]
            
            if len(nfe_plot) == 0: continue
            has_data = True
            plt.scatter(nfe_plot, nrmse_plot, marker=algo_styles[alg]["marker"],
                        color=algo_styles[alg]["color"], s=30, alpha=0.75,
                        label=alg if first_point else None)
            first_point = False
            
    if has_data:
        plt.xlabel("NFE")
        plt.ylabel("Shape NRMSE (%)")
        plt.yscale("log")
        plt.ticklabel_format(axis="x", style="plain", useOffset=False)
        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_nrmse_nfe_all_runs_scatter_{cond}_{timestamp}.pdf", bbox_inches="tight")
        plt.show()  
    plt.close()

global_max_nrmse = 0
for pop in populations:
    for cond in conditions:
        for alg in algorithms:
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                mean_val = np.nanmean(data[pop][cond][alg]["nrmse_all"])
                if mean_val > global_max_nrmse: global_max_nrmse = mean_val
nrmse_max = global_max_nrmse * 1.1 if global_max_nrmse > 0 else 100

for pop in populations:
    plt.figure()
    has_data = False
    for alg in algorithms:
        means = []
        for cond in conditions:
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                means.append(np.nanmean(data[pop][cond][alg]["nrmse_all"]))
            else:
                means.append(np.nan)
        if not np.all(np.isnan(means)):
            plt.plot(conditions, means, marker=algo_styles[alg]["marker"],
                     color=algo_styles[alg]["color"], linestyle=algo_styles[alg]["linestyle"], label=alg)
            has_data = True
            
    if has_data:
        plt.xlabel("Noise Condition")
        plt.ylabel("Mean Shape NRMSE (%)")
        plt.ylim(0, nrmse_max)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_p{pop}_snr_nrmse_{timestamp}.pdf", bbox_inches="tight")
        plt.show()  
    plt.close()

# ============================================================
# [PLOT 7] SNR Median NRMSE
# ============================================================
global_max_median_nrmse = 0
for pop in populations:
    for cond in conditions:
        for alg in algorithms:
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                median_val = np.nanmedian(data[pop][cond][alg]["nrmse_all"])
                if median_val > global_max_median_nrmse: global_max_median_nrmse = median_val
median_nrmse_max = global_max_median_nrmse * 1.1 if global_max_median_nrmse > 0 else 100

for pop in populations:
    plt.figure()
    has_data = False
    for alg in algorithms:
        medians = []
        for cond in conditions:
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                medians.append(np.nanmedian(data[pop][cond][alg]["nrmse_all"]))
            else:
                medians.append(np.nan)
        if not np.all(np.isnan(medians)):
            plt.plot(conditions, medians, marker=algo_styles[alg]["marker"],
                     color=algo_styles[alg]["color"], linestyle=algo_styles[alg]["linestyle"], label=alg)
            has_data = True
            
    if has_data:
        plt.xlabel("Noise Condition")
        plt.ylabel("Median Shape NRMSE (%)")
        plt.ylim(0, median_nrmse_max)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_p{pop}_snr_median_nrmse_{timestamp}.pdf", bbox_inches="tight")
        plt.show()  
    plt.close()

# =====================SNR Success Rate=======================

success_threshold = 10
for pop in populations:
    plt.figure()
    has_data = False
    for alg in algorithms:
        success_rates = []
        for cond in conditions:
            if pop in data and cond in data[pop] and alg in data[pop][cond]:
                nrmse_array = data[pop][cond][alg]["nrmse_all"]
                valid_runs = nrmse_array[np.isfinite(nrmse_array)]
                if len(valid_runs) > 0:
                    sr = np.sum(valid_runs < success_threshold) / len(valid_runs) * 100
                    success_rates.append(sr)
                else:
                    success_rates.append(np.nan)
            else:
                success_rates.append(np.nan)

        if not np.all(np.isnan(success_rates)):
            plt.plot(conditions, success_rates, marker=algo_styles[alg]["marker"],
                     color=algo_styles[alg]["color"], linestyle=algo_styles[alg]["linestyle"], label=alg)
            has_data = True

    if has_data:
        plt.xlabel("SNR (dB)")
        plt.ylabel("Success Rate (%)")
        plt.xticks(conditions, ["10", "20", "30", "Noiseless"])
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{case_name}_p{pop}_snr_success_rate_{timestamp}.pdf", bbox_inches="tight")
        plt.show()
    plt.close()
elapsed = time.perf_counter() - t_start
minutes, seconds = divmod(elapsed, 60)
print(f"Total script execution time: {int(minutes)} min {seconds:.2f} sec")
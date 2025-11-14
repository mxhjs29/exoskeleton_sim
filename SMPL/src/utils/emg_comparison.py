import pandas as pd
import pyemgpipeline as pep
import numpy as np
from matplotlib.figure import SubplotParams

def read_sto(file_path):
    return pd.read_csv(file_path, sep='\t', skiprows=4)

EMG_file_path = "/home/chenshuo/PycharmProjects/LabValidation_withoutVideos/subject2/EMGData/squats1_EMG.sto"
EMG_df = read_sto(EMG_file_path)
activation_file_path = "/SMPL/data/torque_record/muscle_activation.csv"
ACT_df = pd.read_csv(activation_file_path, sep='\t',skiprows=0)
muscle_name = ['soleus_r','gasmed_r','tibant_r','recfem_r',
               'vasmed_r','vaslat140_r','semiten_r','bflh140_r','glmed1_r',
               'soleus_l', 'gasmed_l', 'tibant_l', 'recfem_l',
               'vasmed_l', 'vaslat140_l', 'semiten_l', 'bflh140_l', 'glmed1_l'
               ]

# EMG信号处理
EMG_df.drop(columns=['time'],inplace=True)
random_indices = EMG_df.sample(n=22, random_state=42).index
EMG_df.drop(random_indices,inplace=True)
emg_channel_names = list(EMG_df.columns)
sample_rate = 100
trial_name = 'squat'
EMG_data = EMG_df.to_numpy()
emg_plot_params = pep.plots.EMGPlotParams(
    n_rows=16,
    fig_kwargs={
        'figsize': (18, 16),
        'dpi': 80,
        'subplotpars': SubplotParams(wspace=0, hspace=0.6),
    },
    line2d_kwargs={
        'color': 'red',
    }
)
EMG = pep.wrappers.EMGMeasurement(EMG_data,hz=sample_rate,trial_name=trial_name,
                                channel_names=emg_channel_names,emg_plot_params=emg_plot_params)
EMG.apply_dc_offset_remover()
EMG.apply_full_wave_rectifier()
EMG_max_amplitude = [np.max(EMG.data[:,l]) for l in range(16)]
EMG.apply_amplitude_normalizer(EMG_max_amplitude)
i = 0
for name in emg_channel_names:
    EMG_df[name] = EMG.data[:,i]
    i = i+1
# EMG.plot()

# 仿真激活信号处理
act_channel_names = list(ACT_df.columns)
ACT_data = ACT_df.to_numpy()
act_plot_params = pep.plots.EMGPlotParams(
    n_rows=18,
    fig_kwargs={
        'figsize':(18,18),
        'dpi':80,
        'subplotpars': SubplotParams(wspace=0, hspace=0.6),
    },
    line2d_kwargs={
        'color':'blue',
    }
)
ACT = pep.wrappers.EMGMeasurement(ACT_data,hz=sample_rate,trial_name=trial_name,
                                  channel_names=act_channel_names,emg_plot_params=act_plot_params)
ACT.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=5, bf_cutoff_fq_hi=49)
ACT.apply_full_wave_rectifier()
ACT.apply_linear_envelope(le_order=4,le_cutoff_fq=6)
ACT_max_amplitude = [np.max(ACT.data[:,l]) for l in range(18)]
ACT.apply_amplitude_normalizer(ACT_max_amplitude)
i = 0
for name in act_channel_names:
    ACT_df[name] = ACT.data[:,i]
    i = i+1
# ACT.plot()

# 计算相关性系数
a = ['l','r']
print(emg_channel_names)

corrcoef = {
    "soleus_l":np.corrcoef(EMG_df[f'soleus_l{name}_activation'], ACT_df[f'soleus_l{name}_tendon'])
}
for name in a:
    print(np.corrcoef(EMG_df[f'glmed1_{name}_activation'], ACT_df[f'glmed1_{name}_tendon']))

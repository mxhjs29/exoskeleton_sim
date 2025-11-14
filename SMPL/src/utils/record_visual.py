import mujoco
import mujoco.viewer
import yaml
import time
# model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/mjc/mj_fullbody_with_exo_carrying.xml"
# model = mujoco.MjModel.from_xml_path(model_path)
# data = mujoco.MjData(model)
# mujoco.viewer.launch(model, data)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

while True:
    fig, ax = plt.subplots()
    ax.set_xlabel('Frame',fontsize=12)
    ax.set_ylabel('Torque/N*m',fontsize=12)
    line1, = ax.plot([], [], 'r', label='hip torque')
    line2, = ax.plot([], [], 'b', label='exoskeleton torque')
    legend = ax.legend(loc='upper right')
    x_data, y_data_hip, y_data_exo = [], [], []  # 初始化空的数据列表

    with open("/SMPL/qfrc_actuator_with_exo.yaml", 'r') as file:
        qfrc_without_exo = yaml.safe_load(file)
    x = np.array(qfrc_without_exo['x'])
    y_hip_r = np.array(qfrc_without_exo['qfrc_actuator_r'])
    y_hip_l = np.array(qfrc_without_exo['qfrc_actuator_l'])
    y_exo_l = np.array(qfrc_without_exo['exo_torque_l'])
    y_exo_r = np.array(qfrc_without_exo['exo_torque_r'])


    def init():
        ax.set_xlim(0,800)
        ax.set_ylim(-250,250)
        return line1,line2

    def update(frame):
        x_data.append(frame)
        y_data_hip.append(y_hip_r[frame])
        y_data_exo.append(y_exo_r[frame])
        line1.set_data(x_data,y_data_hip)
        line2.set_data(x_data,y_data_exo)
        return line1,line2

    ani = animation.FuncAnimation(fig, update, frames=x,
                                  init_func=init, blit=False, interval=10,repeat=False)
    plt.show()
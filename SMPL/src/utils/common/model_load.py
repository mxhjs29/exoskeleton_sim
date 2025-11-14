
import sys
import math
from myosuite.physics.sim_scene import SimScene
import os
from os.path import join as pjoin
from SMPL.src.utils.common.quaternion import *
import xml.etree.ElementTree as ET
import tempfile

body_length2 = 0.129 * 2
body_length1 = 0.145 * 2
right_pos = ['01','03','04','05','06','07','08','09', '11','12','13','14','15','16','18','19','22','23']
right_muscle = ['1','3','4','5','6','7','8' '9','11','12','13','14','15','16','18','19','22','23']
mission_id = ['1','2','3','4','5','6','7','8','9','10']

def load_model(model_path, joints, actuators, bodys, sites):
    sim = SimScene.get_sim(model_path)
    joints_id = {name: sim.model.joint(name).id for name in joints}
    actuators_id = {name: sim.model.actuator(name).id for name in actuators}
    # actuators_id = 0
    bodys_id = {name: sim.model.body(name).id for name in bodys}
    sites_id = {name: sim.model.site(name).id for name in sites}
    return sim, joints_id, actuators_id, bodys_id, sites_id

def load_model_gesture(model_path, joints, actuators, bodys):
    sim = SimScene.get_sim(model_path)
    joints_id = {name: sim.model.joint(name).id for name in joints}
    actuators_id = {name: sim.model.actuator(name).id for name in actuators}

    bodys_id = {name: sim.model.body(name).id for name in bodys}

    return sim, joints_id, actuators_id, bodys_id

def data_process(src_dir, index, quat_SMPL_base, base_point, kinetic_chain):
    npy_files = os.listdir(src_dir)
    npy_files = sorted(npy_files)
    data = np.load(pjoin(src_dir, npy_files[index]))

    Spine3 = data[:, kinetic_chain[0], :]
    R_collar = data[:, kinetic_chain[1], :]
    R_shoulder = data[:, kinetic_chain[2], :]
    R_elbow = data[:, kinetic_chain[3], :]
    R_wrist = data[:, kinetic_chain[4], :]
    R_root_inv = qinv_np(Spine3)
    R_root = Spine3
    R_global = qmul_np(R_root, R_collar)

    R_global_shoulder = qmul_np(R_global, R_shoulder)
    R_global_shoulder = qmul_np(R_global_shoulder, R_elbow)
    R_global_wrist = qmul_np(R_global_shoulder, R_wrist)

    R_global_shoulder = qmul_np(R_root_inv, R_global_shoulder)
    R_global_wrist = qmul_np(R_root_inv, R_global_wrist)
    length = len(R_shoulder)

    Quat_base = np.tile(quat_SMPL_base, (length, 1))

    SMPL_position_shoulder = [0, -1 * body_length1, 0]
    SMPL_position_elbow = [0, -1 * body_length2, 0]

    Quat_0 = qmul_np(Quat_base, R_global_shoulder)

    position_elbow = qrot_np(Quat_0, np.tile(SMPL_position_shoulder, (length, 1))) + np.tile(
       base_point, (length, 1))
    Quat_1 = qmul_np(Quat_base, R_global_wrist)
    position_wrist = qrot_np(Quat_1, np.tile(SMPL_position_elbow, (length, 1))) + position_elbow
    return length, Quat_0, Quat_1, position_elbow, position_wrist

def data_process_another_dataset(src_dir, index, quat_SMPL_base, base_point, kinetic_chain, name=None):
    npy_files = os.listdir(src_dir)
    npy_files = sorted(npy_files)

    if(name == None):
        data = np.load(pjoin(src_dir, npy_files[index]))
        file = npy_files[index]
    else:
        data = np.load(pjoin(src_dir, name + '.npy'))
        file = name + '.npy'

    # find id and mission

    start = 0
    while True:
        pos_split = file.find('_', start)
        if(pos_split == -1):
            break
        if(start == 0):
            name_id = file[start:pos_split]
        else:
            mission_id = file[start:pos_split]
        start = pos_split + 1
        
    R_shoulder = data[:, kinetic_chain[0], :]
    R_elbow = data[:, kinetic_chain[1], :]
    R_wrist = data[:, kinetic_chain[2], :]
    R_root_inv = qinv_np(R_shoulder)
    R_root = R_shoulder
    R_global_shoulder = qmul_np(R_root, R_elbow)

    R_global_elbow = qmul_np(R_global_shoulder, R_wrist)

    R_global_shoulder = qmul_np(R_root_inv, R_global_shoulder)
    R_global_elbow = qmul_np(R_root_inv, R_global_elbow)
    length = len(R_shoulder)

    Quat_base = np.tile(quat_SMPL_base, (length, 1))

    SMPL_position_shoulder = [0, 0, -1 * body_length1]
    SMPL_position_elbow = [0, 0, -1 * body_length2]

    Quat_0 = qmul_np(Quat_base, R_global_shoulder)

    position_elbow = qrot_np(Quat_0, np.tile(SMPL_position_shoulder, (length, 1))) + np.tile(
       base_point, (length, 1))
    Quat_1 = qmul_np(Quat_base, R_global_elbow)
    position_wrist = qrot_np(Quat_1, np.tile(SMPL_position_elbow, (length, 1))) + position_elbow
    return length, Quat_0, Quat_1, position_elbow, position_wrist, npy_files, file, name_id, mission_id


def motion_track_right(Quat_0, Quat_1, position_elbow, position_wrist,sim,flag_animation, bodys_id):
    sim.data.mocap_pos[1] = position_elbow[flag_animation, :]
    sim.data.mocap_pos[4] = position_wrist[flag_animation, :]
    sim.data.mocap_quat[4] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[2] = sim.data.xpos[bodys_id['SMPL_17']].copy()
    sim.data.mocap_quat[2] = Quat_0[flag_animation, :]
    sim.data.mocap_quat[3] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[3] = position_elbow[flag_animation, :]


def motion_track_right_withexo(Quat_0, Quat_1, position_elbow, position_wrist,sim,flag_animation, bodys_id, ratio_exo):
    sim.data.mocap_pos[1] = position_elbow[flag_animation, :]
    sim.data.mocap_pos[4] = position_wrist[flag_animation, :]
    sim.data.mocap_quat[4] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[2] = sim.data.xpos[bodys_id['SMPL_17']].copy()
    sim.data.mocap_quat[2] = Quat_0[flag_animation, :]
    sim.data.mocap_quat[3] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[3] = position_elbow[flag_animation, :]

    exo_delta = (position_wrist[flag_animation, :] - position_elbow[flag_animation, :]) * ratio_exo
    sim.data.mocap_pos[5] = exo_delta + position_elbow[flag_animation, :]

    # sim.data.mocap_pos[5] = exo_delta + position_wrist[flag_animation, :]

def motion_track_left(Quat_0, Quat_1, position_elbow, position_wrist,sim,flag_animation, bodys_id):
    sim.data.mocap_pos[6] = position_elbow[flag_animation, :]
    sim.data.mocap_pos[9] = position_wrist[flag_animation, :]
    sim.data.mocap_quat[9] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[7] = sim.data.xpos[bodys_id['SMPL_16']].copy()
    sim.data.mocap_quat[7] = Quat_0[flag_animation, :]
    sim.data.mocap_quat[8] = Quat_1[flag_animation, :]

    sim.data.mocap_pos[8] = position_elbow[flag_animation, :]

def muscle_process_all(src_dir_muscle):
    npy_files = os.listdir(src_dir_muscle)
    npy_files = sorted(npy_files)
    length = len(npy_files)
    all_item = []
    for index in range(length):
        data = np.load(pjoin(src_dir_muscle, npy_files[index]), allow_pickle=True).item()
        file = npy_files[index]

        # find id and mission
        item_dict = {}
        start = 0
        while True:
            pos_split = file.find('_', start)
            if(pos_split == -1):
                break
            if(start == 0):
                name_id = file[start:pos_split]
            else:
                mission_id = file[start:pos_split]
            start = pos_split + 1
        item_dict['name_id'] = int(name_id)
        item_dict['mission_id'] = int(mission_id)
        item_dict['index'] = index
        all_item.append(item_dict)
    return all_item

def muscle_process(src_dir_muscle, index, length,name=None, step = 3):
    npy_files = os.listdir(src_dir_muscle)
    npy_files = sorted(npy_files)
    if(name == None):
        data = np.load(pjoin(src_dir_muscle, npy_files[index]), allow_pickle=True).item()
        file = npy_files[index]
    else:
        data = np.load(pjoin(src_dir_muscle, name + '.npy'), allow_pickle=True).item()
        file = name + '.npy'

    # find id and mission

    start = 0
    while True:
        pos_split = file.find('_', start)
        if(pos_split == -1):
            break
        if(start == 0):
            name_id = file[start:pos_split]
        else:
            mission_id = file[start:pos_split]
        start = pos_split + 1
    
    length_muscle = []
    for key,value in data.items():
        min_val = np.min(value)
        max_val = np.max(value)
        if max_val > min_val:
            normalized_value = (value - min_val) / (max_val - min_val)
        else:
            normalized_value = np.zeros_like(value)
        # data[key] = normalized_value
        length_muscle.append(len(normalized_value))
    frequency = length_muscle[0] / (length * step)
    return data, frequency, file, name_id, mission_id

def action_generate(muscle_dict, name, action, actuators_id, sim, joints_id, kp ,kv):
    if (name == '1'):
        action[muscle_dict['FDS'][0]] = 0
        action[muscle_dict['FDS'][1]] = 0
        action[muscle_dict['FDS'][2]] = 0
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 1 
    elif (name == '2'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 0
    elif (name == '4'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 0 
    elif (name == '3'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 1
    elif (name == '9' or name == '10'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 0
        action[actuators_id['FPL']] = 1  
    else: 
        action[muscle_dict['FDS'][0]] = 0
        action[muscle_dict['FDS'][1]] = 0
        action[muscle_dict['FDS'][2]] = 0
        action[muscle_dict['FDS'][3]] = 0
        action[actuators_id['FPL']] = 0 
    if (name != '8'):
        sim.data.qfrc_applied[joints_id['flexion']] = -kp * sim.data.qpos[joints_id['flexion']]- kv * sim.data.qvel[joints_id['flexion']] 
    else:
        sim.data.qfrc_applied[joints_id['flexion']] = kp * (sim.model.joint('flexion').range[0] - sim.data.qpos[joints_id['flexion']]) - kv * sim.data.qvel[joints_id['flexion']]
    if (name == '4'):
        sim.data.qfrc_applied[joints_id['pro_sup']] = -kp * sim.data.qpos[joints_id['pro_sup']]- kv * sim.data.qvel[joints_id['pro_sup']] 
        sim.data.qfrc_applied[joints_id['cmc_abduction']] = kp * (sim.model.joint('cmc_abduction').range[1] - sim.data.qpos[joints_id['cmc_abduction']]) - kv * sim.data.qvel[joints_id['cmc_abduction']]
    if (name == '2'):
        sim.data.qfrc_applied[joints_id['pro_sup']] = kp * (sim.model.joint('pro_sup').range[1] - sim.data.qpos[joints_id['pro_sup']])- kv * sim.data.qvel[joints_id['pro_sup']] 
        sim.data.qfrc_applied[joints_id['cmc_abduction']] = kp * (sim.model.joint('cmc_abduction').range[1] - sim.data.qpos[joints_id['cmc_abduction']]) - kv * sim.data.qvel[joints_id['cmc_abduction']]
    return action


def action_generate_gesture(muscle_dict, name, action, actuators_id, sim, joints_id, kp ,kv):
    if (name == '1'):
        action[muscle_dict['FDS'][0]] = 0
        action[muscle_dict['FDS'][1]] = 0
        action[muscle_dict['FDS'][2]] = 0
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 1 
    elif (name == '2'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 0
    elif (name == '4'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 0 
    elif (name == '3'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 1
        action[actuators_id['FPL']] = 1
    elif (name == '9' or name == '10'):
        action[muscle_dict['FDS'][0]] = 1
        action[muscle_dict['FDS'][1]] = 1
        action[muscle_dict['FDS'][2]] = 1
        action[muscle_dict['FDS'][3]] = 0
        action[actuators_id['FPL']] = 1  
    else: 
        action[muscle_dict['FDS'][0]] = 0
        action[muscle_dict['FDS'][1]] = 0
        action[muscle_dict['FDS'][2]] = 0
        action[muscle_dict['FDS'][3]] = 0
        action[actuators_id['FPL']] = 0 
    if (name != '8'):
        sim.data.qfrc_applied[joints_id['flexion']] = -kp * sim.data.qpos[joints_id['flexion']]- kv * sim.data.qvel[joints_id['flexion']] 
    else:
        sim.data.qfrc_applied[joints_id['flexion']] = kp * (sim.model.joint('flexion').range[0] - sim.data.qpos[joints_id['flexion']]) - kv * sim.data.qvel[joints_id['flexion']]
    if (name == '4'):
        sim.data.qfrc_applied[joints_id['pro_sup']] = -kp * sim.data.qpos[joints_id['pro_sup']]- kv * sim.data.qvel[joints_id['pro_sup']] 
        sim.data.qfrc_applied[joints_id['cmc_abduction']] = kp * (sim.model.joint('cmc_abduction').range[1] - sim.data.qpos[joints_id['cmc_abduction']]) - kv * sim.data.qvel[joints_id['cmc_abduction']]
    if (name == '2'):
        sim.data.qfrc_applied[joints_id['pro_sup']] = kp * (sim.model.joint('pro_sup').range[1] - sim.data.qpos[joints_id['pro_sup']])- kv * sim.data.qvel[joints_id['pro_sup']] 
        sim.data.qfrc_applied[joints_id['cmc_abduction']] = kp * (sim.model.joint('cmc_abduction').range[1] - sim.data.qpos[joints_id['cmc_abduction']]) - kv * sim.data.qvel[joints_id['cmc_abduction']]
    return action
    
def save_data_dict(data_dict, obs_saved_dir, name):
    os.makedirs(obs_saved_dir, exist_ok=True)
    np.save(pjoin(obs_saved_dir, str(name)+'.npy'), data_dict)

def obs_data_load(obs_true_path, index):
    npy_files = os.listdir(obs_true_path)
    npy_files = sorted(npy_files)
    data = np.load(pjoin(obs_true_path, npy_files[index]), allow_pickle=True).item()
    file = npy_files[index]
    qpos = data['joint_pos']
    qvel = data['joint_vel']
    local_body_pos = data['local_body_pos']
    local_body_rot = data['local_body_rot_obs']
    local_body_vel = data['local_body_vel']
    local_body_angle_vel = data['local_body_ang_vel']
    length = len(qpos)
    return length, qpos, qvel, local_body_pos, local_body_rot, local_body_vel, local_body_angle_vel

def applied_force(kp, kv, qpos_gt, qvel_gt, qpos, qvel, kp_flexion, kv_flexion, id_flexion):
    exteral_force = np.zeros(len(qpos_gt))
    error_pos = qpos_gt[:] - qpos[:]
    # error_vel = qvel_gt[:]-qvel[:]
    error_vel =  -qvel[:]
    exteral_force[:] = kp * (error_pos[:]) + kv * error_vel 
    exteral_force[id_flexion] = kp_flexion * (error_pos[id_flexion]) + kv_flexion * error_vel[id_flexion]
    return error_pos, error_vel, exteral_force

def applied_force_has_force(sim, tau_gt):
    sim.data.qfrc_applied = tau_gt

def applied_mocap(sim, Quat_0_gt, Quat_1_gt, position_elbow_gt, position_wrist_gt, frame, length):
    frame = int(np.floor(frame / 3))
    if(frame > length - 1):
        frame = length -1
    sim.data.mocap_pos[2] = sim.data.xpos[sim.model.body('mocap_17').id].copy()
    sim.data.mocap_quat[2] = Quat_0_gt[frame, :]
    sim.data.mocap_quat[3] = Quat_1_gt[frame, :]

    sim.data.mocap_pos[3] = position_elbow_gt[frame, :]

    sim.data.mocap_pos[4] = position_wrist_gt[frame, :]

    sim.data.mocap_pos[1] = position_elbow_gt[frame, :]
    sim.data.mocap_pos[0] = sim.data.xpos[sim.model.body('mocap_17').id].copy()
    sim.data.mocap_quat[0] = sim.data.xquat[sim.model.body('SMPL_17').id].copy()
    sim.data.mocap_quat[1] = sim.data.xquat[sim.model.body('SMPL_19').id].copy()
    sim.data.mocap_quat[4] = sim.data.xquat[sim.model.body('SMPL_21').id].copy()


   

def muscle_applied(action, muscle_dict, muscle_activate, frame, relative_freq):
    for key, value in muscle_dict.items():
        length_muscle = len(value)
        for j in range(length_muscle):
            if(math.floor(frame * relative_freq) >= len(muscle_activate[key])):
                action[muscle_dict[key][j]] = muscle_activate[key][len(muscle_activate[key]) - 1]
            else: 
                action[muscle_dict[key][j]] = muscle_activate[key][int(math.floor(frame * relative_freq))]
    return action
    # print('qpose:',sim.data.qpos[joints_id['flexion']])


def xml_tailor(random_index, model_path):

    mass_tailor = random_index[0]
    random_factor_muscle = random_index[1]
    random_factor_exo = random_index[2]
    random_obj_density = random_index[3]
    random_factor_vol = random_index[4]
    random_factor_exo_force = random_index[5]
    # 加载 XML 文件
    tree = ET.parse(model_path)
    root = tree.getroot()

    # MuJoCo 的 <body> 元素通常在 <worldbody> 下面
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("未找到 worldbody，检查 XML 结构是否正确！")

    # 要排除的 body 名称
    exclude_names = {"upper_exo", "lower_exo", "upper_exo_mirror", "lower_exo_mirror", 
                     "table", "box_move", "grap_1", "grap_2"}

    # 计算原始总质量（不包括排除项）
    total_mass = 0
    body_inertials = []

    for body in worldbody.iter('body'):
        body_name = body.get('name', '')
     
        if (body_name.startswith("mocap") or
            body_name.startswith("SMPL")):
                continue

        inertial = body.find('inertial')
        if inertial is None:
            continue
        else:
            mass_str = inertial.get('mass', None)
        
        if mass_str is not None:
            try:
                mass = float(mass_str)
                if (body_name not in exclude_names):
                    total_mass = total_mass + mass
       
            except ValueError:
                mass = 0.0

    
        # print(body_name)
        body_inertials.append((body_name, inertial, mass))
    
    # 计算质量缩放比例
    if total_mass == 0:
        raise ValueError("剩余 body 的总质量为 0,无法调整!")

    scale_factor = mass_tailor / total_mass # 计算新的质量比例

    # 按比例调整质量
    for body_name, inertial, mass in body_inertials:
        if (body_name not in exclude_names):
            new_mass = mass * scale_factor
            if (new_mass >= 0.001):
                inertial.set('mass', str(new_mass)) # 更新 XML
        
        if (body_name in exclude_names):
            if "exo" in body_name:
                new_mass = random_factor_exo * mass
                if (new_mass >= 0.001):
                    inertial.set('mass', str(new_mass)) # 更新 XML

    actuator_elem = root.find("actuator")

    for motor in actuator_elem.findall(".//motor"):
        name = motor.get("name")
        if (name == "Exo" or name == "Exo_mirror"):
            gear = float(motor.get("gear"))
            new_gear = random_factor_exo_force * gear
            motor.set("gear", str(new_gear))

    for general in actuator_elem.findall(".//general"):
        name = general.get("name", "")
        tendon = general.get("tendon", None)

        if "Exo" in name or tendon is None:
            continue
            
        gainprm = general.get("gainprm", None)
        if gainprm:
            gainprm_values = gainprm.split()
        if len(gainprm_values) >= 3:
            gainprm_values[2] = str(float(gainprm_values[2]) * random_factor_muscle)
            general.set("gainprm", " ".join(gainprm_values))

        biasprm = general.get("biasprm", None)
        if biasprm:
            biasprm_values = biasprm.split()
        if len(biasprm_values) >= 3:
            biasprm_values[2] = str(float(biasprm_values[2]) * random_factor_muscle)
            general.set("biasprm", " ".join(biasprm_values))

    temp_dir = '/home/mmy/myosuite/myosuite/simhive/mjc'
    new_model_path = os.path.join(temp_dir, "modified_mujoco_model.xml")
    tree.write(new_model_path)

    return new_model_path

def grasp_handle(action, muscle_dict,actuators_id, command,gym_a):
    for i in range(action.shape[0]):
        action[i] = 100
    action[muscle_dict['FDS'][0]] = command
    action[muscle_dict['FDS'][1]] = command
    action[muscle_dict['FDS'][2]] = command
    action[muscle_dict['FDS'][3]] = command
    action[muscle_dict['FDP'][0]] = command
    action[muscle_dict['FDP'][1]] = command
    action[muscle_dict['FDP'][2]] = command
    action[muscle_dict['FDP'][3]] = command
    action[actuators_id['FPL']] = command
    action[muscle_dict['FDS_mirror'][0]] = command
    action[muscle_dict['FDS_mirror'][1]] = command
    action[muscle_dict['FDS_mirror'][2]] = command
    action[muscle_dict['FDS_mirror'][3]] = command
    action[muscle_dict['FDP_mirror'][0]] = command
    action[muscle_dict['FDP_mirror'][1]] = command
    action[muscle_dict['FDP_mirror'][2]] = command
    action[muscle_dict['FDP_mirror'][3]] = command
    action[actuators_id['FPL_mirror']] = command
    j = 0
    for i,a in enumerate(action):
        if a == 100:
            action[i] = gym_a[j]
            j += 1
    return action

def grasp_handle_1(action, muscle_dict,actuators_id, command):
    action[muscle_dict['FDS'][0]] = command
    action[muscle_dict['FDS'][1]] = command
    action[muscle_dict['FDS'][2]] = command
    action[muscle_dict['FDS'][3]] = command
    action[muscle_dict['FDP'][0]] = command
    action[muscle_dict['FDP'][1]] = command
    action[muscle_dict['FDP'][2]] = command
    action[muscle_dict['FDP'][3]] = command
    action[actuators_id['FPL']] = command
    action[muscle_dict['FDS_mirror'][0]] = command
    action[muscle_dict['FDS_mirror'][1]] = command
    action[muscle_dict['FDS_mirror'][2]] = command
    action[muscle_dict['FDS_mirror'][3]] = command
    action[muscle_dict['FDP_mirror'][0]] = command
    action[muscle_dict['FDP_mirror'][1]] = command
    action[muscle_dict['FDP_mirror'][2]] = command
    action[muscle_dict['FDP_mirror'][3]] = command
    action[actuators_id['FPL_mirror']] = command
    return action
   
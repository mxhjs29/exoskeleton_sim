import sys
sys.path.append('/SMPL/src')

import numpy as np
from SMPL.src.utils.common.model_load import *
from SMPL.src.utils.common.quaternion import *
import pickle
import yaml
from tqdm import tqdm
from collections import OrderedDict
import SMPL.src.utils.np_transform_utils as npt_utils

def compute_self_observations(body_pos: np.ndarray, body_rot: np.ndarray, body_vel: np.ndarray, body_ang_vel: np.ndarray) -> OrderedDict:
    obs = OrderedDict()

    root_pos = body_pos[:,0,:]
    root_rot = body_rot[:,0,:]

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:,2:3]
    obs["root_h_obs"] = root_h

    heading_rot_inv_expand = heading_rot_inv[...,None,:]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1],axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    # 计算各个部分的相对位置,朝向统一
    root_pos_expand = root_pos[...,None,:]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    # 不包含根节点位置,因为一直是0
    obs["local_body_pos"] = local_body_pos[...,3:]

    # 相对角度计算
    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    # 相对速度
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"] = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    return obs

data_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data"
data_path_with_base_quat = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data"
left_leg =  tuple([0,1,4,7,10])
right_leg =  tuple([0,2,5,8,11])
right_arm =  tuple([12,17,19,21,41])
left_arm =  tuple([12,16,18,20,26])
body =  tuple([0,12])
joints = ['box_move_tx','pelvis_tx','ankle_angle_r', 'hip_flexion_l','ankle_angle_l','lumbar_rotation' ,'sternoclavicular_r2','md5_flexion_mirror', 'elbow_flexion']
actuators = ['FPL','FDS5','FDS4','FDS3','FDS2', 'FDP5','FDP4','FDP3','FDP2','FDS5_mirror','FDS4_mirror','FDS3_mirror','FDS2_mirror'
             ,'FDP5_mirror','FDP4_mirror','FDP3_mirror','FDP2_mirror', 'FPL_mirror']
bodys = ['SMPL_0', 'SMPL_1', 'SMPL_2', 'SMPL_4', 'SMPL_5', 'SMPL_7', 'SMPL_8', 'SMPL_10','SMPL_11',
            'SMPL_12','SMPL_16','SMPL_18','SMPL_20','SMPL_17','SMPL_19','SMPL_21', 'SMPL_41', 'SMPL_26',
            'mocap_0', 'mocap_1', 'mocap_2', 'mocap_4', 'mocap_5', 'mocap_7', 'mocap_8', 'mocap_10','mocap_11',
            'mocap_12', 'mocap_16', 'mocap_18', 'mocap_20', 'mocap_17', 'mocap_19', 'mocap_21', 'pelvis',
            'grap_1', 'grap_2','box_move', 'table', 'proximal_thumb', 'secondmc', 'thirdmc', 'fourthmc', 'fifthmc',
            'proximal_thumb_mirror', 'secondmc_mirror', 'thirdmc_mirror', 'fourthmc_mirror', 'fifthmc_mirror']
sites = ['SMPL_10', 'head_max']
muscle_dict = {
    'FDS': ['FDS5','FDS4','FDS3','FDS2'],
    'FDP': ['FDP5','FDP4','FDP3','FDP2'],
    'FDS_mirror': ['FDS5_mirror','FDS4_mirror','FDS3_mirror','FDS2_mirror'],
    'FDP_mirror': ['FDP5_mirror','FDP4_mirror','FDP3_mirror','FDP2_mirror'],
}
obs_dict = {
    'joint_pos' : [],
    'joint_vel' : []
}



entries = ["bend.pkl"]
# entries = entries[:-1]
number = len(entries)
# number = 2
entries_all = entries

xml_length = 1
motor_scale = []
pos_scale = []
vel_scale = []
for xml_id in range(xml_length):
    model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying.xml"
    sim, joints_id, actuators_id, bodys_id, sites_id = load_model(model_path, joints, actuators, bodys, sites)
    body_names = ['pelvis','femur_r','tibia_r','talus_r','calcn_r','toes_r','patella_r',
                           'femur_l','tibia_l','talus_l','calcn_l','toes_l','patella_l','torso']
    robot_body_idxes = [sim.model.body(name).id for name in body_names]
    for key, value in muscle_dict.items():
        N_muscle = len(value)
        for i in range(N_muscle):
            muscle_dict[key][i] = actuators_id[muscle_dict[key][i]]
    target_geoms = {"table_collision", "box_visual"}
    geom_id = {name: sim.model.geom(name).id for name in target_geoms}

    # sim.advance(substeps = 1, render = True)
    elbow_new_id = joints_id['elbow_flexion'] - joints_id['sternoclavicular_r2']
    foot_pos = sim.data.site_xpos[sites_id['SMPL_10']].copy()
    head_pos = sim.data.site_xpos[sites_id['head_max']].copy()

    height = head_pos[2] - foot_pos[2]
    total_mass = 0.0
    body_records = []
    # max_pos = sim.data.body_xpos[bodys_id['']]
    # 遍历所有body（包括worldbody）
    for body_id in range(sim.model.nbody):
        # 获取body名称
        body_name = sim.model.body(body_id).name
        
        # 跳过特定前缀的body
        if body_name.startswith(('mocap', 'SMPL')):
            continue
        
        # 获取质量（注意：worldbody的mass为0）
        mass = sim.model.body_mass[body_id]
        if (mass == 0.001):
            continue
        # 记录有效body信息
        body_records.append({
            'name': body_name,
            'id': body_id,
            'mass': mass
        })
        
        # 累加总质量
        total_mass += mass
    print("human_height", height)
    print("human_mass", total_mass)

    index = 0
    with tqdm(total=number) as pbar:
        while index < number:
            name = entries[index]
            print("name", name)
            
            index = index + 1
            pbar.update(1)
            file_path = os.path.join(data_path, name)
            file_path_quat = os.path.join(data_path_with_base_quat, name)
            with open(file_path, "rb") as f:
                two_modata = pickle.load(f)
            with open(file_path_quat, "rb") as f:
                two_modata_with_quat = pickle.load(f)

            two_modata_with_quat = two_modata
           

            base_relative_pos = {'left_leg' : [], 'left_leg_norm' : [], 
                            'right_leg' : [], 'right_leg_norm' : [],
                            'left_leg' : [], 'left_leg_norm' : [], 
                            'right_arm' : [], 'right_arm_norm' : [],
                            'left_arm' : [], 'left_arm_norm' : [], 
                            'body' : [], 'body_norm' : []}
            base_relative_quat = {'left_leg' : [], 'right_leg' : [], 'right_arm' : [], 'left_arm' : [], 'body' : []}
            dataset_relative_pos = {'left_leg' : [], 'right_leg' : [], 'right_arm' : [], 'left_arm' : [], 'body' : [], 'base_pos': []}
            dataset_relative_quat = {'left_leg' : [], 'right_leg' : [], 'right_arm' : [], 'left_arm' : [], 'body' : [], 'base_rot': []}
            base_pos = {'left_leg' : [], 'right_leg' : [], 'right_arm' : [], 'left_arm' : [], 'body' : []}
            base_quat = {'left_leg' : [], 'right_leg' : [], 'right_arm' : [], 'left_arm' : [], 'body' : []}
            
            base_camera = two_modata_with_quat['base_rot_relative'][0, 0, :]
            body_camera_quat = qbetween_np(np.array([0,-1,0]), base_camera)

            cam_id = sim.model.camera_name2id("third_person")
            # print(cam_id)
          
           
            angle = -90
            angle = math.radians(angle)
               
            delta_y = math.sin(angle) * 1
            delta_x = math.cos(angle) * 1
            
            sim.forward()
            for i in range(len(body) - 1):
                sim.reset()
                sim.data.qpos[:] = np.zeros_like(sim.data.qpos[:].copy())
                sim.data.qvel[:] = np.zeros_like(sim.data.qvel[:].copy())
                # sim.data.mocap_pos[38] = np.array([1.5,0,0])
                # sim.advance(substeps = 1, render = True)
                sim.forward()
                
                body_father_left = 'SMPL_' + str(body[i])
                body_child_left = 'SMPL_' + str(body[i+1])
                
                dataset_body = two_modata[body][i]
                dataset_body_norm = dataset_body / np.linalg.norm(dataset_body, axis=-1, keepdims=True)

                relative_body = sim.data.xpos[bodys_id[body_child_left]].copy() \
                                - sim.data.xpos[bodys_id[body_father_left]].copy()
            
                relative_body_quat = qbetween_np(np.array([0,0,1]), relative_body)
                dataset_body_scale = np.linalg.norm(relative_body) * dataset_body_norm
                
                relative_left_broad = np.broadcast_to(np.array([0,0,1]), dataset_body_scale.shape).copy()
                body_rot = qbetween_np(relative_left_broad, dataset_body_scale)

                base_relative_pos['body'].append(relative_body.tolist())
                base_relative_pos['body_norm'].append(np.linalg.norm(relative_body).tolist())
                base_relative_quat['body'].append(relative_body_quat.tolist())

                base_pos['body'].append(sim.data.xpos[bodys_id[body_father_left]].copy().tolist())
                base_quat['body'].append(sim.data.xquat[bodys_id[body_father_left]].copy().tolist())

                if (i == len(body) - 2):
                    base_pos['body'].append(sim.data.xpos[bodys_id[body_child_left]].copy().tolist())
                    base_quat['body'].append(sim.data.xquat[bodys_id[body_child_left]].copy().tolist())     
                
                dataset_relative_pos['body'].append(dataset_body_scale)
                dataset_relative_quat['body'].append(body_rot)


            for i in range(len(left_arm) - 1):
                
                left_arm_father_left = 'SMPL_' + str(left_arm[i])
                right_arm_father_right = 'SMPL_' + str(right_arm[i])
                left_arm_child_left = 'SMPL_' + str(left_arm[i+1])
                right_arm_child_right = 'SMPL_' + str(right_arm[i+1])

                left_leg_father_left = 'SMPL_' + str(left_leg[i])
                right_leg_father_right = 'SMPL_' + str(right_leg[i])
                left_leg_child_left = 'SMPL_' + str(left_leg[i+1])
                right_leg_child_right = 'SMPL_' + str(right_leg[i+1])
                
                dataset_left_arm = two_modata[left_arm][i]
                dataset_right_arm = two_modata[right_arm][i]
                dataset_left_leg = two_modata[left_leg][i]
                dataset_right_leg = two_modata[right_leg][i]
                
                plv_pos = two_modata['base_pose']
                shoulder_dif = two_modata['base_rot_relative']

                dataset_left_arm_norm = dataset_left_arm / np.linalg.norm(dataset_left_arm, axis=-1, keepdims=True)
                dataset_right_arm_norm = dataset_right_arm / np.linalg.norm(dataset_right_arm, axis=-1, keepdims=True)
                dataset_left_leg_norm = dataset_left_leg / np.linalg.norm(dataset_left_leg, axis=-1, keepdims=True)
                dataset_right_leg_norm = dataset_right_leg / np.linalg.norm(dataset_right_leg, axis=-1, keepdims=True)

                dataset_base_rot = qbetween_np(np.broadcast_to(np.array([0,-1,0]), shoulder_dif.shape).copy(), shoulder_dif)
                if (left_leg_father_left == 'SMPL_0'):
                    error = sim.data.xpos[bodys_id[left_leg_father_left]].copy() - plv_pos[0, 0, :]
                    dataset_base_pos =  plv_pos + error
                    dataset_relative_pos['base_pos'].append(dataset_base_pos)
                    dataset_relative_quat['base_rot'].append(dataset_base_rot)

                #leg_process
                relative_left_leg = sim.data.xpos[bodys_id[left_leg_child_left]].copy() \
                                - sim.data.xpos[bodys_id[left_leg_father_left]].copy()
                relative_right_leg = sim.data.xpos[bodys_id[right_leg_child_right]].copy() \
                                - sim.data.xpos[bodys_id[right_leg_father_right]].copy()
                if (left_leg_father_left == 'SMPL_7'):
                    base_vector = np.array([1,0,0])
                else:
                    base_vector = np.array([0,0,-1])
                relative_left_leg_quat = qbetween_np(base_vector, relative_left_leg)
                relative_right_leg_quat = qbetween_np(base_vector, relative_right_leg)
                dataset_left_leg_scale = np.linalg.norm(relative_left_leg) * dataset_left_leg_norm
                dataset_right_leg_scale = np.linalg.norm(relative_right_leg) * dataset_right_leg_norm
                relative_left_broad = np.broadcast_to(base_vector, dataset_left_leg_scale.shape).copy()
                relative_right_broad = np.broadcast_to(base_vector, dataset_right_leg_scale.shape).copy()
                left_leg_rot = qbetween_np(relative_left_broad, dataset_left_leg_scale)
                right_leg_rot = qbetween_np(relative_right_broad, dataset_right_leg_scale)
                base_relative_pos['left_leg'].append(relative_left_leg.tolist())
                base_relative_pos['right_leg'].append(relative_right_leg.tolist())
                base_relative_pos['left_leg_norm'].append(np.linalg.norm(relative_left_leg).tolist())
                base_relative_pos['right_leg_norm'].append(np.linalg.norm(relative_right_leg).tolist())
                base_relative_quat['left_leg'].append(relative_left_leg_quat.tolist())
                base_relative_quat['right_leg'].append(relative_right_leg_quat.tolist())
                base_pos['left_leg'].append(sim.data.xpos[bodys_id[left_leg_father_left]].copy().tolist())
                base_pos['right_leg'].append(sim.data.xpos[bodys_id[right_leg_father_right]].copy().tolist())
                base_quat['left_leg'].append(sim.data.xquat[bodys_id[left_leg_father_left]].copy().tolist())
                base_quat['right_leg'].append(sim.data.xquat[bodys_id[right_leg_father_right]].copy().tolist())
                if (i == len(left_arm) - 2):
                    base_pos['left_leg'].append(sim.data.xpos[bodys_id[left_leg_child_left]].copy().tolist())
                    base_pos['right_leg'].append(sim.data.xpos[bodys_id[right_leg_child_right]].copy().tolist())
                    base_quat['left_leg'].append(sim.data.xquat[bodys_id[left_leg_child_left]].copy().tolist())
                    base_quat['right_leg'].append(sim.data.xquat[bodys_id[right_leg_child_right]].copy().tolist())
                    
                dataset_relative_pos['left_leg'].append(dataset_left_leg_scale)
                dataset_relative_pos['right_leg'].append(dataset_right_leg_scale)
                dataset_relative_quat['left_leg'].append(left_leg_rot)
                dataset_relative_quat['right_leg'].append(right_leg_rot)

                #arm_process
                relative_left_arm = sim.data.xpos[bodys_id[left_arm_child_left]].copy() \
                                - sim.data.xpos[bodys_id[left_arm_father_left]].copy()
                relative_right_arm = sim.data.xpos[bodys_id[right_arm_child_right]].copy() \
                                - sim.data.xpos[bodys_id[right_arm_father_right]].copy()
                if (left_arm_father_left == 'SMPL_12'):
                    base_vector_left = np.array([0,-1,0])
                    base_vector_right = np.array([0,1,0])
                else:
                    base_vector_left = np.array([0,0,-1])
                    base_vector_right = np.array([0,0,-1])
                relative_left_arm_quat = qbetween_np(base_vector_left, relative_left_arm)
                relative_right_arm_quat = qbetween_np(base_vector_right, relative_right_arm)
                dataset_left_arm_scale = np.linalg.norm(relative_left_arm) * dataset_left_arm_norm
                dataset_right_arm_scale = np.linalg.norm(relative_right_arm) * dataset_right_arm_norm
                relative_left_broad = np.broadcast_to(base_vector_left, dataset_left_arm_scale.shape).copy()
                relative_right_broad = np.broadcast_to(base_vector_right, dataset_right_arm_scale.shape).copy()
                left_arm_rot = qbetween_np(relative_left_broad, dataset_left_arm_scale)
                right_arm_rot = qbetween_np(relative_right_broad, dataset_right_arm_scale)
                base_relative_pos['left_arm'].append(relative_left_arm.tolist())
                base_relative_pos['right_arm'].append(relative_right_arm.tolist())
                base_relative_pos['left_arm_norm'].append(np.linalg.norm(relative_left_arm).tolist())
                base_relative_pos['right_arm_norm'].append(np.linalg.norm(relative_right_arm).tolist())
                base_relative_quat['left_arm'].append(relative_left_arm_quat.tolist())
                base_relative_quat['right_arm'].append(relative_right_arm_quat.tolist())
                base_pos['left_arm'].append(sim.data.xpos[bodys_id[left_arm_father_left]].copy().tolist())
                base_pos['right_arm'].append(sim.data.xpos[bodys_id[right_arm_father_right]].copy().tolist())
                base_quat['left_arm'].append(sim.data.xquat[bodys_id[left_arm_father_left]].copy().tolist())
                base_quat['right_arm'].append(sim.data.xquat[bodys_id[right_arm_father_right]].copy().tolist())
                if (i == len(left_arm) - 2):
                    base_pos['left_arm'].append(sim.data.xpos[bodys_id[left_arm_child_left]].copy().tolist())
                    base_pos['right_arm'].append(sim.data.xpos[bodys_id[right_arm_child_right]].copy().tolist())
                    base_quat['left_arm'].append(sim.data.xquat[bodys_id[left_arm_child_left]].copy().tolist())
                    base_quat['right_arm'].append(sim.data.xquat[bodys_id[right_arm_child_right]].copy().tolist())
                    
                dataset_relative_pos['left_arm'].append(dataset_left_arm_scale)
                dataset_relative_pos['right_arm'].append(dataset_right_arm_scale)
                dataset_relative_quat['left_arm'].append(left_arm_rot)
                dataset_relative_quat['right_arm'].append(right_arm_rot)



            N = 1000000


            step = 3
            flag_animation = 0
            end_flag = 0
            frame_all = dataset_relative_pos['left_leg'][0].shape[1]
            frames = []
            q_pos_data_collected = []
            q_vel_data_collected = []
            root_h_obs_collect = []
            local_body_pos_collect = []
            local_body_rot_collect = []
            local_body_vel_collect = []
            local_body_ang_vel_collect = []
            q_pos_list = []
            q_vel_list = []
            tau_data_collected = []
            force_buffer = []
            hand_box_weld_l = 34
            hand_box_weld_r = 37
            grap_flag = 0
            q_frac_norm_list = []
            # sim.data.eq_active[hand_box_weld_r] = 0
            # sim.data.eq_active[hand_box_weld_l] = 0
            table_size = sim.model.geom_size[geom_id['table_collision']]
            box_size = sim.model.geom_size[geom_id['box_visual']]
            for i in range(N):
                if (i % step == 0 and i!=0 and flag_animation < frame_all):
                    sim.data.mocap_pos[0] = dataset_relative_pos['base_pos'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[0] = dataset_relative_quat['body'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[1] = sim.data.mocap_pos[0].copy() + dataset_relative_pos['body'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[1] = sim.data.xquat[bodys_id['SMPL_12']].copy()

                    sim.data.mocap_pos[2] = dataset_relative_pos['base_pos'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[2] = dataset_relative_quat['left_leg'][0][:,flag_animation,:].squeeze()   
                    sim.data.mocap_pos[3] = sim.data.mocap_pos[2].copy() + dataset_relative_pos['left_leg'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[3] = dataset_relative_quat['left_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[4] = sim.data.mocap_pos[3].copy() + dataset_relative_pos['left_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[4] = dataset_relative_quat['left_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[5] = sim.data.mocap_pos[4].copy() + dataset_relative_pos['left_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[5] = dataset_relative_quat['left_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[6] = sim.data.mocap_pos[5].copy() + dataset_relative_pos['left_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[6] = sim.data.xquat[bodys_id['SMPL_10']].copy()
                    # print("left_leg:", sim.data.mocap_pos[6])

                    sim.data.mocap_pos[7] = dataset_relative_pos['base_pos'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[7] = dataset_relative_quat['right_leg'][0][:,flag_animation,:].squeeze()   
                    sim.data.mocap_pos[8] = sim.data.mocap_pos[7].copy() + dataset_relative_pos['right_leg'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[8] = dataset_relative_quat['right_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[9] = sim.data.mocap_pos[8].copy() + dataset_relative_pos['right_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[9] = dataset_relative_quat['right_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[10] = sim.data.mocap_pos[9].copy() + dataset_relative_pos['right_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[10] = dataset_relative_quat['right_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[11] = sim.data.mocap_pos[10].copy() + dataset_relative_pos['right_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[11] = sim.data.xquat[bodys_id['SMPL_11']].copy()
                    # print("right_leg:", sim.data.mocap_pos[11])

                    sim.data.mocap_pos[12] =  sim.data.mocap_pos[1].copy()
                    sim.data.mocap_quat[12] = dataset_relative_quat['right_arm'][0][:,flag_animation,:].squeeze()   
                    
                    # flag_animation = flag_test
                    sim.data.mocap_pos[13] = sim.data.mocap_pos[12].copy() + dataset_relative_pos['right_arm'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[13] = dataset_relative_quat['right_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[14] = sim.data.mocap_pos[13].copy() + dataset_relative_pos['right_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[14] = dataset_relative_quat['right_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[15] = sim.data.mocap_pos[14].copy() + dataset_relative_pos['right_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[15] = dataset_relative_quat['right_arm'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[16] = sim.data.mocap_pos[15].copy() + dataset_relative_pos['right_arm'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[16] = sim.data.xquat[bodys_id['SMPL_41']].copy()

                    sim.data.mocap_pos[17] =  sim.data.mocap_pos[1].copy()
                    sim.data.mocap_quat[17] = dataset_relative_quat['left_arm'][0][:,flag_animation,:].squeeze()   
                    
                    sim.data.mocap_pos[18] = sim.data.mocap_pos[17].copy() + dataset_relative_pos['left_arm'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[18] = dataset_relative_quat['left_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[19] = sim.data.mocap_pos[18].copy() + dataset_relative_pos['left_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[19] = dataset_relative_quat['left_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[20] = sim.data.mocap_pos[19].copy() + dataset_relative_pos['left_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[20] = dataset_relative_quat['left_arm'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[21] = sim.data.mocap_pos[20].copy() + dataset_relative_pos['left_arm'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[21] = sim.data.xquat[bodys_id['SMPL_26']].copy()
                    #track
                    sim.data.mocap_pos[22] = dataset_relative_pos['base_pos'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[22] = dataset_relative_quat['base_rot'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[23] = sim.data.mocap_pos[22].copy() + dataset_relative_pos['body'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[23] = sim.data.xquat[bodys_id['SMPL_12']].copy()
                    sim.data.mocap_quat[23] = dataset_relative_quat['base_rot'][0][:,flag_animation,:].squeeze()

                    sim.data.mocap_pos[24] = sim.data.mocap_pos[22].copy() + dataset_relative_pos['left_leg'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[24] = sim.data.xquat[bodys_id['SMPL_1']].copy()
                    # sim.data.mocap_quat[24] = dataset_relative_quat['left_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[25] = sim.data.mocap_pos[24].copy() + dataset_relative_pos['left_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[25] = sim.data.xquat[bodys_id['SMPL_4']].copy()
                    # sim.data.mocap_quat[25] = dataset_relative_quat['left_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[26] = sim.data.mocap_pos[25].copy() + dataset_relative_pos['left_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[26] = sim.data.xquat[bodys_id['SMPL_7']].copy()
                    # sim.data.mocap_quat[26] = dataset_relative_quat['left_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[27] = sim.data.mocap_pos[26].copy() + dataset_relative_pos['left_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[27] = sim.data.xquat[bodys_id['SMPL_10']].copy()

            
                    sim.data.mocap_pos[28] = sim.data.mocap_pos[22].copy() + dataset_relative_pos['right_leg'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[28] = sim.data.xquat[bodys_id['SMPL_2']].copy()
                    # sim.data.mocap_quat[28] = dataset_relative_quat['right_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[29] = sim.data.mocap_pos[28].copy() + dataset_relative_pos['right_leg'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[29] = sim.data.xquat[bodys_id['SMPL_5']].copy()
                    # sim.data.mocap_quat[29] = dataset_relative_quat['right_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[30] = sim.data.mocap_pos[29].copy() + dataset_relative_pos['right_leg'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[30] = sim.data.xquat[bodys_id['SMPL_8']].copy()
                    # sim.data.mocap_quat[30] = dataset_relative_quat['right_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[31] = sim.data.mocap_pos[30].copy() + dataset_relative_pos['right_leg'][3][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[31] = sim.data.xquat[bodys_id['SMPL_11']].copy()

                    sim.data.mocap_pos[32] = sim.data.mocap_pos[23].copy() + dataset_relative_pos['right_arm'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[32] = dataset_relative_quat['right_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[33] = sim.data.mocap_pos[32].copy() + dataset_relative_pos['right_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[33] = dataset_relative_quat['right_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[34] = sim.data.mocap_pos[33].copy() + dataset_relative_pos['right_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[34] = dataset_relative_quat['right_arm'][3][:,flag_animation,:].squeeze()
                  

                    sim.data.mocap_pos[35] = sim.data.mocap_pos[23].copy() + dataset_relative_pos['left_arm'][0][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[35] = dataset_relative_quat['left_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[36] = sim.data.mocap_pos[35].copy() + dataset_relative_pos['left_arm'][1][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[36] = dataset_relative_quat['left_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_pos[37] = sim.data.mocap_pos[36].copy() + dataset_relative_pos['left_arm'][2][:,flag_animation,:].squeeze()
                    sim.data.mocap_quat[37] = dataset_relative_quat['left_arm'][3][:,flag_animation,:].squeeze()
                    flag_animation = flag_animation + 1
                    if (flag_animation == 345):
                        step = 3
                    if (flag_animation == 147):
                        step = 3
                    if (flag_animation == 148):
                        step = 3
                    if (flag_animation == 346):
                        step = 3
                    
                        end_flag = 1
                
                obs_proprio = OrderedDict()
                body_pos = np.array([sim.data.xpos.copy()[i] for i in robot_body_idxes])[None,]
                body_rot = np.array([sim.data.xquat.copy()[i] for i in robot_body_idxes])[None,]
                body_vel = np.array([sim.data.cvel.copy()[i][:3] for i in robot_body_idxes])[None,]
                body_ang_vel = np.array([sim.data.cvel.copy()[i][3:] for i in robot_body_idxes])[None,]
                obs_proprio = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
                root_h_obs_collect.append(obs_proprio['root_h_obs'])
                local_body_pos_collect.append(obs_proprio['local_body_pos'])
                local_body_rot_collect.append(obs_proprio['local_body_rot_obs'])
                local_body_vel_collect.append(obs_proprio['local_body_vel'])
                local_body_ang_vel_collect.append(obs_proprio['local_body_ang_vel'])

                q_pos_truth = sim.data.qpos[joints_id['pelvis_tx']:joints_id['lumbar_rotation'] + 1].copy()
                q_vel_truth = sim.data.qvel[joints_id['pelvis_tx']:joints_id['lumbar_rotation'] + 1].copy()
                q_pos_data_collected.append(q_pos_truth)
                q_vel_data_collected.append(q_vel_truth)
                grap_pos_l = sim.data.xpos[bodys_id['grap_1']]
                hand_pos_l = sim.data.xpos[bodys_id['SMPL_26']]
                grap_pos_r = sim.data.xpos[bodys_id['grap_2']]
                hand_pos_r = sim.data.xpos[bodys_id['SMPL_41']]
                box_pos = sim.data.xpos[bodys_id['box_move']]
                table_pos = sim.data.xpos[bodys_id['table']]
                dis_l = np.linalg.norm(grap_pos_l - hand_pos_l)
                dis_r = np.linalg.norm(grap_pos_r - hand_pos_r)

                
                
                if (flag_animation >= 147 and end_flag == 0):
               
                    action = np.ones(sim.model.nu) * 0
                    command = 1
                    action = grasp_handle_1(action, muscle_dict,actuators_id, command)
                    sim.data.ctrl[:] = action
                
                if (flag_animation >= 345 and end_flag == 1):
               
                    action = np.ones(sim.model.nu) * 0
                    command = -1
                    action = grasp_handle_1(action, muscle_dict,actuators_id, command)
                    sim.data.ctrl[:] = action
              
                
                q_pos_truth_yaml = sim.data.qpos[joints_id['pelvis_tx']:joints_id['lumbar_rotation'] + 1].copy()
                q_vel_truth_yaml = sim.data.qvel[joints_id['pelvis_tx']:joints_id['lumbar_rotation'] + 1].copy()
                q_frac_constraint = sim.data.qfrc_constraint[joints_id['sternoclavicular_r2']:joints_id['md5_flexion_mirror'] + 1].copy()
                q_frac_norm = np.linalg.norm(q_frac_constraint)
                q_frac_norm_list.append(q_frac_norm)
        
                sim.advance(substeps = 1, render = True)
                sim.forward()
                if (i % step == 0 and i >= 500):
                    q_pos_list.append(q_pos_truth_yaml.tolist())
                    q_vel_list.append(q_vel_truth_yaml.tolist())
                    if (i != 0):
                        avg_force = np.mean(force_buffer, axis=0)
                        tau_data_collected.append(avg_force)
                        force_buffer.clear()
                    flag = 0
                
                tau_truth = sim.data.qfrc_constraint[joints_id['sternoclavicular_r2']:joints_id['md5_flexion_mirror'] + 1].copy() + \
                            sim.data.qfrc_passive[joints_id['sternoclavicular_r2']:joints_id['md5_flexion_mirror'] + 1].copy() - \
                            sim.data.qfrc_bias[joints_id['sternoclavicular_r2']:joints_id['md5_flexion_mirror'] + 1].copy()
                force_buffer.append(tau_truth)

                if (i > (frame_all + 15) * step + 100):
                    break
                
                # print(frame_number)
            q_pos_truth_array = np.array(q_pos_data_collected)
            print("q_pos shape:", len(q_pos_truth_array))
            q_vel_truth_array = np.array(q_vel_data_collected)
            tau_truth_scale = np.array(tau_data_collected) / total_mass
            obs_dict['joint_pos'] = q_pos_truth_array
            obs_dict['joint_vel'] = q_vel_truth_array
            obs_dict['root_h_obs'] = root_h_obs_collect
            obs_dict['local_body_pos'] = local_body_pos_collect
            obs_dict['local_body_rot_obs'] = local_body_rot_collect
            obs_dict['local_body_vel'] = local_body_vel_collect
            obs_dict['local_body_ang_vel'] = local_body_ang_vel_collect
            obs_saved_dir = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/obs_truth"
            name = "obs_truth_long"
            save_data_dict(obs_dict,obs_saved_dir, name)

            data_to_save_yaml = {
                'qpos' : q_pos_list,
                'qvel' : q_vel_list
            }
            initial_yaml_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data/initial_state_with_exo.yaml"
            with open(initial_yaml_path, "w") as file:
                yaml.safe_dump(data_to_save_yaml, file, default_flow_style=False)
        sim.renderer.close()
        del sim




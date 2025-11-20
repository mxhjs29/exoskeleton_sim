from amc_parser import *
import numpy as np
import pickle

asf_smpl_map = {
  'root': 0,
  'lhipjoint' : 2,
  'rhipjoint' : 1,
  'lfemur': 5,
  'rfemur': 4,
  'ltibia': 8,
  'rtibia': 7,
  'lfoot': 11,
  'rfoot': 10,
  'head': 24,
  'lowerneck': 12,
  'upperneck': 15,
  'lclavicle': 17,
  'rclavicle': 16,
  'lhumerus': 19,
  'rhumerus': 18,
  'lwrist': 21,
  'rwrist': 20,
  'lhand': 23,
  'rhand': 22
}
asf_path = './115.asf'
amc_path = './115.amc'
joints = parse_asf(asf_path)
motions = parse_amc(amc_path)

left_leg =  tuple([0,2,5,8,11])
right_leg =  tuple([0,1,4,7,10])
left_arm =  tuple([12,17,19,21,41])
right_arm =  tuple([12,16,18,20,26])
body =  tuple([0,12])

pkl_write = {
    'text' : ['bend and pick up box'],
    body : [],
    left_leg : [],right_leg : [],
    left_arm : [], right_arm : [],
    'base_pose' : np.zeros((1,len(motions), 3), dtype=np.float32),
    'base_rot_relative' : np.zeros((1,len(motions), 3), dtype=np.float32),
    'new_index' : np.zeros((1,len(motions), 3), dtype=np.float32)
}
a = np.array([[0],[1],[2]])
b = [item[0] for item in a]
left_leg_relative = [[[]],[[]],[[]],[[]]]
right_leg_relative = [[[]],[[]],[[]],[[]]]
left_arm_relative = [[[]],[[]],[[]],[[]]]
right_arm_relative = [[[]],[[]],[[]],[[]]]
body_relative = [[[]]]
global x_old
global y_old
global z_old
for frame_idx in range(len(motions)):
  #更新每一帧 关节的世界坐标
  joints['root'].set_motion(motions[frame_idx])
  for joint in joints:
    x_old = joints[joint].coordinate[0].copy()
    y_old = joints[joint].coordinate[1].copy()
    z_old = joints[joint].coordinate[2].copy()
    joints[joint].coordinate[0] = x_old / 20
    joints[joint].coordinate[1] = -z_old / 20
    joints[joint].coordinate[2] = y_old / 20


  pkl_write['base_pose'][0,frame_idx,:] = np.reshape(np.array(joints['root'].coordinate),(1,3))
  shoulder_relative = joints['rclavicle'].coordinate -joints['lclavicle'].coordinate
  pkl_write['base_rot_relative'][0,frame_idx,:] = np.reshape(shoulder_relative,(1,3))
  pkl_write['new_index'][0,frame_idx,:] = np.reshape(shoulder_relative,(1,3))

  c = [item[0] for item in (joints['lhipjoint'].coordinate - joints['root'].coordinate)]

  left_leg_relative[0][0].append([item[0] for item in (joints['lhipjoint'].coordinate - joints['root'].coordinate)])
  left_leg_relative[1][0].append([item[0] for item in (joints['lfemur'].coordinate - joints['lhipjoint'].coordinate)])
  left_leg_relative[2][0].append([item[0] for item in (joints['ltibia'].coordinate - joints['lfemur'].coordinate)])
  left_leg_relative[3][0].append([item[0] for item in (joints['lfoot'].coordinate - joints['ltibia'].coordinate)])

  right_leg_relative[0][0].append([item[0] for item in (joints['rhipjoint'].coordinate - joints['root'].coordinate)])
  right_leg_relative[1][0].append([item[0] for item in (joints['rfemur'].coordinate - joints['rhipjoint'].coordinate)])
  right_leg_relative[2][0].append([item[0] for item in (joints['rtibia'].coordinate - joints['rfemur'].coordinate)])
  right_leg_relative[3][0].append([item[0] for item in (joints['rfoot'].coordinate - joints['rtibia'].coordinate)])

  left_arm_relative[0][0].append([item[0] for item in (joints['lclavicle'].coordinate - joints['lowerneck'].coordinate)])
  left_arm_relative[1][0].append([item[0] for item in (joints['lhumerus'].coordinate - joints['lclavicle'].coordinate)])
  left_arm_relative[2][0].append([item[0] for item in (joints['lwrist'].coordinate - joints['lhumerus'].coordinate)])
  left_arm_relative[3][0].append([item[0] for item in (joints['lthumb'].coordinate - joints['lwrist'].coordinate)])

  right_arm_relative[0][0].append([item[0] for item in (joints['rclavicle'].coordinate - joints['lowerneck'].coordinate)])
  right_arm_relative[1][0].append([item[0] for item in (joints['rhumerus'].coordinate - joints['rclavicle'].coordinate)])
  right_arm_relative[2][0].append([item[0] for item in (joints['rwrist'].coordinate - joints['rhumerus'].coordinate)])
  right_arm_relative[3][0].append([item[0] for item in (joints['rthumb'].coordinate - joints['rwrist'].coordinate)])

  body_relative[0][0].append([item[0] for item in (joints['lowerneck'].coordinate - joints['root'].coordinate)])

  for joint in joints:
    x_old = joints[joint].coordinate[0].copy() * 20
    y_old = joints[joint].coordinate[1].copy() * 20
    z_old = joints[joint].coordinate[2].copy() * 20
    joints[joint].coordinate[0] = x_old
    joints[joint].coordinate[1] = z_old
    joints[joint].coordinate[2] = -y_old


pkl_write[left_leg] = left_leg_relative
pkl_write[left_arm] = left_arm_relative
pkl_write[right_leg] = right_leg_relative
pkl_write[right_arm] = right_arm_relative
pkl_write[body] = body_relative
with open("bend.pkl", "wb") as file_w:
  pickle.dump(pkl_write, file_w)
print("类实例已保存到文件 bend.pkl")
with open("bend.pkl", "rb") as file_r:
  two_modata = pickle.load(file_r)

z=1
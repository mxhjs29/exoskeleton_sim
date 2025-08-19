import mujoco
import mujoco.viewer

model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/Exoskeleton3DOF_URDFmodel/mjmodel.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
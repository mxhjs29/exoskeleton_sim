import mujoco
import mujoco.viewer
model = mujoco.MjModel.from_xml_path(
    "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml")

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.viewer.launch(model,data)


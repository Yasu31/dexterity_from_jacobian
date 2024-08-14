import mujoco
from mujoco import viewer
import numpy as np

"""
run the simulation using the estimated Jacobian based controller
"""

model_path = "shadow_hand/scene_sphere.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# initial hand pose that angles hand downwards and lightly closes the fingers
data.ctrl[:] = [0.08, -0.35,
                0., 1.2, 0, 0.4, 0,
                0, 0.4, 2,
                0, 0.4, 2,
                0, 0.4, 2,
                0, 0, 0.4, 2,]
actuator_num = model.nu

# get the indices to access the robot's state
# get the names of the actuators
actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(actuator_num)]
# get the indices of the corresponding joints
# convert the actuator names to the joint names (specific to the shadow hand)
actuated_joint_names = [actuator_name.replace("_A_", "_") for actuator_name in actuator_names]
actuated_joint_names = [jnt_name.replace("0", "1") for jnt_name in actuated_joint_names]
actuated_joint_ids = []
for joint_name in actuated_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id != -1, f"Joint {joint_name} not found"
    actuated_joint_ids.append(joint_id)
# start addr in 'qvel' for joint's data
actuated_dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in actuated_joint_ids]
# start addr in 'qpos' for joint's data
actuated_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in actuated_joint_ids]
print(f"{actuator_names=}\n{actuated_joint_names=}\n{actuated_joint_ids=}\n{actuated_dof_ids=}\n{actuated_qpos_ids=}")

# get the indices to access the object's state
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
assert body_id != -1, "Object not found"
body_dof_ids = np.arange(model.body_dofadr[body_id], model.body_dofadr[body_id] + model.body_dofnum[body_id])
body_qpos_ids = np.arange(model.body_jntadr[body_id], model.body_jntadr[body_id] + 7)
print(f"{body_id=}\n{body_dof_ids=}\n{body_qpos_ids=}")

# the estimated control Jacobian
J = np.zeros((3, actuator_num))
# covariance of the estimated J
p = np.eye(3) * 1e-1

def control_cb(model, data):
    q = data.qpos[actuated_qpos_ids]
    dq = data.qvel[actuated_dof_ids]
    u = data.qvel[body_dof_ids[:3]]

mujoco.set_mjcb_control(control_cb)


viewer.launch(model, data)

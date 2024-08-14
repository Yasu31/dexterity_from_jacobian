import mujoco
from mujoco import viewer
import numpy as np

"""
run the simulation using the estimated Jacobian based controller
"""

model_path = "shadow_hand/scene_sphere.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# initial commanded hand pose that angles hand downwards and lightly closes the fingers
init_ctrl = [0.08, -0.3,
                0., 1.2, 0, 0.4, 0,
                -0.1, 0.4, 2,
                0.0, 0.4, 2,
                -0.1, 0.4, 2,
                0, -0.2, 0.4, 2,]
data.ctrl[:] = init_ctrl
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

# get the indices to access the ghost object (just to show the target pose)
ghost_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_ghost")
assert ghost_body_id != -1, "Ghost object not found"

mujoco.mj_forward(model, data)
body_init_pose = data.qpos[body_qpos_ids].copy()
print(f"{body_init_pose=}")
body_target_pos = np.zeros(3)

# the estimated control Jacobian
J = np.zeros((3, actuator_num))
# covariance of the estimated J
p = np.ones(actuator_num) * 1e-1

def compute_task_space_command():
    """
    compute the task space command that will bring the system closer to task space goal
    it is supposed to be the desired velocity in the task space
    """
    # the target should slowly draw a circle
    phase = data.time * 3
    body_target_pos[:] = body_init_pose[:3] + 0.01 * np.array([np.sin(phase), np.cos(phase), 0])
    body_target_pos[2] -= 0.02
    # move the mocap object to the target position for visualization
    data.mocap_pos[:] = body_target_pos
    body_pos = data.xpos[body_id]
    
    task_space_vel = (body_target_pos - body_pos) * 4
    return task_space_vel


def control_cb(model, data):
    """
    callback function called on every step, and is used to set the control command
    """
    # don't do anything for the first moments (until ball falls)
    if data.time < 0.5:
        return
    # first update the estimation of the Jacobian
    global J, p
    q = data.qpos[actuated_qpos_ids]
    dq = data.qvel[actuated_dof_ids]
    u = data.qvel[body_dof_ids[:3]]
    r = 1e-3  # observation noise variance
    numerator = (u - J @ dq).reshape((-1, 1)) @ (p * dq).reshape((1, -1))
    denominator = p.T @ (dq * dq) + r
    J += numerator / denominator
    p = p * (1 - p * dq * dq / denominator)

    task_space_vel_desired = compute_task_space_command()
    dt = model.opt.timestep
    eps = 0.003  # how much to weigh the "going back to init pose" term
    ctrl_0 = init_ctrl - data.ctrl
    # Tikhonov regularization with a shifted center
    delta_q = np.linalg.inv(J.T@J + eps*np.eye(actuator_num)) @ (J.T @ task_space_vel_desired + eps * ctrl_0) * dt
    delta_q = np.clip(delta_q, -0.1, 0.1)  # don't move too much in one step
    data.ctrl[:] += delta_q
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

mujoco.set_mjcb_control(control_cb)


viewer.launch(model, data)

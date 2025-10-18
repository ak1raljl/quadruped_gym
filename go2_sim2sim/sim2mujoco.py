from legged_gym.envs.go2.go2_config import Go2Cfg
import math
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import Logger
import torch
from pynput import keyboard
import sys
import time

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = True
joystick_opened = False

def on_press(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    try:
        if key.char == 'l':
            x_vel_cmd += 0.3
        elif key.char == '.':
            x_vel_cmd -= 0.3
        elif key.char == ',':
            yaw_vel_cmd += 0.3
        elif key.char == '/':
            yaw_vel_cmd -= 0.3
        elif key.char == 'k':
            y_vel_cmd += 0.3
        elif key.char == ';':
            y_vel_cmd -= 0.3
        elif key.char == 'm':
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        print(f"Command: {x_vel_cmd:.2f}, {y_vel_cmd:.2f}, {yaw_vel_cmd:.2f}")
    except AttributeError:
        pass

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    base_pos = data.qpos[0:3].astype(np.double)
    dof_pos = data.qpos[7:19].astype(np.double)
    dof_vel = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    base_lin_vel = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    base_ang_vel = data.qvel[3:6].astype(np.double)
    
    projected_gravity = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    return base_pos, dof_pos, dof_vel, quat, base_lin_vel, base_ang_vel, projected_gravity

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q - q) * kp + (target_dq - dq) * kd
    return torque_out

def run_mujoco(policy, cfg):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    
    data.qpos[2] = 0.42
    default_dof_pos = cfg.robot_config.default_dof_pos
    data.qpos[7:19] = default_dof_pos
    mujoco.mj_forward(model, data)
    
    viewer = mujoco.viewer.launch_passive(model, data)

    torques = np.zeros(cfg.env.num_actions, dtype=np.double)
    actions = np.zeros(cfg.env.num_actions, dtype=np.double)
    last_actions = np.zeros(cfg.env.num_actions, dtype=np.double)
    default_dof_vel = np.zeros(cfg.env.num_actions, dtype=np.double)  # 默认关节速度为0

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.float32))
    
    count_lowlevel = 0
    commands = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd])
    
    gait_freq = 3.0  # Hz
    gait_phase = 0.5
    gait_offset = 0.0
    gait_bound = 0.0
    gait_duration = 0.5
    swing_height = 0.3
    body_pitch = 0.0
    body_roll = 0.0

    gait_indices = 0.0 # FL, RL, FR, RR
    clock_inputs = np.zeros(4, dtype=np.float32)
    transformed_indices = np.zeros(4, dtype=np.float32)
    dt = cfg.sim_config.dt
    decim = cfg.sim_config.decimation
    dt_policy = dt * decim

    print(f"观测维度: {cfg.env.num_observations}")

    for step in range(int(cfg.sim_config.sim_duration / dt)):
        base_pos, dof_pos, dof_vel, quat, base_lin_vel, base_ang_vel, projected_gravity = get_obs(data)
        # 1000hz -> 100hz
        if count_lowlevel % decim == 0:
            gait_indices = (gait_indices + gait_freq * dt_policy) % 1.0

            foot_indices = np.array([
                (gait_indices + gait_phase + gait_offset + gait_bound) % 1.0,  # FL
                (gait_indices + gait_offset) % 1.0,  # RL
                (gait_indices + gait_bound) % 1.0,  # FR
                (gait_indices + gait_phase) % 1.0,  # RR
            ], dtype=np.float32)
            # foot_indices = np.remainder(foot_indices, 1.0)
            
            for i in range(4):
                phase = foot_indices[i]
                if phase < gait_duration:
                    transformed_indices[i] = phase * (0.5 / gait_duration)
                else:
                    transformed_indices[i] = 0.5 + (phase - gait_duration) * (0.5 / (1.0 - gait_duration))

            clock_inputs = np.sin(2 * np.pi * transformed_indices)

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            # projected_gravity
            obs[0, 0:3] = projected_gravity
            # commands
            # linear_velocity
            obs[0, 3] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
            # ang_vel_yaw
            obs[0, 5] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
            # body height
            obs[0, 6] = 0.34 * cfg.normalization.obs_scales.body_height_cmd
            # gait frequency
            obs[0, 7] = gait_freq * cfg.normalization.obs_scales.gait_freq_cmd
            # gait phase
            obs[0, 8] = gait_phase * cfg.normalization.obs_scales.gait_phase_cmd
            # gait offset
            obs[0, 9] = gait_offset * cfg.normalization.obs_scales.gait_phase_cmd
            # gait bound
            obs[0, 10] = gait_bound * cfg.normalization.obs_scales.gait_phase_cmd
            # gait duration
            obs[0, 11] = gait_duration * cfg.normalization.obs_scales.gait_phase_cmd
            # footswing height
            obs[0, 12] = swing_height * cfg.normalization.obs_scales.footswing_height_cmd
            # body pitch
            obs[0, 13] = body_pitch * cfg.normalization.obs_scales.body_pitch_cmd
            # body roll
            obs[0, 14] = body_roll * cfg.normalization.obs_scales.body_roll_cmd

            # dof_pos
            obs[0, 15:27] = (dof_pos - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            # dof_vel
            obs[0, 27:39] = dof_vel * cfg.normalization.obs_scales.dof_vel
            # actions
            obs[0, 39:51] = actions
            # last actions
            obs[0, 51:63] = last_actions
            # gait indices
            obs[0, 63] = gait_indices
            # clock inputs
            obs[0, 64:68] = clock_inputs

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            
            hist_obs.append(obs)
            hist_obs.popleft()


            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                start = i * cfg.env.num_single_obs
                end = (i + 1) * cfg.env.num_single_obs
                policy_input[0, start:end] = hist_obs[i][0, :]
            
            last_actions = actions.copy()
            actions[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            actions = np.clip(actions, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            actions_scaled = actions * cfg.control.action_scale
            # print(f"Step {step}, Cmd: [{x_vel_cmd:.2f}, {y_vel_cmd:.2f}, {yaw_vel_cmd:.2f}] -> Actions: {actions_scaled}")
            
        target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
        if step < 50:
            torques = pd_control(
                target_q=default_dof_pos,
                q=dof_pos,
                kp=cfg.robot_config.kps,
                target_dq=target_dq,
                dq=dof_vel,
                kd=cfg.robot_config.kds
            )
        else:
            torques = pd_control(
                target_q=default_dof_pos + actions_scaled,
                q=dof_pos,
                kp=cfg.robot_config.kps,
                target_dq=target_dq,
                dq=dof_vel,
                kd=cfg.robot_config.kds
            )
        torques = np.clip(torques, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = torques
        
        mujoco.mj_step(model, data)
        viewer.sync()
        count_lowlevel += 1
    
    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script')
    parser.add_argument('--load_model', type=str, default='/home/ak1ra/Quadruped/legged_gym/logs/go2/exported/policies/policy_1.pt')
    parser.add_argument('--mujoco_model', type=str, default='/home/ak1ra/Quadruped/go2_sim2sim/go2_description/scene.xml')
    parser.add_argument('--terrain', action='store_true', help='plane or terrain')
    args = parser.parse_args()

    class Sim2simCfg( Go2Cfg ):
        class sim_config:
            mujoco_model_path = args.mujoco_model
            sim_duration = 120.0
            dt = 0.001
            decimation = 20
        
        class robot_config:
            kps = np.array([25.0] * 12, dtype=np.double)  # 与Isaac Gym保持一致：25.0
            kds = np.array([0.6] * 12, dtype=np.double)   # 与Isaac Gym保持一致：0.6
            tau_limit = 40 * np.ones(12, dtype=np.double)
            
            default_dof_pos = np.array([
                0.0, 0.8, -1.5,   # FL
                0.0, 0.8, -1.5,   # RL
                -0.0, 0.8, -1.5,  # FR
                -0.0, 0.8, -1.5,  # RR
            ], dtype=np.double)
        
        class env(Go2Cfg.env):
            frame_stack = 1 
            num_single_obs = 68  # 与 num_observations 相同
        
    policy = torch.jit.load(args.load_model)
    print("Loaded policy from ", args.load_model)
    
    run_mujoco(policy, Sim2simCfg)

        
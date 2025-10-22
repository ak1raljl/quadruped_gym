from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import os
import torch
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg
import numpy as np

class Go2RoughRobot( LeggedRobot ):
    cfg : GO2RoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
    
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # add body height termination criterion
        if self.cfg.rewards.use_terminal_body_height and self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)
        
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.gait_indices[env_ids] = 0
        
        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #TODO add positive and negative reward buffers
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew
        
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        self.obs_buf = torch.cat(( 
            self.projected_gravity, # 3
            self.commands * self.commands_scale, # 12
            (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:, :self.num_actuated_dof]) * self.obs_scales.dof_pos,
            self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf, self.last_actions), dim=-1)
        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf, self.clock_inputs), dim=-1)
        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(self.num_envs, -1) * 1.0), dim=1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        self.privileged_obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.last_actions), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.clock_inputs), dim=-1)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heading), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(self.num_envs, -1) * 1.0), dim=1)
    
    #------------- Callbacks --------------
    def _post_physics_step_callback(self):
        # resample commands
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # push robots
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0: 
            return
        # lin_vel_x
        self.commands[env_ids, 0] = torch_rand_float(
            self.cfg.commands.lin_vel_x[0],
            self.cfg.commands.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # lin_vel_y
        self.commands[env_ids, 1] = torch_rand_float(
            self.cfg.commands.lin_vel_y[0],
            self.cfg.commands.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # ang_vel_yaw
        self.commands[env_ids, 2] = torch_rand_float(
            self.cfg.commands.ang_vel_yaw[0],
            self.cfg.commands.ang_vel_yaw[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # body_height
        self.commands[env_ids, 3] = torch_rand_float(
            self.cfg.commands.body_height_cmd[0],
            self.cfg.commands.body_height_cmd[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # gait frequency
        self.commands[env_ids, 4] = torch_rand_float(
            self.cfg.commands.gait_frequency_cmd_range[0],
            self.cfg.commands.gait_frequency_cmd_range[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # gait phase
        self.commands[env_ids, 5] = 0.5
        # gait offset
        self.commands[env_ids, 6] = 0.0
        # gait bound
        self.commands[env_ids, 7] = 0.0
        # gait duration
        self.commands[env_ids, 8] = 0.5
        # swing height
        self.commands[env_ids, 9] = torch_rand_float(
            self.cfg.commands.footswing_height_range[0],
            self.cfg.commands.footswing_height_range[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # body pitch
        self.commands[env_ids, 10] = torch_rand_float(
            self.cfg.commands.body_pitch_range[0],
            self.cfg.commands.body_pitch_range[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        # body roll
        self.commands[env_ids, 11] = 0.0
        # # stance_width
        # self.commands[env_ids, 12] = torch_rand_float(
        #     self.cfg.commands.stance_width_range[0],
        #     self.cfg.commands.stance_width_range[1],
        #     (len(env_ids), 1),
        #     device=self.device
        # ).squeeze(1)
        # # stance_length
        # self.commands[env_ids, 13] = torch_rand_float(
        #     self.cfg.commands.stance_length_range[0],
        #     self.cfg.commands.stance_length_range[1],
        #     (len(env_ids), 1),
        #     device=self.device
        # ).squeeze(1)

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4] # 频率
            phases = self.commands[:, 5] # 相位偏移
            offsets = self.commands[:, 6] # 偏移量
            bounds = self.commands[:, 7] # 边界
            durations = self.commands[:, 8] # 支撑相占空比
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases
            ]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
            
            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))
            
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  
            for i in range(4):
                foot_phase = torch.remainder(foot_indices[i], 1.0)
                self.desired_contact_states[:, i] = (
                    smoothing_cdf_start(foot_phase) * 
                    (1 - smoothing_cdf_start(foot_phase - 0.5)) +
                    smoothing_cdf_start(foot_phase - 1) * 
                    (1 - smoothing_cdf_start(foot_phase - 0.5 - 1))
                )

    def _reset_root_states(self, env_ids):
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(
                -self.cfg.terrain.x_init_range,
                self.cfg.terrain.x_init_range, 
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 1:2] += torch_rand_float(
                -self.cfg.terrain.y_init_range,
                self.cfg.terrain.y_init_range,
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 0] += self.cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += self.cfg.terrain.y_init_offset
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base yaws
        init_yaws = torch_rand_float(
            -self.cfg.terrain.yaw_init_range,
            self.cfg.terrain.yaw_init_range,
            (len(env_ids), 1),
            device=self.device
        )
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _get_noise_scale_vec(self, cfg):
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((
            torch.ones(3) * noise_scales.gravity * noise_level,
            torch.zeros(self.cfg.commands.num_commands),
            torch.ones(self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
            torch.ones(self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
            torch.zeros(self.num_actions),
        ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec, torch.zeros(self.num_actions)), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec, torch.zeros(4)), dim=0)
        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec, torch.ones(4) * noise_scales.contact_states * noise_level, ), dim=0)

        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 增加刚体状态
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 刚体状态刷新
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:self.num_envs, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # 增加walk this ways buffers
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([
            self.obs_scales.lin_vel, 
            self.obs_scales.lin_vel, 
            self.obs_scales.ang_vel,
            self.obs_scales.body_height_cmd, 
            self.obs_scales.gait_freq_cmd,
            self.obs_scales.gait_phase_cmd, 
            self.obs_scales.gait_phase_cmd,
            self.obs_scales.gait_phase_cmd, 
            self.obs_scales.gait_phase_cmd,
            self.obs_scales.footswing_height_cmd, 
            self.obs_scales.body_pitch_cmd,
            self.obs_scales.body_roll_cmd
            # self.obs_scales.stance_width_cmd,
            # self.obs_scales.stance_length_cmd
        ], device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False, )
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) 
            for name in list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual", "ep_timesteps"]
        }

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1), device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # self.curriculum_thresholds = class_to_dict(self.cfg.curriculum_thresholds)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    #------------ reward functions----------------
    def _reward_tracking_contacts_shaped_force(self):
        '''
            惩罚在摆动相(swing phase)时脚与地面的意外接触
                >当脚应该离地时(desired_contact = 0),如果仍有接触力,给予惩罚
                >当脚应该着地时(desired_contact = 1),不进行惩罚
        '''
        # calculate contact forces on feet
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # get desired_contact_states
        desired_contact = self.desired_contact_states
        reward = 0
        # sum over 4 feet
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        '''
            惩罚在支撑相(stance phase)时脚的滑动
                >当脚应该着地时(desired_contact = 1),如果仍有速度,给予惩罚
                >当脚应该离地时(desired_contact = 0),不进行惩罚
        '''
        # calculate contact forces on feet
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        # get desired_contact_states
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 0.5)))
        
        return reward / 4
    
    def _reward_feet_contact_vel(self):
        '''
            惩罚脚在接近地面时的运动速度,鼓励轻柔着地
                >防止脚猛烈拍击地面 (stomping)
                >鼓励平缓着地 (soft landing)
                >减少冲击力,保护硬件
        '''
        # set reference heights
        reference_heights = 0
        # measure if feet touch the ground
        near_ground = self.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:3], dim=2).view(self.num_envs, -1))
        # penalize high foot velocities when close to the ground
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)

        return rew_contact_vel
    
    def _reward_feet_clearance_cmd_linear(self):
        '''
            在摆动相引导脚按照指定的高度轨迹运动,形成抛物线式的抬腿动作
                >鼓励脚在摆动中期达到命令指定的最大高度
                >保证摆动轨迹平滑
                >避免障碍物,避免过度抬脚
        '''
        # get triangular phases for each foot
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        '''
            获取着地瞬间的垂直冲击速度,惩罚猛烈下落
                >只在实际发生接触时惩罚
                >只惩罚垂直方向的速度
                >使用上一步的速度,避免当前步的接触力影响
                >鼓励零速着地或向上缓冲
        '''
        # calculate vertical foot velocities at previous step
        prev_foot_velocities = self.prev_foot_velocities[:, :, 2].view(self.num_envs, -1)
        # measure if feet are in contact
        contact_states = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        # penalize high downward velocities when in contact
        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        # error = = phases × velocity × (0.5 / frequency)
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.cfg.commands.num_commands >= 13:
            desired_stance_width = self.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        if self.cfg.commands.num_commands >= 14:
            desired_stance_length = self.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.35
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.commands[:, 4]
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        
        # 打印前后脚间距
        # front_x = (footsteps_in_body_frame[:, 0, 0] + footsteps_in_body_frame[:, 1, 0]) / 2  # FR, FL平均
        # rear_x = (footsteps_in_body_frame[:, 2, 0] + footsteps_in_body_frame[:, 3, 0]) / 2   # RR, RL平均
        # stance_length = torch.abs(front_x - rear_x)
        # print(f"Front: {footsteps_in_body_frame[0, :2, 0]}, Rear: {footsteps_in_body_frame[0, 2:, 0]}, Length: {stance_length[0]:.3f}m,", err_raibert_heuristic)
        
        # if self.common_step_counter % 100 == 0:  # 每100步打印一次
        #     print(f"\n=== Step {self.common_step_counter} ===")
        #     print(f"Actual Length: {stance_length[0]:.3f}m")
        #     if self.cfg.commands.num_commands >= 14:
        #         print(f"Desired Length (from cmd): {self.commands[0, 13]:.3f}m")
        #     else:
        #         print(f"Desired Length (default): 0.35m")
        #     print(f"desired_xs_nom: {desired_xs_nom[0]}")
        #     print(f"desired_xs_offset: {desired_xs_offset[0]}")
        #     print(f"Error X: {err_raibert_heuristic[0, :, 0]}")
        #     print(f"Reward: {-torch.sum(torch.square(err_raibert_heuristic[0])):.4f}")
        
        reward = -torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_default_hip_pos(self):
        joint_diff = torch.abs(self.dof_pos[:,0]) + torch.abs(self.dof_pos[:,3]) + torch.abs(self.dof_pos[:,6]) + torch.abs(self.dof_pos[:,9])

        return joint_diff
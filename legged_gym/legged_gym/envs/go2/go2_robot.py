from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.envs.go2.go2_config import Go2Cfg

class Go2Robot( LeggedRobot ):
    cfg : Go2Cfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = None
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        self.init_done = True

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
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

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)
        
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # reset robot states
        self._resample_commands(env_ids)
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)
        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
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
        if self.cfg.commands.command_curriculum:
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]
            self.extras["episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            self.extras["episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            self.extras["episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            self.extras["episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            self.extras["episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            self.extras["episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            self.extras["episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            self.extras["episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            self.extras["episode"]["min_command_frequency"] = torch.min(self.commands[:, 4])
            self.extras["episode"]["max_command_frequency"] = torch.max(self.commands[:, 4])
            self.extras["episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            self.extras["episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            self.extras["episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            self.extras["episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            self.extras["episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            self.extras["episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            if self.cfg.commands.num_commands > 9:
                self.extras["episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                self.extras["episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / curriculum.weights.shape[0]
            self.extras["episode"]["min_actions"] = torch.min(self.actions)
            self.extras["episode"]["max_actions"] = torch.max(self.actions)
            self.extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        
    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )
        # base position
        self.root_states[env_ids] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew
        
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        
        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((  
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:, :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                self.actions
            ), dim=-1)
        else:
            self.obs_buf = torch.cat((  
                self.projected_gravity,
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
        
        # build privileged obs
        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,
                (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
            ), dim=1)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,
                (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
            ), dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,
                (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale
            ), dim=1)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,
                (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale
            ), dim=1)
        assert self.privileged_obs_buf.shape[1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"
   
    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    #------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):
        env_ids = env_ids[env_ids != self.num_envs]
        ret = None
        if len(env_ids) > 0:
            ret = func(env_ids, self.cfg)
        
        return ret
    
    def _randomize_gravity(self, external_force=False):
        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(
                3, 
                dtype=torch.float, 
                device=self.device,
                requires_grad=False
            ) * (max_gravity - min_gravity) + min_gravity
            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _process_rigid_shape_props(self, props, env_id):
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props
    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(
                len(env_ids), 
                dtype=torch.float, 
                device=self.device,
                requires_grad=False
            ) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(
                len(env_ids), 
                3, 
                dtype=torch.float, 
                device=self.device,
                requires_grad=False
            ) * (max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(
                len(env_ids), 
                1, 
                dtype=torch.float, 
                device=self.device,
                requires_grad=False
            ) * (max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(
                len(env_ids), 
                1, 
                dtype=torch.float, 
                device=self.device,
                requires_grad=False
            ) * (max_restitution - min_restitution) + min_restitution








    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
            if self.cfg.commands.pacing_offset:
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + phases]
            else:
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + bounds,
                    self.gait_indices + phases]
            
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
            
            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])


                


    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
  
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec
    
    #------------ reward functions----------------
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 0.5)))
        
        return reward / 4
    
    def _reward_smoothness(self):
        diff = torch.square(self.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0) # ignore first step
        
        return torch.sum(diff, dim=1)
    
    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:3], dim=2).view(self.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)

        return rew_contact_vel
    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0

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
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.cfg.commands.num_commands >= 13:
            desired_stance_width = self.cfg.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        if self.cfg.commands.num_commands >= 14:
            desired_stance_length = self.cfg.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
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

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward



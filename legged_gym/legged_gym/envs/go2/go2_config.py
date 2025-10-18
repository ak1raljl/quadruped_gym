from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = 70
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False
        # additional settings
        observe_gait_commands = True
        observe_two_prev_actions = True
        observe_timing_parameter = True
        observe_clock_inputs = True
        observe_contact_states = False
        observe_yaw = False


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        border_size = 50
        teleport_thresh = 0.3 # height threshold for teleporting robots
        x_init_range = 1.
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.

    class control( LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        use_terminal_body_height = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        kappa_gait_probs = 0.07
        terminal_body_height = 0.05
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 30.0
            tracking_ang_vel = 20.0
            torques = -0.0001
            dof_pos_limits = -10.0
            orientation = -5.0
            # orientation_control = -5.0
            base_height = -180.0
            feet_air_time = 0.0
            collision = -50.0
            tracking_contacts_shaped_force = 40.0
            tracking_contacts_shaped_vel = 40.0
            feet_clearance_cmd_linear = -30.0
            raibert_heuristic = -40.0
            dof_vel = -1e-4
            dof_pos = -0.0
            feet_impact_vel = -0.0
            feet_clearance = -0.0
            feet_clearance_cmd = -0.0
            feet_contact_forces = 0.0
            action_rate = -0.01
            stand_still = -1.0
            default_hip_pos = -2.0

    class commands( LeggedRobotCfg.commands ):
        command_curriculum = True
        resampling_time = 10
        heading_command = False

        num_commands = 14

        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        lin_vel_y = [-1.0, 1.0] # min max [m/s]
        ang_vel_yaw = [-1, 1]   # min max [rad/s]
        heading = [-3.14, 3.14]
        body_height_cmd = [0.32, 0.35]

        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.01, 0.01]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.1, 0.1]
        limit_body_roll = [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 1.0]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.1, 0.1]
        body_roll_range = [-0.0, 0.0]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]

        binary_phases = False

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = False
        max_push_vel_xy = 0.5
        randomize_friction = True
        friction_range = [0.1, 3.0]
        push_interval_s = 15
        gravity_impulse_duration = 0.99

    class normalization( LeggedRobotCfg.normalization ):
        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        clip_actions = 10.0
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            imu = 0.1
            body_height_cmd = 1.0
            gait_freq_cmd = 1.0
            gait_phase_cmd = 1.0
            footswing_height_cmd = 0.15
            body_pitch_cmd = 0.3
            body_roll_cmd = 0.3
            stance_width_cmd = 1.0
            stance_length_cmd = 1.0
    
    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            contact_states = 0.05
            
class Go2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2'
        max_iterations = 1500

import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
from collections import deque
import onnxruntime as ort

from legged_gym.envs import *

default_dof_pos=[0.1,0.8,-1.5 ,-0.1,0.8,-1.5, 0.1,1,-1.5, -0.1,1,-1.5]#默认角度需要与isacc一致


class Sim2simCfg( Go2Cfg ):

    class sim_config:
        # print("{LEGGED_GYM_ROOT_DIR}",{LEGGED_GYM_ROOT_DIR})

        mujoco_model_path = '/home/ak1ra/Quadruped/go2_sim2sim/go2_description/xml/go2.xml'
        
        sim_duration = 60.0
        dt = 0.005
        decimation = 4

    class robot_config:

        kps = np.array(20, dtype=np.double) # PD和isacc内部一致
        kds = np.array(0.5, dtype=np.double)
        tau_limit = 20. * np.ones(12, dtype=np.double) # nm


if __name__ == '__main__':
    policy_model_path = "/home/ak1ra/Quadruped/go2_sim2sim/go2_description/onnx/legged.onnx"
    policy = ort.InferenceSession(policy_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model = mujoco.MjModel.from_xml_path(Sim2simCfg.sim_config.mujoco_model_path)
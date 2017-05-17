#0512_version.

try:
    import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')

import math
import time
import random
import socket
import numpy as np
import sys
from PIL import Image as Im
import datetime
from config import *
from gym import spaces

Shuffle_Count = 0
shuffle_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def add_opts(parser):
    parser.add_argument('--render', default=True, action='store_true')
    parser.add_argument('--action-repeats', type=int, default=1,
                        help="number of action repeats")
    parser.add_argument('--num-cameras', type=int, default=1,
                        help="how many camera points to render; 1 or 2")
    parser.add_argument('--event-log-out', type=str, default=None,
                        help="path to record event log.")
    parser.add_argument('--max-episode-len', type=int, default=200,
                        help="maximum episode len for cartpole")
    parser.add_argument('--use-raw-pixels', default=True, action='store_true',
                        help="use raw pixels as state instead of cart/pole poses")
    parser.add_argument('--render-width', type=int, default=256,
                        help="if --use-raw-pixels render with this width")
    parser.add_argument('--render-height', type=int, default=256,
                        help="if --use-raw-pixels render with this height")
    parser.add_argument('--reward-calc', type=str, default='fixed',
                         help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")

class sim_env(object):
    def __init__(self, opts):
        self.clientID = -1
        self.obj_list = obj_list
        self.handle_obj = HANDLE_OBJ_LIST
        self.handle_robot = HANDLE_ROBOT
        self.handle_joint = HANDLE_JOINT_LIST
        self.handle_tray = HANDLE_TRAY
        self.handle_cam = HANDLE_CAM_LIST
        self.handle_end = HANDLE_END_EFFECTOR
        self.handle_collision = HANDLE_COLLISION_LIST
        self.opts = opts
        self.render = opts.render
        # self.delay = opts.delay if self.gui else 0.0


        # how many cameras to render?
        # if 1 just render from front
        # if 2 render from front and 90deg side
        if opts.num_cameras not in [1, 2]:
            raise ValueError("--num-cameras must be 1 or 2")
        self.num_cameras = opts.num_cameras

        self.repeats = opts.action_repeats
        # whether we are using raw pixels for state or just pole + cart pose

        # in the use_raw_pixels is set we will be rendering
        self.render_width = opts.render_width
        self.render_height = opts.render_height

        # decide observation space
        # in high dimensional case each observation is an RGB images (H, W, 3)
        # we have R repeats and C cameras resulting in (H, W, 3, R, C)
        # final state fed to network is concatenated in depth => (H, W, 3*R*C)
        self.state_shape = (self.render_height, self.render_width, 3,
                            self.num_cameras, self.repeats)

        self.action_space = spaces.Box(opts.joint_angle_low_limit, opts.joint_angle_high_limit,
                                       shape=(1, opts.action_dim))


        # no state until reset.
        self.state = np.empty(self.state_shape, dtype=np.float32)

        self.cur_end_effector_pos = np.empty(3, dtype=np.float32)
        self.end_effector_vel = np.empty(3, dtype=np.float32)
        self.cur_joint_angles = np.empty(6, dtype=np.float32)
        self.joint_angles_vel = np.empty(6, dtype=np.float32)

        if self.opts.use_full_internal_state:
            internal_state = 18
        else:
            internal_state = 9

        self.internal_state = np.empty(internal_state, dtype=np.float32)

        # Simulatior Connect
        self.connect()

    def connect(self):
        print("Connecting to remote API server...", file=sys.stderr)

        lan_ip = socket.gethostbyname(socket.gethostname())

        vrep.simxStopSimulation(-1, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(-1)  # just in case, close all opened connections

        self.clientID = vrep.simxStart(lan_ip, 19997, True, True, 5000, 5)

        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        # Register all handles
        print("Registering all handle of scene... (Rendering = {})".format(self.render), file=sys.stderr)

        # if scene modified, Call this function to update handle list < DEBUG MODE !!!>
        #_chk_handle_list()

        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)

        # Is Render?
        err = vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_display_enabled, self.render,
                                           vrep.simx_opmode_oneshot)

        # Data streaming
        self._data_streaming()

        print(datetime.datetime.now(), "  Complete registered. Simulation ready", file=sys.stderr)

    def shutdown(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        #self.data_streaming(option=False) # Need??
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(self.clientID)

    def set_state_element(self, repeat):
        for camera_idx in range(0, self.num_cameras):
            _, self.state[:, :, :, camera_idx, repeat - 1] = self.get_cam(camera_idx)
        # image.transpose(Im.FLIP_TOP_BOTTOM).save('C:\\tensorflow_code\\Vrep\\approaching_code\\DDPG\imgs\\temp.bmp','bmp')

    def step(self, action, target_obj_idx):
        action = list(action.reshape(-1,))
        self.movejoint(action)
        self.set_state_element(1)
        self.steps += 1
        self.done = False

        obj_pos = list(self.get_obj_pos(target_obj_idx))[1]
        endEffector_pos = self.get_end_pos()

        obj_pos[2] = obj_pos[2] + 0.5  # check!!!!!

        dist = np.linalg.norm(np.array(obj_pos) - np.array(endEffector_pos))
        dist = math.sqrt(dist)

        #######################
        reward = 1.0 / dist

        if obj_pos[2] > endEffector_pos[2]:
            reward = -100

            #        print('dist=',dist)
            #        reward = math.sqrt(1.0/dist)
            #        print('reward=', reward)
            #        reward = math.exp(reward) - 1

        ########################


        if self.opts.use_full_internal_state:
            prev_end_effector_pos = self.cur_end_effector_pos
            self.cur_end_effector_pos = np.asarray(endEffector_pos)
            self.end_effector_vel = self.cur_end_effector_pos - prev_end_effector_pos
            prev_joint_angles = self.cur_joint_angles
            self.cur_joint_angles = self.get_joint()
            self.joint_angles_vel = self.cur_joint_angles - prev_joint_angles
            self.internal_state = np.concatenate(
                (self.cur_joint_angles, self.joint_angles_vel, self.cur_end_effector_pos, self.end_effector_vel))
        else:
            self.internal_state = np.concatenate((self.cur_joint_angles, self.cur_end_effector_pos))

        collision_type = self.collision_read()

        if collision_type == 1:
            reward = -100.0
            self.done = True
            print('Tray collision or Object collision detected')
        elif collision_type == 2:
            reward = -100.0
            self.done = True
            print('Self collision detected')
        elif collision_type == 0:
            self.done = False

        return np.copy(self.state), reward, self.done

    def reset(self):
        self.steps = 0
        self.done = False
        self.movejoint([0, 0, 0, 0, 0, 0]) # Go to initial angles

        self.set_state_element(repeat=1)

        self.cur_end_effector_pos = np.asarray(self.get_end_pos()) # initial end_effector_pos
        self.end_effector_vel[:] = [0, 0, 0]
        self.cur_joint_angles[:] = [0, 0, 0, 0, 0, 0]
        self.joint_angles_vel[:] = [0, 0, 0, 0, 0, 0]

        if self.opts.use_full_internal_state:
            self.internal_state = np.concatenate(
                (self.cur_joint_angles, self.joint_angles_vel, self.cur_end_effector_pos, self.end_effector_vel))
        else:
            self.internal_state = np.concatenate((self.cur_joint_angles, self.cur_end_effector_pos)) # dim 12

        return np.copy(self.state)

    # For update handle list, handle list check function. <Check on Debug mode>
    def _chk_handle_list(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        self.handle_robot = vrep.simxGetObjectHandle(self.clientID, 'UR3', vrep.simx_opmode_blocking)[1]
        # Tray Handle
        self.handle_tray = vrep.simxGetObjectHandle(self.clientID, 'Tray', vrep.simx_opmode_blocking)[1]
        # End effector Handle
        self.handle_end = vrep.simxGetObjectHandle(self.clientID, 'End_Effector', vrep.simx_opmode_blocking)[1]
        # Object Handle
        for i in obj_list:
            self.handle_obj.append(vrep.simxGetObjectHandle(self.clientID, i, vrep.simx_opmode_blocking)[1])
        # UR3 Joint Handle
        for i in range(6):
            self.handle_joint.append(vrep.simxGetObjectHandle(self.clientID, "UR3_joint{}".format(i + 1),
                                                              vrep.simx_opmode_blocking)[1])
        # Vision camera Handle
        for i in range(2):
            self.handle_cam.append(vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor_0%d' % (i + 1),
                                                            vrep.simx_opmode_blocking)[1])
        # Collision Handle
        for i in range(124):
            self.handle_collision.append(vrep.simxGetCollisionHandle(self.clientID, 'Collision%d' % i,
                                                                     vrep.simx_opmode_blocking)[1])

    def _data_streaming(self, option = True):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        if option:
            op = vrep.simx_opmode_streaming
        else :
            op = vrep.simx_opmode_discontinue

        #Object postion
        for i in self.handle_obj:
            vrep.simxGetObjectPosition(self.clientID, i, self.handle_robot, op)
        #Dummy position
        vrep.simxGetObjectPosition(self.clientID, self.handle_end, self.handle_robot, op)
        #vision
        for i in self.handle_cam :
            vrep.simxGetVisionSensorImage(self.clientID, i, 0, op)
        #collision
        for i in self.handle_collision:
            vrep.simxReadCollision(self.clientID, i, op)
        #UR3 Joint
        for i in self.handle_joint:
            vrep.simxGetJointPosition(self.clientID, i, op)

    def get_end_pos(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        _, end_pos = vrep.simxGetObjectPosition(self.clientID, self.handle_end,
                                              self.handle_robot, vrep.simx_opmode_buffer)

        return end_pos # return as degree value

    def get_obj_pos(self, obj_idx): # only for 1 objects
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        if obj_idx in range(len(self.handle_obj)):
            return vrep.simxGetObjectPosition(self.clientID, self.handle_obj[obj_idx], self.handle_robot,
                                              vrep.simx_opmode_buffer)

        else:
            raise ValueError("object index must be 0 ~ 9")

    def shuffle_obj(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        Past_number =[]
        change_num = [-3, -2, -1, 1, 2, 3]

        for i in self.handle_obj:
            SufflePosition = []
            L_x = 0.075 + 0.02*(2*random.random() - 1)
            L_y_positive = (0.25 / 2) * (1 * random.random())
            L_y_negative = (0.25 / 2) * (-1 * random.random())
            L_z_positive = (0.70 / 2) * (1 * random.random())
            L_z_negative = (0.70 / 2) * (-1 * random.random())

            Suffle_part = random.randrange(1, 5)
            if(Suffle_part == Past_number):
                Suffle_part = (Suffle_part + random.choice(change_num))%4
            Past_number = Suffle_part


            if (Suffle_part == 1):
                SufflePosition.append(L_x)
                SufflePosition.append(L_y_positive)
                SufflePosition.append(L_z_positive)
            elif (Suffle_part == 2):
                SufflePosition.append(L_x)
                SufflePosition.append(L_y_negative)
                SufflePosition.append(L_z_positive)
            elif (Suffle_part == 3):
                SufflePosition.append(L_x)
                SufflePosition.append(L_y_positive)
                SufflePosition.append(L_z_negative)
            else:
                SufflePosition.append(L_x)
                SufflePosition.append(L_y_negative)
                SufflePosition.append(L_z_negative)

            vrep.simxSetObjectPosition(self.clientID, i, self.handle_tray, SufflePosition, vrep.simx_opmode_oneshot)
            time.sleep(0.05)

    def movejoint(self, degree):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        if len(degree) != 6:
            raise ValueError("The input's length not matched the number of joint")

        else :
            for idx, deg in enumerate(degree):
                target_joint_rad = math.radians(deg)
                vrep.simxSetJointTargetPosition(self.clientID, self.handle_joint[idx], target_joint_rad,
                                                vrep.simx_opmode_blocking)

    def get_cam(self, cam_idx):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        if cam_idx not in [0, 1]:
            raise ValueError("camera_idx must be 0 or 1")

        # TODO : image PIL.IMage????
        err, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.handle_cam[cam_idx], 0,
                                                          vrep.simx_opmode_buffer)

        image_byte_array = np.asarray(image).astype('uint8')
        result_image = Im.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)

        image = np.reshape(image, (resolution[0], resolution[1], 3))
        np.transpose(image, (1, 0, 2))
        image = image + 128
        image = image / 255

        return result_image, image

    def get_joint(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        cur_joint = []

        for i in self.handle_joint:
            _, rad_joint = vrep.simxGetJointPosition(self.clientID, i, vrep.simx_opmode_buffer)
            cur_joint.append(math.degrees(rad_joint))

        return cur_joint # return as degree value

    def collision_read(self):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        for i in range(124):
                err, collision = vrep.simxReadCollision(self.clientID, self.handle_collision[i], vrep.simx_opmode_buffer)

                if err == vrep.simx_return_ok:
                    #print('collision information imported')
                    if (collision == True and i < 92):
                        #print('Tray collision or Object collision detected')
                        return 1  # tray collision or object collision detected => break
                    elif (collision == True and i > 91):
                        #print('Self collision detected')
                        return 2  # self collision detected => negative reward
                    else:
                        break

                elif err == vrep.simx_return_novalue_flag:
                    print("no collision information")
                else:
                    print('Error')
                    print(err)

        #print('No Collision')

        return 0  # no collision

    def remove_obj(self, remove_obj_idx):
        assert self.clientID != -1, 'Failed to connect to the V-rep server'

        vrep.simxSetObjectPosition(self.clientID, self.handle_obj[remove_obj_idx],
                                   self.handle_robot, [0.8, 0, 0],
                                   vrep.simx_opmode_oneshot)

    def send(self):
        vrep.simxSynchronousTrigger(self.clientID)
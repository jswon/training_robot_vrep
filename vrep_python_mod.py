#0511_version.

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



obj_list = ['O_00_Big_USB', 'O_01_Black_Tape', 'O_02_Blue_Glue', 'O_03_Brown_Box', 'O_04_Green_Glue',
            'O_05_Pink_Box', 'O_06_Red_Cup', 'O_07_Small_USB', 'O_08_White_Tape', 'O_09_Yellow_Cup']

Shuffle_Count = 0
shuffle_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class sim_env():
    def __init__(self):
        self.clientID = -1
        self.obj_handle = []
        self.rob_handle = []
        self.joint_handle = []
        self.tray_handle = []
        self.cam_handle = []
        self.end_effector = []
        self.collision_handle = []
        self.FLAG_CONNECT = False
        # Sim_env Connect
        self.connect()

    def connect(self):
        print("Connecting to remote API server...", file=sys.stderr)

        lan_ip = socket.gethostbyname(socket.gethostname())
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart(lan_ip, 19997, True, True, 5000, 5)
        self.FLAG_CONNECT = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        if self.FLAG_CONNECT is not vrep.simx_return_ok :
            print('Not Connected. Please confirm the simulation', file=sys.stderr)

        else:
            # UR3 Handle
            err, self.rob_handle = vrep.simxGetObjectHandle(self.clientID, 'UR3', vrep.simx_opmode_oneshot_wait)

            # Tray Handle
            err, self.tray_handle = vrep.simxGetObjectHandle(self.clientID, 'Tray', vrep.simx_opmode_oneshot_wait)

            # End effector Handle
            err, self.end_effector = vrep.simxGetObjectHandle(self.clientID, 'End_Effector', vrep.simx_opmode_oneshot_wait)

            # Object Handle
            for i in obj_list:
                err, handle = vrep.simxGetObjectHandle(self.clientID, i, vrep.simx_opmode_oneshot_wait)
                self.obj_handle.append(handle)

            # UR3 Joint Handle
            for i in range(6):
                err, joint_handle = vrep.simxGetObjectHandle(self.clientID, "UR3_joint{}".format(i+1),
                                                             vrep.simx_opmode_oneshot_wait)
                self.joint_handle.append(joint_handle)

            # Vision camera Handle
            for i in range(2):
                err, cam = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor_0%d' %(i+1), vrep.simx_opmode_oneshot_wait)
                self.cam_handle.append(cam)

            # Collision Handle
            for i in range(14):
                err, collision = vrep.simxGetCollisionHandle(self.clientID, 'Collision%d' % i, vrep.simx_opmode_oneshot_wait)
                self.collision_handle.append(collision)

            # Start Simulation
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

            # Data streaming
            self.data_streaming()

            print("Environment ready", file=sys.stderr)


    def shutdown(self):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
        else:
            self.data_streaming(option=False)
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
            time.sleep(1)
            vrep.simxFinish(self.clientID)

    def data_streaming(self, option = True):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
        else:
            if option :
                op = vrep.simx_opmode_streaming
            else :
                op = vrep.simx_opmode_discontinue

            #Object postion
            for i in self.obj_handle:
                vrep.simxGetObjectPosition(self.clientID, i, self.rob_handle, op)

            #Dummy position
            vrep.simxGetObjectPosition(self.clientID, self.end_effector, self.rob_handle, op)

            #vision
            for i in self.cam_handle :
                vrep.simxGetVisionSensorImage(self.clientID, i, 0, op)
            #collision
            for i in self.collision_handle:
                vrep.simxReadCollision(self.clientID, i, op)
            #UR3 Joint
            for i in self.joint_handle:
                vrep.simxGetJointPosition(self.clientID, i, op)


    def get_obj_pos(self, obj_idx): # only for 1 objects

        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
        else:
            if obj_idx in range(len(self.obj_handle)):
                err, obj_pos = vrep.simxGetObjectPosition(self.clientID, self.obj_handle[obj_idx], self.rob_handle,
                                                          vrep.simx_opmode_buffer)
                return obj_pos


            else:
                raise ValueError("object index must be 0 ~ 9")




    def shuffle(self):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
        else:
            Past_number =[]
            change_num = [-3, -2, -1, 1, 2, 3]
            for i in self.obj_handle:
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

                vrep.simxSetObjectPosition(self.clientID, i, self.tray_handle, SufflePosition, vrep.simx_opmode_oneshot)
                time.sleep(0.05)

    def movejoint(self, degree):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
            sys.exit()
        elif len(degree) != 6 :
            raise ValueError("The input's length not matched the number of joint")
        else:
            for idx, deg in enumerate(degree):
                target_joint_rad = math.radians(deg)
                vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[idx], target_joint_rad,
                                                vrep.simx_opmode_blocking)

    def get_cam_img(self, cam_idx):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
            sys.exit()
        else:
           if cam_idx in [0, 1]:
               while (vrep.simxGetConnectionId(self.clientID) != -1):
                   err, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.cam_handle[cam_idx], 0,
                                                                          vrep.simx_opmode_buffer)

                   # Save image to txt file => But it have a problem
                   if err == vrep.simx_return_ok:
                       # save image information list
                       #print('vision information imported')
                       image_byte_array = np.asarray(image).astype('uint8')
                       result_image = Im.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
                       return result_image

                   elif err == vrep.simx_return_novalue_flag:
                       print("no vision information")
                   else:
                       print('Error')
                       print(err)
           else:
               raise ValueError("camera_idx must be 0 or 1")

    def get_joint(self):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
            sys.exit()
        else:
            cur_joint = []

            for i in self.joint_handle:
                ret, rad_joint = vrep.simxGetJointPosition(self.clientID, i, vrep.simx_opmode_buffer)
                cur_joint.append(math.degrees(rad_joint))
            return cur_joint # return as degree value

    # def collsion_Read(self):
    #     if self.FLAG_CONNECT is not vrep.simx_return_ok:
    #         print('Not Connected. Please check the simulator', file=sys.stderr)
    #         sys.exit()
    #     else:
    #         for i in self.collision_handle:
    #             check = []
    #             while (vrep.simxGetConnectionId(self.clientID) != -1):
    #                 err, collision = vrep.simxReadCollision(self.clientID, i, vrep.simx_opmode_buffer)
    #
    #                 if err == vrep.simx_return_ok:
    #                     print('collision information imported')
    #                     del check[:]
    #                     check.append(collision)
    #                     break
    #
    #                 elif err == vrep.simx_return_novalue_flag:
    #                     print("no collision information")
    #                 else:
    #                     print('Error')
    #                     print(err)
    #
    #             if(check[0] == 1):
    #                 print('Collision detected')
    #                 return 1
    #         print('No Collision')
    #         return 0

    def render(self, isrender = True):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)
            sys.exit()
        else:
            err = vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_display_enabled, isrender, vrep.simx_opmode_oneshot)

    def remove_obj(self):
        if self.FLAG_CONNECT is not vrep.simx_return_ok:
            print('Not Connected. Please check the simulator', file=sys.stderr)

        elif not shuffle_num:
            for i in range(10):
                shuffle_num.append(i)
        random_num = random.randrange(0, len(shuffle_num))
        vrep.simxSetObjectPosition(self.clientID, self.obj_handle[shuffle_num[random_num]],
                                   self.rob_handle, [0.8, 0, 0], vrep.simx_opmode_oneshot)

        del shuffle_num[random_num]
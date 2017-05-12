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
from PIL import Image as Im


Objects_name = ['O_00_Big_USB', 'O_01_Black_Tape', 'O_02_Blue_Glue', 'O_03_Brown_Box', 'O_04_Green_Glue',
                    'O_05_Pink_Box', 'O_06_Red_Cup', 'O_07_Small_USB', 'O_08_White_Tape', 'O_09_Yellow_Cup' ]
Objects_handle = []
UR3_handle = []
UR3_joint_handle = []
Tray_handle = []
vision_sensor = []
end_effector = []
collision_handle = []
clientID =[]
Shuffle_Count = 0
shuffle_num = [0,1,2,3,4,5,6,7,8,9]

def connect():
    print("Connecting to remote API server")
    del clientID[:]
    lan_ip = socket.gethostbyname(socket.gethostname())
    vrep.simxFinish(19997)  # just in case, close all opened connections
    #ID = vrep.simxStart('192.168.0.3', 19997, True, True, 5000, 5) #IP address
    ID = vrep.simxStart(lan_ip, 19997, True, True, 5000, 5) #IP address
    clientID.append(ID)
    vrep.simxStartSimulation(clientID[0], vrep.simx_opmode_oneshot)

    if (clientID == -1):
        print('Not connected')
        print('Connect again')
    else:
        del Objects_handle[:]
        del UR3_joint_handle[:]
        del UR3_handle[:]
        del Tray_handle[:]
        del vision_sensor[:]
        del collision_handle[:]
        del end_effector[:]

        #Object Handle
        for i in Objects_name:
            err, Handle = vrep.simxGetObjectHandle(clientID[0], i, vrep.simx_opmode_blocking)
            Objects_handle.append(Handle)
        #UR3 Handle
        err, UR3 = vrep.simxGetObjectHandle(clientID[0], 'UR3', vrep.simx_opmode_blocking)
        UR3_handle.append(UR3)
        #Tray Handle
        err, Tray = vrep.simxGetObjectHandle(clientID[0], 'Tray', vrep.simx_opmode_blocking)
        Tray_handle.append(Tray)
        #End Effector Handle - endeffector
        err, end = vrep.simxGetObjectHandle(clientID[0], 'End_Effector', vrep.simx_opmode_blocking)
        end_effector.append(end)
        #UR3 Joint Handle
        for i in  range(1,7):
            err, Joint_handle = vrep.simxGetObjectHandle(clientID[0], "UR3_joint%d" % i,
                                                             vrep.simx_opmode_oneshot_wait)
            UR3_joint_handle.append(Joint_handle)
        #Vision camera Handle
        for i in range(1,3):
            err, vision = vrep.simxGetObjectHandle(clientID[0], 'Vision_sensor_0%d' % i, vrep.simx_opmode_blocking)
            vision_sensor.append(vision)
        #Collision Handle
        for i in range(0,14):
            err, collision = vrep.simxGetCollisionHandle(clientID[0],'Collision%d' %i,vrep.simx_opmode_oneshot_wait)
            collision_handle.append(collision)
        #Start Simulation
        vrep.simxStartSimulation(clientID[0], vrep.simx_opmode_oneshot)

def shutdown():
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else :
        vrep.simxStopSimulation(clientID[0], vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(clientID[0])
        del Objects_handle[:]
        del UR3_joint_handle[:]
        del UR3_handle[:]
        del Tray_handle[:]
        del end_effector[:]
        del clientID[:]

def startDataStreaming():
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        #Position
        for i in Objects_handle :
            vrep.simxGetObjectPosition(clientID[0], i, UR3_handle[0], vrep.simx_opmode_streaming)

        #Dummy position
        vrep.simxGetObjectPosition(clientID[0], end_effector[0], UR3_handle[0], vrep.simx_opmode_streaming)
        #vision
        for i in vision_sensor :
            vrep.simxGetVisionSensorImage(clientID[0], i, 0, vrep.simx_opmode_streaming)
        #collision
        for i in collision_handle :
            vrep.simxReadCollision(clientID[0], i, vrep.simx_opmode_streaming)
        #UR3 Joint
        for i in UR3_joint_handle :
            vrep.simxGetJointPosition(clientID[0],i,vrep.simx_opmode_streaming)


def stopDataStreaming():
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        #Position
        for i in Objects_handle :
            vrep.simxGetObjectPosition(clientID[0], i, UR3_handle, vrep.simx_opmode_discontinue)
        #Dummy position
        vrep.simxGetObjectPosition(clientID[0], end_effector[0], UR3_handle, vrep.simx_opmode_discontinue)
        #vision
        for i in vision_sensor :
            vrep.simxGetVisionSensorImage(clientID[0], i, 0, vrep.simx_opmode_discontinue)
        #collision
        for i in collision_handle :
            vrep.simxReadCollision(clientID[0], i, vrep.simx_opmode_discontinue)


def getPosition(Objects_num): # only for objects
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        if(Objects_num <10 and Objects_num >= 0):
            Object_Handle = Objects_handle[Objects_num]
        else:
            print('wrong object number')
            return 0

        while (vrep.simxGetConnectionId(clientID[0]) != -1):

            err, Output_Postion = vrep.simxGetObjectPosition(clientID[0], Object_Handle, UR3_handle[0], vrep.simx_opmode_buffer)

            if err == vrep.simx_return_ok:
                time.sleep(1)
                print('Position')
                print(Output_Postion)
                return Output_Postion

            elif err == vrep.simx_return_novalue_flag:
                print("no value")


            else:
                print("not yet")

def getEndEffector(): # only for objects
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        Object_Handle = end_effector[0]

        while (vrep.simxGetConnectionId(clientID[0]) != -1):

            err, Output_Postion = vrep.simxGetObjectPosition(clientID[0], Object_Handle, UR3_handle[0], vrep.simx_opmode_buffer)

            if err == vrep.simx_return_ok:
                time.sleep(1)
                print('Position')
                print(Output_Postion)
                return Output_Postion

            elif err == vrep.simx_return_novalue_flag:
                print("no value")


            else:
                print("not yet")


def shuffleObject():##############################check
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        Past_number =[]
        change_num = [-3, -2, -1, 1, 2, 3]
        for i in Objects_handle:
            SufflePosition = []
            L_x = 0.075 + 0.02*(2*random.random() - 1)
            L_y_positive = (0.25 / 2) * (1 * random.random())
            L_y_negative = (0.25 / 2) * (-1 * random.random())
            L_z_positive = (0.70 / 2) * (1 * random.random())
            L_z_negative = (0.70 / 2) * (-1 * random.random())

            Suffle_part = random.randrange(1,5)
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

            vrep.simxSetObjectPosition(clientID[0], i, Tray_handle[0], SufflePosition, vrep.simx_opmode_oneshot)
            time.sleep(0.05)



def movejoint(Degree):  # move joint of ur3
    #############################################################################check
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    elif(len(Degree) != 6) :
        print('Wrong input')
    else:
        num = 0  # maximum of num is 6
        for i in Degree:
            JointTargetPosition = math.radians(i)
            vrep.simxSetJointTargetPosition(clientID[0], UR3_joint_handle[num], JointTargetPosition,
                                            vrep.simx_opmode_blocking)
            num = num + 1




def getimageinformation(sensor_index):
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
       if(sensor_index == 1 or sensor_index == 2) :
           while (vrep.simxGetConnectionId(clientID[0]) != -1):
               err, resolution, image = vrep.simxGetVisionSensorImage(clientID[0], vision_sensor[sensor_index - 1], 0,
                                                                      vrep.simx_opmode_buffer)

               # Save image to txt file => But it have a problem
               if err == vrep.simx_return_ok:
                   # save image information list
                   print('vision information imported')
                   image_byte_array = np.asarray(image).astype('uint8')
                   result_image = Im.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
                   return result_image

               elif err == vrep.simx_return_novalue_flag:
                   print("no vision information")
               else:
                   print('Error')
                   print(err)
       else:
           print('Wrong index number')



def all_jointangle_list(): ########### check
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        angle_list = []

        for i in UR3_joint_handle:
            ret, UR3_joint_CurrentAngle = vrep.simxGetJointPosition(clientID[0], i ,
                                                                    vrep.simx_opmode_buffer)
            angle_list.append(math.degrees(UR3_joint_CurrentAngle))
        return angle_list #return as degree value

def collsion_Read():
    if (not clientID):
        print('connect first')
    elif(clientID[0] == -1):
        print('connect first')
    else:
        for i in collision_handle :
            check = []
            while (vrep.simxGetConnectionId(clientID[0]) != -1):
                err, collision = vrep.simxReadCollision(clientID[0], i, vrep.simx_opmode_buffer)

                if err == vrep.simx_return_ok:
                    print('collision information imported')
                    del check[:]
                    check.append(collision)
                    break

                elif err == vrep.simx_return_novalue_flag:
                    print("no collision information")
                else:
                    print('Error')
                    print(err)

            if(check[0] == 1):
                print('Collision detected')
                return 1
        print('No Collision')
        return 0


def rendering(True_or_False):
    if (not clientID):
        print('connect first')
    elif (clientID[0] == -1):
        print('connect first')
    else:
        err = vrep.simxSetBooleanParameter(clientID[0],vrep.sim_boolparam_display_enabled , True_or_False, vrep.simx_opmode_oneshot)


def kickoff() :
    if (not shuffle_num):
        for i in range(10):
            shuffle_num.append(i)
    random_num = random.randrange(0,len(shuffle_num))
    vrep.simxSetObjectPosition(clientID[0],Objects_handle[shuffle_num[random_num]],UR3_handle[0],[0.8,0,0],vrep.simx_opmode_oneshot)
    del shuffle_num[random_num]


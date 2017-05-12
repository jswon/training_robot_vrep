import vrep_python_mod
from PIL import Image
import time
try :
    env = vrep_python_mod.sim_env()

#    joint_ang = env.get_joint()

#    print(joint_ang)

    #env.movejoint([90, 0, 0, 90, 90, 0])

#    img0 = env.get_cam_img(0)

#    img0.show('test')
    time.sleep(5)
    obj1 = env.get_obj_pos(0)
    print(obj1)
    while True :
        a =1
    env.shutdown()


except :
    print("test")
    env.shutdown()
    # TODO : Force quit  -> Call saver !!!
    # Function()

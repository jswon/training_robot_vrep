import vrep_python_mod
from PIL import Image
import time
import datetime
try :

    print(datetime.datetime.now())
    env = vrep_python_mod.sim_env(isrender=False)

    tar_joint = [0, 0, 10, 10, 10, 0]
    env.movejoint(tar_joint)

    print('movejoint')

    print('target :{}'.format(tar_joint))
    print('result :{}'.format(env.get_joint()))

    img0 = env.get_cam_img(0)
    img0.show('img_cut_0')

    tar_joint = [0, 0, 20, 20, 20, 0]
    env.movejoint(tar_joint)
    img0 = env.get_cam_img(0)
    img0.show('img_cut_1')
    print('target :{}'.format(tar_joint))
    print(env.get_joint())

    tar_joint = [0, 0, 30, 30, 30, 0]
    env.movejoint(tar_joint)
    img0 = env.get_cam_img(0)
    img0.show('img_cut_2')
    print('target :{}'.format(tar_joint))
    print('result :{}'.format(env.get_joint()))

    obj1 = env.get_obj_pos(0)
    print(obj1)

    print(datetime.datetime.now())

    while True:
        pass

except :
    print("non standard exit")
    env.shutdown()
    # TODO : Force quit  -> Call saver !!!
    # Function()

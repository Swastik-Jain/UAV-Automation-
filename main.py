import os
import time
import math
import numpy as np
import cv2
import airsim
import tensorflow as tf
from keras import layers, models, optimizers


# ------------------------
#CONFIG
# ------------------------
REMOTE_IP = "192.168.209.21"
PORT = 41451
IMG_H, IMG_W = 84, 84        # CNN input
ACTION_DIM = 2               # continuous vx, vy (m/s)
ACTION_SCALE = 3.0           # scale output to real m/s
DT = 0.3                     # command duration (seconds)
ROLLOUT_STEPS = 1024         # steps per update
MINIBATCH_SIZE = 64
UPDATE_EPOCHS = 8
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
MAX_TRAIN_ITERS = 1000
SAVE_DIR = "ppo_airsim_checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------   AIRSIM ENV WRAPPER -----------------------
# -----------------------
class AirSimDroneEnv:
    def __init__(self, ip=REMOTE_IP, port=PORT, img_h=IMG_H, img_w=IMG_W, dt=DT):
        self.client = airsim.MultirotorClient(ip=REMOTE_IP,port=PORT)
        print("Connecting to AirSim at", ip, port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.img_h = img_h
        self.img_w = img_w
        self.dt = dt
        self.reset()

    def _get_image(self):
        # returns HxWx3 float32 [0,1]
        responses = self.client.simGetImages([ #gets image from AirSim
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False) 
        ])
        if len(responses) == 0: #checks if image is returned from AirSim
            raise RuntimeError("No image returned from AirSim.")
        
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8) #converts image to numpy array
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)  #resizes image to desired dimensions
        img_rgb = cv2.resize(img_rgb, (self.img_w, self.img_h))  #resizes image to desired dimensions
        return img_rgb / 255.0    #normalizes pixel in range of [0,1]

    def reset(self):
        try:
            self.client.reset()
            time.sleep(0.2)
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-5, 2).join()
            time.sleep(0.2)

        except Exception as e:
            print("Reset warning:", e)
            
        return self._get_image()
        
    def compute_reward(self,vx,collided):

        if collided:
            return -10.0 ,True,{'collision' :True }
        else:
            reward = 0.01+0.001*max(0.0,vx)
            return reward,False,{'collision' :False }
        
    def step(self,action):
        vx,vy = action

        try :
            self.client.moveByVelocityAsync(vx,vy,0,duration=self.dt).join()
        except Exception as e:
            print("Step warning:", e)

        obs = self._get_image()

        colinfo = self.client.getCollisionInfo()
        collided = colinfo.has_collided

        reward ,done ,info = self.compute_reward(vx,collided)
        return obs,reward,done,info
        
    def close(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
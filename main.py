# import gymnasium as gym
# from gym import spaces
# import time 
import airsim
import numpy as np
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


client = airsim.MultirotorClient()
client.confirmConnection()

# Takeoff for safety
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Move forward 5s
client.moveByVelocityAsync(5, 0, 0, 5).join()

# Move right 5s
client.moveByVelocityAsync(0, 5, 0, 5).join()

# Move up for 3s
client.moveByVelocityAsync(0, 0, -3, 3).join()

# Rotate drone while moving forward
client.rotateByYawRateAsync(30, 5).join()
client.moveByVelocityAsync(3, 0, 0, 5).join()
0


client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()
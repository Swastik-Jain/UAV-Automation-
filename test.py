"""
ppo_airsim_tf2.py
TensorFlow2 PPO training for AirSim City (remote).
Replace REMOTE_IP with your AirSim server IP.
"""

import os
import time
import math
# Removed incorrect import
import numpy as np
import cv2
import airsim
import tensorflow as tf
from keras import layers, models, optimizers
from collections import deque

# -----------------------
# CONFIG
# -----------------------
# from airsim_config import REMOTE_IP, REMOTE_PORT
REMOTE_IP = "127.0.0.1"
REMOTE_PORT = 41451

PORT = REMOTE_PORT
NUM_DRONES = 3

#------------Environment Parameters------------
# --- Environment ---
IMG_H, IMG_W = 84, 84
ACTION_DIM = 2
ACTION_SCALE = 3.0
DT = 0.3

MIN_DIST_TO_GOAL = 5.0
GOAL_RADIUS = 2.0

# --- Reward Coefficients ---
GOAL_REWARD = 5.0
COLLISION_PENALTY = -5.0
TIME_LIMIT_PENALTY = -1.0
TIME_PENALTY = -0.01
PROGRESS_COEFF = 2.5
ALIGNMENT_COEFF = 0.1
SMOOTHNESS_PENALTY_WEIGHT = 0.01

# --- PPO Hyperparameters ---
ROLLOUT_STEPS = 512           # Steps per agent per update
MINIBATCH_SIZE = 64 * NUM_DRONES
UPDATE_EPOCHS = 8
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
MAX_TRAIN_ITERS = 1000
SAVE_DIR = "ppo_airsim_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# AIRSIM ENV WRAPPER
# -----------------------
class AirSimDroneEnv:
    def __init__(self,client,vehicle_name, ip=REMOTE_IP, port=PORT, img_h=IMG_H, img_w=IMG_W, dt=DT):
        self.vehicle_name = vehicle_name
        self.client = client
        self.img_h = img_h
        self.img_w = img_w
        self.dt = dt
        self.prev_dist = None
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        # episode management
        # self.max_steps = 500  <------------------------------------------->
        self.step_count = 0

        
    def _get_image(self):
        # returns HxWx3 float32 [0,1]
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ],vehicle_name=self.vehicle_name)
        if len(responses) == 0:
            raise RuntimeError("No image returned from AirSim.")
        resp = responses[0]
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        # some builds could return empty image if camera not ready
        if img1d.size == 0:
            # return black image to avoid crash
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)
        img = img1d.reshape(resp.height, resp.width, 3)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        return img

    def reset(self):
        # reset simulation / drone pose
        while True:
            x = np.random.uniform(-20,20)
            y = np.random.uniform(-20,20)
            z = -5.0

            # Goal Co ordinates
            gx = np.random.uniform(-20,20)
            gy = np.random.uniform(-20,20)
            gz = -5.0

            if np.linalg.norm(np.array([x,y,z]) - np.array([gx,gy,gz])) > MIN_DIST_TO_GOAL:
                break
            
        quat = airsim.to_quaternion(0,0,0)
        pose = airsim.Pose(airsim.Vector3r(x,y,z),quat)
        self.client.simSetVehiclePose(pose,ignore_collision=True,vehicle_name=self.vehicle_name)

        self.goal=np.array([gx,gy,gz],dtype=np.float32)
        self.prev_action = np.zeros(ACTION_DIM,dtype=np.float32)
        self.step_count = 0

        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        

        if(self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided):
            print("Spawned in a collided state! Retrying reset...")
            return self.reset()
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        # Initialize previous horizontal distance after altitude settle
        state_now = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos_now = np.array([
            state_now.kinematics_estimated.position.x_val,
            state_now.kinematics_estimated.position.y_val,
            state_now.kinematics_estimated.position.z_val
        ], dtype=np.float32)
        self.prev_dist = np.linalg.norm((pos_now[:2] - self.goal[:2]))
        
        return self._get_image()

    def compute_reward(self,state,collided,action):
        
        pos = np.array([state.kinematics_estimated.position.x_val,
                        state.kinematics_estimated.position.y_val,
                        state.kinematics_estimated.position.z_val], dtype=np.float32)
        # Use horizontal distance only since control is in x,y
        pos_xy = pos[:2]
        goal_xy = self.goal[:2]
        dist = np.linalg.norm(pos_xy - goal_xy)
        # Ensure prev_dist is initialized (e.g., first step after reset)
        if self.prev_dist is None:
            self.prev_dist = dist
        progress = self.prev_dist - dist
        
        if dist < GOAL_RADIUS:  #goal reached
            print("Goal reached")
            return GOAL_REWARD,True,{'goal_reached':True}
        
        if collided: #collided 
            print("Collided")
            return COLLISION_PENALTY ,True,{'collision' :True }
        
        reward = 0.0
        # per-step time penalty to encourage faster completion
        reward -= TIME_PENALTY

        # progress toward goal in horizontal plane
        reward += PROGRESS_COEFF * progress

        # encourage velocity alignment toward goal direction (horizontal)
        dir_vec = goal_xy - pos_xy
        dir_norm = np.linalg.norm(dir_vec) + 1e-8
        dir_unit = dir_vec / dir_norm

        # commanded horizontal velocity vector (m/s)
        v_vec = np.array([
            float(np.clip(action[0], -1.0, 1.0) * ACTION_SCALE),
            float(np.clip(action[1], -1.0, 1.0) * ACTION_SCALE)
        ], dtype=np.float32)
        speed = np.linalg.norm(v_vec)
        if speed > 1e-6:
            vel_unit = v_vec / speed
            alignment = float(np.dot(vel_unit, dir_unit))  # cosine in [-1, 1]
            reward += ALIGNMENT_COEFF * speed * alignment

        # action smoothness
        reward -= SMOOTHNESS_PENALTY_WEIGHT * np.linalg.norm(action - self.prev_action)
            
        self.prev_action = action
        self.prev_dist = dist
        return reward,False,{'collision' :False }
        
    def step(self,action):
        vx = float(np.clip(action[0], -1.0, 1.0) * ACTION_SCALE)
        vy = float(np.clip(action[1], -1.0, 1.0) * ACTION_SCALE)

        

        try :
            self.client.moveByVelocityAsync(vx,vy,0,duration=self.dt,vehicle_name=self.vehicle_name).join()
        except Exception as e:
            print("Step warning:", e)

        # increment step counter for time-limit termination
        self.step_count += 1

        obs = self._get_image()

        colinfo = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        collided = colinfo.has_collided


        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        reward ,done ,info = self.compute_reward(state,collided,action)
        # time-limit termination
        if not done and self.step_count >= self.max_steps:
            done = True
            info = dict(info)
            info['time_limit'] = True
            reward -= TIME_LIMIT_PENALTY
        return obs,reward,done,info
        

    def get_collision(self):
        return self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided

    def close(self):
        try:
            self.client.reset(vehicle_name=self.vehicle_name)
        except:
            pass


# -----------------------
# VECTORIZED ENV WRAPPER (Manages all drones)
# -----------------------
class VectorizedAirSimEnv:
    def __init__(self,num_drones):

        self.num_drones = num_drones
        self.drone_names = [f"Drone{i+1}" for i in range(num_drones)]
        
        self.client = airsim.MultirotorClient(ip=REMOTE_IP, port=PORT)
        print("Connecting to AirSim...")
        self.client.confirmConnection()

        self.envs = [AirSimDroneEnv(self.client,name)for name in self.drone_names]

    def reset(self):
        obs_list=[env.reset()for env in self.envs]
        return np.stack(obs_list)

    def step(self,actions):
        for i , env in enumerate(self.envs):
            action = actions[i]
            vx = float(np.clip(action[0],-1.0,1.0)*ACTION_SCALE)
            vy = float(np.clip(action[1],-1.0,1.0)*ACTION_SCALE)
            self.client.moveByVelocityAsync(vx,vy,duration =DT,vehicle_name = env.vehicle_name)

        time.sleep(DT)

        obs_batch, rew_batch,done_batch,info_batch=[],[],[],[]

        for env in self.envs:
            obs = env._get_image()
            state = env.client.getMultirotorState(vehicle_name=env.vehicle_name)
            collided = env.client.simGetCollisionInfo(vehicle_name = env.vehicle_name).has_collided

            
            current_action =action[i]
            reward, done, info = env.compute_reward(state, collided, current_action)
            env.step_count +=1
            
            if not done and env.step_count >= env.max_steps:
                done = True
                info['time_limit'] = True
                reward += TIME_LIMIT_PENALTY

            if done:
                obs = env.reset() # Reset individual env if done
            
            obs_batch.append(obs)
            rew_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)

        return np.stack(obs_batch), np.array(rew_batch), np.array(done_batch), info_batch


    

# -----------------------
# MODEL: CNN Encoder + ActorCritic (Keras)
# -----------------------
def build_actor_critic(input_shape=(IMG_H, IMG_W, 3), action_dim=ACTION_DIM):
    # shared CNN
    inp = layers.Input(shape=input_shape, name="image")
    x = layers.Conv2D(32, 8, strides=4, activation='relu')(inp)
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)

    # actor head (mu) and log_std (trainable variable)
    mu = layers.Dense(action_dim, activation='tanh', name='mu')(x)  # in [-1,1]
    # critic head
    value = layers.Dense(1, name='value')(x)

    model = models.Model(inputs=inp, outputs=[mu, value])
    # create log_std as a trainable variable (initialized small)
    log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), dtype=tf.float32, trainable=True, name='log_std')
    return model, log_std

# -----------------------
# PPO UTILITIES
# -----------------------
@tf.function
def gaussian_log_prob(mu, log_std, actions):
    # mu: [B, D], log_std: [D], actions: [B, D]
    std = tf.exp(log_std)
    pre_sum = -0.5 * (((actions - mu) / std) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    """
    rewards: [T]
    values: [T]
    dones: [T] (1.0 if not done, 0.0 if done)
    last_value: scalar
    returns: advs [T], returns [T]
    """
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = dones[t]
        next_value = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advs[t] = last_gae
    returns = advs + values
    return advs, returns

# -----------------------
# TRAINING LOOP
# -----------------------
def train():
    env = VectorizedAirSimEnv(num_drones=NUM_DRONES)
    model, log_std = build_actor_critic()
    optimizer = optimizers.Adam(learning_rate=LR)
    


    obs = env.reset()

    ep_returns = deque(maxlen =10*NUM_DRONES)
    drone_ep_rets = np.zeros(NUM_DRONES,dtype=np.float32)

    for itr in range(1, MAX_TRAIN_ITERS + 1):
        # storage
        obs_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES,IMG_H,IMG_W,3),dtype=np.float32)
        act_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES,ACTION_DIM),dtype=np.float32)
        rew_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        val_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        logp_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        done_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)

        # Collect rollout
        for step in range(ROLLOUT_STEPS):
            
            mu, value = model(obs, training=False)
            std = np.exp(log_std.numpy())
            # sample action from Gaussian policy
            action = mu + np.random.randn(*mu.shape) * std
            action = np.clip(action, -1.0, 1.0)  # keep in range
            # compute log prob
            logp = gaussian_log_prob(mu,log_std,action)

            # step the env
            next_obs, reward, done, info = env.step(action.numpy())
            obs_buf[step] = obs
            act_buf[step] = action.numpy()
            rew_buf[step] = reward
            val_buf[step] = tf.squeeze(value).numpy()
            logp_buf[step] = logp.numpy()
            done_buf[step] = 1.0-done

            obs = next_obs
            drone_ep_rets += reward
            for i,done in enumerate(done):
                if done:
                    ep_returns.append(drone_ep_rets[i])
                    print(f"[Itr {itr}, Drone {i+1}] Episode finished. Return: {drone_ep_rets[i]:.2f}")
                    drone_ep_rets[i] = 0

        # Bootstrap last value
        _, last_val = model(obs, training=False)
        last_val = tf.squeeze(last_val).numpy()
        
        # Reshape buffers for GAE and training
        # New shape: (num_drones, rollout_steps, ...)
        obs_buf = np.swapaxes(obs_buf, 0, 1)
        act_buf = np.swapaxes(act_buf, 0, 1)
        rew_buf = np.swapaxes(rew_buf, 0, 1)
        val_buf = np.swapaxes(val_buf, 0, 1)
        logp_buf = np.swapaxes(logp_buf, 0, 1)
        done_buf = np.swapaxes(done_buf, 0, 1)
        
        advs, returns = compute_gae(rew_buf, val_buf, done_buf, last_val)
        
        # Flatten for training
        obs_flat = obs_buf.reshape(-1, IMG_H, IMG_W, 3)
        act_flat = act_buf.reshape(-1, ACTION_DIM)
        logp_flat = logp_buf.reshape(-1)
        adv_flat = advs.reshape(-1)
        ret_flat = returns.reshape(-1)
        
        # Normalize advantages
        adv_flat = (adv_flat - np.mean(adv_flat)) / (np.std(adv_flat) + 1e-8)
        
        # Training
        dataset_size = ROLLOUT_STEPS * NUM_DRONES
        inds = np.arange(dataset_size)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                mb_inds = inds[start:start+MINIBATCH_SIZE]
                with tf.GradientTape() as tape:
                    mu_batch, val_batch = model(obs_flat[mb_inds], training=True)
                    val_batch = tf.squeeze(val_batch)
                    logp_new = gaussian_log_prob(mu_batch, log_std, act_flat[mb_inds])
                    
                    ratio = tf.exp(logp_new - logp_flat[mb_inds])
                    surr1 = ratio * adv_flat[mb_inds]
                    surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat[mb_inds]
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    value_loss = tf.reduce_mean((ret_flat[mb_inds] - val_batch)**2)
                    loss = policy_loss + 0.5 * value_loss
                    
                vars_to_train = model.trainable_variables + [log_std]
                grads = tape.gradient(loss, vars_to_train)
                optimizer.apply_gradients(zip(grads, vars_to_train))

        mean_ret = np.mean(ep_returns) if ep_returns else 0.0
        print(f"[Itr {itr}] Update complete. LastMeanReturn: {mean_ret:.2f}")

    # cleanup
    model.save_weights(os.path.join(SAVE_DIR, f"model_final.h5"))
    np.save(os.path.join(SAVE_DIR, f"log_std_final.npy"), log_std.numpy())
    env.close()
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()

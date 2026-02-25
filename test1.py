import os
import time
import math
import numpy as np
import cv2
import airsim
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque

import argparse
import shutil
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision


# Enable GPU memory growth for smoother training
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(" GPU memory growth enabled.")
    except RuntimeError as e:
        print(" GPU memory growth config error:", e)

# mixed_precision.set_global_policy('mixed_float16')

tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)
 
# -----------------------
# CONFIG
# -----------------------
REMOTE_IP = "127.0.0.1"
REMOTE_PORT = 41451

PORT = REMOTE_PORT
NUM_DRONES = 3

#------------Environment Parameters------------
# --- Environment ---
# City curriculum: start small, expand per phase (8 → 15 → 25)
SPAWN_RANGE = 8.0     # Phase 1: ±8m — goals always reachable, no blocked LOS
IMG_H, IMG_W = 84, 84
ACTION_DIM = 2
### NEW: Define shape of the new goal vector (distance, angle) ###
VECTOR_DIM = 2
# This is the normalization constant for distance.
# Choose a value slightly larger than your max spawn-to-goal distance.
GOAL_DIST_NORMALIZATION = 50.0 

ACTION_SCALE = 1.5
DT = 0.3

MIN_DIST_TO_GOAL = 5.0
GOAL_RADIUS = 2.0

# --- Reward Coefficients ---
GOAL_REWARD = 100.0
COLLISION_PENALTY = -150.0  # City: heavier penalty — dense buildings make collisions frequent
# FIX: Was 100 (same magnitude as GOAL_REWARD, creating wrong incentive). Reduced to 10.
TIME_LIMIT_PENALTY = 10.0
TIME_PENALTY = 0.05
PROGRESS_COEFF = 5.0
ALIGNMENT_COEFF = 1.5
SMOOTHNESS_PENALTY_WEIGHT = 0.1


# --- PPO Hyperparameters ---
ROLLOUT_STEPS = 1024
MINIBATCH_SIZE = 128 * NUM_DRONES
UPDATE_EPOCHS = 8
GAMMA = 0.995
LAM = 0.97
CLIP_EPS = 0.2
LR = 3e-4
# FIX: Raised start so the policy explores more before committing (was 0.02)
ENTROPY_COEFF_START = 0.05
ENTROPY_COEFF_END   = 0.005
VALUE_CLIP_EPS = 0.2       # FIX: Added value-function clipping (PPO best practice)
MAX_KL = 0.015             # FIX: KL early-stopping threshold per epoch
MAX_GRAD_NORM = 0.5        # FIX: Gradient clip norm to prevent divergence
# FIX: Freeze CNN layers for first N iters so goal-vector branch can warm up
CNN_FREEZE_ITERS = 20  # City: unfreeze CNN sooner — visual obstacle awareness is critical
# FIX: Was 260 — bumped to 500+ for CNN+goal navigation convergence
MAX_TRAIN_ITERS = 500
SAVE_DIR = "ppo_airsim_checkpoints"
### NEW: Path to your old model for transfer learning ###
OLD_MODEL_PATH = os.path.join(SAVE_DIR, "model_final.h5") 
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
        self.max_steps = 500  
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
        if img1d.size == 0:
            # MODIFIED: Fallback to 3-channel zeros
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)
        img = img1d.reshape(resp.height, resp.width, 3)
        
        # MODIFIED: Removed grayscale conversion to allow transfer learning
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.expand_dims(img, axis=-1)
        
        # Resize the 3-channel image
        img = cv2.resize(img, (self.img_w, self.img_h))
        
        # MODIFIED: Ensure resize didn't drop a dim if img was weird
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            # if we are here, it's grayscale, so make it 3-channel to match model
            img = np.concatenate([img, img, img], axis=-1) 
            
        img = img.astype(np.float32) / 255.0
        return img

    ### NEW: Function to get the full state (Image + Vector) ###
    def _get_state(self):
        """Returns (image, goal_vector)"""
        # 1. Get Image
        img = self._get_image()
        
        # 2. Get Vector
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        
        # Calculate vector from drone to goal
        goal_x_local = self.goal[0] - pos.x_val
        goal_y_local = self.goal[1] - pos.y_val
        
        distance_to_goal = np.linalg.norm([goal_x_local, goal_y_local])
        angle_to_goal = np.arctan2(goal_y_local, goal_x_local)
        
        # Normalize the vector (CRITICAL!)
        norm_dist = np.clip(distance_to_goal / GOAL_DIST_NORMALIZATION, 0.0, 1.0) 
        norm_angle = angle_to_goal / np.pi # Normalized from [-pi, pi] to [-1, 1]
        
        goal_vector = np.array([norm_dist, norm_angle], dtype=np.float32)
        
        return img, goal_vector

    def reset(self):
        while True:
            x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            z = -15.0  # City: 15m altitude clears most building rooftops

            gx = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            gy = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            gz = -15.0

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
        
        # FIX: Give takeoffAsync enough time; 1 s was sometimes not enough.
        self.client.takeoffAsync(vehicle_name=self.vehicle_name)
        time.sleep(2)  # FIX: Extended from 1 s → 2 s so drone is airborne before episode starts

        # City safety check: if drone barely lifted it spawned inside geometry — retry
        state_check = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        achieved_z = state_check.kinematics_estimated.position.z_val
        if achieved_z > -3.0:
            print(f"[{self.vehicle_name}] Takeoff stalled (z={achieved_z:.1f}m) — likely inside building. Retrying...")
            return self.reset()

        state_now = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos_now = np.array([
            state_now.kinematics_estimated.position.x_val,
            state_now.kinematics_estimated.position.y_val,
            state_now.kinematics_estimated.position.z_val
        ], dtype=np.float32)
        self.prev_dist = np.linalg.norm((pos_now[:2] - self.goal[:2]))
        self.initial_dist = self.prev_dist
        
        ### MODIFIED: Return the full state (img, vec) ###
        return self._get_state()

    def compute_reward(self,state,collided,action):
        
        pos = np.array([state.kinematics_estimated.position.x_val,
                        state.kinematics_estimated.position.y_val,
                        state.kinematics_estimated.position.z_val], dtype=np.float32)
        pos_xy = pos[:2]
        goal_xy = self.goal[:2]
        dist = np.linalg.norm(pos_xy - goal_xy)
        
        if self.prev_dist is None:
            self.prev_dist = dist
        progress = self.prev_dist - dist

        # FIX: Clamp initial_dist to avoid divide-by-zero producing NaN
        init_d = max(self.initial_dist, 1e-3)
        progress_norm = progress / init_d

        if dist < GOAL_RADIUS:
            print(f"[{self.vehicle_name}] Goal reached")
            bonus = max(0, (self.max_steps - self.step_count) * 0.1)
            return GOAL_REWARD + bonus, True, {'goal_reached': True}

        if collided:
            print(f"[{self.vehicle_name}] Collided")
            return COLLISION_PENALTY, True, {'collision': True}

        reward = 0.0

        # Time penalty — slightly scaled by progress so early on it's less punishing
        progress_fraction = np.clip(1.0 - (dist / init_d), 0.0, 1.0)
        time_penalty_scaled = TIME_PENALTY * (0.5 + progress_fraction)
        reward -= time_penalty_scaled

        # Progress reward — adaptive coefficient
        adaptive_coeff = PROGRESS_COEFF * (1.0 + 0.5 * (1.0 - dist / init_d))
        reward += adaptive_coeff * progress_norm

        # Alignment reward — uses ACTUAL measured velocity (not commanded action)
        # FIX: commanded action != actual physics velocity due to drag/PID lag
        dir_vec = goal_xy - pos_xy
        dir_norm = np.linalg.norm(dir_vec) + 1e-8
        dir_unit = dir_vec / dir_norm
        meas_vel = state.kinematics_estimated.linear_velocity
        v_vec = np.array([meas_vel.x_val, meas_vel.y_val], dtype=np.float32)
        speed = np.linalg.norm(v_vec)
        if speed > 1e-6:
            vel_unit = v_vec / speed
            alignment = float(np.dot(vel_unit, dir_unit))
            reward += ALIGNMENT_COEFF * speed * alignment

        # Smoothness penalty
        reward -= SMOOTHNESS_PENALTY_WEIGHT * np.linalg.norm(action - self.prev_action)

        # Stagnation penalty — FIX: threshold raised to 5cm (was 1cm, never fired)
        # Step size is ~45cm, so 5cm is a realistic "barely moved" threshold
        if abs(progress) < 0.05:
            reward -= 0.2

        self.prev_action = action
        self.prev_dist = dist

        # FIX: REMOVED the np.clip(reward/10.0, -1.0, 1.0) that was destroying the
        # goal-direction signal. Progress toward goal and away from goal now produce
        # meaningfully different reward magnitudes, so the policy CAN learn direction.
        return reward, False, {'collision': False}
        
    def step(self,action):
        vx = float(np.clip(action[0], -1.0, 1.0) * ACTION_SCALE)
        vy = float(np.clip(action[1], -1.0, 1.0) * ACTION_SCALE)

        try :
            self.client.moveByVelocityAsync(vx,vy,0,duration=self.dt,vehicle_name=self.vehicle_name)
        except Exception as e:
            print(f"[{self.vehicle_name}] Step warning:", e)

        self.step_count += 1

        ### MODIFIED: Get the full (img, vec) state ###
        img, vec = self._get_state()
        obs = (img, vec) # Package as a tuple

        colinfo = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        collided = colinfo.has_collided

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        reward ,done ,info = self.compute_reward(state,collided,action)
        
        if not done and self.step_count >= self.max_steps:
            done = True
            info = dict(info)
            info['time_limit'] = True
            reward -= TIME_LIMIT_PENALTY
            
        return obs,reward,done,info
        
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
        ### MODIFIED: Handle (img, vec) tuple from reset ###
        # This will run resets in parallel
        state_list = [env.reset() for env in self.envs]
        obs_list = [s[0] for s in state_list] # List of images
        vec_list = [s[1] for s in state_list] # List of vectors
        # Return two stacked arrays
        return (np.stack(obs_list), np.stack(vec_list))

    def step(self,actions):
        # 1. Asynchronously send move commands to all drones
        for i , env in enumerate(self.envs):
            action = actions[i]
            vx = float(np.clip(action[0],-1.0,1.0)*ACTION_SCALE)
            vy = float(np.clip(action[1],-1.0,1.0)*ACTION_SCALE)
            self.client.moveByVelocityAsync(vx,vy,0,duration =DT,vehicle_name = env.vehicle_name)
        
        # 2. Wait for the duration of the step for commands to execute
        time.sleep(DT)

        ### MODIFIED: Handle (img, vec) tuple from step ###
        obs_batch, vec_batch, rew_batch, done_batch, info_batch = [], [], [], [], []

        # 3. Collect results from all drones
        for i, env in enumerate(self.envs):
            ### MODIFIED: Get full state ###
            img, vec = env._get_state()
            
            state = env.client.getMultirotorState(vehicle_name=env.vehicle_name)
            collided = env.client.simGetCollisionInfo(vehicle_name=env.vehicle_name).has_collided
            
            action_i = actions[i]
            reward, done, info = env.compute_reward(state, collided, action_i)
            env.step_count +=1
            
            if not done and env.step_count >= env.max_steps:
                done = True
                info['time_limit'] = True
                reward -= TIME_LIMIT_PENALTY

            if done:
                ### MODIFIED: Handle (img, vec) tuple from reset ###
                img, vec = env.reset() # Reset individual env if done
            
            obs_batch.append(img)
            vec_batch.append(vec)
            rew_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)

        ### MODIFIED: Return two stacked arrays for state ###
        return (np.stack(obs_batch), np.stack(vec_batch)), np.array(rew_batch), np.array(done_batch), info_batch

# -----------------------
# MODEL: CNN Encoder + Vector Input + ActorCritic
# -----------------------
### MODIFIED: Model now accepts two inputs ###
def build_actor_critic(
    img_shape=(IMG_H, IMG_W, 3), # MODIFIED: Back to 3 channels for RGB
    vec_shape=(VECTOR_DIM,), 
    action_dim=ACTION_DIM
):
    # === Image Input Branch (CNN) ===
    # We will give these layers specific names so we can load weights
    inp_img = layers.Input(shape=img_shape, name="image_input")
    x_img = layers.Conv2D(32, 8, strides=4, activation='relu', name="conv2d_0")(inp_img)
    x_img = layers.Conv2D(64, 4, strides=2, activation='relu', name="conv2d_1")(x_img)
    x_img = layers.Conv2D(64, 3, strides=1, activation='relu', name="conv2d_2")(x_img)
    x_img = layers.Flatten(name="flatten")(x_img)
    x_img = layers.Dense(512, activation='relu', name="dense_img")(x_img)

    # === Vector Input Branch ("GPS") ===
    inp_vec = layers.Input(shape=vec_shape, name="vector_input")
    x_vec = layers.Dense(32, activation='relu', name="dense_vec")(inp_vec)

    # === Combine and Finish ===
    combined = layers.Concatenate(name="concat")([x_img, x_vec])
    x = layers.Dense(512, activation='relu', name="dense_combined")(combined)

    # Actor head (mu)
    mu = layers.Dense(action_dim, activation='tanh', name='mu')(x)
    # Critic head
    value = layers.Dense(1, name='value')(x)

    # Define the full model
    model = models.Model(inputs=[inp_img, inp_vec], outputs=[mu, value])
    
    # Create log_std as a trainable variable
    log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), dtype=tf.float32, trainable=True, name='log_std')
    
    return model, log_std

### NEW: Function to load old weights into the new model ###
def load_transfer_weights(new_model, old_model_path):
    if not os.path.exists(old_model_path):
        print(f"ℹ No old model found at {old_model_path}. Skipping transfer learning.")
        return

    try:
        # Load the old model (which has a different architecture)
        old_model = load_model(old_model_path, compile=False)
        print(f"✅ Loaded old model from {old_model_path} for weight transfer.")
        
        # Map to new layer names
        # Default Keras names: conv2d, conv2d_1, conv2d_2, flatten, dense
        # Our new model names: conv2d_0, conv2d_1, conv2d_2, flatten, dense_img
        
        layer_map = {
            "conv2d": "conv2d_0",
            "conv2d_1": "conv2d_1",
            "conv2d_2": "conv2d_2",
            "flatten": "flatten",
            "dense": "dense_img" 
        }

        weights_transferred = 0
        for old_layer in old_model.layers:
            if old_layer.name in layer_map:
                new_layer_name = layer_map[old_layer.name]
                try:
                    new_layer = new_model.get_layer(new_layer_name)
                    new_layer.set_weights(old_layer.get_weights())
                    print(f"  > Transferred weights for layer: {old_layer.name} -> {new_layer_name}")
                    weights_transferred += 1
                except Exception as e:
                    print(f"  ⚠ Could not transfer weights for {old_layer.name}: {e}")

        if weights_transferred > 0:
            print(f"✅ Successfully transferred weights for {weights_transferred} layers.")
        else:
            print("⚠ No weights were transferred. Check layer names in old vs new model.")

    except Exception as e:
        print(f" FAILED to load or transfer weights from old model: {e}")


# -----------------------
# PPO UTILITIES
# -----------------------
@tf.function
def gaussian_log_prob(mu, log_std, actions):
    mu = tf.cast(mu, tf.float32)
    log_std = tf.cast(log_std, tf.float32)
    actions = tf.cast(actions, tf.float32)
    std = tf.exp(log_std)
    pre_sum = -0.5 * (((actions - mu) / std) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    N, T = rewards.shape
    advs = np.zeros((N, T), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    next_values = np.hstack([values[:, 1:], last_value[:, None]])
    
    for t in reversed(range(T)):
        mask_t = dones[:, t]
        delta = rewards[:, t] + gamma * next_values[:, t] * mask_t - values[:, t]
        last_gae = delta + gamma * lam * mask_t * last_gae
        advs[:, t] = last_gae
        
    returns = advs + values
    return advs, returns


# -----------------------
# CHECKPOINT HELPERS
# -----------------------
def save_checkpoint(model, log_std, optimizer, itr, save_dir):
    try:
        opt_weights = optimizer.get_weights()
    except Exception:
        opt_weights = None

    checkpoint = {
        'model_weights': model.get_weights(),
        'log_std': log_std.numpy(),
        'optimizer_weights': opt_weights,
        'iteration': itr
    }
    np.save(os.path.join(save_dir, f'checkpoint_itr_{itr}.npy'), checkpoint)
    print(f"Checkpoint saved at iteration {itr}")

def save_full_model(model, log_std, save_dir, tag="final", itr=None):
    os.makedirs(save_dir, exist_ok=True)
    fname = f"model_final.h5" if tag == "final" else f"model_itr_{itr}.h5"
    model.save(os.path.join(save_dir, fname), save_format="h5")
    np.save(os.path.join(save_dir, f"log_std_{tag}.npy"), log_std.numpy())
    print(f"Full model saved: {fname} and log_std_{tag}.npy")

def load_checkpoint(model, log_std, optimizer, save_dir, force_start=False):
    if force_start:
         print("🔁 force_start=True -> Starting fresh (no checkpoint loaded).")
         return 0, False # 0 iterations, False = did not load
    
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_itr_')]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))[-1]
        try:
            checkpoint = np.load(os.path.join(save_dir, latest), allow_pickle=True).item()
            model.set_weights(checkpoint['model_weights'])
            log_std.assign(checkpoint['log_std'])
            if checkpoint['optimizer_weights'] is not None:
                try:
                    optimizer.set_weights(checkpoint['optimizer_weights'])
                except Exception as e:
                    print(" Could not restore optimizer state:", e)
            print(f"✅ Loaded checkpoint from {latest}")
            return checkpoint.get('iteration',0), True # Iteration, True = did load
        except Exception as e:
            print(f" Failed to load checkpoint {latest}. Error: {e}")
            print("Starting fresh.")
            return 0, False

    print("ℹ No checkpoints found for new model. Starting fresh.")
    return 0, False # 0 iterations, False = did not load
    
# -----------------------
# TRAINING LOOP
# -----------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Start training fresh (ignore checkpoints and transfer)")
    parser.add_argument("--save-full-every", type=int, default=10, help="Save full model every N iterations")
    args, unknown = parser.parse_known_args()

    SAVE_FULL_EVERY = args.save_full_every
    
    ### MODIFIED: Build the new model ###
    model, log_std = build_actor_critic()
    # FIX: Added clipnorm for gradient clipping — prevents early divergence
    optimizer = optimizers.Adam(learning_rate=LR, clipnorm=MAX_GRAD_NORM)
    model.summary()

    env = VectorizedAirSimEnv(num_drones=NUM_DRONES)

    # Initialize optimizer
    _ = optimizer.apply_gradients(zip([tf.zeros_like(var) for var in model.trainable_variables] + [tf.zeros_like(log_std)], model.trainable_variables + [log_std]))
        
    os.makedirs(SAVE_DIR, exist_ok=True)

    ### MODIFIED: Checkpoint loading logic ###
    start_itr, loaded_checkpoint = load_checkpoint(model, log_std, optimizer, SAVE_DIR, force_start=args.fresh)
    
    # If we didn't start fresh AND we didn't load a new checkpoint, try transfer learning
    if not args.fresh and not loaded_checkpoint:
        print("\nAttempting Transfer Learning from old model...")
        load_transfer_weights(model, OLD_MODEL_PATH)
        print("Transfer learning attempt finished.\n")

    print(f"Training starting from iteration {start_itr + 1}")

    ### MODIFIED: obs is now (img_obs, vec_obs) ###
    img_obs, vec_obs = env.reset()

    ep_returns = deque(maxlen=50*NUM_DRONES)   # FIX: wider window for stable metrics
    drone_ep_rets = np.zeros(NUM_DRONES,dtype=np.float32)

    iters = []
    mean_returns = []
    success_rates = []
    episodes_per_iter = []


    # ---- CNN Freeze Schedule ----
    CNN_LAYER_NAMES = ["conv2d_0", "conv2d_1", "conv2d_2", "flatten", "dense_img"]

    for itr in range(start_itr + 1, MAX_TRAIN_ITERS + 1):
        # FIX: Compute freeze state each iter relative to start_itr so that
        # checkpoint resumes correctly inherit the freeze/unfreeze schedule.
        iters_since_start = itr - start_itr
        cnn_frozen = (iters_since_start <= CNN_FREEZE_ITERS)
        for lname in CNN_LAYER_NAMES:
            try:
                model.get_layer(lname).trainable = not cnn_frozen
            except:
                pass
        if iters_since_start == 1:
            print(f"[CNN Schedule] Freezing CNN layers for first {CNN_FREEZE_ITERS} iters.")
        elif iters_since_start == CNN_FREEZE_ITERS + 1:
            print("[CNN Schedule] Unfreezing CNN layers for full joint training.")

        # FIX: Linearly decay entropy coefficient
        frac = min(1.0, (itr - 1) / max(MAX_TRAIN_ITERS - 1, 1))
        ENTROPY_COEFF = ENTROPY_COEFF_START + frac * (ENTROPY_COEFF_END - ENTROPY_COEFF_START)
        true_goal_successes = 0
        episodes_this_iter = 0   # FIX: track actual episodes completed this rollout
        
        ### MODIFIED: Buffers for (img, vec) state ###
        # MODIFIED: Correct channel dim
        img_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES,IMG_H,IMG_W,3),dtype=np.float32)
        vec_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES,VECTOR_DIM),dtype=np.float32) # New buffer
        
        act_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES,ACTION_DIM),dtype=np.float32)
        rew_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        val_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        logp_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)
        done_buf = np.zeros((ROLLOUT_STEPS,NUM_DRONES),dtype=np.float32)

        # Collect rollout
        for step in range(ROLLOUT_STEPS):
            
            ### MODIFIED: Model call with two inputs ###
            mu, value = model([img_obs, vec_obs], training=False)
            
            std = np.exp(log_std.numpy())
            action = mu + np.random.randn(*mu.shape) * std
            action = np.clip(action, -1.0, 1.0)
            logp = gaussian_log_prob(mu,log_std,action)

            ### MODIFIED: env.step now returns (img, vec) ###
            (next_img_obs, next_vec_obs), reward, done, info = env.step(action)
            
            ### MODIFIED: Store in separate buffers ###
            img_buf[step] = img_obs
            vec_buf[step] = vec_obs
            act_buf[step] = action
            rew_buf[step] = reward
            val_buf[step] = tf.squeeze(value).numpy()
            logp_buf[step] = logp.numpy()
            done_buf[step] = 1.0-done

            ### MODIFIED: Update state ###
            img_obs, vec_obs = next_img_obs, next_vec_obs
            
            drone_ep_rets += reward
            for i, done_flag in enumerate(done):
                if done_flag:
                    ep_returns.append(drone_ep_rets[i])
                    episodes_this_iter += 1  # FIX: count episodes per rollout
                    if isinstance(info, list) and i < len(info):
                        if info[i].get('goal_reached', False):
                            print(f"[Itr {itr}, Drone {i+1}] Episode finished. GOAL REACHED! Return: {drone_ep_rets[i]:.2f}")
                            true_goal_successes += 1
                        else:
                            print(f"[Itr {itr}, Drone {i+1}] Episode finished. Return: {drone_ep_rets[i]:.2f}")
                    else:
                        print(f"[Itr {itr}, Drone {i+1}] Episode finished. Return: {drone_ep_rets[i]:.2f}")
                    drone_ep_rets[i] = 0

            last_info_batch = info

        ### MODIFIED: Bootstrap last value with two inputs ###
        _, last_val = model([img_obs, vec_obs], training=False)
        last_val = tf.squeeze(last_val).numpy()
        
        ### MODIFIED: Reshape buffers ###
        img_buf = np.swapaxes(img_buf, 0, 1)
        vec_buf = np.swapaxes(vec_buf, 0, 1)
        act_buf = np.swapaxes(act_buf, 0, 1)
        rew_buf = np.swapaxes(rew_buf, 0, 1)
        val_buf = np.swapaxes(val_buf, 0, 1)
        logp_buf = np.swapaxes(logp_buf, 0, 1)
        done_buf = np.swapaxes(done_buf, 0, 1)
        
        advs, returns = compute_gae(rew_buf, val_buf, done_buf, last_val)
 
        ### MODIFIED: Flatten buffers ###
        # MODIFIED: Correct channel dim
        img_flat = img_buf.reshape(-1, IMG_H, IMG_W, 3) 
        vec_flat = vec_buf.reshape(-1, VECTOR_DIM)
        act_flat = act_buf.reshape(-1, ACTION_DIM)
        logp_flat = logp_buf.reshape(-1)
        adv_flat = advs.reshape(-1)
        ret_flat = returns.reshape(-1)
        
        adv_flat = (adv_flat - np.mean(adv_flat)) / (np.std(adv_flat) + 1e-8)
        
        dataset_size = ROLLOUT_STEPS * NUM_DRONES
        inds = np.arange(dataset_size)
        
        # Store old log-probs for KL check
        old_logp_flat = logp_flat.copy()

        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            kl_epoch = 0.0
            kl_count = 0
            early_stop = False

            for start in range(0, dataset_size, MINIBATCH_SIZE):
                mb_inds = inds[start:start+MINIBATCH_SIZE]
                with tf.GradientTape() as tape:
                    mu_batch, val_batch = model([
                        img_flat[mb_inds],
                        vec_flat[mb_inds]
                    ], training=True)

                    val_batch = tf.squeeze(val_batch)
                    logp_new = gaussian_log_prob(mu_batch, log_std, act_flat[mb_inds])

                    ratio = tf.exp(logp_new - logp_flat[mb_inds])
                    surr1 = ratio * adv_flat[mb_inds]
                    surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat[mb_inds]
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    # FIX: Value function clipping (PPO best practice)
                    val_old = ret_flat[mb_inds] - adv_flat[mb_inds]  # approx old value
                    val_clipped = val_old + tf.clip_by_value(
                        val_batch - val_old, -VALUE_CLIP_EPS, VALUE_CLIP_EPS
                    )
                    value_loss = tf.reduce_mean(tf.maximum(
                        (ret_flat[mb_inds] - val_batch) ** 2,
                        (ret_flat[mb_inds] - val_clipped) ** 2
                    ))

                    std = tf.exp(log_std)
                    entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * np.e * std ** 2))
                    entropy_loss = -ENTROPY_COEFF * entropy

                    policy_loss = tf.cast(policy_loss, tf.float32)
                    value_loss = tf.cast(value_loss, tf.float32)
                    entropy_loss = tf.cast(entropy_loss, tf.float32)
                    loss = policy_loss + 0.5 * value_loss + entropy_loss

                vars_to_train = [v for v in model.trainable_variables] + [log_std]
                grads = tape.gradient(loss, vars_to_train)
                optimizer.apply_gradients(zip(grads, vars_to_train))

                # FIX: Track approx KL for early stopping
                kl_approx = tf.reduce_mean(logp_flat[mb_inds] - logp_new).numpy()
                kl_epoch += kl_approx
                kl_count += 1

            # FIX: KL early stopping — if policy changed too much, stop update epochs
            mean_kl = kl_epoch / max(kl_count, 1)
            if mean_kl > MAX_KL:
                print(f"[Itr {itr}] KL early stop at epoch {epoch+1} (KL={mean_kl:.4f})")
                break

        mean_ret = np.mean(ep_returns) if ep_returns else 0.0
        # FIX: Divide by THIS-ROLLOUT episodes, not the deque size (was 5x too low)
        success_rate = (true_goal_successes / max(episodes_this_iter, 1)) * 100

        mean_ep_len = (ROLLOUT_STEPS * NUM_DRONES) / max(episodes_this_iter, 1)
        print(f"[Itr {itr}] SR: {success_rate:.1f}% | MeanRet: {mean_ret:.2f} | "
              f"EpLen: {mean_ep_len:.0f} | Episodes: {episodes_this_iter} | "
              f"Entropy: {ENTROPY_COEFF:.4f}")

        iters.append(itr)
        mean_returns.append(mean_ret)
        success_rates.append(success_rate)
        episodes_per_iter.append(len(ep_returns))
        
        # Save checkpoint (for the new model)
        save_checkpoint(model, log_std, optimizer, itr, SAVE_DIR)

        if itr % SAVE_FULL_EVERY == 0:
            save_full_model(model, log_std, SAVE_DIR, tag=f"itr_{itr}", itr=itr)
            
    # Save final model (new architecture)
    save_full_model(model, log_std, SAVE_DIR, tag="final")
    save_checkpoint(model, log_std, optimizer, itr, SAVE_DIR)

    # ---- Plotting ----
    plt.figure(figsize=(8, 5))
    plt.plot(iters, mean_returns, label='Mean Return', marker='o')
    plt.title("Drone Training Progress (Mean Return)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "mean_return_plot.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(iters, success_rates, label='Success Rate (%)', color='green', marker='o')
    plt.title("Drone Training Progress (Success Rate)")
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "success_rate_plot.png"))
    plt.show()
    
    print("Training finished and model saved successfully.")

if __name__ == "__main__":
    train()
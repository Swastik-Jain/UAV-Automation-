"""
ppo_airsim_tf2.py
TensorFlow2 PPO training for AirSim City (remote).
Replace REMOTE_IP with your AirSim server IP.
"""

import os
import time
import math
import numpy as np
import cv2
import airsim
import tensorflow as tf
from keras import layers, models, optimizers

# -----------------------
# CONFIG
# -----------------------
REMOTE_IP = "127.0.0.1"   # <<< R120EPLACE with Windows machine IP
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
GOAL_POS = np.array([50.0, 0.0, -5.0])

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# AIRSIM ENV WRAPPER
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
        # small safety: ensure takeoff
        try:
            self.client.takeoffAsync().join()
            time.sleep(0.5)
        except Exception as e:
            print("Warning: takeoff failed (maybe already flying).", e)

    def _get_image(self):
        # returns HxWx3 float32 [0,1]
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
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

    def step(self, action):
        """
        action: np.array shape (2,) in [-1,1]; mapped to vx, vy
        returns obs, reward, done, info
        """
        vx = float(np.clip(action[0], -1.0, 1.0) * ACTION_SCALE)
        vy = float(np.clip(action[1], -1.0, 1.0) * ACTION_SCALE)
        # send velocity command for dt seconds and block
        try:
         self.client.moveByVelocityZAsync(vx, vy, GOAL_POS[2], duration=self.dt).join()
        except Exception as e:
         print("moveByVelocityZAsync exception:", e)

        obs = self._get_image()

        # reward: simple survival + forward progress proxy
        # For City environment we don't have a specific goal by default.
        # We'll use negative reward on collisions and small positive step reward.
        pos = self.client.getMultirotorState().kinematics_estimated.position
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        dist_to_goal = np.linalg.norm(drone_pos - GOAL_POS)

        reward = -0.01 * dist_to_goal   # encourage being closer
        done = False
        info = {}

        #check collision
        colinfo = self.client.simGetCollisionInfo()
        if colinfo.has_collided:
            reward = -20.0
            done = True
            info['collision'] = True
        
        # check goal
        elif dist_to_goal < 2.0:  # within 2m of goal
          reward = +50.0
          done = True
          info['goal_reached'] = True
        else:
          info['collision'] = False
          info['goal_reached'] = False

        return obs, reward, done, info

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
    env = AirSimDroneEnv()
    model, log_std = build_actor_critic()
    optimizer = optimizers.Adam(learning_rate=LR)

    # For saving
    ckpt_prefix = os.path.join(SAVE_DIR, "ppo_ckpt")
    best_mean_return = -1e9

    obs = env.reset()
    episode_rewards = []
    ep_ret = 0.0
    ep_len = 0

    for itr in range(1, MAX_TRAIN_ITERS + 1):
        # storage
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        logp_buf = []
        done_buf = []

        # Collect rollout
        for step in range(ROLLOUT_STEPS):
            img = obs.astype(np.float32)
            img_tensor = tf.expand_dims(img, axis=0)  # [1,H,W,3]
            mu, value = model(img_tensor, training=False)
            mu = mu[0].numpy()  # shape (action_dim,)
            std = np.exp(log_std.numpy())
            # sample action from Gaussian policy
            action = mu + np.random.randn(*mu.shape) * std
            action = np.clip(action, -1.0, 1.0)  # keep in range
            # compute log prob
            logp = gaussian_log_prob(tf.constant(mu.reshape(1,-1), dtype=tf.float32),
                                     log_std, tf.constant(action.reshape(1,-1), dtype=tf.float32)).numpy()[0]

            # step the env
            next_obs, reward, done, info = env.step(action)
            obs_buf.append(img)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value[0][0].numpy())
            logp_buf.append(logp)
            done_buf.append(0.0 if done else 1.0)

            ep_ret += reward
            ep_len += 1

            obs = next_obs
            if done:
                # record and reset
                episode_rewards.append(ep_ret)
                print(f"[Itr {itr}] Episode finished. Return: {ep_ret:.2f}, Length: {ep_len}")
                ep_ret = 0.0
                ep_len = 0
                obs = env.reset()

        # bootstrap last value
        img_tensor = tf.expand_dims(obs.astype(np.float32), axis=0)
        _, last_val = model(img_tensor, training=False)
        last_val = float(last_val[0][0].numpy())

        # convert buffers to arrays
        obs_arr = np.array(obs_buf, dtype=np.float32)
        acts_arr = np.array(act_buf, dtype=np.float32)
        rews_arr = np.array(rew_buf, dtype=np.float32)
        vals_arr = np.array(val_buf, dtype=np.float32)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        dones_arr = np.array(done_buf, dtype=np.float32)

        # compute advantages and returns (GAE)
        advs, returns = compute_gae(rews_arr, vals_arr, dones_arr, last_val, GAMMA, LAM)
        # normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Training / update epochs with minibatches
        dataset_size = len(rews_arr)
        inds = np.arange(dataset_size)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                mb_inds = inds[start:start+MINIBATCH_SIZE]
                mb_obs = obs_arr[mb_inds]
                mb_acts = acts_arr[mb_inds]
                mb_advs = advs[mb_inds]
                mb_rets = returns[mb_inds]
                mb_logp_old = logp_arr[mb_inds]

                with tf.GradientTape() as tape:
                    mu_batch, val_batch = model(mb_obs, training=True)
                    # mu_batch: [B, D], val_batch: [B,1]
                    val_batch = tf.squeeze(val_batch, axis=1)
                    # broadcast log_std
                    logp_new = gaussian_log_prob(mu_batch, log_std, tf.constant(mb_acts, dtype=tf.float32))
                    # policy loss (clipped)
                    ratio = tf.exp(logp_new - mb_logp_old)
                    surr1 = ratio * mb_advs
                    surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    # value loss
                    value_loss = tf.reduce_mean((mb_rets - val_batch) ** 2)
                    # entropy bonus
                    entropy = tf.reduce_mean(0.5 * (tf.math.log(2.0 * np.pi) + 1.0 + 2.0 * log_std))
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # gather trainable variables: model weights + log_std
                vars = model.trainable_variables + [log_std]
                grads = tape.gradient(loss, vars)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                optimizer.apply_gradients(zip(grads, vars))

        # Logging
        mean_ret = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0
        print(f"[Itr {itr}] Update complete. LastMeanReturn(10): {mean_ret:.2f} | AvgRewardPerStep: {np.mean(rews_arr):.4f}")

        # Save checkpoint occasionally
        if itr % 10 == 0:
            # save model weights and log_std
            model.save_weights(os.path.join(SAVE_DIR, f"model_itr{itr}.h5"))
            np.save(os.path.join(SAVE_DIR, f"log_std_itr{itr}.npy"), log_std.numpy())
            print(f"Saved checkpoint at iter {itr}")

        # Early exit if solved-ish (custom criterion)
        if mean_ret > 50.0 and itr > 50:
            print("Performance threshold reached -> stopping training.")
            break

    # cleanup
    model.save_weights(os.path.join(SAVE_DIR, f"model_final.h5"))
    np.save(os.path.join(SAVE_DIR, f"log_std_final.npy"), log_std.numpy())
    env.close()
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()

from http import client
import airsim
import asyncio
import cv2
import json
import base64
import numpy as np
import tensorflow as tf
from keras.models import load_model
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os

# === Configuration ===
IMG_H, IMG_W = 84, 84
ACTION_DIM = 2
ACTION_SCALE = 1.5  # This is the MAXIMUM scale
REMOTE_IP = "10.46.134.21"
SAVE_DIR = "/Users/sj/Desktop/project/ppo_airsim_checkpoints"

### NEW: Must match the normalization from your training script ###
GOAL_DIST_NORMALIZATION = 50.0 

# === Globals ===
mission_active = False  # Shared flag to control mission state

# === FastAPI setup ===
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://10.46.134.21:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Helper: Capture AirSim image ===
async def get_image(client, vehicle_name="Drone1"):
    # This function is already correct for 3-channel (84, 84, 3) images
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name=vehicle_name)

    if not responses or len(responses[0].image_data_uint8) == 0:
        return None

    resp = responses[0]
    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    if img1d.size == 0:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.float32) # Match 3-channel
        
    img = img1d.reshape(resp.height, resp.width, 3)
    img = cv2.resize(img, (IMG_H, IMG_W))
    img = img.astype(np.float32) / 255.0
    return img


# === Helper: Setup AirSim client ===
async def setup_airsim_client(remote_ip: str = REMOTE_IP, vehicle_name: str = "Drone1"):
    client = airsim.MultirotorClient(ip=remote_ip)
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    print("AirSim client connected and API enabled.")
    return client

async def reset_drone(client, vehicle_name="Drone1"):
    print("Resetting drone...")
    await asyncio.sleep(0.2)
    try:
        client.hoverAsync(vehicle_name=vehicle_name).join()
        await asyncio.sleep(0.5)
        client.landAsync(vehicle_name=vehicle_name).join()
    except Exception as e:
        print(f" Hover/Land error: {e}")

    try:
        client.armDisarm(False, vehicle_name=vehicle_name)
        client.enableApiControl(False, vehicle_name=vehicle_name)
        client.reset()
        await asyncio.sleep(1.0)
        print(" Drone reset complete.")
    except Exception as e:
        print(f" Reset error: {e}")


# === Live camera video streaming ===
async def stream_camera_feed(websocket: WebSocket, client, vehicle_name="Drone1"):
    print("Starting live camera feed...")
    try:
        while True:
            global mission_active
            if not mission_active:
                await asyncio.sleep(0.1)
                continue  # Wait until mission starts again

            img = await get_image(client, vehicle_name)
            if img is None:
                continue

            # Convert image to base64-encoded JPEG
            frame = (img * 255.0).astype(np.uint8)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64 = base64.b64encode(buffer).decode('utf-8')

            try:
                await websocket.send_json({
                    "type": "video_frame",
                    "frame": f"data:image/jpeg;base64,{b64}"
                })
            except Exception as e:
                print(f" Video stream send error: {e}")
                break

            await asyncio.sleep(0.05)  # ~20 FPS

    except asyncio.CancelledError:
        print(" Video stream stopped.")
    except Exception as e:
        print(f" Video stream error: {e}")


# === Main inference loop (Velocity-only control) ===
async def run_inference_loop(websocket: WebSocket, goal_pos: dict, client):
    if model is None:
        await websocket.send_json({"error": "Model not loaded. Cannot start mission."})
        return

    try:
        goal_x = float(goal_pos["x"])
        goal_y = float(goal_pos["y"])
    except Exception:
        await websocket.send_json({"error": "Invalid goal format. Expected {x:..., y:...}"})
        return

    print(f"🎯 Starting mission towards goal: ({goal_x:.2f}, {goal_y:.2f})")

    try:
        print("Preparing drone position...")

        # === ✅ FIX 3: Safe spawn elevation ===
        safe_pose = airsim.Pose(airsim.Vector3r(0, 0, -8), airsim.to_quaternion(0, 0, 0))
        client.simSetVehiclePose(safe_pose, True)
        await asyncio.sleep(0.5)
        # =====================================

        print("Arming drone and taking off...")
        client.armDisarm(True)

        state = client.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            print("Landed state detected. Taking off...")
            client.takeoffAsync().join()

        # Go to stable hover altitude
        client.moveToZAsync(-6, 1).join()
        await asyncio.sleep(2)
        print("Drone is hovering. Starting inference loop.")

        
        target_altitude = -6.0
        max_horiz_speed = 1.5
        max_vert_speed = 1.0
        loop_duration = 0.5
        
        # Start counter at 1 for ramp-up
        loop_counter = 1 
        ramp_up_steps = 10 

        while True:
            global mission_active
            if not mission_active:
                print(" Mission flag set to False — exiting inference loop.")
                break
            
            # MODIFIED: Check for cancellation
            try:
                await asyncio.sleep(0) # This is a checkpoint to allow cancellation
            except asyncio.CancelledError:
                print("Inference loop cancelled.")
                break

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            # 1. Get Image
            obs_img = await get_image(client) # Renamed to obs_img
            if obs_img is None:
                print("No image from AirSim.")
                await asyncio.sleep(0.1)
                continue

            ### MODIFIED: Calculate the Goal Vector ###
            # 2. Get Vector (The "GPS")
            goal_x_local = goal_x - pos.x_val
            goal_y_local = goal_y - pos.y_val
            
            
            distance_to_goal_vec = np.linalg.norm([goal_x_local, goal_y_local])
            angle_to_goal = np.arctan2(goal_y_local, goal_x_local)
            
            # Normalize the vector EXACTLY like in training
            norm_dist = np.clip(distance_to_goal_vec / GOAL_DIST_NORMALIZATION, 0.0, 1.0) 
            norm_angle = angle_to_goal / np.pi # Normalized from [-pi, pi] to [-1, 1]
            
            # Create the vector with a batch dimension (1, 2)
            goal_vector = np.array([[norm_dist, norm_angle]], dtype=np.float32)
            ### ================================== ###


            ### MODIFIED: Model call with two inputs ###
            # Create tensors for both inputs
            obs_img_tensor = tf.expand_dims(obs_img, axis=0)
            vec_tensor = tf.convert_to_tensor(goal_vector, dtype=tf.float32)
            
            # Pass both tensors to the model
            mu, _ = model([obs_img_tensor, vec_tensor], training=False)
            raw_action = mu.numpy()[0]
            ### ================================== ###


            # === THIS IS THE CRITICAL NAN GUARD FIX ===
            # Check if the model output is invalid (NaN or Inf)
            if not np.all(np.isfinite(raw_action)):
                print(f"DETECTED INVALID MODEL OUTPUT (NaN/Inf): {raw_action}")
                print("Stopping mission to prevent crash.")
                await websocket.send_json({
                    "status": "collided",
                    "summary": {"outcome": "Failure: Model output was NaN/Inf."},
                })
                break # Exit the loop
            # ==========================================

            # Corrected Ramp-up Logic
            ramp_factor = min(loop_counter, ramp_up_steps) / ramp_up_steps
            current_scale = ACTION_SCALE * ramp_factor

            vz = 0.5 * (target_altitude - pos.z_val)
            vz = float(np.clip(vz, -max_vert_speed, max_vert_speed))

            vx = float(np.clip(raw_action[0] * current_scale, -max_horiz_speed, max_horiz_speed))
            vy = float(np.clip(raw_action[1] * current_scale, -max_horiz_speed, max_horiz_speed))

            dist_to_goal = np.linalg.norm([pos.x_val - goal_x, pos.y_val - goal_y])
            if dist_to_goal < 10.0:
                scale_factor = max(0.2, dist_to_goal / 10.0)
                vx *= scale_factor
                vy *= scale_factor

            client.moveByVelocityAsync(vx, vy, vz, duration=loop_duration).join()

            # Update telemetry
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            dist_to_goal = np.linalg.norm([pos.x_val - goal_x, pos.y_val - goal_y])
            
            # Use obs_img (the one we got from get_image) for telemetry
            img_frame = (obs_img * 255.0).astype(np.uint8) 
            _, buffer = cv2.imencode('.jpg', img_frame)
            buffer_img = base64.b64encode(buffer).decode('utf-8')

            telemetry = {
                "drone_position": {"x": pos.x_val, "y": pos.y_val},
                "image_src": f"data:image/jpeg;base64,{buffer_img}",
                "status": "en_route",
                "altitude": -pos.z_val,
                "velocity": {"x": vel.x_val, "y": vel.y_val, "z": vel.z_val},
                "coordinates": {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
                "speed": np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]),
                "action": {"vx": float(raw_action[0]), "vy": float(raw_action[1])},
                "distance_to_goal": dist_to_goal,
            }

            try:
                await websocket.send_json(telemetry)
            except Exception as e:
                print(f"WebSocket send failed: {e}")
                break

            if dist_to_goal < 2.0:
                print(" Goal reached successfully.")
                await websocket.send_json({
                    "status": "goal_reached",
                    "summary": {"outcome": "Success: Goal Reached!", "final_distance": f"{dist_to_goal:.2f} m"},
                })
                break

            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                obj = collision_info.object_name or ""
                if "Landscape" in obj or "terrain" in obj.lower():
                    print(f" Ignored landscape collision at altitude {pos.z_val:.2f}m.")
                else:
                    print(f" Drone collided with: {obj} at altitude {pos.z_val:.2f}m")
                    await websocket.send_json({
                        "status": "collided",
                        "summary": {"outcome": f"Failure: Drone Collided with {obj}"},
                    })
                    break
            
            # This is the debug log
            print(f"Step {loop_counter}: dist={dist_to_goal:.1f}m | "
                  f"Raw Action=({raw_action[0]:.2f},{raw_action[1]:.2f}) | "
                  f"Scale={current_scale:.2f} | "
                  f"Final V=({vx:.2f},{vy:.2f},{vz:.2f})")
            
            # Increment loop counter at the end
    except asyncio.CancelledError:
        print("Inference loop cancelled by STOP_MISSION.")
    except Exception as e:
        print(f"Inference loop error: {e}")

        
    finally:
        
        mission_active = False  # Ensure flag reset

        # Give AirSim a short pause before reset (to complete any pending velocity commands)
        await asyncio.sleep(0.2)

        try:
            await reset_drone(client)
        except Exception as e:
            print(f"Reset during finally failed: {e}")

        try:
            await websocket.send_json({
                "status": "stopped",
                "summary": {"outcome": "Mission stopped by user or completed reset."}
            })
        except Exception as e:
            print(f"Could not send final status: {e}")

        print("Mission ended and drone reset successfully.")



# === Load model ===
print("Loading trained model...")
model_path = os.path.join(SAVE_DIR, "model_final.h5")
model = None
if os.path.exists(model_path):
    try:
        # This will now load the new model with 2 inputs
        model = load_model(model_path, compile=False) 
        print("Model loaded successfully.")
        model.summary() # Print summary to confirm it has 2 inputs
    except Exception as e:
        print(f" Failed to load model: {e}")
else:
    print(f" Model not found at {model_path}. Running without model.")
print("Model ready for inference.")


# === WebSocket endpoint ===
@app.websocket("/ws/mission")
async def websocket_endpoint(websocket: WebSocket):
    global mission_active
    await websocket.accept()
    airsim_client = None
    
    # MODIFIED: Task tracking
    video_task = None
    inference_task = None

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "CONNECT":
                try:
                    airsim_client = await setup_airsim_client(REMOTE_IP)
                    await websocket.send_json({"action": "CONNECT_ACK", "drone_con": True})
                except Exception as e:
                    airsim_client = None
                    await websocket.send_json({
                        "action": "CONNECT_ACK", "drone_con": False, "error": str(e)
                    })

            elif action == "START_MISSION":
                if airsim_client is None:
                    await websocket.send_json({"error": "Drone not connected"})
                    continue
                goal_position = data.get("goal")
                
                mission_active = True
                
                
                video_task = asyncio.create_task(stream_camera_feed(websocket, airsim_client))
                inference_task = asyncio.create_task(run_inference_loop(websocket, goal_position, airsim_client))
                
                print("🚀 Mission and Video tasks started.")

            elif action == "STOP_MISSION":
                print(f"Received {action} from frontend.")
                mission_active = False  # Signal loops to stop
                
                # MODIFIED: Cancel running tasks
                if inference_task and not inference_task.done():
                    inference_task.cancel()
                    print("Inference task cancellation sent.")
                if video_task and not video_task.done():
                    video_task.cancel()
                    print("Video task cancellation sent.")
                
                # Wait for tasks to actually cancel
                try:
                    await asyncio.wait_for(asyncio.gather(inference_task, video_task, return_exceptions=True), timeout=2.0)
                    print("Tasks successfully cancelled.")
                except asyncio.TimeoutError:
                    print("Task cancellation timed out.")
                except Exception:
                    pass # Ignore other errors on gather, we are stopping anyway
                
                if airsim_client:
                    await reset_drone(airsim_client) # Reset the drone
                    
                await websocket.send_json({"action": f"{action}_ACK"})
                # DO NOT BREAK. Stay connected for a new mission.

            elif action == "DISCONNECT":
                print(f"Received {action} from frontend.")
                mission_active = False
                
                # MODIFIED: Cancel tasks just like STOP
                if inference_task and not inference_task.done():
                    inference_task.cancel()
                if video_task and not video_task.done():
                    video_task.cancel()
                
                if airsim_client:
                    await reset_drone(airsim_client)
                
                await websocket.send_json({"action": f"{action}_ACK"})
                break # NOW we break, because client wants to disconnect.


            elif action == "PING":
                await websocket.send_json({"action": "PONG"})

    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        # This 'finally' block runs when the loop breaks (DISCONNECT) or on error
        print("Client disconnecting, ensuring all tasks are stopped.")
        mission_active = False
        if inference_task and not inference_task.done():
            inference_task.cancel()
        if video_task and not video_task.done():
            video_task.cancel()
            
        if airsim_client:
            # Run final reset in a separate task to avoid blocking close
            asyncio.create_task(reset_drone(airsim_client))
            airsim_client.enableApiControl(False)
        print("Client disconnected.")


# === Run server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
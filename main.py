import airsim
import numpy as np
import cv2
import signal
import sys

# Global client variable
client = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down drone...')
    if client:
        try:
            client.armDisarm(False)  # Disarm drone
            client.enableApiControl(False)  # Release API control
            print("Drone safely landed and API control released.")
        except Exception as e:
            print(f"Error during shutdown: {e}")
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def initialize_drone():
    """Initialize drone connection and takeoff"""
    global client
    client = airsim.MultirotorClient(ip="192.168.40.21", port=41451)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    print("Drone initialized and took off!")

def get_drone_image():
    """Get image from drone camera"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    img_rgb = cv2.resize(img_rgb, (84, 84))   # resize for CNN
    return img_rgb / 255.0   # normalize

# Initialize drone when script runs
if __name__ == "__main__":
    try:
        initialize_drone()
        print("Drone ready! Press Ctrl+C to safely stop.")
        # Your main loop would go here
        # For example:
        # while True:
        #     image = get_drone_image()
        #     # Process image...
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if client:
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass


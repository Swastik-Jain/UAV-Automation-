import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient(ip="192.168.40.21", port=41451)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

def get_drone_image():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    img_rgb = cv2.resize(img_rgb, (84, 84))   # resize for CNN
    return img_rgb / 255.0   # normalize


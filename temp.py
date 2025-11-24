# import sys
# import time

# try:
#     import airsim
# except Exception as e:
#     print("ERROR: cannot import airsim:", e, file=sys.stderr)
#     raise SystemExit(1)

# print("Imported airsim from:", getattr(airsim, "__file__", "built-in/unknown"))

# client = airsim.MultirotorClient()
# client.timeout = 5  # seconds, set short for quick failure

# # Try confirmConnection and show result
# try:
#     ok = client.confirmConnection()
#     print("confirmConnection() returned:", ok)
# except Exception as e:
#     print("confirmConnection() raised exception:", repr(e))

# # Try to list vehicles
# try:
#     vehicles = client.listVehicles()
#     print("Vehicles:", vehicles)
# except Exception as e:
#     print("listVehicles() raised exception:", repr(e))

# print("Script finished.")



# import airsim

# client = airsim.MultirotorClient()
# client.confirmConnection()
# print("Vehicles available:", client.listVehicles())

import airsim
c = airsim.MultirotorClient()
c.confirmConnection()
print(c.listVehicles())

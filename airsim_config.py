"""
AirSim Remote Connection Configuration
Choose the connection method that works best for you and your friend.
"""

# =============================================================================
# CONNECTION METHODS
# =============================================================================

# Method 1: VPN Connection (Recommended)
# Both you and your friend connect to the same VPN, then use VPN IP
VPN_IP = "10.0.0.100"  # Replace with your friend's VPN IP
VPN_PORT = 41451

# Method 2: Public IP with Port Forwarding
# Your friend needs to forward port 41451 on their router
PUBLIC_IP = "203.0.113.1"  # Replace with your friend's public IP
PUBLIC_PORT = 41451

# Method 3: SSH Tunneling
# Your friend sets up SSH server, you create tunnel
SSH_HOST = "your-friend-server.com"  # Replace with SSH server
SSH_PORT = 22
SSH_USER = "username"  # Replace with SSH username
LOCAL_TUNNEL_PORT = 41451  # Local port for tunnel

# Method 4: Cloud/Remote Desktop
# Use Parsec, TeamViewer, etc. - no code changes needed
REMOTE_DESKTOP = True  # Set to True if using remote desktop

# =============================================================================
# CURRENT CONFIGURATION
# =============================================================================

# Choose your connection method:
CONNECTION_METHOD = "VPN"  # Options: "VPN", "PUBLIC", "SSH", "REMOTE_DESKTOP"

# Active configuration (will be set based on CONNECTION_METHOD)
if CONNECTION_METHOD == "VPN":
    REMOTE_IP = VPN_IP
    REMOTE_PORT = VPN_PORT
elif CONNECTION_METHOD == "PUBLIC":
    REMOTE_IP = PUBLIC_IP
    REMOTE_PORT = PUBLIC_PORT
elif CONNECTION_METHOD == "SSH":
    REMOTE_IP = "127.0.0.1"  # SSH tunnel creates local connection
    REMOTE_PORT = LOCAL_TUNNEL_PORT
elif CONNECTION_METHOD == "REMOTE_DESKTOP":
    REMOTE_IP = "127.0.0.1"  # AirSim runs locally on friend's machine
    REMOTE_PORT = 41451
else:
    # Default to local for testing
    REMOTE_IP = "127.0.0.1"
    REMOTE_PORT = 41451

# =============================================================================
# CONNECTION TESTING
# =============================================================================

def test_connection():
    """Test if we can connect to the remote AirSim server"""
    import socket
    import time
    
    print(f"Testing connection to {REMOTE_IP}:{REMOTE_PORT}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex((REMOTE_IP, REMOTE_PORT))
        sock.close()
        
        if result == 0:
            print("✅ Connection successful!")
            return True
        else:
            print("❌ Connection failed - AirSim server not reachable")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    print(f"Current configuration: {CONNECTION_METHOD}")
    print(f"Connecting to: {REMOTE_IP}:{REMOTE_PORT}")
    test_connection()

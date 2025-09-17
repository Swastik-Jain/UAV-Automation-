# AirSim Remote Connection Setup Instructions

## For Your Friend (AirSim Server Owner)

### Option 1: VPN Setup (Recommended)
1. **Choose a VPN service** that both of you can use:
   - Free options: ProtonVPN, Windscribe
   - Paid options: NordVPN, ExpressVPN
   
2. **Connect to the same VPN server** (same location)
3. **Find your VPN IP address**:
   ```bash
   # Windows Command Prompt
   ipconfig
   
   # Look for VPN adapter IP (usually 10.x.x.x or 192.168.x.x)
   ```
4. **Share your VPN IP** with your friend
5. **Start AirSim** and make sure it's running on port 41451

### Option 2: Port Forwarding Setup
⚠️ **Security Warning**: This exposes AirSim to the internet

1. **Find your public IP**:
   - Visit: https://whatismyipaddress.com/
   
2. **Set up port forwarding on your router**:
   - Access router admin panel (usually 192.168.1.1 or 192.168.0.1)
   - Go to "Port Forwarding" or "Virtual Server"
   - Forward external port 41451 to internal port 41451
   - Set target IP to your computer's local IP
   
3. **Configure Windows Firewall**:
   ```cmd
   # Run as Administrator
   netsh advfirewall firewall add rule name="AirSim" dir=in action=allow protocol=TCP localport=41451
   ```
   
4. **Share your public IP** with your friend

### Option 3: SSH Tunneling Setup
1. **Install OpenSSH Server** (Windows 10/11):
   ```powershell
   # Run as Administrator
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
   
2. **Configure SSH**:
   - Edit `C:\ProgramData\ssh\sshd_config`
   - Add: `AllowTcpForwarding yes`
   - Restart SSH service
   
3. **Set up port forwarding**:
   ```bash
   # Your friend will run this command:
   ssh -L 41451:localhost:41451 username@your-public-ip
   ```

### Option 4: Remote Desktop Setup
1. **Install Parsec** (recommended for gaming):
   - Download from: https://parsec.app/
   - Create account and install on both machines
   - Share your computer with your friend
   
2. **Alternative: TeamViewer**:
   - Download from: https://www.teamviewer.com/
   - Set up unattended access
   - Share credentials with your friend

## For You (Client)

### Step 1: Configure Connection
Edit `airsim_config.py` and set:
```python
CONNECTION_METHOD = "VPN"  # or "PUBLIC", "SSH", "REMOTE_DESKTOP"
```

### Step 2: Update IP Address
Based on your chosen method, update the IP in `airsim_config.py`:
- **VPN**: Use your friend's VPN IP
- **Public**: Use your friend's public IP
- **SSH**: Use "127.0.0.1" (tunnel creates local connection)
- **Remote Desktop**: Use "127.0.0.1" (AirSim runs locally on friend's machine)

### Step 3: Test Connection
```bash
python airsim_config.py
```

### Step 4: Run AirSim Script
```bash
python test.py
```

## Troubleshooting

### Connection Refused
- Check if AirSim is running on your friend's machine
- Verify the IP address and port
- Check firewall settings
- For VPN: Make sure both are on the same VPN server

### High Latency
- Use VPN servers closer to both of you
- Consider using remote desktop for better performance
- Check your internet connection speed

### Security Concerns
- VPN is the most secure option
- Avoid using public IP with port forwarding for extended periods
- Use SSH tunneling for better security than direct port forwarding

## Quick Start Commands

```bash
# Test connection
python airsim_config.py

# Run the training script
python test.py

# Check if AirSim is reachable
telnet [FRIEND_IP] 41451
```

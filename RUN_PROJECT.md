# Project Setup and Run Instructions

This project consists of:
- **Backend**: FastAPI server that connects to AirSim (drone simulator)
- **Frontend**: React/Vite web interface for drone control
- **Training**: PPO (Proximal Policy Optimization) training scripts for the drone

---

## Prerequisites

1. **Python 3.11** (virtual environment is already set up in `env/`)
2. **Node.js** and **npm** (for the frontend)
3. **AirSim** (Microsoft AirSim drone simulator) - either locally or remotely configured

---

## Step 1: Activate Python Virtual Environment

```bash
# Navigate to project directory
cd /Users/sj/Desktop/project

# Activate virtual environment
source env/bin/activate
```

You should see `(env)` in your terminal prompt.

---

## Step 2: Install Python Dependencies

**Recommended: Install from requirements file:**
```bash
# Install backend dependencies (FastAPI, etc.)
pip install -r requirements_backend.txt

# Install additional dependencies for AirSim and training (if needed)
pip install msgpack-rpc-python airsim tensorflow keras opencv-python numpy
```

**Or install individually:**
- `fastapi` - Backend API server
- `uvicorn` - ASGI server
- `websockets` - WebSocket support
- `starlette` - ASGI framework (FastAPI dependency)
- `pydantic` - Data validation (FastAPI dependency)
- `airsim` - AirSim Python client (for full backend)
- `tensorflow` - For neural network training
- `keras` - Deep learning framework
- `opencv-python` (cv2) - Image processing
- `numpy` - Numerical computing

**Note:** If you encounter import errors, the packages may be corrupted. Try:
```bash
pip install --force-reinstall -r requirements_backend.txt
```

---

## Step 3: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd Interface/frontend

# Install npm packages
npm install

# Go back to project root
cd ../..
```

---

## Step 4: Configure AirSim Connection

Edit `airsim_config.py` to set up your AirSim connection:

1. **For local AirSim** (default):
   - Keep `CONNECTION_METHOD = "VPN"` or set to local settings
   - Set `REMOTE_IP = "127.0.0.1"` for local

2. **For remote AirSim**:
   - Choose connection method (VPN, PUBLIC, SSH, or REMOTE_DESKTOP)
   - Update IP addresses accordingly
   - See `SETUP_INSTRUCTIONS.md` for detailed remote setup

**Test connection:**
```bash
python airsim_config.py
```

---

## Step 5: Run the Project

### Option A: Run Frontend Interface (Recommended for UI)

**Terminal 1 - Backend Server:**
```bash
# Make sure virtual environment is activated
source env/bin/activate

# Run the simple backend (mock mode - no AirSim needed)
cd Interface/backend
python simple_backend.py

# OR run the full backend with AirSim:
# python "back_main copy.py"
```

The backend will start on `http://localhost:8000`

**Terminal 2 - Frontend Server:**
```bash
# Navigate to frontend directory
cd Interface/frontend

# Start development server
npm run dev
```

The frontend will start on `http://localhost:3000`

**Open your browser** and navigate to `http://localhost:3000`

---

### Option B: Run Training Script

**For PPO Training:**
```bash
# Activate virtual environment
source env/bin/activate

# Run training
python test.py
```

**For simpler training (main.py):**
```bash
python main.py
```

---

## Step 6: Verify Everything is Running

1. **Backend**: Check terminal for "Starting mock backend server on http://localhost:8000"
2. **Frontend**: Check terminal for Vite dev server URL (usually http://localhost:3000)
3. **AirSim**: If using AirSim, make sure it's running and accessible on the configured IP/port

---

## Troubleshooting

### Backend Issues
- **Port 8000 already in use**: Change port in backend file (`uvicorn.run(app, host="0.0.0.0", port=8001)`)
- **Import errors (FastAPI/uvicorn)**: The virtual environment may have corrupted packages. Try:
  ```bash
  pip install --force-reinstall -r requirements_backend.txt
  ```
  Or reinstall specific packages:
  ```bash
  pip install --force-reinstall fastapi uvicorn starlette pydantic click anyio sniffio
  ```
- **AirSim connection errors**: Check `airsim_config.py` settings and test with `python airsim_config.py`

### Frontend Issues
- **Port 3000 already in use**: Vite will automatically try the next available port
- **npm install fails**: Try `rm -rf node_modules package-lock.json && npm install`
- **CORS errors**: Check that backend CORS settings in backend files allow `http://localhost:3000`

### AirSim Issues
- **Connection refused**: Verify AirSim is running and IP/port configuration is correct
- **No image returned**: Check AirSim camera settings and vehicle name
- See `SETUP_INSTRUCTIONS.md` for detailed remote connection troubleshooting

---

## Project Structure

```
project/
├── Interface/
│   ├── backend/
│   │   ├── simple_backend.py      # Mock backend (no AirSim)
│   │   ├── mock_backend.py        # Alternative mock backend
│   │   └── back_main copy.py      # Full backend with AirSim
│   └── frontend/
│       ├── src/
│       │   └── App.jsx            # Main React component
│       ├── package.json
│       └── vite.config.js
├── main.py                        # Simple training script
├── test.py                        # Full PPO training script
├── airsim_config.py              # AirSim connection configuration
└── SETUP_INSTRUCTIONS.md         # Remote AirSim setup guide
```

---

## Quick Start (Mock Mode - No AirSim)

1. Activate environment: `source env/bin/activate`
2. Start backend: `cd Interface/backend && python simple_backend.py`
3. Start frontend: `cd Interface/frontend && npm run dev`
4. Open browser: `http://localhost:3000`

---

## Next Steps

- Train the drone model: `python test.py`
- Use trained model with backend: Update `back_main copy.py` to load your trained model
- Configure remote AirSim: Follow `SETUP_INSTRUCTIONS.md` for remote setup


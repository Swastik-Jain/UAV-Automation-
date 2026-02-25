import React, { useState, useEffect, useRef } from 'react';
import { Home, Map, Activity, Settings, Power, Video, Navigation, User } from 'lucide-react';

// --- Configuration ---
// MODIFIED: Use window.location.hostname to connect to the correct server IP
const WEBSOCKET_URL = `ws://${window.location.hostname}:8000/ws/mission`;

// --- Sidebar Component ---
const Sidebar = ({ isOpen, view, setView }) => {
    return (
      <div className={`${isOpen ? 'w-64' : 'w-0'} bg-black border-r border-red-900/30 transition-all duration-300 overflow-hidden`}>
        <div className="p-6">
          <h2 className="text-xl font-bold mb-8 text-red-500 tracking-wider">DRONE NAV</h2>
          <nav className="space-y-3">
            <button 
              onClick={() => setView('hud')}
              className={`flex items-center gap-3 w-full p-3 rounded-lg font-medium transition-all ${view === 'hud' ? 'bg-red-900/30 text-red-400 border border-red-900/50' : 'text-gray-400 hover:bg-gray-900'}`}
            >
              <Home size={20} />
              <span>HUD</span>
            </button>
            <button className="flex items-center gap-3 w-full p-3 rounded-lg font-medium text-gray-400 hover:bg-gray-900 transition-all">
              <Map size={20} />
              <span>Maps</span>
            </button>
            <button className="flex items-center gap-3 w-full p-3 rounded-lg font-medium text-gray-400 hover:bg-gray-900 transition-all">
              <Activity size={20} />
              <span>Telemetry</span>
            </button>
            <button className="flex items-center gap-3 w-full p-3 rounded-lg font-medium text-gray-400 hover:bg-gray-900 transition-all">
              <Settings size={20} />
              <span>Settings</span>
            </button>
          </nav>
        </div>
      </div>
    );
  };

// --- TopBar Component ---
const TopBar = ({ startConnection, battery, setSidebarOpen, isConnected }) => {
    return (
      <div className="bg-black border-b border-red-900/30 p-4 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <button 
            onClick={() => setSidebarOpen(prev => !prev)}
            className="p-2 hover:bg-gray-900 rounded-lg transition-all"
          >
            <div className="space-y-1.5">
              <div className="w-6 h-0.5 bg-gray-400 rounded"></div>
              <div className="w-6 h-0.5 bg-gray-400 rounded"></div>
              <div className="w-6 h-0.5 bg-gray-400 rounded"></div>
            </div>
          </button>
          
          <button 
            onClick={startConnection}
            className={`px-6 py-2.5 rounded-lg font-semibold tracking-wide transition-all ${
              isConnected 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-gray-900 border border-gray-700 text-gray-300 hover:bg-gray-800'
            }`} >
            <Power size={16} className="inline mr-2" />
            {isConnected ? 'DISCONNECT' : 'CONNECT'}
          </button>
          <div className={`flex items-center px-6 py-2.5 rounded-lg border font-medium tracking-wide transition-all ${
            isConnected 
              ? 'border-red-500/50 bg-red-950/30 text-red-400 animate-pulse' 
              : 'border-gray-800 text-gray-600'
          }`}>
            <span className="text-xs opacity-70">STATUS:</span> {isConnected ? 'ACTIVE' : 'STANDBY'}
          </div>
        </div>
  
        <div className="flex items-center gap-5">
          <div className="text-gray-300 font-medium">
            <span className="text-xs text-gray-500">BATTERY:</span> {battery.toFixed(0)}%
          </div>
          <button className="p-2.5 hover:bg-gray-900 rounded-lg transition-all border border-gray-800">
            <User size={18} className="text-gray-400" />
          </button>
        </div>
      </div>
    );
  };

// --- LiveFeed Component ---
// FIXED: Now receives 'isConnected' AND 'isActive'
const LiveFeed = ({ isConnected, isActive, imageSrc }) => {
  return (
    <div className="border border-red-900/40 rounded-xl overflow-hidden bg-black relative shadow-lg shadow-red-900/10 flex flex-col">
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/90 to-transparent p-3 border-b border-gray-800/50 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Video size={18} className="text-gray-400" />
            <span className="text-xs font-semibold text-gray-400 tracking-wide">LIVE FEED</span>
          </div>
          {/* FIXED: This logic is now correct because 'isActive' is passed in */}
          {isActive && (
            <div className="flex items-center gap-2 bg-red-600 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></span>
              <span className="text-xs font-bold text-white">Live</span>
            </div>
          )}
        </div>
      </div>
      
      {/* FIXED: Added robust display logic for all states */}
      <div className="flex-1 w-full h-full flex items-center justify-center bg-black">
        {isActive && imageSrc ? (
          // Case 1: Mission active and receiving image
          <img 
            src={imageSrc} 
            alt="Live Feed" 
            className="w-full h-full object-cover transition-all duration-75" 
          />
        ) : isActive && !imageSrc ? (
          // Case 2: Mission active, but no image
          <div className="text-center text-gray-500 animate-pulse">
            <Video size={48} className="mx-auto mb-3" />
            <p className="font-semibold">CONNECTING TO STREAM...</p>
          </div>
        ) : isConnected ? (
          // Case 3: Connected, but mission not active
          <div className="text-center text-gray-600">
             <Video size={48} className="mx-auto mb-3" />
            <p className="font-semibold">DRONE ON STANDBY</p>
            <p className="text-sm">Set a goal on the map and press START.</p>
          </div>
        ) : (
          // Case 4: Not connected at all
          <div className="text-center text-gray-700">
            <Video size={48} className="mx-auto mb-3" />
            <p className="font-semibold">FEED OFFLINE</p>
            <p className="text-sm">Press CONNECT to start.</p>
          </div>
        )}
      </div>
    </div>
  );
};

// --- MapView Component ---
const MapView = ({ isActive, dronePosition, goalPosition, onMapClick, toggleDrone ,isConnected}) => {
  const canvasRef = useRef(null);
  const [mapBounds] = useState({ minX: -100, maxX: 100, minY: -100, maxY: 100 });

  const worldToCanvas = (worldX, worldY, canvas) => {
    const percentX = (worldX - mapBounds.minX) / (mapBounds.maxX - mapBounds.minX);
    const percentY = (1.0 - (worldY - mapBounds.minY) / (mapBounds.maxY - mapBounds.minY)); // Invert Y-axis for canvas
    return {
      x: percentX * canvas.width,
      y: percentY * canvas.height
    };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    // Clear canvas and draw background (dark)
    ctx.fillStyle = '#0a0a0a'; // Dark background
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
        let x = (i / 10) * width;
        let y = (i / 10) * height;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // Draw home base (0,0)
    const home = worldToCanvas(0, 0, canvas);
    ctx.fillStyle = 'cyan';
    ctx.beginPath();
    ctx.arc(home.x, home.y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('Home', home.x + 8, home.y + 4);


    // Draw goal
    if (goalPosition) {
      const goal = worldToCanvas(goalPosition.x, goalPosition.y, canvas);
      ctx.fillStyle = 'lime'; // Bright green for goal
      ctx.beginPath();
      ctx.moveTo(goal.x, goal.y - 8);
      ctx.lineTo(goal.x - 8, goal.y + 8);
      ctx.lineTo(goal.x + 8, goal.y + 8);
      ctx.closePath();
      ctx.fill();
      ctx.fillText('Goal', goal.x + 10, goal.y + 12);
    }

    // Draw drone
    if (dronePosition) {
      const drone = worldToCanvas(dronePosition.x, dronePosition.y, canvas);
      ctx.fillStyle = 'red';
      ctx.beginPath();
      ctx.arc(drone.x, drone.y, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw line from home to drone
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(home.x, home.y);
      ctx.lineTo(drone.x, drone.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [dronePosition, goalPosition, mapBounds]);

  const handleCanvasClick = (event) => {
  if (!isConnected || isActive) return; // Only allow setting goal when connected but not active

  const canvas = canvasRef.current;
  const rect = canvas.getBoundingClientRect();

  // 🔧 FIX: use rect.width/height instead of canvas.width/height
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  // Get click coordinates in canvas coordinate space
  const clickX = (event.clientX - rect.left) * scaleX;
  const clickY = (event.clientY - rect.top) * scaleY;

  // Convert to world coordinates
  const percentX = clickX / canvas.width;
  const percentY = 1.0 - (clickY / canvas.height); // invert Y-axis

  const worldX = mapBounds.minX + percentX * (mapBounds.maxX - mapBounds.minX);
  const worldY = mapBounds.minY + percentY * (mapBounds.maxY - mapBounds.minY);

  onMapClick({ x: worldX, y: worldY });
  };


  return (
    <div className="border border-red-900/40 rounded-xl overflow-hidden bg-black relative shadow-lg shadow-red-900/10 flex flex-col">
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/90 to-transparent p-3 border-b border-gray-800/50 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Navigation size={18} className="text-gray-400" />
            <span className="text-xs font-semibold text-gray-400 tracking-wide">NAVIGATION MAP</span>
          </div>
          <button
            onClick={toggleDrone}
            disabled={!isConnected || (!isActive && !goalPosition)} // Disable if not connected, or if trying to start without a goal
            className={`px-6 py-2.5 rounded-lg font-semibold tracking-wide transition-all ${
              isActive
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-gray-900 border border-gray-700 text-gray-300 hover:bg-gray-800'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <Power size={16} className="inline mr-2" />
            {isActive ? 'STOP MISSION' : 'START MISSION'}
          </button>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={600} // Internal resolution
        height={400} // Internal resolution
        onClick={handleCanvasClick}
        className={`w-full h-full ${isConnected && !isActive ? 'cursor-crosshair' : 'cursor-default'}`}
      />
    </div>
  );
};

// --- TelemetryPanel Component ---
const TelemetryPanel = ({ speed, altitude, lat, lng }) => {
  return (
    <div className="bg-black border-t border-red-900/30 p-5">
      <div className="grid grid-cols-6 gap-4">
        <div className="border border-gray-800 rounded-lg p-3 bg-gray-950/50">
          <div className="text-xs text-gray-600 mb-1 font-medium tracking-wide">SPEED</div>
          <div className="text-2xl font-bold text-gray-300">{speed.toFixed(1)}</div>
          <div className="text-xs text-gray-600 mt-0.5">m/s</div>
        </div>
        <div className="border border-gray-800 rounded-lg p-3 bg-gray-950/50">
          <div className="text-xs text-gray-600 mb-1 font-medium tracking-wide">ALTITUDE</div>
          <div className="text-2xl font-bold text-gray-300">{altitude.toFixed(1)}</div>
          <div className="text-xs text-gray-600 mt-0.5">meters</div>
        </div>
        <div className="border border-gray-800 rounded-lg p-3 bg-gray-950/50 col-span-2">
          <div className="text-xs text-gray-600 mb-1 font-medium tracking-wide">COORDINATES (X, Y)</div>
          <div className="text-lg font-bold text-gray-300">X: {lat.toFixed(4)}</div>
          <div className="text-lg font-bold text-gray-300">Y: {lng.toFixed(4)}</div>
        </div>
        <div className="border border-gray-800 rounded-lg p-3 bg-gray-950/50 col-span-2">
          <div className="text-xs text-gray-600 mb-1 font-medium tracking-wide">MISSION STATUS</div>
          <div className="text-lg font-bold text-green-400">NOMINAL</div>
        </div>
      </div>
    </div>
  );
};

// --- WebSocket Hook ---
// FIXED: This hook now sends the CONNECT message on its own
const useWebSocket = (url, isConnected) => { // Renamed 'isActive' to 'isConnected' for clarity
  const [data, setData] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);

  useEffect(() => {
    // If not supposed to be connected, close any existing connection
    if (!isConnected) {
      if (wsRef.current) {
        console.log('Closing WebSocket connection.');
        wsRef.current.close();
        wsRef.current = null;
        setConnectionStatus('disconnected');
      }
      return;
    }

    // If already connected or connecting, do nothing
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    // Start new connection
    setConnectionStatus('connecting');
    console.log('Attempting to connect to WebSocket...');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setConnectionStatus('connected');
      // FIXED: Send CONNECT message *after* connection is open
      console.log('Sending CONNECT action to backend.');
      ws.send(JSON.stringify({ action: 'CONNECT' }));
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setConnectionStatus('error');
    };

    ws.onmessage = (event) => {
      try {
        const receivedData = JSON.parse(event.data);

        // 🔹 Handle video frames separately
        if (receivedData.type === "video_frame" && receivedData.frame) {
          // Custom event dispatch (so the rest of the app can update the feed)
          const videoEvent = new CustomEvent("droneVideoFrame", { detail: receivedData.frame });
          window.dispatchEvent(videoEvent);
          return; // Skip normal telemetry update
        }

        // Normal telemetry or status data
        setData(receivedData);
      } catch (error) {
        console.error('Error parsing WebSocket data:', error);
      }
    };


    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setConnectionStatus('disconnected');
      wsRef.current = null; // Clear the ref on close
    };

    // Cleanup
    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        console.log('Cleaning up WebSocket.');
        wsRef.current.close();
      }
      wsRef.current = null;
    };
  }, [url, isConnected]); // Effect depends on 'isConnected' state

  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.error('Cannot send message: WebSocket is not open.');
    }
  };

  return { data, connectionStatus, sendMessage };
};

// --- Main App Component ---
function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isActive, setIsActive] = useState(false); // Mission active state
  const [speed, setSpeed] = useState(0);
  const [altitude, setAltitude] = useState(0);
  const [lat, setLat] = useState(0); // Use 0,0 as default
  const [lng, setLng] = useState(0);
  const [battery, setBattery] = useState(100);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState('hud');
  const [goalPosition, setGoalPosition] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [dronePosition, setDronePosition] = useState(null);
  const frameRef = useRef(null);
  const { data: wsData, connectionStatus, sendMessage } = useWebSocket(WEBSOCKET_URL, isConnected);

  // FIXED: This is the new effect to handle incoming data
  useEffect(() => {
    if (wsData) {
      // console.log('Received data:', wsData); // For debugging
      
      // Update telemetry
      if (wsData.speed) setSpeed(wsData.speed);
      if (wsData.altitude) setAltitude(wsData.altitude);
      if (wsData.coordinates) {
        setLat(wsData.coordinates.x);
        setLng(wsData.coordinates.y);
      }
      
      // Update map and feed
      if (wsData.image_src) setImageSrc(wsData.image_src);
      if (wsData.drone_position) setDronePosition(wsData.drone_position);
      
      // Handle status messages
      if (wsData.status === 'goal_reached' || wsData.status === 'collided' || wsData.status === 'stopped') {
        if (wsData.summary) {
            console.log('Mission ended:', wsData.summary.outcome);
        }
        setIsActive(false); // Automatically stop the mission on the UI
      }
      
      if (wsData.action === 'STOP_MISSION_ACK') {
          console.log('Backend acknowledged STOP_MISSION.');
          setIsActive(false);
          // Clear UI data
          setSpeed(0);
          setAltitude(0);
          setImageSrc(null);
          setDronePosition(null);
      }

      // Handle connection acknowledgment from backend
      if (wsData.action === 'CONNECT_ACK') {
        if (wsData.drone_con) {
          console.log('Backend confirmed drone connection.');
        } else {
          console.error('Backend failed to connect to drone:', wsData.error);
          setIsConnected(false); // Auto-disconnect UI if backend fails
        }
      }
    }
  }, [wsData]); // This effect runs every time new data arrives

  // === Unified live video event listener ===
  useEffect(() => {
    const handleFrame = (event) => {
      // MODIFIED: Only update image if mission is active
      if (!isActive && !imageSrc) return; // Don't show stale frames
      if (!isActive && imageSrc) {
        setImageSrc(null); // Clear image if mission just stopped
        return;
      }

      // Avoid redundant re-renders
      if (frameRef.current !== event.detail) {
        frameRef.current = event.detail;
        setImageSrc(event.detail);
      }
    };

    window.addEventListener("droneVideoFrame", handleFrame);
    return () => {
      window.removeEventListener("droneVideoFrame", handleFrame);
    };
  }, [isConnected, isActive, imageSrc]); // MODIFIED: Added isActive and imageSrc dependencies

  
  const startConnection = () => {
    // Just toggle the state. The hook will handle all connection/disconnect logic.
    const newIsConnected = !isConnected;
    setIsConnected(newIsConnected);

    if (!newIsConnected) {
      // If we are DISCONNECTING (current state isConnected=true)
      console.log('🔌 Sending DISCONNECT action.');
      sendMessage({ action: 'DISCONNECT' }); // <-- MODIFIED: Was 'CONNECT'
      
      // Also stop any active mission and clear UI
      setIsActive(false);
      setGoalPosition(null);
      setDronePosition(null);
      setImageSrc(null);
      setSpeed(0);
      setAltitude(0);
    }
    // If we are connecting, the useWebSocket hook handles sending 'CONNECT'
  };

  const toggleDrone = () => {
    if (!isActive && !goalPosition) {
      console.warn('⚠️ Cannot start mission: No goal position set');
      return;
    }
    
    const newActiveState = !isActive;
    // setIsActive(newActiveState); // MODIFIED: Let the backend confirm the state
    
    if (newActiveState) {
      // Starting mission
      console.log('🚀 Starting mission with goal:', goalPosition);
      setIsActive(true); // Optimistically set UI to active
      sendMessage({ 
        action: 'START_MISSION',
        goal: {
          x: goalPosition.x,
          y: goalPosition.y
        }
      });
    } else {
      // Stopping mission
      console.log('🛑 Stopping mission');
      sendMessage({ action: 'STOP_MISSION' }); 
      
      // MODIFIED: Instantly clear the UI data
      // We will wait for the STOP_MISSION_ACK to set isActive(false)
      // but we can clear the telemetry immediately.
      setSpeed(0);
      setAltitude(0);
      setImageSrc(null);
      setDronePosition(null);
      // We keep the goalPosition so the user can see where they were headed
    }
  };

  const handleMapClick = (position) => {
    if (!isConnected || isActive) return;
    
    console.log('🎯 Goal position set:', position);
    setGoalPosition(position);
  };
    
  return (
    <div className="flex h-screen bg-black text-gray-400 font-mono overflow-hidden">
      <Sidebar isOpen={sidebarOpen} view={view} setView={setView} />

      <div className="flex-1 flex flex-col">
        <TopBar 
          startConnection={startConnection} 
          battery={battery} 
          setSidebarOpen={setSidebarOpen}
          isConnected={isConnected}
        />
        <div className="flex-1 p-5 grid grid-cols-2 gap-5 overflow-auto">
          {/* FIXED: Pass both isConnected and isActive */}
          <LiveFeed 
            isConnected={isConnected} 
            isActive={isActive} 
            imageSrc={imageSrc} 
          />
          <MapView 
            isActive={isActive}
            isConnected={isConnected} // Pass isConnected to MapView
            dronePosition={dronePosition}
            goalPosition={goalPosition}
            onMapClick={handleMapClick}
            toggleDrone={toggleDrone} // Pass toggleDrone to MapView
          />
        </div>
        <TelemetryPanel 
          speed={speed} 
          altitude={altitude} 
          lat={lat} 
          lng={lng}
        />
      </div>
    </div>
  );
}

export default App;
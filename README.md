# Smart Grid AI Framework

A comprehensive web application for power system simulation and AI-driven grid management.

## Project Structure

```
app/
├── backend/                 # FastAPI backend
│   ├── api/
│   │   ├── routes/         # REST API endpoints
│   │   │   ├── grid.py         # Grid operations (load, state, topology)
│   │   │   ├── simulation.py   # Simulation control
│   │   │   ├── forecasting.py  # ML forecasting (placeholder)
│   │   │   └── diagnostics.py  # Fault detection (placeholder)
│   │   └── websockets/     # Real-time communication
│   │       └── handlers.py
│   ├── models/             # Pydantic schemas
│   ├── services/           # Business logic
│   │   ├── opendss_service.py  # OpenDSS integration
│   │   └── simulation_service.py
│   ├── config.py           # Configuration settings
│   ├── main.py             # FastAPI application
│   └── requirements.txt
│
└── frontend/               # React + TypeScript frontend
    ├── src/
    │   ├── components/     # React components
    │   │   ├── layout/     # Layout components
    │   │   ├── dashboard/  # Dashboard widgets
    │   │   └── ui/         # UI components
    │   ├── pages/          # Page components
    │   │   ├── Dashboard.tsx
    │   │   ├── GridViewer.tsx
    │   │   ├── SelfHealing.tsx
    │   │   ├── Forecasting.tsx
    │   │   ├── Diagnostics.tsx
    │   │   ├── NetLoad.tsx
    │   │   └── Settings.tsx
    │   ├── hooks/          # Custom React hooks
    │   ├── services/       # API and WebSocket services
    │   ├── stores/         # Zustand state management
    │   └── types/          # TypeScript type definitions
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── tsconfig.json
```

## Research Components

This application supports four research sub-components:

| Component | Student | Focus | Status |
|-----------|---------|-------|--------|
| Self-Healing Grid | IT22053350 | MARL + GNN for autonomous restoration | Placeholder |
| Solar Forecasting | IT22360182 | Stacked Ensemble ML model | Placeholder |
| Fault Diagnostics | IT22577924 | CNN-Transformer + R-GNN | Placeholder |
| Net Load Forecast | IT22891204 | ICEEMDAN + Transformer + GP-RML | Placeholder |

## Setup Instructions

### Prerequisites

- Python 3.9+ with pip
- Node.js 18+ with npm
- OpenDSS (installed via opendssdirect.py)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd app/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python main.py
   # Or with uvicorn:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at:
- REST API: http://localhost:8000/api/v1
- WebSocket: ws://localhost:8000/ws
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd app/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at: http://localhost:5173

## API Endpoints

### Grid API (`/api/v1/grid`)
- `POST /load` - Load OpenDSS model
- `GET /state` - Get current grid state
- `GET /topology` - Get network topology
- `POST /load-multiplier` - Set load multiplier
- `POST /generation-multiplier` - Set generation multiplier
- `POST /inject-fault` - Inject a test fault

### Simulation API (`/api/v1/simulation`)
- `POST /start` - Start simulation
- `POST /stop` - Stop simulation
- `POST /pause` - Pause simulation
- `POST /resume` - Resume simulation
- `POST /step` - Execute single step
- `GET /status` - Get simulation status
- `GET /history` - Get simulation history

### Forecasting API (`/api/v1/forecasting`)
- `POST /load` - Forecast load demand
- `POST /solar` - Forecast solar generation
- `POST /net-load` - Forecast net load
- `GET /imbalance-detection` - Detect imbalance state

### Diagnostics API (`/api/v1/diagnostics`)
- `POST /detect` - Run fault detection
- `GET /self-healing/status` - Get self-healing status
- `POST /self-healing/trigger` - Trigger self-healing
- `GET /hif-detection` - Detect high-impedance faults

## WebSocket Communication

Connect to `ws://localhost:8000/ws` for real-time updates.

### Client → Server Messages:
```json
{"action": "start", "params": {"hours": 24, "speed": 10}}
{"action": "stop"}
{"action": "pause"}
{"action": "resume"}
{"action": "step"}
{"action": "get_state"}
{"action": "get_status"}
```

### Server → Client Messages:
```json
{"type": "state_update", "data": {...}, "timestamp": "..."}
{"type": "status", "data": {...}}
{"type": "error", "message": "..."}
{"type": "info", "message": "..."}
```

## Technology Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **opendssdirect.py** - OpenDSS Python interface
- **Pydantic** - Data validation
- **WebSockets** - Real-time communication

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Flow** - Network visualization
- **Recharts** - Charts and graphs
- **Zustand** - State management
- **TanStack Query** - Data fetching

## Integrating ML Models

The application is designed for seamless ML model integration. Each research component has placeholder endpoints that can be replaced with actual model inference:

1. **Self-Healing (MARL + GNN)**
   - Update `api/routes/diagnostics.py`
   - Load PyTorch/TensorFlow models in service layer
   - Implement agent inference in `trigger_self_healing()`

2. **Solar Forecasting (Stacked Ensemble)**
   - Update `api/routes/forecasting.py`
   - Load trained ensemble models
   - Implement prediction in `forecast_solar()`

3. **Fault Diagnostics (CNN-Transformer + R-GNN)**
   - Update `api/routes/diagnostics.py`
   - Load PyTorch Geometric models
   - Implement inference in `detect_fault()`

4. **Net Load Forecasting (ICEEMDAN + Transformer + GP)**
   - Update `api/routes/forecasting.py`
   - Load trained models
   - Implement prediction in `forecast_net_load()`

## License

Research project for SLIIT - Project ID: 25-26J-092

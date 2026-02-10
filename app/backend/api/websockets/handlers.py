"""
WebSocket handlers for real-time communication.
Manages client connections and broadcasts simulation state updates.
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Set
import json
import asyncio
import logging
from datetime import datetime

from services import simulation_service

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._callbacks: Dict[WebSocket, Any] = {}  # Track callbacks per connection
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

        # Create and store callback for this connection
        callback = self._create_broadcast_callback(websocket)
        self._callbacks[websocket] = callback
        simulation_service.subscribe(callback)

        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection and clean up its callback."""
        async with self._lock:
            self.active_connections.discard(websocket)

            # Unsubscribe the callback for this connection
            if websocket in self._callbacks:
                callback = self._callbacks.pop(websocket)
                simulation_service.unsubscribe(callback)

        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    def _create_broadcast_callback(self, websocket: WebSocket):
        """Create a callback for broadcasting to a specific client."""
        async def callback(data: Dict[str, Any]):
            try:
                message = {
                    "type": "state_update",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        return callback

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = set()

        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        async with self._lock:
            self.active_connections -= disconnected

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time grid simulation data.

    Supports the following message types:

    Client -> Server:
    - {"action": "start", "params": {"hours": 24, "speed": 1.0}}
    - {"action": "stop"}
    - {"action": "pause"}
    - {"action": "resume"}
    - {"action": "step"}
    - {"action": "set_speed", "params": {"speed": 2.0}}
    - {"action": "get_state"}
    - {"action": "get_status"}
    - {"action": "ping"}

    Server -> Client:
    - {"type": "state_update", "data": {...}, "timestamp": "..."}
    - {"type": "status", "data": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    - {"type": "info", "message": "..."}
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        await manager.send_personal(websocket, {
            "type": "info",
            "message": "Connected to Smart Grid AI Framework",
            "timestamp": datetime.now().isoformat()
        })

        # Send current status
        status = simulation_service.get_status()
        await manager.send_personal(websocket, {
            "type": "status",
            "data": status,
            "timestamp": datetime.now().isoformat()
        })

        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_json()
                await handle_client_message(websocket, data)
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": "Invalid JSON message"
                })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


async def _broadcast_status():
    """Broadcast current simulation status to all connected clients."""
    status = simulation_service.get_status()
    await manager.broadcast({
        "type": "status",
        "data": status,
        "timestamp": datetime.now().isoformat()
    })


async def handle_client_message(websocket: WebSocket, data: Dict[str, Any]):
    """Handle incoming client messages."""
    action = data.get("action", "").lower()
    params = data.get("params", {})

    try:
        if action == "ping":
            await manager.send_personal(websocket, {"type": "pong"})

        elif action == "start":
            hours = params.get("hours", 24)
            speed = params.get("speed", 1.0)
            mode = params.get("mode", "synthetic")
            result = await simulation_service.start(hours=hours, speed=speed, mode=mode)
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "start",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
            # Broadcast updated status to ALL clients
            await _broadcast_status()

        elif action == "stop":
            result = await simulation_service.stop()
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "stop",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
            # Broadcast updated status to ALL clients
            await _broadcast_status()

        elif action == "pause":
            result = await simulation_service.pause()
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "pause",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
            # Broadcast updated status to ALL clients
            await _broadcast_status()

        elif action == "resume":
            result = await simulation_service.resume()
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "resume",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
            # Broadcast updated status to ALL clients
            await _broadcast_status()

        elif action == "step":
            result = await simulation_service.step()
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "step",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })

        elif action == "set_speed":
            speed = params.get("speed", 1.0)
            result = simulation_service.set_speed(speed)
            await manager.send_personal(websocket, {
                "type": "response",
                "action": "set_speed",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })

        elif action == "get_state":
            state = simulation_service.current_state
            if state:
                state_dict = simulation_service._state_to_dict(state)
                await manager.send_personal(websocket, {
                    "type": "state_update",
                    "data": state_dict,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await manager.send_personal(websocket, {
                    "type": "info",
                    "message": "No simulation state available"
                })

        elif action == "get_status":
            status = simulation_service.get_status()
            await manager.send_personal(websocket, {
                "type": "status",
                "data": status,
                "timestamp": datetime.now().isoformat()
            })

        elif action == "get_history":
            limit = params.get("limit", 100)
            history = simulation_service.get_history(limit=limit)
            await manager.send_personal(websocket, {
                "type": "history",
                "data": {"history": history},
                "timestamp": datetime.now().isoformat()
            })

        else:
            await manager.send_personal(websocket, {
                "type": "error",
                "message": f"Unknown action: {action}"
            })

    except Exception as e:
        logger.error(f"Error handling action '{action}': {e}")
        await manager.send_personal(websocket, {
            "type": "error",
            "message": str(e)
        })

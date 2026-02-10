/**
 * WebSocket Service for real-time grid simulation updates
 */

import type { WSMessage, WSControlMessage, GridState } from '../types';

type MessageHandler = (message: WSMessage) => void;
type StateUpdateHandler = (state: GridState) => void;
type ConnectionHandler = () => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: Set<MessageHandler> = new Set();
  private stateUpdateHandlers: Set<StateUpdateHandler> = new Set();
  private connectHandlers: Set<ConnectionHandler> = new Set();
  private disconnectHandlers: Set<ConnectionHandler> = new Set();
  private pingInterval: NodeJS.Timeout | null = null;

  constructor(url: string = 'ws://localhost:8000/ws') {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startPing();
        this.connectHandlers.forEach(handler => handler());
        resolve();
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.stopPing();
        this.disconnectHandlers.forEach(handler => handler());
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    });
  }

  disconnect(): void {
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private handleMessage(message: WSMessage): void {
    // Notify all message handlers
    this.messageHandlers.forEach(handler => handler(message));

    // Handle state updates specifically
    if (message.type === 'state_update' && message.data) {
      this.stateUpdateHandlers.forEach(handler =>
        handler(message.data as GridState)
      );
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect().catch(() => {
        // Reconnection failed, will try again
      });
    }, delay);
  }

  private startPing(): void {
    this.pingInterval = setInterval(() => {
      this.send({ action: 'ping' });
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  send(message: WSControlMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }

  // ============== Subscription Methods ==============

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onStateUpdate(handler: StateUpdateHandler): () => void {
    this.stateUpdateHandlers.add(handler);
    return () => this.stateUpdateHandlers.delete(handler);
  }

  onConnect(handler: ConnectionHandler): () => void {
    this.connectHandlers.add(handler);
    return () => this.connectHandlers.delete(handler);
  }

  onDisconnect(handler: ConnectionHandler): () => void {
    this.disconnectHandlers.add(handler);
    return () => this.disconnectHandlers.delete(handler);
  }

  // ============== Control Methods ==============

  startSimulation(hours: number = 24, speed: number = 1.0, mode: string = 'synthetic'): void {
    this.send({ action: 'start', params: { hours, speed, mode } });
  }

  stopSimulation(): void {
    this.send({ action: 'stop' });
  }

  pauseSimulation(): void {
    this.send({ action: 'pause' });
  }

  resumeSimulation(): void {
    this.send({ action: 'resume' });
  }

  stepSimulation(): void {
    this.send({ action: 'step' });
  }

  setSpeed(speed: number): void {
    this.send({ action: 'set_speed', params: { speed } });
  }

  getState(): void {
    this.send({ action: 'get_state' });
  }

  getStatus(): void {
    this.send({ action: 'get_status' });
  }

  // ============== Status ==============

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }
}

export const wsService = new WebSocketService();
export default wsService;

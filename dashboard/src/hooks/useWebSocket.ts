/**
 * WebSocket hook for real-time progress updates.
 * Manages WebSocket connection lifecycle, heartbeat, and auto-reconnect with exponential backoff.
 */

import { useEffect, useRef, useState } from 'react';
import type { RunProgress } from '../types/pipeline';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'error';

interface UseWebSocketOptions {
  onMessage?: (data: RunProgress) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onStateChange?: (state: ConnectionState) => void;
  autoReconnect?: boolean;
  heartbeatInterval?: number;
  staleConnectionTimeout?: number;
}

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  isConnected: boolean;
  lastMessage: RunProgress | null;
  error: Event | null;
  reconnect: () => void;
}

const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = `${WS_PROTOCOL}://${window.location.host}/ws/progress`;

const MIN_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const DEFAULT_HEARTBEAT_INTERVAL = 30000;
const DEFAULT_STALE_TIMEOUT = 120000;  // 2 minutes (increased from 90s)

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    autoReconnect = true,
    heartbeatInterval = DEFAULT_HEARTBEAT_INTERVAL,
    staleConnectionTimeout = DEFAULT_STALE_TIMEOUT,
  } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [lastMessage, setLastMessage] = useState<RunProgress | null>(null);
  const [error, setError] = useState<Event | null>(null);

  // Store all callbacks in refs to avoid re-renders
  const optionsRef = useRef(options);
  optionsRef.current = options;  // Update immediately, not in useEffect

  // Connection management refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const heartbeatIntervalRef = useRef<number | null>(null);
  const staleCheckIntervalRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(true);
  const reconnectAttemptsRef = useRef(0);
  const lastMessageTimeRef = useRef<number>(Date.now());
  const isConnectingRef = useRef(false);  // Prevent multiple simultaneous connections

  // Manual reconnect function
  const reconnectRef = useRef<() => void>();

  useEffect(() => {
    // Helper to update state
    const updateState = (newState: ConnectionState) => {
      setConnectionState(prevState => {
        if (prevState !== newState) {
          console.log(`[WebSocket] ${prevState} → ${newState}`);
          optionsRef.current.onStateChange?.(newState);
        }
        return newState;
      });
    };

    // Clear all intervals
    const clearIntervals = () => {
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
      if (staleCheckIntervalRef.current) {
        clearInterval(staleCheckIntervalRef.current);
        staleCheckIntervalRef.current = null;
      }
    };

    // Send heartbeat ping
    const sendHeartbeat = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'ping' }));
        } catch (err) {
          console.error('[WebSocket] Failed to send ping:', err);
        }
      }
    };

    // Check for stale connection
    const checkStale = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const timeSinceLastMessage = Date.now() - lastMessageTimeRef.current;
        if (timeSinceLastMessage > staleConnectionTimeout) {
          console.warn(`[WebSocket] Stale connection (${timeSinceLastMessage}ms), closing...`);
          wsRef.current.close();
        }
      }
    };

    // Start heartbeat and stale check
    const startMonitoring = () => {
      clearIntervals();

      heartbeatIntervalRef.current = window.setInterval(sendHeartbeat, heartbeatInterval);
      staleCheckIntervalRef.current = window.setInterval(checkStale, 60000); // Check every minute

      console.log('[WebSocket] Monitoring started');
    };

    // Calculate reconnect delay with exponential backoff
    const getReconnectDelay = () => {
      return Math.min(
        MIN_RECONNECT_DELAY * Math.pow(2, reconnectAttemptsRef.current),
        MAX_RECONNECT_DELAY
      );
    };

    // Main connection function
    const connect = () => {
      if (!shouldReconnectRef.current || isConnectingRef.current) {
        console.log('[WebSocket] Skipping connect - shouldReconnect:', shouldReconnectRef.current, 'isConnecting:', isConnectingRef.current);
        return;
      }

      // Close existing connection
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore errors when closing
        }
        wsRef.current = null;
      }

      isConnectingRef.current = true;
      console.log(`[WebSocket] Connecting to ${WS_URL}...`);
      updateState(reconnectAttemptsRef.current > 0 ? 'reconnecting' : 'connecting');

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[WebSocket] Connected');
          isConnectingRef.current = false;
          updateState('connected');
          setError(null);
          reconnectAttemptsRef.current = 0;
          lastMessageTimeRef.current = Date.now();
          optionsRef.current.onConnect?.();
          startMonitoring();
        };

        ws.onmessage = (event) => {

          try {
            const data = JSON.parse(event.data);
            lastMessageTimeRef.current = Date.now();

            if (data.type === 'ping') {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'pong' }));
              }
              return;
            }

            if (data.type === 'pong') {
              return;
            }

            setLastMessage(data);
            optionsRef.current.onMessage?.(data);
          } catch (err) {
            console.error('[WebSocket] Parse error:', err);
          }
        };

        ws.onerror = (event) => {
          console.error('[WebSocket] Error');
          isConnectingRef.current = false;
          setError(event);
          updateState('error');
          optionsRef.current.onError?.(event);
        };

        ws.onclose = (event) => {
          console.log(`[WebSocket] Closed (code: ${event.code})`);
          isConnectingRef.current = false;
          clearIntervals();
          updateState('disconnected');
          optionsRef.current.onDisconnect?.();

          // Auto-reconnect
          if (shouldReconnectRef.current && autoReconnect) {
            const delay = getReconnectDelay();
            reconnectAttemptsRef.current++;
            reconnectTimeoutRef.current = window.setTimeout(() => {
              if (shouldReconnectRef.current) {
                connect();
              }
            }, delay);
          }
        };
      } catch (err) {
        console.error('[WebSocket] Connection error:', err);
        isConnectingRef.current = false;
        updateState('error');
      }
    };

    // Manual reconnect function
    reconnectRef.current = () => {
      console.log('[WebSocket] Manual reconnect');
      reconnectAttemptsRef.current = 0;

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      connect();
    };

    // Initialize connection
    shouldReconnectRef.current = true;
    connect();

    // Cleanup
    return () => {
      console.log('[WebSocket] Cleanup - disabling reconnect and closing connection');
      shouldReconnectRef.current = false;
      isConnectingRef.current = false;  // Reset flag so next mount can connect

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      clearIntervals();

      // Only close if connection is established or still connecting
      // Don't interfere with connections that are in progress
      if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
        console.log('[WebSocket] Closing WebSocket in cleanup, readyState:', wsRef.current.readyState);
        try {
          wsRef.current.close();
        } catch (e) {
          console.error('[WebSocket] Error closing in cleanup:', e);
        }
      }
    };
  }, []); // Empty deps - only run once on mount!

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    lastMessage,
    error,
    reconnect: () => reconnectRef.current?.(),
  };
};

export default useWebSocket;

import os
import queue
from dotenv import load_dotenv
# --- FIX: Import the ORIGINAL WebSocket Manager and Client ---
from binance import ThreadedWebsocketManager
from binance.client import Client

class BinanceStreamer:
    """
    Connects to Binance.US WebSocket and puts kline data into a thread-safe queue.
    """
    def __init__(self, data_queue: queue.Queue):
        load_dotenv()
        self.api_key = os.environ.get("BINANCE_API_KEY")
        self.api_secret = os.environ.get("BINANCE_API_SECRET")
        
        # --- FIX: Initialize the WebSocket Manager with tld='us' ---
        self.twm = ThreadedWebsocketManager(tld='us')
        self.queue = data_queue
        self.symbol = 'BTCUSDT'
        print("Binance.US Streamer initialized.")

    def _handle_socket_message(self, msg):
        """
        Callback function to handle incoming kline messages.
        """
        if msg.get('e') == 'kline':
            kline = msg.get('k', {})
            if kline.get('i') == '1m' and kline.get('x'): # If the kline is closed
                processed_data = {
                    'time': int(kline['t']),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                self.queue.put(processed_data)

    def start_stream(self):
        """
        Starts the WebSocket stream in a background thread.
        """
        print(f"Starting WebSocket stream for {self.symbol} on Binance.US...")
        self.twm.start()
        
        # Pass API keys to the start_kline_socket for authentication [3]
        self.twm.start_kline_socket(
            callback=self._handle_socket_message,
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        self.twm.join()

    def stop_stream(self):
        """
        Stops the WebSocket stream.
        """
        print("Stopping WebSocket stream...")
        self.twm.stop()

if __name__ == "__main__":
    # Test the streamer
    import time
    import threading
    
    test_queue = queue.Queue()
    streamer = BinanceStreamer(data_queue=test_queue)
    
    stream_thread = threading.Thread(target=streamer.start_stream, daemon=True)
    stream_thread.start()
    
    print("Stream started. Waiting for 3 minutes to collect data...")
    
    try:
        for _ in range(3):
            time.sleep(60)
            print(f"\nChecking queue... Queue size: {test_queue.qsize()}")
            while not test_queue.empty():
                data = test_queue.get()
                print(f"  {data}")
                
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        streamer.stop_stream()
        print("Stream stopped.")
from binance import ThreadedWebsocketManager
from binance.client import Client
import queue

class BinanceStreamer:
    """
    Connects to Binance WebSocket and puts kline data into a thread-safe queue.
    """
    def __init__(self, data_queue: queue.Queue):
        self.twm = ThreadedWebsocketManager(tld='us')
        self.queue = data_queue
        self.symbol = 'BTCUSDT'
        print("Binance Streamer initialized.")

    def _handle_socket_message(self, msg):
        """
        Callback function to handle incoming kline messages.
        """
        # We only care about kline data
        if msg['e'] == 'kline':
            kline = msg['k']
            
            # We are interested in 1-minute klines
            if kline['i'] == '1m':
                # We only put the message in the queue if the kline is closed
                if kline['x']: 
                    processed_data = {
                        'time': int(kline['t']),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }
                    # Put the processed data into the queue
                    self.queue.put(processed_data)

    def start_stream(self):
        """
        Starts the WebSocket stream in a background thread.
        """
        print(f"Starting WebSocket stream for {self.symbol}...")
        self.twm.start()
        
        # Subscribe to the 1-minute kline socket
        self.twm.start_kline_socket(
            callback=self._handle_socket_message,
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE
        )
        
        # This join() keeps the thread alive
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
    
    test_queue = queue.Queue()
    streamer = BinanceStreamer(data_queue=test_queue)
    
    # We must run start_stream in a separate thread, just as we will in Streamlit
    import threading
    stream_thread = threading.Thread(target=streamer.start_stream, daemon=True)
    stream_thread.start()
    
    print("Stream started. Waiting for 3 minutes to collect data...")
    
    try:
        for _ in range(3):
            time.sleep(60) # Wait for 1 minute
            print(f"\nChecking queue... Queue size: {test_queue.qsize()}")
            while not test_queue.empty():
                data = test_queue.get()
                print(f"  {data}")
                
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        streamer.stop_stream()
        print("Stream stopped.")
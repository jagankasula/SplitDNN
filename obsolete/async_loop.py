import tornado.ioloop
import tornado.locks

count = 0
event = tornado.locks.Event()

async def process_data():
    global count
    await event.wait()  # Wait until the event is set
    event.clear()  # Reset the event for the next iteration
        
    # Your code to process the data goes here
    print(f"All data processed: {count}")
    

def set_count():
    global count
    if count < 5:
        # Simulating some operation that sets the count
        print(f"Inside set_count: {count}")
        count += 1
    if count == 5:
        event.set()  # Set the event to signal that count has been updated

def main():
    # Create a periodic callback to call set_count() every second
    periodic_callback = tornado.ioloop.PeriodicCallback(set_count, 1000)  # 1000ms = 1s
    periodic_callback.start()
    
    # Start the IOLoop and run process_data() coroutine
    ioloop = tornado.ioloop.IOLoop.current()
    ioloop.add_callback(process_data)
    ioloop.start()

if __name__ == "__main__":
    main()

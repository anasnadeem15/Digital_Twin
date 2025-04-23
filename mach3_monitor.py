# mach3_monitor.py (Updated)
import os
import time
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GCodeHandler(FileSystemEventHandler):
    def __init__(self, api_url):
        self.api_url = api_url
        self.processed_files = set()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.nc', '.txt', '.gcode')):
            self.process_file(event.src_path)
    
    def process_file(self, path):
        try:
            if path in self.processed_files:
                return
            
            with open(path, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/upload",
                    files={'file': (os.path.basename(path), f)}
                )
            
            if response.status_code == 200:
                self.processed_files.add(path)
                print(f"Processed {path} successfully")
            else:
                print(f"Error processing {path}: {response.text}")
        
        except Exception as e:
            print(f"Failed to process {path}: {str(e)}")

def start_monitoring(directory, api_url):
    event_handler = GCodeHandler(api_url)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring(
        directory="C:\\Mach3\\GCode",
        api_url="http://localhost:5000"
    )
import os
import time
import requests

# Path to Mach3's G-code directory (update this path!)
MACH3_GCODE_DIR = r"C:\Mach3\Gcode"  # Use raw string to avoid issues with backslashes

# URL of your web app's G-code processing endpoint
WEB_APP_URL = "http://127.0.0.1:5000/process_gcode"  # Local Flask app URL

def upload_gcode(file_path):
    """
    Uploads a G-code file to the web app for processing.
    """
    try:
        with open(file_path, 'r') as file:
            gcode = file.read()
            print(f"Uploading file: {file_path}")
            response = requests.post(WEB_APP_URL, json={"gcode": gcode})
            
            if response.status_code == 200:
                print(f"Processed: {file_path}")
                print(f"Response: {response.json()}")
            else:
                print(f"Failed to process: {file_path}. Status code: {response.status_code}")
                print(f"Response: {response.text}")  # Debug: Print the server's response
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def monitor_directory():
    """
    Monitors the Mach3 G-code directory for new files.
    """
    print("Monitoring Mach3 G-code directory...")
    processed_files = set()  # Track processed files to avoid duplicates

    while True:
        for filename in os.listdir(MACH3_GCODE_DIR):
            if filename.endswith((".nc", ".txt")) and filename not in processed_files:
                file_path = os.path.join(MACH3_GCODE_DIR, filename)
                upload_gcode(file_path)
                processed_files.add(filename)  # Mark file as processed
        time.sleep(5)  # Check the directory every 5 seconds

if __name__ == "__main__":
    monitor_directory()
import subprocess
import threading
import time
import pyautogui
import pygetwindow as gw
import ctypes
from inspect import getsourcefile
from os.path import dirname

# Make the application DPI-aware
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except AttributeError:
    # Fallback if DPI-awareness is not available
    pass


def start_openvibe_acquisition_server():
    # Path to the OpenViBE Acquisition Server executable
    ov_acquisition_server_path = "C:/Program Files/openvibe-3.6.0-64bit/bin/openvibe-acquisition-server.exe"

    # Start the Acquisition Server
    subprocess.run([ov_acquisition_server_path])


def run_openvibe_xml(cmd, xml_path):
        try:
            # Run OpenViBE with the specified XML file
            fullCommand = f"{cmd} --open {xml_path}"
            # f"{cmd} --play {xml_path}"
            print(fullCommand)
            subprocess.run(fullCommand, shell=True)
            print("OpenViBE process completed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error: OpenViBE process failed with exit code", e.returncode)


def get_connect_play_position():  
    try:
        # Get the window by title
        windows = gw.getAllWindows()
        window = [window for window in windows if "Acquisition Server v" in str(window.title)][0]
        if not window:
            return "Window not found."
    except Exception as e:
        return str(e)
    
    top, left, width, height = window.top, window.left, window.width, window.height

    x = left + 0.92*width
    y_c = top + 0.32*height
    y_s = top + 0.41*height
    print(x, y_c, y_s)
    
    return x, y_c, y_s

def automate_gui_interaction():
    # Then press "Connect" and "Play"
    x, y_c, y_s = get_connect_play_position()
    pyautogui.moveTo(x=x,y=y_c)
    pyautogui.click()
    pyautogui.moveTo(x=x,y=y_s)
    pyautogui.click()


# Get location of this file to find path to .xml files
currentDic = dirname(getsourcefile(lambda:0))

# Get the actual.xml files
openvibe_path = "C:/Program Files/openvibe-3.6.0-64bit/bin/openvibe-designer.exe"
xml_file = f"{currentDic}/code/main.xml"

# Create threads for each function
thread1 = threading.Thread(target=start_openvibe_acquisition_server)
thread2 = threading.Thread(target=automate_gui_interaction)
thread3 = threading.Thread(target=run_openvibe_xml,args=[openvibe_path, xml_file])

# Start threads
thread1.start()
time.sleep(0.5)
thread2.start()
time.sleep(0.5)
thread3.start()

# run_openvibe_xml(openvibe_path, xml_file)
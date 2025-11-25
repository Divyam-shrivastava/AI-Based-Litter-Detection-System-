# **AI-Based Litter Detection System 🚯📷**

This project is an automated litter detection system that uses Computer Vision and AI to identify trash in real-time. It utilizes an **ESP32-CAM** module for video streaming and a **YOLOv8** model (converted to ONNX) for object detection. When litter is detected, the system automatically sends an alert with an image via WhatsApp.

## **🌟 Features**

* **Real-time Detection:** Uses opencv-python and a custom trained .onnx model to detect objects like bottles, cups, and trash.  
* **Wireless Streaming:** Fetches video feed wirelessly from an ESP32-CAM.  
* **Smart Alerts:** Sends a WhatsApp message with a snapshot of the litter using pywhatkit.  
* **Cooldown System:** Prevents spamming by implementing a timer between alerts.

## **🛠️ Hardware Required**

* **ESP32-CAM Module**: For capturing video.  
* **FTDI Programmer (USB to TTL)**: To upload code to the ESP32-CAM (as seen in the project photos).  
* **5V Power Supply**: To power the camera (or via USB).  
* **Jumper Wires & Breadboard**.

## **⚙️ Installation & Setup**

### **1\. Python Setup**

Ensure you have Python 3.x installed. Clone this repository and install the required dependencies:

pip install \-r requirements.txt

### **2\. Model Setup**

Ensure the litter.onnx file is placed in the root directory of this project. This file contains the trained YOLOv8 weights.

### **3\. ESP32-CAM Setup**

1. Flash your ESP32-CAM with the standard "Camera Web Server" example from the Arduino IDE.  
2. Note the IP address printed in the Serial Monitor (e.g., http://192.168.1.15).  
3. Ensure your PC and the ESP32-CAM are on the **same WiFi network**.

### **4\. Configuration**

Open main02.py and update the following lines:

* **Stream URL:** Add your ESP32-CAM's IP address to the CANDIDATE\_URLS list.  
  CANDIDATE\_URLS \= \[  
      "\[http://192.168.1.\](http://192.168.1.)X/",  \# Replace X with your camera's IP  
      "\[http://192.168.1.\](http://192.168.1.)X:81/stream"  
  \]

* **WhatsApp:** Update the target phone number for alerts.  
  WHATSAPP\_NUMBER \= "+91XXXXXXXXXX"

## **🚀 Usage**

Run the main script:

python main02.py

The system will open a window showing the live feed. If specific litter classes (e.g., bottle, cup, waste) are detected with high confidence, an alert will be triggered.

## **📁 Project Structure**

* main02.py: Main script handling stream capture, inference, and alerting.  
* litter.onnx: The AI model weights.  
* requirements.txt: Python dependencies.

## **📷 Hardware Setup**

(Reference your uploaded images here if you add them to the repo)  
The ESP32-CAM is connected via an FTDI adapter for power and data. Ensure the IO0 pin is not connected to GND when trying to run the camera code.

## **🤝 Contributing**

Feel free to submit issues or pull requests if you have ideas to improve the detection accuracy or alerting mechanisms.
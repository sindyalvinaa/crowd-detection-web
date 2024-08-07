# Import necessary modules
from flask import Flask, render_template, Response, request, jsonify, send_file
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
import time
import uuid
import asyncio
import logging
import time
from ultralytics import YOLO
import numpy as np
import threading

import pandas as pd

# Create a Flask app instance
app = Flask(__name__, static_url_path='/static')

# Set to keep track of RTCPeerConnection instances
pcs = set()

# set path here 
path_to_model = "/Users/macbook/tugas-akhir-yolo/skripsi.pt"
path_to_mask = "/Users/macbook/tugas-akhir-yolo/mask.jpg"
path_to_report = "/Users/macbook/tugas-akhir-yolo/report.csv"

model = YOLO(path_to_model)
mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)


kondisi = "proses"
area = mask.copy()
def draw_boxes(result, frame):
    blank = np.zeros(frame.shape, dtype=np.uint8)
    image = frame.copy()
    car, motorcycle, person = 0, 0, 0
    for box in result.boxes:
        x1, y1, x2, y2 = [
        round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        # for every box draw a rectangle for contours
        # get class name
        # set color for each class
        if class_id == 0:
            color = (0, 255, 0)
            class_name = "Person"
            person += 1
        elif class_id == 1:
            color = (0, 0, 255)
            class_name = "Motorcycle"
            motorcycle += 1
        elif class_id == 2:
            color = (255, 0, 0)
            class_name = "Car"
            car += 1
        cv2.putText(image, f'{class_name} {prob}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), -1)
    blank = cv2.bitwise_and(blank, blank, mask=mask)
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    sum_object = [person, motorcycle, car]
    return blank, image, sum_object

def draw(result, frame):
    blank = frame.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = [
        round(x) for x in box.xyxy[0].tolist()
        ]
        # for every box draw a rectangle for contours
        cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return blank

def detect(frame, show=True):
    image = frame.copy()
    masked = cv2.bitwise_and(image, image, mask=mask)
    result = model(masked, conf=0.25, verbose=False)
    area , image, object= draw_boxes(result[0], image)
    percentage = round((np.sum(area) / np.sum(mask)) * 100, 2)
    return image, percentage, object
    
    
# Function to generate video frames from the camera
def generate_frames():
    while True:
        global area
        ret, buffer = cv2.imencode('.jpg', area)
        frame = buffer.tobytes()
        # Concatenate frame and yield for streaming
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def run_yolo():
    camera = cv2.VideoCapture('http://stream.cctv.malangkota.go.id/WebRTCApp/streams/134679292061611148844449.m3u8?token=null')
    time_percentage = time.time()
    sum_car = []
    sum_motor = []
    sum_person = []
    sum_percentage = []
    time_area = time.time()
    while True:
        global kondisi
        global area
        start_time = time.time()
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 384))
            result = model(frame, conf=0.25, verbose=False)
            area, percentage, object = detect(frame, show=False)
            area = cv2.rectangle(area, (0, 0), (170, 50), (0, 0, 0), -1)
            area = cv2.putText(area, f"Area: {percentage}%", (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            elapsed_time = time.time() - start_time
            area = cv2.putText(area, f"FPS: {1/elapsed_time:.2f}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            area = cv2.putText(area, f"Kondisi: {kondisi}", (7, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            elapsed_time = time.time() - start_time
            # setiap berapa detik update persentase
            if time.time() - time_percentage > 1:
                sum_percentage.append(percentage)
                sum_car.append(object[2])
                sum_motor.append(object[1])
                sum_person.append(object[0])
                # print(f"Car: {sum_car}, Motorcycle: {sum_motor}, Person: {sum_person}")
                time_percentage = time.time() 
                # setiap berapa detik update kondisi lalu simpan ke csv
                if time.time() - time_area > 10:
                    if np.mean(sum_percentage) > 30:
                        kondisi = "ramai"
                    else:
                        kondisi = "tidak ramai"
                    # append to csv named report.csv
                    with open(path_to_report, "a") as f:
                        datetime = time.strftime("%d-%m-%Y %H:%M:%S")
                        f.write(f"{datetime},{int(np.mean(sum_car))},{int(np.mean(sum_motor))},{int(np.mean(sum_person))},{np.mean(sum_percentage)},{kondisi}\n")
                    time_area = time.time()
                print(elapsed_time)
# Route to render the HTML template
@app.route('/')
def index():
    global kondisi
    return render_template('index.html', kondisi = kondisi)
    # return redirect(url_for('video_feed')) #to render live stream directly
    
@app.route("/kondisi", methods = ["GET"])
def cek_kondisi():
    global kondisi
    return kondisi

@app.route('/download')
def download():
    path = path_to_report
    return send_file(path, as_attachment=True)

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = path_to_report
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                              encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html',
                           data_var=uploaded_df_html)
# Asynchronous function to handle offer exchange
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create a data channel named "chat"
    # pc.createDataChannel("chat")

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

# Wrapper function for running the asynchronous offer function
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    threading.Thread(target = run_yolo).start()
    app.run(debug=True, host='0.0.0.0')
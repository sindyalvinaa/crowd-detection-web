# Import modul
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
import os
import csv
import pandas as pd

# Buat instance aplikasi Flask
app = Flask(__name__, static_url_path='/static')

# Atur untuk melacak instance RTCPeerConnection
pcs = set()

# Mengatur path model, mask, report
path_to_model = "/Users/macbook/crowd-detection-web/yolov8n.pt"
path_to_mask = "/Users/macbook/crowd-detection-web/mask.jpg"
path_to_report = "/Users/macbook/crowd-detection-web/report.csv"

# Memuat model YOLO
model = YOLO(path_to_model)

# Membaca Mask Gambar dengan OpenCV, ubah ke Grayscale
mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)

# Definisi dan Inisialisasi Variabel
kondisi = "Proses"
area = mask.copy()

# Fungsi draw_boxes untuk membuat bounding box
def draw_boxes(result, frame):
    blank = np.zeros(frame.shape, dtype=np.uint8)
    image = frame.copy()
    car, motorcycle, person = 0, 0, 0

    # Looping setiap hasil deteksi
    for box in result.boxes:
        x1, y1, x2, y2 = [
        round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)

        # Klasifikasi dan Pewarnaan Objek
        if class_id == 0:
            color = (0, 255, 0) # Biru
            class_name = "Car"
            car += 1
        elif class_id == 1:
            color = (0, 0, 255) # Merah
            class_name = "Motorcycle"
            motorcycle += 1
        elif class_id == 2:
            color = (255, 0, 0) # Hijau
            class_name = "Person"
            person += 1

        # Menambahkan Teks dan Bounding Box dalam Gambar
        cv2.putText(image, f'{class_name} {prob}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Menggunakan Mask dan Konversi Warna
    blank = cv2.bitwise_and(blank, blank, mask=mask) # Mempertahankan area putih
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)

    # Menghitung total objek 
    sum_object = [car, motorcycle, person]
    return blank, image, sum_object

# Fungsi Draw: menggambar Bounding Box putih tanpa informasi
def draw(result, frame):
    blank = frame.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = [
        round(x) for x in box.xyxy[0].tolist()
        ]
        cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return blank

# Fungsi Detect: Informasi bounding box, persentase area, jumlah objek terdeteksi
def detect(frame, show=True):
    image = frame.copy()
    masked = cv2.bitwise_and(image, image, mask=mask)
    result = model(masked, conf=0.25, verbose=False)
    area , image, object= draw_boxes(result[0], image)
    percentage = round((np.sum(area) / np.sum(mask)) * 100, 2)
    return image, percentage, object
    
# Mengubah video menjadi frame
def generate_frames():
    while True:
        global area
        ret, buffer = cv2.imencode('.jpg', area)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Fungsi untuk menulis data statistik ke file CSV
def write_data_with_header(path_to_report, datetime, sum_car, sum_motor, sum_person, sum_percentage, kondisi):
    header_needed = not os.path.exists(path_to_report) or os.stat(path_to_report).st_size == 0

    with open(path_to_report, "a", newline='') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["Tanggal/Jam", "Jumlah Mobil", "Jumlah Motor", "Jumlah Orang", "Persentase Area", "Kondisi"])
        writer.writerow([datetime, int(np.mean(sum_car)), int(np.mean(sum_motor)), int(np.mean(sum_person)), np.mean(sum_percentage), kondisi])

# Inisialisasi Kamera dan Variabel Waktu
def run_yolo():
    camera = cv2.VideoCapture('http://stream.cctv.malangkota.go.id/WebRTCApp/streams/134679292061611148844449.m3u8?token=null')
    time_percentage = time.time()
    sum_car = []
    sum_motor = []
    sum_person = []
    sum_percentage = []
    time_area = time.time()

    # Loop Utama: memproses stream video secara real-time menggunakan model.
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

            # Menambahkan Informasi Teks ke halaman CCTV
            area, percentage, object = detect(frame, show=False)
            area = cv2.rectangle(area, (0, 0), (170, 50), (0, 0, 0), -1)
            area = cv2.putText(area, f"Area: {percentage}%", (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            elapsed_time = time.time() - start_time
            area = cv2.putText(area, f"FPS: {1/elapsed_time:.2f}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            area = cv2.putText(area, f"Kondisi: {kondisi}", (7, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Memeriksa apakah telah berlalu lebih dari 1 detik sejak time_percentage
            # time_time: Waktu saat ini
            # time_percentage: Waktu yang dicatat sebelumnya
            if time.time() - time_percentage > 1:
                sum_percentage.append(percentage)
                sum_car.append(object[2])
                sum_motor.append(object[1])
                sum_person.append(object[0])
                time_percentage = time.time()

            # Mengatur pembaruan kondisi dan penyimpanan data ke CSV secara berkala
            # time area : waktu yang dicatat sebelumnya
            if time.time() - time_area > 5:  # Memeriksa apakah telah berlalu lebih dari 5 detik sejak time_area
                if np.mean(sum_percentage) > 30:
                    kondisi = "Ramai"
                else:
                    kondisi = "Tidak Ramai"

                # Mendapatkan tanggal dan waktu saat ini
                # Format hari-bulan-tahun jam:menit
                datetime = time.strftime("%d-%m-%Y %H:%M:%S")
                
                # Menulis data statistik ke file CSV dengan header
                write_data_with_header(path_to_report, datetime, sum_car, sum_motor, sum_person, sum_percentage, kondisi)

                # Mereset daftar data setelah menulis ke CSV
                sum_car.clear()
                sum_motor.clear()
                sum_person.clear()
                sum_percentage.clear()

                # Memperbarui Waktu Interval
                time_area = time.time()

# Route untuk merender template HTML
@app.route('/')
def index():
    global kondisi
    return render_template('index.html', kondisi = kondisi)
    # return redirect (url_for('video_feed')) 
    # to render live stream directly
    
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
    data_file_path = path_to_report
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    uploaded_df.insert(0, 'No', range(1, 1 + len(uploaded_df)))
    uploaded_df_html = uploaded_df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

# Fungsi offer_async()
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

# Fungsi offer()
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

# Jalankan Aplikasi Flask
if __name__ == "__main__":
    threading.Thread(target = run_yolo).start()
    app.run(debug=True, host='0.0.0.0')
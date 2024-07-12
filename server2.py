# Mengimpor modul yang diperlukan
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

# Membuat instance aplikasi Flask
app = Flask(__name__, static_url_path='/static')

# Mengatur untuk melacak instance RTCPeerConnection
pcs = set()

# Mendefinisikan jalur untuk model, mask, dan laporan
path_to_model = "/Users/macbook/crowd-detection-web/yolov8n.pt"
path_to_mask = "/Users/macbook/crowd-detection-web/mask.jpg"
path_to_report = "/Users/macbook/crowd-detection-web/report.csv"

# Memuat Model YOLO
model = YOLO(path_to_model)

# Membaca Mask Gambar dengan OpenCV
mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)

# Definisi dan Inisialisasi Variabel
kondisi = "Proses"
area = mask.copy()

# Fungsi untuk menggambar kotak deteksi pada frame
def draw_boxes(result, frame):
    blank = np.zeros(frame.shape, dtype=np.uint8)
    image = frame.copy()
    car, motorcycle, person = 0, 0, 0

    # Looping melalui hasil deteksi
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)

        # Klasifikasi dan Pewarnaan Objek
        if class_id == 0:
            color = (0, 255, 0)
            class_name = "Car"
            car += 1
        elif class_id == 1:
            color = (0, 0, 255)
            class_name = "Motorcycle"
            motorcycle += 1
        elif class_id == 2:
            color = (255, 0, 0)
            class_name = "Person"
            person += 1

        # Menampilkan Teks dan Menggambar Kotak pada Gambar
        cv2.putText(image, f'{class_name} {prob}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Menggunakan Mask dan Konversi Ke grayscale
    blank = cv2.bitwise_and(blank, blank, mask=mask)
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)

    # Menghitung Total Objek dan Mengembalikan Nilai
    sum_object = [car, motorcycle, person]
    return blank, image, sum_object

# Fungsi untuk menggambar kontur pada frame
# def draw(result, frame):
#     blank = frame.copy()
#     for box in result.boxes:
#         x1, y1, x2, y2 = [
#             round(x) for x in box.xyxy[0].tolist()
#         ]

#         # Untuk setiap kotak, gambar persegi panjang untuk kontur
#         cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     return blank


# Fungsi untuk mendeteksi objek dalam frame
def detect(frame, show=True):
    image = frame.copy()
    masked = cv2.bitwise_and(image, image, mask=mask)
    result = model(masked, conf=0.25, verbose=False)
    area, image, object = draw_boxes(result[0], image)
    percentage = round((np.sum(area) / np.sum(mask)) * 100, 2)
    return image, percentage, object


# Fungsi untuk menghasilkan frame yang akan dikirimkan ke klien
def generate_frames():
    while True:
        global area
        ret, buffer = cv2.imencode('.jpg', area)
        frame = buffer.tobytes()

        # Menggabungkan frame dan mengirimkan untuk streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Fungsi untuk menulis data dengan header ke file CSV
def write_data_with_header(path_to_report, datetime, sum_car, sum_motor, sum_person, sum_percentage, kondisi):

    # Memeriksa apakah file ada dan jika kosong
    if not os.path.exists(path_to_report) or os.stat(path_to_report).st_size == 0:
        with open(path_to_report, "w") as f:

            # Menulis header
            f.write("Tanggal/Jam,Jumlah Mobil,Jumlah Motor,Jumlah Orang,Persentase Area,Kondisi\n")
    with open(path_to_report, "a", newline='') as f:
        writer = csv.writer(f)
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

    # Loop Utama
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

            # Menambahkan Informasi Teks ke Gambar
            area, percentage, object = detect(frame, show=False)
            area = cv2.rectangle(area, (0, 0), (170, 50), (0, 0, 0), -1)
            area = cv2.putText(area, f"Area: {percentage}%", (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            elapsed_time = time.time() - start_time
            area = cv2.putText(area, f"FPS: {1/elapsed_time:.2f}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            area = cv2.putText(area, f"Kondisi: {kondisi}", (7, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Mengumpulkan data statistik selama interval waktu tertentu
            if time.time() - time_percentage > 1:
                sum_percentage.append(percentage)
                sum_car.append(object[2])
                sum_motor.append(object[1])
                sum_person.append(object[0])
                time_percentage = time.time()

            # Update condition and save to CSV every certain time interval
            # Setiap berapa detik update kondisi lalu simpan ke CSV
            if time.time() - time_area > 5:  # Kurangi interval waktu menjadi 5 detik
                if np.mean(sum_percentage) > 30:
                    kondisi = "Ramai"
                else:
                    kondisi = "Tidak Ramai"

                # Dapatkan waktu saat ini
                datetime = time.strftime("%d-%m-%Y %H:%M:%S")
                
                # Tulis data dengan header
                write_data_with_header(path_to_report, datetime, sum_car, sum_motor, sum_person, sum_percentage, kondisi)

                # Kosongkan list setelah menulis data
                sum_car.clear()
                sum_motor.clear()
                sum_person.clear()
                sum_percentage.clear()
                time_area = time.time()

# Route to render the index page
# Route untuk merender halaman index
@app.route('/')
def index():
    return render_template('index.html')

# Route to check crowd condition
# Route untuk mengecek kondisi keramaian
@app.route("/kondisi", methods=["GET"])
def cek_kondisi():
    global kondisi
    return kondisi

# Route to download the report CSV file
# Route untuk mendownload file laporan CSV
@app.route('/download')
def download():
    path = path_to_report
    return send_file(path, as_attachment=True)

# Route to display data from the CSV file
# Route untuk menampilkan data dari file CSV
@app.route('/show_data')
def showData():
    data_file_path = path_to_report
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    uploaded_df.insert(0, 'No', range(1, 1 + len(uploaded_df)))
    uploaded_df_html = uploaded_df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

# Asynchronous function to handle offer exchange
# Fungsi asynchronous untuk menangani pertukaran offer
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    # Membuat instance RTCPeerConnection
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    # Menghasilkan ID unik untuk RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create and set the local description
    # Membuat dan mengatur deskripsi lokal
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    # Menyiapkan data respons dengan SDP dan tipe lokal
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

# Wrapper function to run offer_async function asynchronously
# Fungsi pembungkus untuk menjalankan fungsi offer_async secara asynchronous
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle offer requests
# Route untuk menangani permintaan offer
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

# Route to stream video frames
# Route untuk streaming video frame
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask application
# Jalankan Aplikasi Flask
if __name__ == "__main__":
    threading.Thread(target=run_yolo).start()
    app.run(debug=True, host='0.0.0.0')

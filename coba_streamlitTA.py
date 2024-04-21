import cv2
#import math
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go
from LR import LogisticRegression
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model
from io import StringIO
import sklearn.metrics as metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split

def main():
    st.title("Tugas akhir - Deteksi Toileting untuk Anak Cerebral Palsy")
    st.subheader("Muhammad Asyarie Fauzie / 5023201049")
    st.markdown("##### Pencitraan dan Pengolahan Citra Medika")
    menu = ["Home", "ML-Convolutional Neural Network", "ML-Logistic Regression"]
    choice = st.sidebar.selectbox("Menu", menu)

    def Rumus1(rumus,xa1,xa2,ya1,ya2):
        Deltax = abs(xa1-xa2)
        Deltay = abs(ya1-ya2)
        if rumus == 'sudut': #Rumus Sudut
            theta = np.arctan2(Deltay, Deltax)
            # sudut = np.radians(theta)
            sudut = theta*(180/np.pi)
            return sudut
        elif rumus == 'kemiringan': #Rumus Slope
            slope = abs(Deltay/Deltax)
            return slope
        elif rumus == 'jarak': #Rumus Jarak
            jarak = np.sqrt((Deltax)**2 + (Deltay)**2)
            return jarak
    # Fungsi untuk plot face landmarks
    def plot_face_landmarks(
        fig,
        xlandmarks,
        ylandmarks,
    ):
        landmark_point = []
        right_arm_list = [11, 13, 15, 17, 19, 21]
        left_arm_list = [12, 14, 16, 18, 20, 22]
        right_body_list = [11, 23, 25]
        left_body_list = [12, 24, 26]
        shoulder_point = [11, 12]
        waist_point = [23, 24]

        for index , (x, y) in enumerate(zip(xlandmarks, ylandmarks)):
            landmark_point.append((x,y))

        right_arm_x = []
        right_arm_y = []
        for index in right_arm_list : 
            try:
                point = landmark_point[index]
                right_arm_x.append(point[0])
                right_arm_y.append(point[1])
            except IndexError:
                pass
        left_arm_x = []
        left_arm_y = []
        for index in left_arm_list : 
            try:
                point = landmark_point[index]
                left_arm_x.append(point[0])
                left_arm_y.append(point[1])
            except IndexError:
                pass
        right_body_x = []
        right_body_y = []
        for index in right_body_list : 
            try : 
                point = landmark_point[index]
                right_body_x.append(point[0])
                right_body_y.append(point[1])
            except IndexError:
                pass
        left_body_x = []
        left_body_y = []
        for index in left_body_list : 
            try : 
                point = landmark_point[index]
                left_body_x.append(point[0])
                left_body_y.append(point[1])
            except IndexError:
                pass
        shoulder_point_x = []
        shoulder_point_y = []
        for index in shoulder_point : 
            try :
                point = landmark_point[index]
                shoulder_point_x.append(point[0])
                shoulder_point_y.append(point[1])
            except IndexError:
                pass
        waist_point_x = []
        waist_point_y = []
        for index in waist_point : 
            try :
                point = landmark_point[index]
                waist_point_x.append(point[0])
                waist_point_y.append(point[1])
            except IndexError:
                pass
        fig.add_trace(go.Scatter(x=right_arm_x, y=right_arm_y, mode='lines+markers', name='Right Arm'))
        fig.add_trace(go.Scatter(x=left_arm_x, y=left_arm_y, mode='lines+markers', name='Left Arm'))
        fig.add_trace(go.Scatter(x=right_body_x, y=right_body_y, mode='lines+markers', name='Right Body'))
        fig.add_trace(go.Scatter(x=left_body_x, y=left_body_y, mode='lines+markers', name='Left Body'))
        fig.add_trace(go.Scatter(x=shoulder_point_x, y=shoulder_point_y, mode='lines+markers', name='Shoulder'))
        fig.add_trace(go.Scatter(x=waist_point_x, y=waist_point_y, mode='lines+markers', name='Waist'))
        return
    
    def display_rumus():
        st.write("#### Rumus Sudut")
        st.write("theta = np.arctan2(Deltay, Deltax) * (180/np.pi)")
        st.write("#### Rumus Kemiringan")
        st.write("slope = abs(Deltay/Deltax)")
        st.write("#### Rumus Jarak")
        st.write("jarak = np.sqrt((Deltax)**2 + (Deltay)**2)")
        # Inisialisasi detektor pose
    
    def extract_landmarks(video_path):
    # Inisialisasi detektor pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
                            model_complexity=2,
                            min_detection_confidence=0.3,
                            min_tracking_confidence=0.3)

        # Membaca video
        cap = cv2.VideoCapture(video_path)

        # Membuat direktori untuk menyimpan gambar landmark
        #output_dir = 'Hasil Test Recoding\Hasil landmark\Toilet'
        #os.makedirs(output_dir, exist_ok=True)

        # df = pd.DataFrame(columns=['Frame Number', 'Jarak antar dua siku', 'Jarak antar dua lutut', 
        #                            'Kemiringan antar dua bahu', 'Jarak antar dua tangan', 'Sudut antara bahu dan lengan kiri', 
        #                            'Sudut antara bahu dan lengan kanan', 'Jarak tangan kiri dari tubuh', 'Jarak tangan kanan dari tubuh'])
        frame_number = 0

        while cap.isOpened():
            # Membaca frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses deteksi pose
            results = pose.process(frame)
            
            # Inisialisasi landmark
            x_landmarks = []
            y_landmarks = []

            # Menggambar pose landmarks pada frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for landmark in results.pose_landmarks.landmark:
                    # Dikonversi dari besaran pixel agar sesuai dengan gambar
                    x = int(landmark.x * frame.shape[1]) # Dikali dengan lebar
                    y = int(landmark.y * frame.shape[0]) # Dikali dengan tinggi
                    x_landmarks.append(x)
                    y_landmarks.append(y)
            try:
                antar_siku = Rumus1('jarak',x_landmarks[13], x_landmarks[14], y_landmarks[13], y_landmarks[14])
                antar_lutut = Rumus1('jarak', x_landmarks[25], x_landmarks[26], y_landmarks[25], y_landmarks[26])
                Kemiringan_bahu = Rumus1('kemiringan',x_landmarks[11], x_landmarks[12], y_landmarks[11], y_landmarks[12])
                jarak_dua_tangan = Rumus1('jarak',x_landmarks[15], x_landmarks[16], y_landmarks[15], y_landmarks[16])
                sudut_bahu_lengan_kiri = Rumus1('sudut',x_landmarks[12], x_landmarks[14], y_landmarks[12], y_landmarks[14])
                sudut_bahu_lengan_kanan = Rumus1('sudut',x_landmarks[11], x_landmarks[13], y_landmarks[11], y_landmarks[13])
                jarak_tangan_kiri_tubuh = Rumus1('jarak', x_landmarks[15], x_landmarks[11], y_landmarks[15], y_landmarks[11])
                jarak_tangan_kanan_tubuh = Rumus1('jarak', x_landmarks[16], x_landmarks[12], y_landmarks[16], y_landmarks[12])
            except IndexError:
                pass

            # Plot landmarks menggunakan Plotly
            fig = go.Figure()
            plot_face_landmarks(fig, x_landmarks, y_landmarks)
            fig.update_yaxes(autorange="reversed")
            
            # Menyimpan gambar landmark
            # img_path = os.path.join(output_dir, f'landmark_frame_{frame_number}.png')
            # fig.write_image(img_path)

            # df = df.append({'Frame Number': frame_number,
            #                 'Jarak antar dua siku': antar_siku,
            #                 'Jarak antar dua lutut': antar_lutut,
            #                 'Kemiringan antar dua bahu': Kemiringan_bahu,
            #                 'Jarak antar dua tangan': jarak_dua_tangan,
            #                 'Sudut antara bahu dan lengan kiri': sudut_bahu_lengan_kiri,
            #                 'Sudut antara bahu dan lengan kanan': sudut_bahu_lengan_kanan,
            #                 'Jarak tangan kiri dari tubuh': jarak_tangan_kiri_tubuh,
            #                 'Jarak tangan kanan dari tubuh': jarak_tangan_kanan_tubuh},
            #                 ignore_index=True)
            
            # Menampilkan frame dengan landmark menggunakan OpenCV
            cv2.imshow('Frame with Landmarks', frame)
            
            # Tombol q untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
        
        # Simpan dataframe ke dalam file Excel
        # excel_file = os.path.join(output_dir, 'hasil_perhitungan_landmark.xlsx')
        # df.to_excel(excel_file, index=False)
        
        # Melepaskan sumber daya
        cap.release()
        cv2.destroyAllWindows()

    if choice == "Home":
        display_rumus()
        if st.button("Mulai Pembacaan Video"):
            video_path = 'D:\Kode python\Hasil Test Recoding\Testrafli.avi'
            # function landmark
            extract_landmarks(video_path)
            img3 = Image.open("streamlitfotoTA3.jpg")
            img2 = Image.open("streamlitfotoTA2.jpg")
            img = Image.open("streamlitfotoTA.png")
            #st.write("Contoh Landmark")
            st.image(img3, caption="Contoh Input")
            st.image(img2, caption= "Setelah diberi Landmark")
            st.image(img, caption= "Output Landmark Tanpa Gambar")

    elif choice == "ML-Convolutional Neural Network":
        st.subheader("Convolutional Neural Network")
        img_height = 150
        img_width = 150
        batch_size = 20
        validation_dat = tf.keras.utils.image_dataset_from_directory(
            "Hasil Test Recoding\Hasil landmark\Validation", #38 dan 17
            image_size = (img_height, img_width),
            batch_size=batch_size
        )

        test_dat = tf.keras.utils.image_dataset_from_directory(
            "Hasil Test Recoding\Hasil landmark\Testing", #70 dan 41
            image_size = (img_height, img_width),
            batch_size=batch_size
        )
        model = load_model("cobamodel1.h5")
        st.write("Architechture CNN")
        # Capture model summary = StringIO
        model_summary_str = StringIO()
        model.summary(print_fn=lambda x: model_summary_str.write(x + '\n'))
        model_summary_str = model_summary_str.getvalue()
        st.text(model_summary_str)
        #model.evaluate(test_dat)
        st.subheader("Evaluasi Kinerja model")
        if st.button("Evaluate"):
            st.write("#### Evaluasi model terhadap test data")
            loss, accuracy = model.evaluate(test_dat)
            st.write("Test Loss:", loss)
            st.write("Test Accuracy:", accuracy)
            predicted_labels = model.predict(validation_dat)
            predicted_labels = np.where(predicted_labels > 0.5, 1, 0)  # Konversi probabilitas menjadi label biner (0 atau 1)

            true_labels = []
            for images, labels in validation_dat:
                true_labels.extend(labels.numpy())

            # Menghitung metrik evaluasi
            precision = metrics.precision_score(true_labels, predicted_labels, average='macro')
            recall = metrics.recall_score(true_labels, predicted_labels, average='macro')
            accuracy = metrics.accuracy_score(true_labels, predicted_labels)
            f1_score = metrics.f1_score(true_labels, predicted_labels, average='macro')
            st.write("#### Model Evaluation Metrics")
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("Accuracy:", accuracy)
            st.write("F1-Score:", f1_score)

    elif choice == "ML-Logistic Regression":
        st.subheader("Logistic Regression")
        data = pd.read_excel("hasil_perhitungan_landmark.xlsx")
        data.drop(labels = [data.columns[0]], axis = 1, inplace=True)
        st.write(data)
        st.subheader("LOGISTIC NYA MASIH OVERFITTINGGG")
        X = data.drop("Target", axis=1).values
        y = data["Target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        st.write(f'Number of training examples: {len(X_train)}')
        st.write(f'Number of testing examples: {len(X_test)}')
        if st.button("Train Model"):
            # Create and train the model
            model = LogisticRegression(n_input_features=X_train.shape[-1])

            costs, accuracies, weights, bias = model.train(X_train, y_train,
                                epochs=5000,
                                learning_rate=0.001,
                                minibatch_size=None,
                                verbose=True)

            # Predict the test labels
            predictions = model.predict(X_test)
            accuracy = model.accuracy(predictions, y_test)
            st.write(f"Model test prediction accuracy: {accuracy:0.2f}%")
            predictions = np.where(predictions > 0.5, 1, 0)
            precision = metrics.precision_score(y_test, predictions, average='macro')
            recall = metrics.recall_score(y_test, predictions, average='macro')
            accuracy = metrics.accuracy_score(y_test, predictions)
            f1_score = metrics.f1_score(y_test, predictions, average='macro')

            # Menampilkan metrik evaluasi di Streamlit
            st.write("### Model Evaluation Metrics")
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("Accuracy:", accuracy)
            st.write("F1-Score:", f1_score)
            
if __name__ == '__main__':
    main()
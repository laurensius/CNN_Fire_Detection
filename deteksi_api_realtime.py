import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/20240923172600_akurasi09688_loss01095/model.keras')

with open('model/20240923172600_akurasi09688_loss01095/labels.txt', 'r') as f:
    class_labels = f.read().splitlines()

def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0 
    frame_expanded = np.expand_dims(frame_normalized, axis=0) 
    return frame_expanded

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Webcam tidak dapat diakses.")
    exit()

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    display_frame = frame.copy()
    
    processed_frame = preprocess_frame(frame, target_size=(224, 224)) 
    predictions = model.predict(processed_frame)
    
    predicted_label_index = np.argmax(predictions[0])
    label_name = class_labels[predicted_label_index]  
    confidence = predictions[0][predicted_label_index]  

    label_text = f"Label: {label_name}, Confidence: {confidence:.2f}"
    cv2.putText(display_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Deteksi Api Secara Realtime', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
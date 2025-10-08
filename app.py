import os
import streamlit as st

HOME = os.getcwd()
player_class = 1

from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model
model = YOLO(f"{HOME}/model.pt")

frame_shape = (640, 640)

@st.cache_data(show_spinner=False)
def get_colors(frames, _model, lab=[1, 2], return_detections=False):
    player_avg_colors = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _model(frame, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results).with_nms()
        player_detections = detections[detections.class_id == player_class]

        for _, detection in enumerate(player_detections.xyxy):
            x_0_, y_0_, x_1_, y_1_ = detection
            x_0_, y_0_, x_1_, y_1_ = int(x_0_), int(y_0_), int(x_1_), int(y_1_)

            x_size = x_1_ - x_0_
            y_size = y_1_ - y_0_

            x_0 = x_0_ + int(0.33 * x_size)
            x_1 = x_0_ + int(0.66 * x_size)
            y_0 = y_0_ + int(0.33 * y_size)
            y_1 = y_0_ + int(0.66 * y_size)

            # Get player mask and mean color
            player_mask = frame_rgb[y_0:y_1, x_0:x_1].copy()
            player_mask_mean = player_mask.mean(axis=(0, 1))

            # Convert color to lab space
            player_mask_mean_lab = cv2.cvtColor(np.uint8([[player_mask_mean]]), cv2.COLOR_RGB2LAB)

            player_avg_colors.append(player_mask_mean_lab[0, 0][np.array(lab)])

    if return_detections:
        return player_avg_colors, detections
    return player_avg_colors

@st.cache_data(show_spinner=False)
def process_clip(clip_file, _model, lab=[1, 2]):
    # Save the uploaded file to a temporary location
    with open("temp_clip.mp4", "wb") as f:
        f.write(clip_file.getbuffer())

    # Load the video
    cap = cv2.VideoCapture("temp_clip.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract 100 random frames
    rng = np.random.default_rng(42)
    random_frame_indices = rng.choice(frame_count, size=100, replace=False)
        
    frames = []
    for idx in sorted(random_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    # Reshape frames for processing
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, frame_shape)
        processed_frames.append(frame_resized)

    player_avg_colors = get_colors(processed_frames, _model, lab=lab)

    # Remove outliers
    player_avg_colors = np.array(player_avg_colors)
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    outliers = iso_forest.fit_predict(player_avg_colors)
    player_avg_colors = player_avg_colors[outliers == 1]
    
    # Train models
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=2).fit(player_avg_colors)

    return frame_count, kmeans


st.title("Detector de jugadores de balonmano y clasificación de equipos")
st.write("Esta aplicación detecta jugadores de balonmano en imágenes y los clasifica en equipos según los colores de sus camisetas.")

# Sidebar for uploading a handball clip
with st.sidebar:
    st.header("Opciones")
    clip_file = st.file_uploader("Carga un clip de balonmano", type=["mp4", "mov", "avi"])


    lab_channels = st.multiselect("Seleccionar canales de color LAB", options=["L", "A", "B"], default=["A", "B"])
    # replace L, A,B with 0, 1, 2 array
    lab = []
    if "L" in lab_channels:
        lab.append(0)
    if "A" in lab_channels:
        lab.append(1)
    if "B" in lab_channels:
        lab.append(2)

# Once the file is uploaded, extract 20 random frames and process them
import cv2
import numpy as np

if clip_file is not None:
    with st.spinner('Procesando el clip...'):
        
        # Get teams color
        frame_count, kmeans = process_clip(clip_file, model, lab=lab)


    # Create annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.2)

    # Select frame to show results
    frame_idx = st.slider("Selecciona el fotograma", 0, frame_count - 1, frame_count // 2)

    cap = cv2.VideoCapture("temp_clip.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Process the selected frame 
        frame_resized = cv2.resize(frame, frame_shape)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        player_avg_colors, detections = get_colors([frame_resized], model, lab=lab, return_detections=True)
        predictions_km = np.array(kmeans.predict(player_avg_colors))

        predictions_km[predictions_km == 0] = 3

        detections.class_id[detections.class_id == player_class] = predictions_km

        detections.data["class_name"][detections.class_id == 1] = "team 1"
        detections.data["class_name"][detections.class_id == 3] = "team 2"

        annotated_image = frame_resized.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, width='stretch')

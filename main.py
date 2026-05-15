import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from vpython import *

# --- CONFIGURATION 3D ---
scene.title = "Contrôle 3D : Milieu des Épaules"
scene.width = 600
scene.height = 400
balle = sphere(pos=vector(0,0,0), radius=1, color=color.cyan)
balle_23 = sphere(pos=vector(0,0,0), radius=0.8, color=color.red)
balle_24 = sphere(pos=vector(0,0,0), radius=0.8, color=color.red)
balle_12 = sphere(pos=vector(0,0,0), radius=0.8, color=color.yellow)
balle_11 = sphere(pos=vector(0,0,0), radius=0.8, color=color.yellow)
balle_16 = sphere(pos=vector(0,0,0), radius=0.8, color=color.blue)
balle_15 = sphere(pos=vector(0,0,0), radius=0.8, color=color.blue)
balle_14 = sphere(pos=vector(0,0,0), radius=0.8, color=color.cyan)
balle_13 = sphere(pos=vector(0,0,0), radius=0.8, color=color.cyan)
balle_0 = sphere(pos=vector(0,0,0), radius=0.8, color=color.red)
LISSAGE = 0.2  
AMPLITUDE_X = 25
AMPLITUDE_Y = 20
AMPLITUDE_Z = -30

model_path = "pose_landmarker_lite.task"
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

class PoseHandler:
    latest_result = None

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: setattr(PoseHandler, 'latest_result', result)
)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)
                
        if PoseHandler.latest_result and PoseHandler.latest_result.pose_landmarks:
            for pose_landmarks in PoseHandler.latest_result.pose_landmarks:
                
                def get_3d_pos(landmark):
                    tx = (landmark.x - 0.5) * AMPLITUDE_X
                    ty = (0.5 - landmark.y) * AMPLITUDE_Y
                    tz = landmark.z * AMPLITUDE_Z
                    return vector(tx, ty, tz)
                
                p11 = pose_landmarks[11]
                p12 = pose_landmarks[12]

                mid_x = (p11.x + p12.x) / 2
                mid_y = (p11.y + p12.y) / 2
                mid_z = (p11.z + p12.z) / 2


                target_torse = vector((mid_x-0.5)*AMPLITUDE_X, (0.5-mid_y)*AMPLITUDE_Y, mid_z*AMPLITUDE_Z)
                balle.pos += (target_torse - balle.pos) * LISSAGE
                
                target_23 = get_3d_pos(pose_landmarks[23])
                target_24 = get_3d_pos(pose_landmarks[24])

                balle_23.pos += (target_23 - balle_23.pos) * LISSAGE
                balle_24.pos += (target_24 - balle_24.pos) * LISSAGE
                
                target_12 = get_3d_pos(pose_landmarks[12])
                target_11 = get_3d_pos(pose_landmarks[11])
                
                balle_12.pos += (target_12 - balle_12.pos) * LISSAGE
                balle_11.pos += (target_11 - balle_11.pos) * LISSAGE
                
                target_16 = get_3d_pos(pose_landmarks[16])
                target_15 = get_3d_pos(pose_landmarks[15])
                
                balle_16.pos += (target_16 - balle_16.pos) * LISSAGE
                balle_15.pos += (target_15 - balle_15.pos) * LISSAGE
                
                target_14 = get_3d_pos(pose_landmarks[14])
                target_13 = get_3d_pos(pose_landmarks[13])
                
                balle_14.pos += (target_14 - balle_14.pos) * LISSAGE
                balle_13.pos += (target_13 - balle_13.pos) * LISSAGE
                
                target_0 = get_3d_pos(pose_landmarks[0])
                
                balle_0.pos += (target_0 - balle_0.pos) * LISSAGE

                # DESSIN SUR L'IMAGE (Cercle bleu pour le milieu)
                h, w, _ = frame.shape
                cx, cy = int(mid_x * w), int(mid_y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                cv2.putText(frame, "Centre Torse", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                for landmark in pose_landmarks:
                    lx, ly = int(landmark.x * w), int(landmark.y * h)
                    if landmark.presence > 0.5:
                        cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

        cv2.imshow("Detection Corps Direct", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    while True:
        rate(60)

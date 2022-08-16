#!/usr/bin/env python
# coding: utf-8


import cv2
import mediapipe as mp
import module.feature_extract as feature
import pandas as pd
from pycaret.classification import *

landmark_names = [
        'nose',
        'left_eye_inner',
        'left_eye',
        'left_eye_outer',
        'right_eye_inner',
        'right_eye',
        'right_eye_outer',
        'left_ear',
        'right_ear',
        'mouth_left',
        'mouth_right',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

landmark_list = []
for i in landmark_names:
    for j in ["_x", "_y", "_z"]:
        landmark_list.append(i + j)
        

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose




def visualizer(video_path = None, webcam = False, show_keypoint = False, model_path = None,
               save_path = False, save_name = False, class_dict = None, show_video = True):
   
    label = 0
    score = 0
    Nan_data = False
    
    if save_path != False and save_name != False:
        save = True
    else:
        save = False
        
    if webcam == True:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
        
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_frame=cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_path + save_name + ".avi", fourcc, int(video_frame), (int(video_width), int(video_height)))
    loaded_model = load_model(model_path)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
    
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            results = pose.process(image)
            normalize_points = []
            
            for i in range(len(landmark_names)):
                try:
                    normalize_points.append(results.pose_world_landmarks.landmark[i].x)
                    normalize_points.append(results.pose_world_landmarks.landmark[i].y)
                    normalize_points.append(results.pose_world_landmarks.landmark[i].z)
                     
                except:
                    normalize_points.append(None)
                    normalize_points.append(None)
                    normalize_points.append(None)
                    Nan_data = True 

            if Nan_data == False:
                points = pd.DataFrame(columns = landmark_list, data = [normalize_points])
                feature_angle = feature.compute_angle_feature(dataframe = points)
                feature_len = feature.compute_len_feature(dataframe = points)
                feature_diff = feature.compute_diff_feature(dataframe =points)
                feature_result = pd.concat([feature_angle, feature_diff, feature_len], axis = 1)
                result = predict_model(loaded_model, data = feature_result)
                label = class_dict[result["Label"][0]]
                score = result["Score"][0]
                
            Nan_data = False
    
 

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            if(show_keypoint == True):
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.flip(image, 1) 
            
              
        
   
        
            cv2.putText(image, str(label), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 0, 0), 2)
            cv2.putText(image, "Score : %f" % score, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 0, 0), 2)
            
            if(save):
                out.write(image) 
            if(show_video):
                cv2.imshow('result', image)
            
           
    
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
    
    if(save):
        out.release()
    print("Done")
    cap.release()
    cv2.destroyWindow('result')



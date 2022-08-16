#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from mpl_toolkits.mplot3d import Axes3D
import os


# In[ ]:


# extract keypoint
def extract_keypoint(path, normalize_extraction = True, real_extraction = False, label = True, save_path = None, show_video = False, save_file_name = None):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
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
    
    
    results_normalize_points = []
    results_real_points = []
    
    cap = cv2.VideoCapture(path)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Done [%s] " % path)
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            results = pose.process(image)
            
            
            #list생성
            if(normalize_extraction == True):
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
                results_normalize_points.append(normalize_points)
                

            
            if(real_extraction == True):
                real_points = []
                for i in range(len(landmark_names)):
                    try:
                        real_points.append(results.pose_landmarks.landmark[i].x * image_width)
                        real_points.append(results.pose_landmarks.landmark[i].y * image_height)
                        real_points.append(results.pose_landmarks.landmark[i].z)
                        
                    except:
                        real_points.append(None)
                        real_points.append(None)
                        real_points.append(None)
                        
                results_real_points.append(real_points)
            
            if(show_video == True):
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                    
    cv2.destroyAllWindows()
    cap.release()
    
    if(save_path != None and save_file_name != None):
        #Normalize only
        if(normalize_extraction == True and real_extraction == False):
            point_normalize_dataframe = pd.DataFrame(columns = landmark_list, data = results_normalize_points)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            point_normalize_dataframe.to_csv(save_path + save_file_name + "_noramlize.csv", index = False)
            return point_normalize_dataframe
        #real only
        elif(normalize_extraction == False and real_extraction == True):
            point_real_dataframe = pd.DataFrame(columns = landmark_list, data = results_real_points)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            point_real_dataframe.to_csv(save_path + save_file_name + "_real.csv", index= False)
            return point_real_dataframe
    
        elif(normalize_extraction == True and real_extraction == True):
            point_normalize_dataframe = pd.DataFrame(columns = landmark_list, data = results_normalize_points)
            point_real_dataframe = pd.DataFrame(columns = landmark_list, data = results_real_points)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            point_normalize_dataframe.to_csv(save_path + save_file_name + "_noramlize.csv", index = False)
            point_real_dataframe.to_csv(save_path + save_file_name + "_real.csv", index= False)
            return point_normalize_dataframe, point_real_dataframe
        
    else:
        #Normalize only
        if(normalize_extraction == True and real_extraction == False):
            point_normalize_dataframe = pd.DataFrame(columns = landmark_list, data = results_normalize_points)
            return point_normalize_dataframe
        #real only
        elif(normalize_extraction == False and real_extraction == True):
            point_real_dataframe = pd.DataFrame(columns = landmark_list, data = results_real_points)
            return point_real_dataframe
    
        elif(normalize_extraction == True and real_extraction == True):
            point_normalize_dataframe = pd.DataFrame(columns = landmark_list, data = results_normalize_points)
            point_real_dataframe = pd.DataFrame(columns = landmark_list, data = results_real_points)
            return point_normalize_dataframe, point_real_dataframe



# In[ ]:


#default start = 23 (hip, knee, ankle, heel, foot)
#
#default start = 23 (hip, knee, ankle, heel, foot)
#
def compute_len_feature(dataframe = None, path = None,
                    drop_na = True, start = 23, end = 32, save_path = None, save_file_name = None):

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
    
    

    if(path != None):
        dataframe = pd.read_csv(path)
            
    if(drop_na == True):
        dataframe.dropna(axis = 0)
    
    result = pd.DataFrame()    
    landmark_list = list(dataframe.columns)
    
   
    for i in range(start, end, 2):
        x = dataframe[landmark_list[(i+1) * 3]] - dataframe[landmark_list[i * 3]]
        y = dataframe[landmark_list[(i+1) * 3 + 1]] - dataframe[landmark_list[i * 3 + 1]]
        point_len = np.sqrt(np.power(x, 2) + np.power(y, 2))
        #point_depth = dataframe[landmark_list[(i+1) * 3 + 2]] - dataframe[landmark_list[i * 3 + 2]]
            
        string = landmark_names[i]
        string = string.replace('left', '')
        string = string.replace('right', '')
        string = string.replace('_', '')
        result[string + "_length"] = point_len
        #result[string + "_depth"] = point_depth
            
    
    if(save_path != None and save_file_name != None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        result.to_csv(save_path + save_file_name + "_len_feauture.csv", index = False)
        return result
    else:
        return result

# In[ ]:


#default start = 23 (hip, knee, ankle, heel, foot)
#
#default start = 23 (hip, knee, ankle, heel, foot)
#
def compute_diff_feature(dataframe = None, path = None,
                    drop_na = True, start = 23, end = 32, save_path = None, save_file_name = None):

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
    
    
    if(path != None):
        dataframe = pd.read_csv(path)
            
    if(drop_na == True):
        dataframe.dropna(axis = 0)
    
    result = pd.DataFrame()    
    landmark_list = list(dataframe.columns)
    
   
    for i in range(start, end, 2):
        x = dataframe[landmark_list[(i+1) * 3]] - dataframe[landmark_list[i * 3]]
        y = dataframe[landmark_list[(i+1) * 3 + 1]] - dataframe[landmark_list[i * 3 + 1]]
        #point_len = np.sqrt(np.power(x, 2) + np.power(y, 2))
        depth = dataframe[landmark_list[(i+1) * 3 + 2]] - dataframe[landmark_list[i * 3 + 2]]
            
        string = landmark_names[i]
        string = string.replace('left', '')
        string = string.replace('right', '')
        string = string.replace('_', '')
        result[string + "_x_diff"] = x
        result[string + "_y_diff"] = y
        result[string + "_depth_diff"] = depth
            
    
    if(save_path != None and save_file_name != None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        result.to_csv(save_path + save_file_name + "_diff_feauture.csv", index = False)
        return result
    else:
        return result

# In[ ]:


def compute_anlge(dataframe, midle_point, point_1, point_2):
    
    landmark_list = list(dataframe.columns)
    angle = []
    
    p0 = np.array([dataframe[landmark_list[point_1 * 3]].tolist(), dataframe[landmark_list[(point_1 * 3) + 1]].tolist()])
    p1 = np.array([dataframe[landmark_list[midle_point * 3]].tolist(), dataframe[landmark_list[(midle_point * 3) + 1]].tolist()])
    p2 = np.array([dataframe[landmark_list[point_2 * 3]].tolist(), dataframe[landmark_list[(point_2 * 3) + 1]].tolist()])
    
    for i in range(len(dataframe)):
        v0 = np.array(p0[:, i]) - np.array(p1[:, i])
        v1 = np.array(p2[:, i]) - np.array(p1[:, i])
        angle.append(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
    angle = np.array(angle)
    angle = np.degrees(angle)
    return angle


# In[ ]:


def compute_angle_feature(dataframe = None, path = None, drop_na = True, data_abs = False,
                          hip = True, knee = True, ankle = True, save_path = None, save_file_name = None):
    result = pd.DataFrame()  
    
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
    

    if(path != None):
        dataframe = pd.read_csv(path)
            
    if(drop_na == True):
        dataframe.dropna(axis = 0)
        
    if(hip == True):
        angle_left_hip = compute_anlge(dataframe, 23, 11, 25)
        angle_right_hip = compute_anlge(dataframe, 24, 12, 26)
        result['left_hip_anlge'] = angle_left_hip
        result['right_hip_anlge'] = angle_right_hip
       
    if(knee == True):
        #angle_left_1_knee, angle_left_2_knee, angle_left_3_knee = compute_anlge(dataframe, 25, 23, 27, knee = True)
        #result['left_knee_anlge_1'], result['left_knee_anlge_2'], result['left_knee_anlge_3'] = angle_left_1_knee, angle_left_2_knee, angle_left_3_knee
        angle_left_knee = compute_anlge(dataframe, 25, 23, 27)
        result['left_knee_anlge'] = angle_left_knee

        #angle_right_1_knee, angle_right_2_knee, angle_right_3_knee = compute_anlge(dataframe, 26, 24, 28, knee =True)
        #result['right_knee_anlge_1'], result['right_knee_anlge_2'], result['right_knee_anlge_3'] = angle_right_1_knee, angle_right_2_knee, angle_right_3_knee
        angle_right_knee = compute_anlge(dataframe, 26, 24, 28)
        result['right_knee_anlge'] = angle_right_knee
        
    
    if(ankle == True):
        
        angle_left_1_ankle = compute_anlge(dataframe, 27, 25, 31)
        angle_left_2_ankle = compute_anlge(dataframe, 27, 25, 29) 
        angle_right_1_ankle = compute_anlge(dataframe, 28, 26, 32)
        angle_right_2_ankle = compute_anlge(dataframe, 28, 26, 30)
        result['left_ankle_angle_1'] = angle_left_1_ankle
        result['left_ankle_angle_2'] = angle_left_1_ankle
        result['right_ankle_angle_1'] = angle_right_1_ankle 
        result['right_ankle_angle_2'] = angle_right_2_ankle
    
    if(save_path != None and save_file_name != None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
    if(data_abs == True):
        result = np.abs(result)
    result_col = list(result.columns)
    for i in result_col:
        result[i] =  [360 + angle if angle < 0 else angle for angle in result[i]] 
    
    
    if(save_path != None and save_file_name != None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        result.to_csv(save_path + save_file_name + "_angle_feauture.csv", index = False)
        return result
    else:
        return result
        

        
def cut_dataframe(dataframe = None, drop = 0.1):
    length = len(dataframe)
    cut_range = int(length * drop)
    dataframe = dataframe.drop(dataframe.index[(length - cut_range):])
    dataframe = dataframe.drop(dataframe.index[:cut_range])
    dataframe = dataframe.reset_index(drop = True)
    
    return dataframe



import module.feature_extract as feature
import pandas as pd
from pycaret.classification import *

def gait_classification(video_path = None, model_path = None, class_dict = None):
    normalize = feature.extract_keypoint(path = video_path, show_video = False)
    test = feature.cut_dataframe(normalize, drop = 0.1)

    feature_angle = feature.compute_angle_feature(dataframe = test)
    feature_diff =  feature.compute_diff_feature(dataframe = test)
    
    feature_len = feature.compute_len_feature(dataframe = test)
    feature_result = pd.concat([feature_angle, feature_diff, feature_len], axis = 1)
    loaded_model = load_model(model_path)
    print("\n-------------------")
    print("Gait Classification")
    print("-------------------\n")
    result = predict_model(loaded_model, data = feature_result)
    
    check_result = pd.Series(result["Label"].value_counts())
    check_result = check_result.to_dict()
    
    check_result = pd.Series(result["Label"].value_counts())
    check_result = check_result.to_dict()
    
    gait = 0
    classification_result = {}

    for i in check_result.keys():
        if(gait < check_result[i]):
            gait = check_result[i]
            label = i
        print("%s : %f %%" % (class_dict[i], (float(check_result[i]) / float(len(result))) * 100))
        classification_result[class_dict[i]] =  (float(check_result[i]) / float(len(result))) * 100
    classification_result = sorted(classification_result.items(), key = lambda item: item[1], reverse = True)

    acc = (float(gait) / float(len(result))) * 100
    print("\n============================================")
    print("Result : %s / %f %%" % (class_dict[label], acc))
    print("============================================")
    
    return  classification_result





import torch
import tensorflow as tf
import os
import cv2
import numpy as np

class ModelManager:
    def __init__(self):
        self.yolo_model = self._load_yolo_model()
        self.recycling_model = self._load_recycling_model()
        self.class_names = ['plastic', 'glass', 'metal', 'paper', 'cardboard']

    def _load_yolo_model(self):
        try:
            print("Loading YOLOv5 model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.eval()
            print("YOLOv5 model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading YOLOv5 model: {str(e)}")
            return None

    def _load_recycling_model(self):
        model_path = 'recycling_model.h5'
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print(f"Recycling model loaded from {model_path}")
                return model
            else:
                print(f"No saved recycling model found at {model_path}")
                return None
        except Exception as e:
            print(f"Error loading recycling model: {str(e)}")
            return None

    def get_object_detections(self, frame):
        if self.yolo_model is None:
            print("YOLO model not loaded")
            return []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                results = self.yolo_model(frame_rgb)
            detections = results.xyxy[0].cpu().numpy()
            return self.non_max_suppression(detections)
        except Exception as e:
            print(f"Error during object detection: {str(e)}")
            return []

    def non_max_suppression(self, boxes, overlap_thresh=0.5):
        if len(boxes) == 0:
            return []
    
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
        return boxes[pick]

    def get_recycling_prediction(self, obj_img):
        if self.recycling_model is None:
            print("Recycling model not loaded")
            return None, 0.0
        
        try:
            prediction = self.recycling_model.predict(obj_img)
            predicted_class = prediction.argmax(axis=1)[0]
            confidence = prediction[0][predicted_class]
            return self.class_names[predicted_class], confidence
        except Exception as e:
            print(f"Error during recycling prediction: {str(e)}")
            return None, 0.0


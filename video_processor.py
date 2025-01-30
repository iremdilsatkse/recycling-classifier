import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

class VideoProcessor:
    def __init__(self, model_manager):
        print(f"OpenCV version: {cv2.__version__}")
        self.model_manager = model_manager
        self.frame_rate = 30
        self.detection_interval = 0.5  # Perform detection every 0.5 seconds
        self.last_detection_time = 0
        self.tracked_objects = {}  # Store tracked objects
        self.centroid_tracker = CentroidTracker()

    def process_frame(self, frame):
        current_time = time.time()
        height, width = frame.shape[:2]
        
        # Perform detection at regular intervals
        if current_time - self.last_detection_time >= self.detection_interval:
            self.last_detection_time = current_time
            detections = self.model_manager.get_object_detections(frame)
            
            rects = []
            new_tracked_objects = {}
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, class_id = det
                if conf > 0.25:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    obj_img = frame[y1:y2, x1:x2]
                    obj_img = cv2.resize(obj_img, (224, 224))
                    obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
                    obj_img = obj_img / 255.0
                    obj_img = np.expand_dims(obj_img, axis=0)
                    class_name, class_conf = self.model_manager.get_recycling_prediction(obj_img)
                    
                    if class_name:
                        rects.append((x1, y1, x2, y2))
                        new_tracked_objects[i] = (class_name, class_conf, (x1, y1, x2, y2))

            objects = self.centroid_tracker.update(rects)
            
            # Update tracked_objects with new detections and positions
            self.tracked_objects = {}
            for (objectID, centroid) in objects.items():
                if objectID in new_tracked_objects:
                    class_name, class_conf, bbox = new_tracked_objects[objectID]
                    self.tracked_objects[objectID] = (class_name, class_conf, bbox)
                elif objectID in self.tracked_objects:
                    class_name, class_conf, _ = self.tracked_objects[objectID]
                    x, y = centroid
                    bbox = (x - 50, y - 50, x + 50, y + 50)  # Estimate new bounding box
                    self.tracked_objects[objectID] = (class_name, class_conf, bbox)

        # Draw bounding boxes and labels for all tracked objects
        for (objectID, (class_name, class_conf, bbox)) in self.tracked_objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} ({class_conf:.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add a timestamp and object count to the frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        object_count = len(self.tracked_objects)
        cv2.putText(frame, f"Objects: {object_count}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_time = 1 / self.frame_rate

        while cap.isOpened():
            start_time = time.time()
            
            success, frame = cap.read()
            if not success:
                break

            processed_frame = self.process_frame(frame)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Control frame rate
            processing_time = time.time() - start_time
            if processing_time < frame_time:
                time.sleep(frame_time - processing_time)

        cap.release()


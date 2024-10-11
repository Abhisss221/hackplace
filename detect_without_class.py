import cv2
from ultralytics import YOLO


def get_color_by_class(class_id):
    
    person_class = 0  
    vehicle_classes = [2, 3, 5, 7]  
    plant_classes = []  
    drone_class = []  
    animal_classes = [16, 17, 18]  

    
    if class_id == person_class:
        return (0, 0, 255)  
    elif class_id in vehicle_classes:
        return (0, 255, 0)  
    elif class_id in plant_classes:
        return (255, 0, 0)  
    elif class_id in drone_class:
        return (0, 0, 0)  
    elif class_id in animal_classes:
        return (0, 255, 255)  
    else:
        return (255, 255, 255)  


model = YOLO("yolov8n.pt")


input_path = "wfr.mp4"  
output_path = "output.avi"  


cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    for result in results:
        for obj in result.boxes:
            
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  
            class_id = int(obj.cls)  
            
            
            color = get_color_by_class(class_id)
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    
    out.write(frame)

    
    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

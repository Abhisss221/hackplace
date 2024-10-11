import cv2
from ultralytics import YOLO


def get_color_by_class(class_id):
    
    if class_id == 0:  
        return (0, 0, 255)  
    elif class_id == 2 or class_id == 3:  
        return (0, 255, 0)  
    else:
        return (0, 255, 255)  


model = YOLO("yolov8n.pt")


input_path = "wfr.mp4"  
output_path = "output2.avi"  


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
            confidence = obj.conf[0]  
            
            
            color = get_color_by_class(class_id)
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    out.write(frame)

    
    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

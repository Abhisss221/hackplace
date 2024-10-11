import cv2
from ultralytics import YOLO

# Define a function to assign colors to different categories
def get_color_by_class(class_id):
    # YOLOv8 class IDs for specific objects:
    person_class = 0  # Person
    vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
    plant_classes = []  # You will need a specific dataset to detect plants/trees
    drone_class = []  # Add the appropriate class ID for drones
    animal_classes = [16, 17, 18]  # Bird, Cat, Dog (other animals can be added)

    # Assign colors based on categories
    if class_id == person_class:
        return (0, 0, 255)  # Red for person
    elif class_id in vehicle_classes:
        return (0, 255, 0)  # Green for vehicles (car/bike/truck)
    elif class_id in plant_classes:
        return (255, 0, 0)  # Blue for plants/trees
    elif class_id in drone_class:
        return (0, 0, 0)  # Black for drones
    elif class_id in animal_classes:
        return (0, 255, 255)  # Yellow for animals
    else:
        return (255, 255, 255)  # White for others

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Load the input video or image
input_path = "path_to_video.mp4"  # Path to the video file
output_path = "output.avi"  # Output file to save the video results

# Open the video
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model(frame)

    # Annotate each detected object
    for result in results:
        for obj in result.boxes:
            # Extract bounding box and class id
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Bounding box coordinates
            class_id = int(obj.cls)  # Object class id
            confidence = obj.conf[0]  # Confidence score
            
            # Get color for the class
            color = get_color_by_class(class_id)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label the object
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

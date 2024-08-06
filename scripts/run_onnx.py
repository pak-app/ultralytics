import onnxruntime as ort
import cv2
import numpy as np
import time 

def read_img(source_path:str, source_type:str):
    if source_type == "image":
        return cv2.imread(source_path)

def preprocessing(img, input_size:tuple =(640, 640)):
    
    if img.shape[0] != input_size[0] or img.shape[1] != input_size[1]:
        input_img_resized = cv2.resize(img, input_size)
    
    input_img_rgb = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2RGB)

    input_img_normalized = input_img_rgb / 255.0

    input_tensor = input_img_normalized.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

def load_model(model_path:str):
    return ort.InferenceSession(model_path)

def post_process(outputs):
    # Example post-processing code, adjust according to your model's output format
    boxes, scores, class_ids = outputs  # Split the output tensors if necessary

    # Thresholding and NMS to remove unnecessary boxes
    threshold = 0.5
    nms_threshold = 0.4

    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, nms_threshold)

    # Extract the detected objects
    detected_objects = []
    for i in indices:
        box = boxes[i[0]]
        score = scores[i[0]]
        class_id = class_ids[i[0]]
        detected_objects.append((box, score, class_id))
    
    return detected_objects

def visulize_objects(input_image, detected_objects):
    # Draw the bounding boxes on the image
    for box, score, class_id in detected_objects:
        x, y, w, h = box
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f'{class_id}: {score:.2f}'
        cv2.putText(input_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Detected Objects', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    source_path = r"C:\Users\p.alishah\Desktop\Projects\python\models\ultralytics\data\test\val_all_data-56-_jpg.rf.fd1cd988f8460e5e0f2f51399df2c348.jpg"
    model_path = r"C:\Users\p.alishah\Desktop\Projects\python\models\ultralytics\runs\detect\train8\weights\best.onnx"

    input_img = read_img(source_path, "image")
    preprocessed_img = preprocessing(input_img.copy())
    
    ort_session = load_model(model_path)
    start_time = time.time()
    outputs = ort_session.run(None, {'images': preprocessed_img})
    end_time = time.time()
    
    print("Latency:", (end_time - start_time))

    # detected_objects = post_process(outputs)
    # visulize_objects(input_img, detected_objects)
    print(len(outputs))
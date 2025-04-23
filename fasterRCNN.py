import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import easygui  

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush'
]

def get_class_id(class_name):
    if class_name in COCO_CLASSES:
        return COCO_CLASSES.index(class_name)
    else:
        return None

user_input = input("Enter a class name (e.g., 'dog'): ").lower()

class_id = get_class_id(user_input)

if class_id is None:
    print("Class name not found. Please try again.")
else:
    print(f"Class ID for '{user_input}': {class_id}")

    fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    fasterrcnn.eval()

    image_path = easygui.fileopenbox(title="Select an image", filetypes=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"])

    if not image_path:
        print("No image selected. Exiting...")
    else:

        image = Image.open(image_path)

        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)  

        with torch.no_grad():  
            prediction = fasterrcnn(image_tensor)

        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']

        score_threshold = 0.5  

        matching_boxes = boxes[(labels == class_id) & (scores > score_threshold)]
        matching_scores = scores[(labels == class_id) & (scores > score_threshold)]

        if matching_boxes.numel() == 0:
            print(f"No {user_input} detected in the image.")
        else:

            plt.imshow(np.array(image))

            for i in range(len(matching_boxes)):
                box = matching_boxes[i].tolist()
                score = matching_scores[i].item()

                x_min, y_min, x_max, y_max = box

                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                                  fill=False, color='red', linewidth=3))

                plt.text(x_min, y_min, f'{user_input.capitalize()}: {score:.2f}', color='white', fontsize=12, 
                          bbox=dict(facecolor='red', alpha=0.5))

            plt.title(f"Detected {user_input.capitalize()}s")
            plt.show()
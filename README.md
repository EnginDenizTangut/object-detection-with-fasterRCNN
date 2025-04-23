# Object Detection with Faster R-CNN and Class-Specific Filtering

This project utilizes a pretrained Faster R-CNN model based on ResNet-50 with a Feature Pyramid Network (FPN) backbone to perform object detection on input images. The primary goal is to detect and visualize only a specific class, chosen by the user, among the 91 COCO categories.

## Overview

Faster R-CNN (Region-based Convolutional Neural Network) is a two-stage object detection architecture. The first stage, the Region Proposal Network (RPN), proposes candidate object bounding boxes, and the second stage classifies these proposals and refines their spatial locations.

In this implementation:

- A user inputs a class name (e.g., `dog`, `car`, `person`).
- The model detects all objects in the image.
- Only the objects corresponding to the selected class and above a confidence threshold are visualized.

## Features

- Uses the pretrained `fasterrcnn_resnet50_fpn` model from `torchvision.models`.
- Accepts image input from a file dialog interface.
- Detects and visualizes bounding boxes for a user-specified class.
- Displays detection score for each predicted bounding box.
- Filters detections based on a confidence score threshold (default: 0.5).

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- torchvision
- Pillow
- matplotlib
- tkinter (usually bundled with Python installations)

Install the required libraries using pip:

```bash
pip install torch torchvision pillow matplotlib

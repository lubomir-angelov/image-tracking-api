import datetime
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import torch
from typing import List
import cv2


"""
Helpful links:
    https://huggingface.co/An-619/FastSAM
    https://docs.ultralytics.com/guides/object-cropping/#visuals => Explains mask & box management, image cropping
    https://www.youtube.com/watch?v=yHNPyqazYYU => Fast Segment Anything (FastSAM) vs SAM | Is it 50x faster?
"""


class FastSegment:
    def __init__(self):
        """
        Builds the Fast SAM models.
        Optionally specify which device to use. Options: cpu, cuda (for gpu)

        Args:
            None

        Attributes:
            everything_results: A YOLO Results object
            results: A list of YOLO results
            prompt_process: A FastSAMPrompt - uses the "everything prompt"
            fast_sam_modesl: FastSAM - optimized Segmentation Model based on yolov8
            image: The image to be processed as an numpy ndarray
            device: str The device to use ("cpu", "cuda"), defaults to cpu.
        """
        self.everything_results = None
        self.results = None
        self.prompt_process = None
        self.fastsam_model = FastSAM("../models/FastSAM-x.pt")
        self.image = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_masks(self, image: np.ndarray) -> List:
        """
        Creates masks of a given image in vector representation.
        
        Args:
            image: np.ndarray A vector representation of an image
    
        Returns:
            masks: A list of masks on cpu or gpu(cuda) depending on the way the model is initialized.
        """
        self.image = image
        # image, device=self.device, retina_masks=True, imgsz=1024, conf=0.8, iou=0.1
        self.everything_results = self.fastsam_model(
            image, device=self.device, retina_masks=True, imgsz=512, conf=0.8, iou=0.4
        )
        self.prompt_process = FastSAMPrompt(
            image, self.everything_results, device=self.device
        )
        self.results = self.prompt_process.results[0]
        # save detections to file
        annotations = self.prompt_process.everything_prompt()
        self.prompt_process.plot(annotations=annotations, output_path=f"output_images/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_annotated_image.jpg")
        
        if self.device == "cuda":
            masks = self.results.boxes.cls.cuda().tolist()
        else:
            masks = self.results.boxes.cls.cpu().tolist()

        return masks

    def get_boxes(self) -> List:
        """
        Retrieves the boxes of the detected segmentations depening on the device type.

        Args:
            None

        Returns: 
            boxes: List a list of bounding box coordinates.
        """
        if self.device == "cuda":
            boxes = self.results.boxes.xyxy.cuda().tolist()
        else:
            boxes = self.results.boxes.xyxy.cpu().tolist()
        
        return boxes
    
    def get_img(self):
        """
        Gets the original image being processed.

        Args:
            None
        
        Returns:
            orig_image: ndarray Original image as a numpy array.
            https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
        """
        return self.everything_results[0].orig_img
    
    def get_box_coordinates(self):
        """
        Gets the segmented boundig boxes in format xyxy.

        Args:
            None
        
        Returns:
            boxes.xyxyx: Return the boxes in xyxy format.
            https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.xyxy
        """
        return self.everything_results[0].boxes.xyxy



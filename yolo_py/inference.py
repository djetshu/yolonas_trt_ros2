import pycuda.driver as cuda
from typing import Iterable, Tuple
import cv2
import numpy as np

from yolo_py.dataset import COCO_DETECTION_CLASSES_LIST
from yolo_py.mod_image import visualize_image
from yolo_py.yolo_mem import allocate_buffers, load_engine

#### From super_gradients

# Class for Inference with TRT
class InferenceSession:
    def __init__(self, engine_file, inference_shape: Tuple[int,int], trt_logger):
        self.engine = load_engine(engine_file, trt_logger)
        self.context = None
        self.inference_shape = inference_shape
        self.initialize()

    def initialize(self):
        self.context = self.engine.create_execution_context()
        assert self.context

        self.context.set_input_shape('input', (1, 3, *self.inference_shape))
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        return self

    def preprocess(self, image):
        image = np.array(image)
        rows, cols = self.inference_shape
        original_shape = image.shape[:2]
        # Resize image to fixed size
        image = cv2.resize(np.array(image), dsize=(cols, rows))
        # Switch from HWC to to CHW order
        return np.moveaxis(image, 2, 0), original_shape

    def postprocess(self, detected_boxes, original_shape: Tuple[int, int]):
        sx = original_shape[1] / self.inference_shape[1]
        sy = original_shape[0] / self.inference_shape[0]
        detected_boxes[:, :, [0, 2]] *= sx
        detected_boxes[:, :, [1, 3]] *= sy
        return detected_boxes

    def __call__(self, image):
        batch_size = 1
        input_image, original_shape = self.preprocess(image)

        self.inputs[0].host[:np.prod(input_image.shape)] = np.asarray(input_image).ravel()

        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
        success = self.context.execute_v2(bindings=self.bindings)
        assert success
        [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]

        num_detections, detected_boxes, detected_scores, detected_labels = [o.host for o in self.outputs]

        num_detections = num_detections.reshape(-1)
        num_predictions_per_image = len(detected_scores) // batch_size
        detected_boxes  = detected_boxes.reshape(batch_size, num_predictions_per_image, 4)
        detected_scores = detected_scores.reshape(batch_size, num_predictions_per_image)
        detected_labels = detected_labels.reshape(batch_size, num_predictions_per_image)

        detected_boxes = self.postprocess(detected_boxes, original_shape) # Scale coordinates back to original image shape
        return num_detections, detected_boxes, detected_scores, detected_labels


    def cleanup(self, exc_type=None, exc_val=None, exc_tb=None): 
        del self.inputs, self.outputs, self.bindings, self.stream, self.context
        
    def __del__(self):
        self.cleanup()

    @staticmethod
    def show_predictions_from_batch_format(image, predictions):
        image_index, pred_boxes, pred_scores, pred_classes = next(iter(InferenceSession.iterate_over_detection_predictions_in_batched_format(predictions)))

        predicted_boxes = np.concatenate([pred_boxes, pred_scores[:, np.newaxis], pred_classes[:, np.newaxis]], axis=1)

        image = visualize_image(
            image_np=np.array(image),
            class_names=COCO_DETECTION_CLASSES_LIST,
            pred_boxes=predicted_boxes
        )
        return image, predicted_boxes, COCO_DETECTION_CLASSES_LIST

    @staticmethod
    def iterate_over_detection_predictions_in_batched_format(
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterate over object detection predictions in batched format.
        This method is suitable for iterating over predictions of object detection models exported to ONNX format
        with postprocessing. An exported object detection model can have 'flat' or 'batched' output format.
        A batched output format means that all detections from all images in batch are padded and stacked together.
        So one should iterate over all detections and filter out detections for each image separately which this method does.

        >>> predictions = model(batch_of_images)
        >>> for image_detections in iterate_over_detection_predictions_in_batched_format(predictions):
        >>>     image_index, pred_bboxes, pred_scores, pred_labels = image_detections
        >>>     # Do something with predictions for image with index image_index
        >>>     ...

        :param predictions:    A tuple of (num_detections, bboxes, scores, labels)
            num_detections: A 1D array of shape (batch_size,) containing number of detections per image
            bboxes:         A 3D array of shape (batch_size, max_detections, 4) containing bounding boxes in format (x1, y1, x2, y2)
            scores:         A 2D array of shape (batch_size, max_detections) containing class scores
            labels:         A 2D array of shape (batch_size, max_detections) containing class labels
        :return:               A generator that yields (image_index, bboxes, scores, labels) for each image in batch
                            image_index: An index of image in batch
                            bboxes: A 2D array of shape (num_predictions, 4) containing bounding boxes in format (x1, y1, x2, y2)
                            scores: A 1D array of shape (num_predictions,) containing class scores
                            labels: A 1D array of shape (num_predictions,) containing class labels. Class labels casted to int.

        """
        num_detections, detected_bboxes, detected_scores, detected_labels = predictions
        num_detections = num_detections.reshape(-1)
        batch_size = len(num_detections)

        detected_bboxes = detected_bboxes.reshape(batch_size, -1, 4)
        detected_scores = detected_scores.reshape(batch_size, -1)
        detected_labels = detected_labels.reshape(batch_size, -1)

        detected_labels = detected_labels.astype(int)

        for image_index in range(batch_size):
            num_detection_in_image = num_detections[image_index]

            pred_bboxes = detected_bboxes[image_index, :num_detection_in_image]
            pred_scores = detected_scores[image_index, :num_detection_in_image]
            pred_labels = detected_labels[image_index, :num_detection_in_image]

            yield image_index, pred_bboxes, pred_scores, pred_labels

    

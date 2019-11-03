import os
import sys
import json
import time
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#Import COCO Config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

important_classes = {
  1 : {'label':'person', 'color':(255,0,0)},
  33 : {'label':'sports ball', 'color':(0,0,255)},
  39 : {'label':'tennis racket', 'color':(0,255,0)}
}

def process_video(model, input_path, output_path):
    assert input_path
    assert output_path

    # Video capture
    vcapture = cv2.VideoCapture(input_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # try to determine the total number of frames in the video file
    try:
      prop = cv2.CAP_PROP_FRAME_COUNT
      num_frames = int(vcapture.get(prop))
      print("[INFO] {} total frames in video".format(num_frames))
    except:
      print("[INFO] could not determine # of frames in video")
      num_frames = -1

    # Define codec and create video writer
    # XVID is for .avi files
    vwriter = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              fps, (width, height))

    writer = open(output_path + ".tab", "w") #todo file to output coordinates

    count = 0
    coordinates = []
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            start = time.time()
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            # Annotate image using masks, labels, and bounding boxes from model
            tennis_objects = get_tennis_objects(image, r['rois'], r['masks'], r['class_ids'], r['scores'])

            annotated = image.copy()
            for obj in tennis_objects:
              obj["frame"] = count
              annotate_image(annotated, obj)
              writer.write(str(obj) + "\n")
            
            # RGB -> BGR to save image to video
            annotated = annotated[..., ::-1]
            # Add image to video writer
            vwriter.write(annotated)

            end = time.time()
            if count == 0:
              print("[INFO] single frame took {:.4f} seconds".format(end - start))
              print("[INFO] estimated total time to finish: {:.4f}".format((end - start) * num_frames))
            count += 1
    vwriter.release()
    writer.close()


def get_tennis_objects(image, boxes, masks, class_ids, scores):
  num_instances = boxes.shape[0]
  if not num_instances:
    print("\n*** No instances to display *** \n")
  else:
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

  tennis_objects = []
  for i in range(num_instances):
    class_id = class_ids[i]
    if class_id in important_classes:
      if not np.any(boxes[i]):
        continue
      y1, x1, y2, x2 = boxes[i]
      obj = {}
      obj['x1'] = x1
      obj['y1'] = y1
      obj['x2'] = x2
      obj['y2'] = y2
      obj['score'] = scores[i] if scores is not None else None
      obj['class_id'] = class_id
      #obj['mask'] = boxmasks[:, :, i]
      tennis_objects.append(obj)
  return tennis_objects


def annotate_image(annotated, obj):
  color = important_classes[obj['class_id']]['color']
  label = important_classes[obj['class_id']]['label'] 

  cv2.rectangle(annotated, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), color, 2)

  #draw caption
  score = obj['score']
  caption = "{} {:.3f}".format(label, score) if score else label
  cv2.putText(annotated, caption, (obj['x1'], obj['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
  #draw mask
  #mask = obj['mask']
  #todo


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--weights', required=False,
                        default=COCO_WEIGHTS_PATH,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--input', required=True,
                        metavar="path or URL to video",
                        help='Video to apply Mask RCNN to')
    parser.add_argument('--output', required=True,
                        metavar="path to video",
                        help='output filename for annotated video')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    class InferenceConfig(coco.CocoConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    model.load_weights(args.weights, by_name=True)

    process_video(model, args.input, args.output)

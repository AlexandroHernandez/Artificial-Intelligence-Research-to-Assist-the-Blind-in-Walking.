### imports

#Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from threading import Thread

# For measuring the inference time.
import time

# For subprocess
import subprocess

import cv2
import os
### methods and fuctions

def test_tensorflow():
    print(f"Tensorflow version = {tf.__version__}")
    print(tf.test.gpu_device_name())
    print(f"Tensorflow hub version = {hub.__version__}")

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)

def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
  pil_image_rgb = pil_image.convert("RGB")
  path = '/home/alexandro/Desktop/asistencia_ciegos/img/test_img.jpg'
  pil_image_rgb.save(path, format="JPEG", quality=90)
  print("Image downloaded to %s." % path)
  if display:
    display_image(pil_image)
  return path

def download_image():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
    downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
    return downloaded_image_path

def load_Centernet_hourglass():
    # CenterNet_hourgalss = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1")
    # CenterNet_URL = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1"
    # tf.saved_model.save(CenterNet_hourgalss, '/home/alexandro/Desktop/asistencia_ciegos/centernet-hourglass-model')
    loaded_model = tf.saved_model.load('/home/alexandro/Desktop/asistencia_ciegos/centernet-hourglass-model')
    return loaded_model

def load_tf_models():
    try:
        subprocess.run(["python", "src/tf_subprocess.py"], check=True)
        print("tf_subprocess.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing tf_subprocess.py: {e}")

def import_object_detection_libraries():
    #import dependencies from the API
    
    
    
    from object_detection.utils import label_map_util

def create_library_for_labels():
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import label_map_util
    PATH_TO_LABELS = '/home/alexandro/Desktop/asistencia_ciegos/models/research/object_detection/data/mscoco_complete_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return category_index

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def image_classification_ObjectDetectionAPI(detector, path, category_index,filename):
  from object_detection.utils import ops as utils_ops
  from object_detection.utils import visualization_utils as viz_utils
  img = load_img(path)[tf.newaxis, ...]
  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(img)
  end_time = time.time()
  result = {key:value.numpy() for key,value in result.items()}

  label_id_offset = 0
  image_np_with_detections = img.numpy().copy()

  # Use keypoints if available in detections
  keypoints, keypoint_scores = None, None
  if 'detection_keypoints' in result:
    keypoints = result['detection_keypoints'][0]
    keypoint_scores = result['detection_keypoint_scores'][0]

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores)
  cv2.imshow('Image', image_np_with_detections[0])
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  output_path = '/home/alexandro/Desktop/asistencia_ciegos/img_detected'
  cv2.imwrite(os.path.join(output_path, filename), image_np_with_detections[0])
  print(f"Image saved: {filename}")

def main():
    test_tensorflow()
    img_path = download_image()
    import_object_detection_libraries()
    category_index = create_library_for_labels()
    CenterNet_hourgalss = load_Centernet_hourglass()
    # image_classification_ObjectDetectionAPI(CenterNet_hourgalss, img_path, category_index)
    # thread = Thread(target=start_video(CenterNet_hourgalss, category_index))
    # thread.start()
    # thread.join()
    folder_path = '/home/alexandro/Desktop/asistencia_ciegos/test_coco_imgs'
    # Ensure the folder path is valid
    if os.path.exists(folder_path):
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a photo (you can customize this check based on file extensions)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                # Full path to the photo
                photo_path = os.path.join(folder_path, filename)
                
                # Your logic here with the photo_path
                print(f"Processing photo: {photo_path}")
                image_classification_ObjectDetectionAPI(CenterNet_hourgalss, photo_path, category_index,filename)
                
    else:
        print(f"The folder path '{folder_path}' does not exist.")
        print("Success")

if __name__ == "__main__":
    main()
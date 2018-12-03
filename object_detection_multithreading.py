import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
#MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 500

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, image_tensor, boxes, scores, classes, num_detections, oneImg=False):
#def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    if (oneImg):
        image_np_expanded = np.expand_dims(image_np, axis=0)
    else:
        image_np_expanded = image_np
    #print("len in detect_objects:", len(image_np))
    '''
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    '''
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    #print("len after detect_objects:", len(image_np))
    return dict(boxes=boxes, scores=scores, classes=classes)
    # Visualization of the results of a detection.'
    '''
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    '''
    '''
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np
    '''
   #return category_index

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  #to keep tf from allocating too much gpu memory
        sess = tf.Session(config=config, graph=detection_graph)
        #sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
   # '''
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #'''
    while True:
        fps.update()
        frm = input_q.get(block=True)
      #  if frm is None:
      #      continue

    #    print("len of frames before ==> ", len(frm), "\n")
        data = detect_objects(frm, sess, image_tensor, boxes, scores, classes, num_detections)
     #   print("len of frames after==> ", len(frm), "\n")
     #   print("len of boxes ==> ", len(data['boxes']), "\n")
        if (len(frm) != len(data['boxes'])):
            for img in frames:
                r = detect_objects(img, sess, image_tensor, boxes, scores, classes, num_detections, oneImg=True)
        #data = detect_objects(frame_rgb, sess, detection_graph)
        output_q.put(data)

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-file', '--file', dest='video_file', type=str,
                        default='', help='video file.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-interval', '--iv', dest='interval', type=int,
                        default=100, help='video stream interval in test')
    parser.add_argument('-gui', '--gui', dest='gui', type=bool,
                        default=True, help='show gui or not')

    args = parser.parse_args()

    input_q = Queue(15)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    '''
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()
    '''
    Thread(target=worker, args=(input_q, output_q)).start()

    batch = 1
    video_capture = WebcamVideoStream(#src=args.video_source,
                                      src=0,
                                      #src=args.video_file,
                                      #src="C:\\Users\\songjue\\Videos\\RealTimes\\RealDownloader\\BirdReaction.mp4",
                                      #src="C:\\Users\\songjue\\Videos\\1.mp4",
                                      width=args.width,
                                      #eight=args.height)
                                      height=args.height, batch=batch).start()
    fps = FPS().start()

    def stop():
        import signal
        fps.stop()
        print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        video_capture.stop()
        cv2.destroyAllWindows()
        os.kill(os.getpid(), signal.SIGKILL)
    '''
    import threading
    timer = threading.Timer(args.interval, stop)
    timer.start()
    '''

    while True:
        #frames = []
        frames = video_capture.read()
        if frames[0] is None:
            print("end of video")
            stop()
        '''
        for i in range(batch):
            frame = video_capture.read_one_frame()
            frames.append(frame)
        '''

        input_q.put(frames)

        #input_q.put(frame)

        t = time.time()
          
        data = output_q.get()

        for f in range(len(frames)):
            frame = frames[f]
          #  '''
            rect_points, class_names, class_colors = draw_boxes_and_labels(
                boxes=np.squeeze(data['boxes'][f]),
                classes=np.squeeze(data['classes'][f]).astype(np.int32),
                scores=np.squeeze(data['scores'][f]),
                category_index=category_index,
                min_score_thresh=.5
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            for point, name, color in zip(rect_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font, 0.3, (0, 0, 0), 1)
          #  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           # '''
            fps.update()

            if fps.get_frames_num() % 100 == 0:
                print('[INFO] frame processed: ', fps.get_frames_num())

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
            if args.gui:
                #if (fps.get_frames_num()) % 10 != 0:
                #    continue
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop()

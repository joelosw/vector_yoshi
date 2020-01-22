import configparser
import anki_vector
import time
import threading
from anki_vector.util import degrees, distance_mm, speed_mmps, Pose, Angle
from anki_vector import behavior
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import cv2
from anki_vector.connection import ControlPriorityLevel
import io
import navigation
from navigation import BALLOON_SIZE_MM
import tensorflow as tf
from PIL import Image
from offline_predict import TFObjectDetection


INITIALIZED = False

##### CLOSE IMAGES WITH ENTER TO CONTINUE ##########
class img_prediction(object):

    def __init__(self, config_file_path=r'azure_config.txt'):
        config_parser = configparser.RawConfigParser()
        config_parser.read(config_file_path)

        self.ENDPOINT = config_parser.get('CustomVision', 'endpoint')

        self.trainiimage_to_predictng_key = config_parser.get('CustomVision', 'training_key')
        self.prediction_key = config_parser.get('CustomVision', 'prediction_key')
        self.prediction_resource_id = config_parser.get('CustomVision', 'ressource_id')
        self.publish_iteration_name = config_parser.get('CustomVision', 'publish_iteration_name')
        self.project_id = config_parser.get('CustomVision', 'project_id')

      
        
        self.predictor = CustomVisionPredictionClient(self.prediction_key, self.ENDPOINT)



    def take_picture(self, robot):

        #image = robot.camera.latest_image.raw_image
        image = robot.camera.capture_single_image().raw_image
        print("image Type: ", type(image))

        with io.BytesIO() as output:
            image.save(output, 'BMP')
            image_to_predict = output.getvalue()

        return image_to_predict

    def take_picture_offline(self, robot):

        image = robot.camera.capture_single_image().raw_image

        return image

    def predict_picture(self, binary_image):
      # Open the image and get back the prediction results as a dict with tuple (left, top, width, height)
        results = self.predictor.detect_image('002e7a08-8696-4ca8-8769-fe0cbc2bd9b0', self.publish_iteration_name, binary_image)
        probability = 0.5
        # Display the results, and return them as a dict (Tuple of four for ecery Tag)
        tag_dict = dict()   
        for prediction in results.predictions:
            print("\t ----ONLINE PREDICTION---" + prediction.tag_name +
            ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100,
            prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))

            if prediction.probability > probability:
                probability = prediction.probability
                tag_dict[prediction.tag_name] = (prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
        return tag_dict

class offline_img_prediction(object):
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('model7.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    print('--------OFFLINE PREDICT INITIIALIZED--------')
    # Load labels
    with open('labels.txt', 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print('opened labels: ', labels)
    od_model = TFObjectDetection(graph_def, labels)

    @staticmethod
    def offline_predict(image):
        # image1 = Image.open("testImage.jpg")
        # image2 = image
        # print("img1: ", type(image1), "\timg2: ", type(image2))

        tag_dict = dict()
        predictions = offline_img_prediction.od_model.predict_image(image)
        print('---OFFLINE RESULTS---\n', predictions)
        for prediction in predictions:
                tag_dict[prediction['tagName']] = (prediction['boundingBox']['left'], prediction['boundingBox']['top'], prediction['boundingBox']['width'], prediction['boundingBox']['height'])
        return tag_dict


def draw_bounding_boxes(im_path, result_dict):
    '''
    Nicht mehr genutzt.
    '''
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    for result in result_dict.values():
        left = result[0]* width
        top = result[1]* height
        w=result[2]* width
        h = result[3]* height
        cv2.rectangle(img,(int(left), int(top)),(int(left + w),int(top + h)), (0, 0, 255), 5)
  
    window = cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def robot_initiate(robot):
    robot.behavior.drive_off_charger()
    robot.behavior.set_head_angle(degrees(0.0))
    robot.behavior.set_lift_height(1.0)
    
    #robot.camera.init_camera_feed()
    #robot.camera.image_streaming_enabled()


def return_from_cliff(robot):
    robot.behavior.turn_in_place(180)
    robot.behavior.drive_straight(100)

def drive_towards_baloon(robot, data, MAX_DRIVING_DISTANCE):
    robot.behavior.turn_in_place(degrees(data[0]))
    #robot.behavior.drive_straight(distance_mm(data[1]), speed_mmps(500))
    v_0 = 200
    a = 3/2 * (v_0**2 / data[1])
    print("*******************Acceleration:",a)
    robot.motors.set_wheel_motors(v_0, v_0, 0, 0)
    #pose = Pose(x = data[1], y = 0, z = 0)
    #robot.behavior.go_to_pose(pose, relative_to_robot=True)
    t = time.time()
    spoken = False
    while (time.time() < t + (v_0/a)): #(data[1]/65)):
        #print(time.time()-t)
        #if not spoken: 
            #if data[1] > 400:                
                #spoken = threading.Thread(target=shutdown(robot))
        if (robot.status.is_cliff_detected):
            robot.motors.set_wheel_motors(-10,-10)
            return_from_cliff(robot)
    robot.motors.stop_all_motors()
    
def shutdown(robot):
    try:
        robot.audio.stream_wav_file("vector_shutdown.wav", 100)
        return True
    except:
        return False


def evaluate_picture(robot, img_prediction, balloon_size = BALLOON_SIZE_MM):
    #online_image = img_prediction.take_picture(robot)
    offline_image = img_prediction.take_picture_offline(robot)

    #t = time.time()
    #result2 = img_prediction.predict_picture(online_image)
    #elapsed = time.time() - t
    #print('----------Time for Online Prediction: ', on_pred, '------------')
    #on_pred.append(elapsed)
    t = time.time()
    results = offline_img_prediction.offline_predict(offline_image)
    elapsed = time.time() - t
    print('----------Time for Offline Prediction: ', elapsed, '------------')
    #off_pred.append(elapsed)
    try:
        results['balloon']

    except KeyError:
        return None

    baloon_left = results['balloon'][0]
    baloon_right = baloon_left + results['balloon'][2]
    baloon_midlle = (baloon_left + baloon_right)/2

    try:
        results['robot']
        robot_left = results['robot'][0]
        robot_right = robot_left + results['balloon'][2]
        robot_middle = (robot_left + robot_right)/2

        if not INITIALIZED:
            navigation.BALLOON_SIZE_MM = calculateBalloonSize(results['robot'][3], results['balloon'][3])
            print("--------------new balloon size------------",navigation.BALLOON_SIZE_MM)
            balloon_size = BALLOON_SIZE_MM

    except KeyError:
        results['robot'] = None
        pass
        #return(-30, 500)


    relation =""
    #TODO: enhanced adaption
    if results['robot']:
        relation = evaluate_relation_balloon_robot(baloon_left, baloon_right, baloon_midlle, robot_left, robot_right, robot_middle)
    else:
        relation = "back"

    turn_degree, distance = define_move(relation, baloon_midlle, balloon_size, results['balloon'][2])

    return (turn_degree, distance)


def evaluate_relation_balloon_robot(baloon_left, baloon_right, baloon_midlle, robot_left, robot_right, robot_middle):
    relation = "";

    if robot_middle > baloon_left and robot_middle < baloon_right:
        relation = "front"
    elif robot_middle > baloon_left and robot_middle > baloon_right:
        relation = "to the right"
    elif robot_middle < baloon_left:
        relation = "to the left"

    return relation


def define_move(relation, baloon_midlle, balloon_size, balloon_width):
    if relation is "back":
        turn_degree = 48 - baloon_midlle * 96
    elif relation is "to the right":
        turn_degree = 48 - baloon_midlle * 96 - 5
    elif relation is "to the left":
        turn_degree = 48 - baloon_midlle * 96 + 5
    #TODO: Vorschlag: Roboter weicht minimal aus und gibt Vollgas, damit er nicht in die Position des "Verfolgten" gerät
    else:
        turn_degree = 2

    distance = balloon_size / (2 * balloon_width * 0.466307658155)
    print("Distanz:", distance)
    return turn_degree, distance


########### not used ############
def drive_and_check(robot, correction, distance=10):
    robot.behavior.drive_straight(distance_mm(distance), speed_mmps(500))

def calculateBalloonSize(robotHeight, balloonHeight):
    INITIALIZED = True
    return balloonHeight/robotHeight*660
"""
All necessary utilities (take and evaluate a picture, estimate position, drive towards position,...)
are implemented in this module
"""

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


class img_prediction(object):
    """ 
    A class to perform a neccessary step to evaluate a picture:
    Initialize a connection to cloud model
    1. Take a picture (in online or offline format)
    2. Evaluate the picture online
    and filter results


    """

    def __init__(self, config_file_path=r'azure_config.txt'):
    """ 
    Instanciate an image_prediction class,
    by setting up a predictor that is connected to azure custom vision

    Parameters
    ----------
    config_file_path: str
        Path to a textfile with the azure credentials


    """
      config_parser = configparser.RawConfigParser()
       config_parser.read(config_file_path)

        self.ENDPOINT = config_parser.get('CustomVision', 'endpoint')
        self.prediction_key = config_parser.get(
            'CustomVision', 'prediction_key')
        self.prediction_resource_id = config_parser.get(
            'CustomVision', 'ressource_id')
        self.publish_iteration_name = config_parser.get(
            'CustomVision', 'publish_iteration_name')
        self.project_id = config_parser.get('CustomVision', 'project_id')

        self.predictor = CustomVisionPredictionClient(
            self.prediction_key, self.ENDPOINT)

    def take_picture(self, robot):
    """ 
    Takes a picture with  the given robot and process it binary

    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should take the picture


    Returns
    -------
    BytesIO
        Binary IO stream that can be handed to the cloud predictor

    """
      image = robot.camera.capture_single_image().raw_image

       with io.BytesIO() as output:
            image.save(output, 'BMP')
            image_to_predict = output.getvalue()

        return image_to_predict

    def take_picture_offline(self, robot):
    """ 
    Takes a picture with  the given robot and process it binary
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should take the picture

    
    Returns
    -------
    PIL
        PIL image that can be handed to the tensorflow model (respective the preprocessing)    
    """

      image = robot.camera.capture_single_image().raw_image

       return image

    def predict_picture(self, binary_image):
    """ 
    Evaluate a given image. This is done using an Azure Custom Vision predictor. 
    Looping through the results of the JSON answer. 
    Only considering results with a probability >0.5. 
    Saving only the instance of each (balloon, robot) with highest probability. 
    
    Parameters
    ----------
    binary_image: 
        The picture that should be evaluated

    
    Returns
    -------
    dict
        Dictionary with the bounding boxes for ballon / robot
    
    """

        results = self.predictor.detect_image(
            '002e7a08-8696-4ca8-8769-fe0cbc2bd9b0', self.publish_iteration_name, binary_image)
        probability_b = 0.5
        probability_r = 0.5
        tag_dict = dict()
        for prediction in results.predictions:
            print("\t ----ONLINE PREDICTION---" + prediction.tag_name +
                  ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100,
                                                                                                                           prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))

            if prediction.tag_name == 'balloon':
                if prediction.probability > probability_b:
                    probability_b = prediction.probability
                    tag_dict[prediction.tag_name] = (
                        prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)

            if prediction.tag_name == 'robot':
                if prediction.probability > probability_r:
                    probability_r = prediction.probability
                    tag_dict[prediction.tag_name] = (
                        prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)

        return tag_dict


class offline_img_prediction(object):
    """ 
    A class  that is necessary if the offline prediction is used. 
    Initializes the TensorFlow model by the graph definition. 
    """
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('model9.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())

    with open('labels.txt', 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print('opened labels: ', labels)
    od_model = TFObjectDetection(graph_def, labels)

    @staticmethod
    def offline_predict(image):
    """ 
    Evaluates the given picture using the initialized offline model. 
    Calling the class that includes pre&postpreoccesing as well
    
    Parameters
    ----------
    image: PIL
        The robot instance that should be controlled

    
    Returns
    -------
    dict
        Dictionary with the bounding boxes for ballon / robot
    
    """
        tag_dict = dict()
        predictions = offline_img_prediction.od_model.predict_image(image)
        print('---OFFLINE RESULTS---\n', predictions)
        for prediction in predictions:
            tag_dict[prediction['tagName']] = (prediction['boundingBox']['left'], prediction['boundingBox']
                                               ['top'], prediction['boundingBox']['width'], prediction['boundingBox']['height'])
        return tag_dict


def draw_bounding_boxes(im_path, result_dict):
    """
    NOT IN USE
    Can be used to draw bounding boxes to a given image

    Parameters
    ----------
    im_path: str
        Path to the image that should be used in the background
    
    result:dict: dict
        Dictionary with bounding boxes as values. 

    """
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    for result in result_dict.values():
        left = result[0] * width
        top = result[1] * height
        w = result[2] * width
        h = result[3] * height
        cv2.rectangle(img, (int(left), int(top)), (int(left + w),int(top + h)), (0, 0, 255), 5)

    window = cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def robot_initiate(robot):
    """ 
    Sets the robot in a starting position
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used
    """

    robot.behavior.set_head_angle(degrees(0.0))
    robot.behavior.set_lift_height(1.0)
    robot.behavior.drive_off_charger()


def return_from_cliff(robot):
    """ 
    Brings the robot back from a  cliff
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used
    """

    robot.behavior.turn_in_place(180)
    robot.behavior.drive_straight(100)


def drive_towards_baloon(robot, data, MAX_DRIVING_DISTANCE=600):
    """ 
    Drive the  robot straight towards the given position, 
    directly setting the motor speed (for parralelization). 
    Capturing the time to stop the motors when the distance is reached

    Optional: Uncomment 274-276 to check for cliff while driving --> causes control loss if cliff is detected
    Optional: Comment 271-273 if the robot should not try to shutddown the other robot
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used
    data: tuple
        Tuple consiting of the degree and distance to the estmated position   
    MAX_DRIVING_DISTANCE: int
        Setting a maximum driving distance, since it is likely that the setup changes when driving for too long.
    """

    robot.behavior.turn_in_place(degrees(data[0]))
    v_0 = 200
    a = 3/2 * (v_0**2 / data[1])

    robot.motors.set_wheel_motors(v_0, v_0, 0, 0)
    t = time.time()
    spoken = False
    while (time.time() < t + (v_0/a)):  # (data[1]/65)):
        print(time.time()-t)
        if not spoken:
            if data[1] > 400:
            spoken = threading.Thread(target=shutdown(robot))
        # if (robot.status.is_cliff_detected):
        #    robot.motors.set_wheel_motors(-10,-10)
        #    return_from_cliff(robot)
    robot.motors.stop_all_motors()


def shutdown(robot):
    """
    Playing the soundfile "Hey Vector, shutdown!"
    Outsourced to a function to enable threading
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used

    Returns
    -------
    bool
        True  if sound was played succesfull.

    """
    try:
        robot.audio.stream_wav_file("vector_shutdown.wav", 100)
        return True
    except:
        return False


def drive_towards_pose(robot, data, MAX_DRIVING_DISTANCE=600):
    """
    Using a simplified path planing, in case the robots are right in front of each other. 
    In this case the robot moves out of the way a little and attacks the other robot from behind.

    For every other relation the robot drives straight towards the estimated position.

    Optional: Uncomment 351-369 to enable path planing for all possible relations of robot and ballon

    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used

    data: tuple
        Tuple consiting of the degree, distance to the estmated position and the relation of the other robot and balloon 

    MAX_DRIVING_DISTANCE: int
        Setting a maximum driving distance, since it is likely that the setup changes when driving for too long.
    """
    direct = data[0]
    dist = min(data[1], MAX_DRIVING_DISTANCE)
    relation = data[2]

    print(relation)
    if relation == 'front':
        robot.behavior.turn_in_place(degrees(data[0]))
        robot.behavior.turn_in_place(degrees(45))
        robot.behavior.drive_straight(distance_mm(dist/2+100), speed_mmps(250))
        robot.behavior.turn_in_place(degrees(-90))
        robot.behavior.drive_straight(distance_mm(dist/2+100), speed_mmps(250))
        robot.behavior.turn_in_place(degrees(-135))

    else:
        drive_towards_baloon(robot, (direct, dist), MAX_DRIVING_DISTANCE)

    # if relation == 'back':
    #     pose = Pose(x = dist, y = 0, z = 0, angle_z = Angle(degrees = 0))
    #     robot.behavior.go_to_pose(pose, relative_to_robot=True)
    #     robot.behavior.set_lift_height(1.0)
    # else:
    #     if relation == 'front':
    #         pose1 = Pose(x = dist/2, y = max(dist/4,50), z = 0, angle_z=anki_vector.util.Angle(degrees=0))
    #         pose2 = Pose(x = dist/2, y = -max(dist/4,50), z = 0, angle_z = Angle(degrees = 30))

    #     elif relation == 'to the left':
    #         pose1 = Pose(x = dist/2, y = -max(dist/4,50), z = 0, angle_z=anki_vector.util.Angle(degrees=0))
    #         pose2 = Pose(x = dist/2, y = max(dist/4,50), z = 0, angle_z = Angle(degrees = 80))

    #     elif relation == 'to the right':
    #         pose1 = Pose(x = dist/2, y = max(dist/4,50), z = 0, angle_z=anki_vector.util.Angle(degrees=0))
    #         pose2 = Pose(x = dist/2, y = -max(dist/4,50), z = 0, angle_z = Angle(degrees = 260))

    #     robot.behavior.go_to_pose(pose1, relative_to_robot=True)
    #     robot.behavior.go_to_pose(pose2, relative_to_robot=True)
    #     robot.behavior.set_lift_height(1.0)


def evaluate_picture(robot, img_prediction, balloon_size=BALLOON_SIZE_MM):
    """ 
    Fundamental function that does the entire picture evaluation process
    this includes: 
        1. Taking a picture
        2. Getting information about the content
        3. Dynamically setting the balloon size
        4. Calculating the turn degree and distance
        5. Evaluating the relation of robot and balloon

    Optional: Uncomment for online image prediction
    Optional: Uncomment 438 to prevent constant recalculation of the ballon size
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be used

    img_prediction: img_prediction
        Instance of (offline_)image_prediction that can be used to take pictures and predict boxes.

    balloon_size: int
        Which balloon size (in mm) to use for calculating the direction, if it can not be determined dynamically
    
    Returns
    -------
    tuple
        Tuple with turn degree, distance and relation of balloon and robot
    """
    balloon_size = BALLOON_SIZE_MM
    #online_image = img_prediction.take_picture(robot)
    offline_image = img_prediction.take_picture_offline(robot)

    #t = time.time()
    #results2 = img_prediction.predict_picture(online_image)
    #elapsed = time.time() - t
    #print('----------Time for Online Prediction: ', on_pred, '------------')
    # on_pred.append(elapsed)

    t = time.time()
    results = img_prediction.predict_picture(online_image)
    elapsed = time.time() - t
    print('----------Time for Online Prediction: ', elapsed, '------------')

    #t = time.time()
    #result2 = offline_img_prediction.offline_predict(offline_image)
    #elapsed = time.time() - t
    #print('----------Time for Offline Prediction: ', elapsed, '------------')

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
        robot_right = robot_left + results['robot'][2]
        robot_middle = (robot_left + robot_right)/2

         if not INITIALIZED:
             navigation.BALLOON_SIZE_MM = calculateBalloonSize(results['robot'][3], results['balloon'][3])
             #INITIALIZED = True
             print("--------------new balloon size------------",navigation.BALLOON_SIZE_MM)

    except KeyError:
        results['robot'] = None
        pass

    relation = ""
    if results['robot']:
        print('ROBOT GEFUNDEN')
        relation = evaluate_relation_balloon_robot(
            baloon_left, baloon_right, baloon_midlle, robot_left, robot_right, robot_middle)
    else:
        relation = "back"
    print(relation)

    turn_degree, distance, angle = define_pose(
        relation, baloon_midlle, balloon_size, results['balloon'][2])

    return (turn_degree, distance, relation)


def evaluate_relation_balloon_robot(baloon_left, baloon_right, baloon_midlle, robot_left, robot_right, robot_middle):
    """
    Determine the discrete relation of the adversarial robot and balloon

    Parameters
    ----------
    coordinates of balloon and robot 

    Returns
    -------
    str
        String with one of  the 4 discrete relations

    """
    relation = ""

    if robot_middle > baloon_left and robot_middle < baloon_right:
        relation = "front"
    elif robot_middle > baloon_left and robot_middle > baloon_right:
        relation = "to the right"
    elif robot_middle < baloon_left:
        relation = "to the left"

    return relation


def define_move(relation, baloon_midlle, balloon_size, balloon_width):
    """
    UNUSED Legacy function
    """
    if relation == "back":
        turn_degree = 48 - baloon_midlle * 96
    elif relation == "to the right":
        turn_degree = 48 - baloon_midlle * 96 - 5
    elif relation == "to the left":
        turn_degree = 48 - baloon_midlle * 96 + 5
    else:
        turn_degree = 2

    distance = balloon_size / (2 * balloon_width * 0.466307658155)
    print("Distanz:", distance)
    return turn_degree, distance


def define_pose(relation, balloon_middle, balloon_size, balloon_width):
    """
    Calculating the estimated position of thhe ballon and best attack angle

    Parameters
    ----------
    relation: str
        relation of balloon and robot
    
    ballon parameters

    Returns
    -------
    tuple
        With turn degree, distance and optimal attack angle

    """
    if relation is "back":
        turn_degree = 48 - balloon_middle * 96
        angle = 0
    elif relation is "to the right":
        turn_degree = 48 - balloon_middle * 96 - 5
        angle = 270
    elif relation is "to the left":
        turn_degree = 48 - balloon_middle * 96 + 5
        angle = 90
    else:
        # Robot is in front of the balloon
        turn_degree = 48 - balloon_middle * 96
        angle = 180

    distance = balloon_size / (2 * balloon_width * 0.466307658155)
    return turn_degree, distance, angle


def calculateBalloonSize(robotHeight, balloonHeight):
    """
    Dynamically Calculating the balloon size, bsed on the known robot size

    Parameters
    ---------
    robotHeight: float
        relative height of the robot in the picture
    balloonHeight: float
        relative height of the balloon in the picture

    Returns
    -------
    float
        estimated absolute ballon size in mm

    """
    print('Calculated BALLOON SIZE: ', balloonHeight/robotHeight*66)
    return balloonHeight/robotHeight*66

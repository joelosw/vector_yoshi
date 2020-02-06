""" 
The main class for the yoshi vector. The elementary logic is implemented here, 
while most utility functions are located in support.py
"""


import anki_vector
from support import *
import support
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import behavior

BALLOON_SIZE_MM = 100
PICTURE_PATH = './balloon_pic.jpg'
MAX_DRIVING_DISTANCE = 1500;
AZURE_CONFIG_FILE = './azure_config.txt'

ROBOT_HEIGHT = 657
probability = 0.5


def search(robot, predictor):
    """ 
    Keeps turning around and looking for the ballon/robot. 
    If there has been no result for 4 rotations (more than 360°) 
    the robot moves a few centimeters to be a harder target
    
    Parameters
    ----------
    robot: anki_vector.Robot
        The robot instance that should be controlled
    
    predictor: support.img_prediction or support.offline_img_preiction
        An initialized instance that can make predictions, using the azure cloud or alternatively GCV.
    
    
    Returns
    -------
    json
        The bounding boxes of detected robots/balloons
    
    """

    result_of_search = None
    
    i = 1

    while result_of_search is None:
        if i>2:
            robot.behavior.drive_straight(distance_mm(150), speed_mmps(250))
            i=0
        i += 1
        print('Keep Searching, taking new picture')
        if i>=4:
            robot.behavior.drive_straight(distance_mm(200), speed_mmps(150))
            i=1
        robot.behavior.turn_in_place(degrees(96))
        result_of_search = evaluate_picture(robot, predictor, BALLOON_SIZE_MM)
        print('Result of Search: ', result_of_search)
        i+=1
    return result_of_search

if __name__ == '__main__':
    """ 
    Main functions that controls the fundamental behaviour.
    First the robot is initialized with maximum priority and surpressed idle behaviour. 
    In a loop:
        Take initial picture
        If no balloon found:
            Keeps searching for an robot/balloon. 
        When a balloon is found, the position is estimated. 
        Then calling a method to drive to the calculated position.
    """
    args = anki_vector.util.parse_command_args()
    with behavior.ReserveBehaviorControl(serial = '008014c1', ip='192.168.0.106'):

        with anki_vector.Robot(serial = '008014c1', name='Vector-N8G2', ip='192.168.0.106',
                            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot_initiate(robot)

            predictor = img_prediction(AZURE_CONFIG_FILE)


            while True:
                result = evaluate_picture(robot, predictor, BALLOON_SIZE_MM)
                if result is None:
                    result = search(robot, predictor)

                support.drive_towards_pose(robot, result, MAX_DRIVING_DISTANCE)
            

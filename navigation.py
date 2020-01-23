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

def search(robot, predictor):
    result_of_search = None
    i = 1
    while result_of_search is None:
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
    args = anki_vector.util.parse_command_args()
    with behavior.ReserveBehaviorControl(serial = '008014c1', ip='192.168.0.106'):

        with anki_vector.Robot(serial = '008014c1', name='Vector-N8G2', ip='192.168.0.106',
                            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot_initiate(robot)

            predictor = img_prediction(AZURE_CONFIG_FILE)


            while True: #not robot.status.is_cliff_detected:
                result = evaluate_picture(robot, predictor, BALLOON_SIZE_MM)
                if result is None:
                    result = search(robot, predictor)

                support.drive_towards_pose(robot, result, MAX_DRIVING_DISTANCE)
            
probability = 0.5
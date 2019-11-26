import anki_vector
from support import *
import support
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import behavior
import time

BALLOON_SIZE_MM = 100
PICTURE_PATH = './balloon_pic.jpg'
MAX_DRIVING_DISTANCE = 100;
AZURE_CONFIG_FILE = './azure_config.txt'


def search(robot, predictor):
    result_of_search = None
    while result_of_search is None:
        robot.behavior.turn_in_place(degrees(50))
        result_of_search = evaluate_picture(robot, predictor, BALLOON_SIZE_MM, PICTURE_PATH)
    i=0
    while i<10:
        print(i)
        robot.behavior.drive_straight(distance_mm(100), speed_mmps(500))
        i = i+1

    return result_of_search

if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with behavior.ReserveBehaviorControl():

        with anki_vector.Robot(args.serial,
                            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot_initiate(robot)

            predictor = img_prediction(AZURE_CONFIG_FILE)


            while True:
                t  = time.time()
                result = evaluate_picture(robot, predictor, BALLOON_SIZE_MM, PICTURE_PATH)
                elapsed = time.time() - t
                print('Time for Evaluation: ', elapsed)
                if result is None:
                    result = search(robot, predictor)

                support.drive_towards_baloon(robot, result, MAX_DRIVING_DISTANCE)


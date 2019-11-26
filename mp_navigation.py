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
result_of_search = None

def drive_searching(robot, predictor):
    global result_of_search
    result_of_search = None
    while result_of_search is None:
        robot.motors.set_wheel_motors(100, 150)
        result_of_search = evaluate_picture(robot, predictor, BALLOON_SIZE_MM, PICTURE_PATH)
    
    robot.motors.set_wheel_motors(0, 0)
    robot.behavior.turn_in_place(degrees(-30))
    robot.behavior.say_text("Stopped Searching")
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
                    result = drive_searching(robot, predictor)

                support.drive_towards_baloon(robot, result, MAX_DRIVING_DISTANCE)




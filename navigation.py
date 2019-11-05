import anki_vector
from find_object import img_prediction
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import behavior

BALLOON_SIZE_MM = 150
PICTURE_PATH = './balloon_pic.jpg'
MAX_DRIVING_DISTANCE = 100;

def search(robot, predictor):
    result_of_search = None
    while result_of_search is None:
        robot.behavior.turn_in_place(degrees(50))
        result_of_search = evaluate_picture(robot, predictor, BALLOON_SIZE_MM, PICTURE_PATH)
    return result_of_search

if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with behavior.ReserveBehaviorControl():

        with anki_vector.Robot(args.serial,
                            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot_initiate(robot)

            predictor = img_prediction(AZURE_CONFIG_FILE)
            while True:
                result = evaluate_picture(robot, predictor, BALLOON_SIZE_MM, PICTURE_PATH)

                if result is None:
                    result = search(robot)

                drive_towards_baloon(robot, result, MAX_DRIVING_DISTANCE)


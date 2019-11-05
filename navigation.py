import anki_vector
from find_object import img_prediction
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import behavior

<<<<<<< HEAD
BALLOON_SIZE_MM = 150
PICTURE_PATH = './balloon_pic.jpg'
MAX_DRIVING_DISTANCE = 100;
=======
BALLOON_SIZE_MM = 100

def drive_towards_baloon(bboxes, robot, max_distance=100):
    print(bboxes)
    baloon_left = bboxes['balloon'][0]
    baloon_right = baloon_left + bboxes['balloon'][2]
    baloon_midlle = baloon_left + 0.5 * baloon_right
    print('baloon_midlle ', baloon_midlle)
    turn_degree = 25 - baloon_midlle * 50
    distance = BALLOON_SIZE_MM / (2 * bboxes['balloon'][2] * 0.466307658155)
    distance = min(distance, max_distance)
    robot.behavior.turn_in_place(degrees(turn_degree*1.3))
    print("Distanz:")
    print(distance)
    print(distance)
    robot.behavior.drive_straight(distance_mm(distance), speed_mmps(500))



def find_balloon(robot):
    balloon = None
    blue = None
    try:
        results = prediction.predict('./balloon_and_robot.jpg', robot)
        balloon = results['balloon']

    except KeyError:
        print('no balloon')
        pass
    print('balloon: ', balloon)
    print("searchin'...")
    if balloon == None:
        robot.behavior.turn_in_place(degrees(50))
        find_balloon(robot)
    else:
        drive_to_baloon(results, robot)
>>>>>>> 23cef476261071854869eddb4058b15a91f8a915

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


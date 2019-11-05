import anki_vector
from find_object import img_prediction
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import behavior

BALLOON_SIZE_MM = 100

def drive_to_baloon(bboxes, robot):
    print(bboxes)
    baloon_left = bboxes['balloon'][0]
    baloon_right = baloon_left + bboxes['balloon'][2]
    baloon_midlle = baloon_left + 0.5 * baloon_right
    print('baloon_midlle ', baloon_midlle)
    turn_degree = 25 - baloon_midlle * 50
    distance = BALLOON_SIZE_MM / (2 * bboxes['balloon'][2] * 0.466307658155)
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

def turn(robot):
    print("Turn")
    robot.behavior.turn_in_place(degrees(90))
    turn(robot)

if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with behavior.ReserveBehaviorControl():

        with anki_vector.Robot(args.serial,
                            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot.behavior.set_head_angle(degrees(0.0))
            robot.behavior.set_lift_height(1.0)
            robot.behavior.say_text("Start")
            robot.behavior.drive_off_charger()
            prediction = img_prediction()
            find_balloon(robot)

            robot.behavior.say_text("Ende")

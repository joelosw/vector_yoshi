import anki_vector
import find_object
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel


def drive_to_baloon(bboxes, robot):
    baloon_left = bboxes['balloon'][0]
    baloon_right = baloon_left + bboxes['balloon'][2]
    baloon_midlle = baloon_left + 0.5* baloon_right
    print('baloon_midlle ' , baloon_midlle)
    turn_degree = 25 - baloon_midlle*50
    distance = 200/(2*bboxes['balloon'][2]*0.466307658155)
    print(turn_degree)
    robot.behavior.turn_in_place(degrees(turn_degree))
    print(distance)
    robot.behavior.drive_straight(distance_mm(distance), speed_mmps(500))

def keep_searching(robot):
    #Turns Vector in place at 50 degrees (left)
    robot.behavior.turn_in_place(degrees(50))
    
if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial, behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
        robot.behavior.say_text("Start")
        robot.behavior.drive_off_charger()
        predictions = img_prediction()
        balloon = None
        while  balloon is None:
            try:
                results = prediction.predict()
                balloon = results['balloon']
                drive_to_baloon(balloon, robot)
            except KeyError:
                pass
        robot.behavior.say_text("End")


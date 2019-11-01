import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel


def drive_to_baloon(bboxes, robot):
    baloon_left = bboxes['balloon'][0]
    baloon_right = baloon_left + bboxes['balloon'][2]
    baloon_midlle = baloon_left * 0.5* baloon_right
    turn_degree = 25 - baloon_midlle*50
    distance = 200/(2*bboxes['balloon'][2]*0.466307658155)
    print(turn_degree)
    robot.behavior.turn_in_place(degrees(turn_degree))
    print(distance)
    robot.behavior.drive_straight(distance_mm(distance), speed_mmps(500))
    
if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial, behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
        robot.behavior.say_text("Start")
        drive_to_baloon({'balloon': (0.1, 0.3, 0.3, 0.4)}, robot)
        robot.behavior.say_text("End")


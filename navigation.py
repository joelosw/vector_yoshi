import anki_vector
from anki_vecctor.util import degreees, distance_mm, speed_mmps



def drive_to_baloon(bboxes, robot):
    baloon_left = bboxes[baloon][0]
    baloon_right = baloon_left + bboxes[balloon][2]
    
    robot.behavior.turn_in_place(degrees())
    robot.behavior.drive_straight(distance_mm(200), speed_mmps(50))
    
if __name__ == '__main__':
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial) as robot:
        drive_to_baloon({'balloon': (0.1, 0.3, 0.3, 0.4)}, robot)


50 degree v 
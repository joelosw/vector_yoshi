import anki_vector

with anki_vector.Robot() as robot:
    robot.camera.init_camera_feed()
    image = robot.camera.latest_image
    image.raw_image.show()


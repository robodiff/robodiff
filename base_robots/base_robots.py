import taichi as ti
import robodiff_startup as didv
import simulator as sim
import IPython


def robot(scene, add_object=False):
  ROBOT_X_START = .3
  ROBOT_Y_START = 0.07 if didv.args.terrain else (sim.bound+0.84) * sim.dx
  ROBOT_ASPECT_RATIO = 1.0 if didv.args.square_body else 0.7
  ROBOT_WIDTH = 0.2
  ROBOT_HEIGHT = ROBOT_WIDTH * ROBOT_ASPECT_RATIO
  ROBOT_RESOLUTION =  128 if didv.args.high_resolution_robot else 64
  scene.set_offset(ROBOT_X_START, ROBOT_Y_START)
  if add_object:
    obj_size_frac_w = 4
    obj_height_ratio = 1
    obj_w = ROBOT_WIDTH/obj_size_frac_w
    obj_cnt_w = ROBOT_RESOLUTION // obj_size_frac_w

    obj_h = obj_w * obj_height_ratio
    obj_cnt_h = int(obj_cnt_w * obj_height_ratio)

    start_x = (ROBOT_WIDTH -  obj_h) * 1/2
    start_y = ROBOT_HEIGHT + 2/ROBOT_RESOLUTION
    # IPython.embed()
    circle_lambda =  lambda x, y: (x-ROBOT_X_START-start_x-obj_w/2)**2 + (y- ROBOT_Y_START - start_y-obj_w/2)**2 < (obj_w/2)**2 #((particle_y - ( y + h/2 + self.offset_y))**2+ (particle_x - (x  + self.offset_x + CIRCLE_RADIUS/2 ))**2 > CIRCLE_RADIUS**2)
    yes_lambda = lambda x, y: True
    scene.add_rect(start_x, start_y, obj_w, obj_h, -1, _w_count = obj_cnt_w, _h_count = obj_cnt_h, terrain=True, filter_in=circle_lambda)

  scene.add_rect(0.0, 0.0, ROBOT_WIDTH, ROBOT_HEIGHT, 0, _w_count = ROBOT_RESOLUTION, _h_count = int(ROBOT_RESOLUTION*ROBOT_ASPECT_RATIO))

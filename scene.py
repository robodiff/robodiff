import numpy as np

def default_terrain_filter(x,y):
    border = 1/128 * 0.25
    return x > border and x < 1-border and y > border and y < 1-border


class Scene:
  def __init__(self, x, v, m, F, C, particle_type, dx, dim, terrain, terrain_mode=1,seed=0):
    self.n_particles = 0
    self.n_solid_particles = 0
    self.n_terrain_particles = 0

    self.x = []
    self.actuator_id = []
    self.particle_type = []
    self.offset_x = 0
    self.offset_y = 0
    self.dx = dx
    self.w_count = 0
    self.h_count = 0
    self.robot_aspect_ratio = 0
    assert dim == 2 or dim == 3
    self.dim = dim
    self.terrain = terrain
    self.terrain_mode = terrain_mode

    self._x_t = x
    self._v_t = v
    self._m_t = m
    self._F_t = F
    self._C_t = C
    self._particle_type_t = particle_type

    np.random.seed(seed)
    if self.terrain:
      self.add_terrain()

  def add_rect(self, x, y, w, h, actuation, ptype=1, _w_count=None, _h_count=None, terrain=False, filter_in = lambda x, y: True):
    if ptype == 0:
      assert actuation == -1

    if _w_count is None:
      self.w_count = int(w / self.dx) * 2
    else:
      self.w_count = _w_count
        
    if _h_count is None:
      self.h_count = int(h / self.dx) * 2
    else:
      self.h_count = _h_count

    self.robot_aspect_ratio = self.w_count / self.h_count

    real_dx = w / self.w_count
    real_dy = h / self.h_count
    for i in range(self.w_count):
      for j in range(self.h_count):
        particle_x =  x + (i + 0.5) * real_dx + self.offset_x
        particle_y = y + (j + 0.5) * real_dy + self.offset_y
        # if ((particle_y - ( y + h/2 + self.offset_y))**2+ (particle_x - (x  + self.offset_x + CIRCLE_RADIUS/2 ))**2 > CIRCLE_RADIUS**2): # + w/2
        if filter_in(particle_x, particle_y):
          self.add_particle([particle_x, particle_y], actuation, ptype, terrain)
    


  def add_particle(self, loc, act_id, ptype, terrain):
    self.x.append(loc)
    self.actuator_id.append(act_id)
    self.particle_type.append(ptype)
    self.n_particles += 1
    self.n_solid_particles += int(ptype == 1)
    self.n_terrain_particles += int(terrain == 1)

  def add_terrain(self):
      
    PARTICLE_DENSITY = 64 / 0.2
    WIDTH = 1.0
    HEIGHT_ADDTL = 0.01*3
    HEIGHT_BASE = 3.5 * 1/128
    HEIGHT = HEIGHT_BASE + HEIGHT_ADDTL
    TERRAIN_X_START = 0.0
    TERRAIN_Y_START = 0.0


    def terrain_ramp_down_filter(x, y):
        dx =  1- (x - TERRAIN_X_START)
        dy = y - TERRAIN_Y_START
        return dy * 6 < dx # US ADA max ramp grade would be 12
    
    def terrain_ramp_filter(x, y):
      dx = x - TERRAIN_X_START
      dy = y - TERRAIN_Y_START
      
      return dy * 12 < dx # US ADA max ramp grade
    
    def terrain_flat_filter(x,y):
        return y < 0.1

    def terrain_cosine_filter(x,y, hill_height = 0.02, hill_lambda=10):
        
        dx = x - TERRAIN_X_START
        dy = y - TERRAIN_Y_START

        _cosine_dx = np.cos(dx * hill_lambda) * hill_height
        cosine_dx = _cosine_dx
        # max(0.0, )
        return dy  < cosine_dx + HEIGHT - hill_height

    def terrain_boulder_filter(x,y):
        hill_height = 0.007
        hill_lambda = 100
        dx = x - TERRAIN_X_START
        dy = y - TERRAIN_Y_START

        sine_dx = np.sin(dx * hill_lambda) * hill_height

        # if sine_dx < 0:
        #     return True
        
        return dy  < sine_dx + HEIGHT - hill_height
    particle_count_terrain = int(WIDTH * PARTICLE_DENSITY)*int(HEIGHT * PARTICLE_DENSITY) * 2
    if self.terrain_mode == 1:
      self.internal_add_terrain(particle_count_terrain, terrain_cosine_filter,  hill_height = 0.02, hill_lambda=10)
    elif self.terrain_mode == 2:
      self.internal_add_terrain(particle_count_terrain, terrain_cosine_filter,  hill_height = 0.01, hill_lambda=50)

    # self.internal_add_terrain(int(WIDTH * PARTICLE_DENSITY)*int(HEIGHT * PARTICLE_DENSITY)*2, terrain_ramp_down_filter)
    # self.internal_add_terrain(particle_count_terrain, terrain_ramp_filter)

  def internal_add_terrain(self, terrain_count,  terrain_filter_in, **kwargs):
      terrain_added = 0
      while terrain_added < terrain_count:
          x, y = np.random.random(2)
          if default_terrain_filter(x,y) and terrain_filter_in(x,y, **kwargs):
              self.add_particle([x, y], -1, 1, True)
              terrain_added+=1

  def reset(self):
    #reset scene
    for i in range(self.n_particles):
      self._x_t[0, i] = self.x[i]
      if self.dim == 2:
        self._F_t[0, i] = [[1, 0], [0, 1]]
      else: #dim == 3
        self._F_t[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      self._particle_type_t[i] = self.particle_type[i]
    
  def get_x(self):
    return self._x_t
    
  def get_v(self):
    return self._v_t

  def get_m(self):
    return self._m_t
    
  def get_F(self):
    return self._F_t
    
  def get_C(self):
    return self._C_t
    
  def get_particle_type(self):
    return self._particle_type_t
    
  def set_offset(self, x, y):
    self.offset_x = x
    self.offset_y = y

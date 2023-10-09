# -----------------------------------------------------------------------------
# Visualization based on source code written by Nicolas P. Rougier.
#
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

from glumpy.transforms.viewport import Viewport
import numpy as np
from glumpy import app, gl, gloo, glm
app.use("sdl")
from glumpy.transforms import TrackballPan, PanZoom, Position
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import SegmentCollection, PathCollection, GlyphCollection
import sys
import os

import IPython

from visualization.terrain_height_maps import * 
IS_ON_DM_MACHINE = os.getenv("USER", "") == "lynx"

try:
  import pygame
  using_pygame = True
except:
  print('pygame not available')
  using_pygame = False

points_vertex = """
#version 120
uniform float linewidth;
uniform float antialias;
attribute vec4  fg_color;
attribute vec4  bg_color;
attribute float radius;
attribute vec3  position;
varying float v_pointsize;
varying float v_radius;
varying float v_z;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
void main (void)
{
    v_radius = radius;
    v_fg_color = fg_color;
    v_bg_color = bg_color;
    gl_Position = <transform>;
    v_z = gl_Position.z;
    gl_PointSize = 2 * (v_radius + linewidth + 1.5*antialias);
}
"""

points_fragment = """
#version 120
uniform float linewidth;
uniform float antialias;
varying float v_radius;
varying float v_z;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
float marker(vec2 P, float size)
{
   const float SQRT_2 = 1.4142135623730951;
   float x = SQRT_2/2 * (P.x - P.y);
   float y = SQRT_2/2 * (P.x + P.y);
   float r1 = max(abs(x)- size/2, abs(y)- size/10);
   float r2 = max(abs(y)- size/2, abs(x)- size/10);
   float r3 = max(abs(P.x)- size/2, abs(P.y)- size/10);
   float r4 = max(abs(P.y)- size/2, abs(P.x)- size/10);
   return min( min(r1,r2), min(r3,r4));
}
void main()
{
    float r = (v_radius + linewidth + 1.5*antialias);
    float t = linewidth/2.0 - antialias;
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - v_radius;
  //  float signed_distance = marker((gl_PointCoord.xy - vec2(0.5,0.5))*r*2, 2*v_radius);

    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    vec2 p = (gl_PointCoord.xy - vec2(0.5, 0.5)) * 2;
    float len_p = length(p);
    gl_FragDepth = 0.5 * v_z  + 0.5* (len_p)*v_radius / 64.0;
    vec3 normal = normalize(vec3(p.xy, 1.0 - len_p));
    vec3 direction = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(0.0, dot(direction, normal));
    float specular = pow(diffuse, 24.0);
    vec4 bg_color = vec4(max(diffuse*v_bg_color.rgb, specular*vec3(1.0)), 1);

    // Inside shape
    if( signed_distance < 0 ) {
        // Fully within linestroke
        if( border_distance < 0 ) {
             gl_FragColor = v_fg_color;
        } else {
            gl_FragColor = mix(bg_color, v_fg_color, alpha);
        }
    // Outside shape
    } else {
        discard;
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else if( abs(signed_distance) < (linewidth/2.0 + antialias) ) {
            gl_FragColor = vec4(v_fg_color.rgb, v_fg_color.a * alpha);
        } else {
            discard;
        }
    }
}
"""

def add_xlines(ticks, grid_x, xmin, ymin, xmax, ymax, z):
    num_lines = grid_x+1
    x = np.linspace(xmin, xmax, num_lines)
    num_minor_levels = np.ceil(np.log2(num_lines))
    linewidth = np.ones(num_lines)
    already_active_mask = np.zeros(num_lines)
    for minor_level in range(int(num_minor_levels)):

        active_mask = ((x*(1<<minor_level)) % 1 == 0) & (already_active_mask==0)
        already_active_mask[active_mask] = 1
        linewidth[active_mask] = 8 /(1<<minor_level)
    P0 = np.zeros((num_lines,3))
    P1 = np.zeros((num_lines,3))

    P0[:,0] = x
    P0[:,1] = ymin
    P0[:,2] = z
    P1[:,0] = x
    P1[:,1] = ymax
    P1[:,2] = z
    ticks.append(P0, P1, linewidth=linewidth, color=(0,0,0,1))

def add_ylines(ticks, grid_y, xmin, ymin, xmax, ymax, z):
    num_lines = grid_y+1
    y = np.linspace(ymin, ymax, num_lines)
    num_minor_levels = np.ceil(np.log2(num_lines))
    linewidth = np.ones(num_lines)
    already_active_mask = np.zeros(num_lines)
    for minor_level in range(int(num_minor_levels)):

        active_mask = ((y*(1<<minor_level)) % 1 == 0) & (already_active_mask==0)
        already_active_mask[active_mask] = 1
        linewidth[active_mask] = 8 /(1<<minor_level)
    P0 = np.zeros((num_lines,3))
    P1 = np.zeros((num_lines,3))

    P0[:,0] = xmin
    P0[:,1] = y
    P0[:,2] = z
    P1[:,0] = xmax
    P1[:,1] = y
    P1[:,2] = z
    ticks.append(P0, P1, linewidth=linewidth, color=(0,0,0,1))

def draw_floor(ticks, grid_x, xmin, xmax, z, height_map= lambda x: x):
    num_lines = grid_x+1
    x = np.linspace(xmin, xmax, num_lines)
    y = height_map(x)

    P0 = np.zeros((num_lines-1,3))
    P1 = np.zeros((num_lines-1,3))

    P0[:,0] = x[1:]
    P0[:,1] = y[1:]
    P0[:,2] = z
    P1[:,0] = x[:-1]
    P1[:,1] = y[:-1]
    P1[:,2] = z
    ticks.append(P0, P1, linewidth=16, color=(1,0,1,1))

    pass

def add_rect(ticks, xmin, ymin, xmax, ymax, z, lw=2):
    # Frame grid
    P0 = [(xmin,ymin,z), (xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z)]
    P1 = [(xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z), (xmin,ymin,z)]
    ticks.append(P0, P1, linewidth=lw)

def add_grid(ticks, grid_x, grid_y, floorticks=None):
    z = 0
    dx = 1/grid_y

    xmin,xmax = 0, dx * grid_x
    ymin,ymax = 0, 1

    add_xlines(ticks, grid_x, xmin, ymin, xmax, ymax, z)
    add_ylines(ticks, grid_y, xmin, ymin, xmax, ymax, z)
    if floorticks is not None:
        draw_floor(floorticks, grid_x, xmin, xmax, z, height_map=ramp)

def visualize(pos=None,
            radius=None,
            colors=None,
            fg_color=(0,0,0,1),
            title=None,
            initial_stepsize=0,
            enable_ticks_and_text=True,
            gui_auto_close=False,
            arrow_offsets = None,
            arrow_centers = None,
            arrow_dts = None,
            arrow_widths=None,
            arrow_colors=None):
    """
    pos: (frames, particles, position dimensions)
    radius: (particles, size) OR (frames, particles, size)
    colors: (particles, G / RGB / RGBA) OR (frames, particles, G / RGB / RGBA)
    """
    global current_frame, num_frames, step_size, radius_backup, _title, radius_multiplier

    assert not ((arrow_dts is None) ^ (arrow_centers is None)) # ensure both parameters are in the same state.

    assert isinstance(pos, np.ndarray)
    assert len(pos.shape) == 3
    #import IPython


    num_frames = pos.shape[0]
    current_frame = 0
    step_size = initial_stepsize

    num_particles = pos.shape[1]
    pos_num_dims = pos.shape[2]

    if radius is None:
        radius = 10 * np.ones(num_particles)
    radius_backup = radius.copy()

    if colors is None:
        colors = np.random.random((num_particles, 4))
        colors[:, 3] = 1

    # if len(radius.shape) == 1:
    #     radius = radius.reshape(-1, 1)
    radius_num_dims = len(radius.shape)
    _title=title

    if len(colors.shape) == 1: # if Greyscale + does not change from frame to frame
        colors = colors.reshape(-1, 1)
    colors_num_dims = len(colors.shape)
    colors_num_channels = colors.shape[-1]
    colors_num_channels_fill = 3 if colors_num_channels != 4 else colors_num_channels

    assert 1 <= pos_num_dims <= 3
    assert 1 <= radius_num_dims < 3
    assert 2 <= colors_num_dims <= 3
    assert colors_num_channels in [1,3,4]

    
    scale_factor = 2 if IS_ON_DM_MACHINE else 1
    window = app.Window(width=1920*scale_factor, height=1080*scale_factor, color=(1,1,1,1), vsync=True, title="test_title")
    
    if sys.platform == "darwin":
        window._hidpi=True

    program = gloo.Program(points_vertex, points_fragment, count=num_particles)
    view = np.eye(4, dtype=np.float32)
    glm.translate(view, 0, 0, -5)

    program['fg_color'] = 0,0,0,1
    program['linewidth'] = 0.0
    program['antialias'] = 0.0

    program['position'] = np.zeros((num_particles, 3))
    # program['radius']   = np.zeros(num_particles)
    program['bg_color'] = np.ones((num_particles, 4))

    program['position'][:, :pos_num_dims] = pos[current_frame] # load first frame
    program['radius'] = radius if radius_num_dims == 1 else radius[current_frame]

    color_frame = colors if colors_num_dims == 2 else colors[current_frame]
    program['bg_color'][:, :colors_num_channels_fill] = color_frame


    # create an instance of the TrackballPan object.
    # Use trackball for 3D world.
    # trackball = TrackballPan(Position("position"), znear=3, zfar=10, distance=5)
    # trackball2 = TrackballPan(Position(), znear=3, zfar=10, distance=5)

    start_dist = 5
    trackball = PanZoom(Position("position"), znear=3, zfar=10, distance=start_dist)
    trackball2 = PanZoom(Position(), znear=3, zfar=10, distance=start_dist)
    viewport = Viewport()
    
    program['transform'] = trackball
    labels = None if not enable_ticks_and_text else GlyphCollection(transform=trackball2, viewport=viewport)
    ticks = None if not enable_ticks_and_text else SegmentCollection(mode="agg", transform=trackball2, viewport=viewport, 
                         linewidth='local', color='local')
    floorticks = None if not enable_ticks_and_text else SegmentCollection(mode="agg", transform=trackball2, viewport=viewport, 
                         linewidth='local', color='local')
    arrows = None if arrow_dts is None else SegmentCollection(mode="agg", transform=trackball2, viewport=viewport, 
                         linewidth='local', color='local')

    if enable_ticks_and_text:
        add_grid(ticks, 128, 128) #  floorticks=floorticks)
        # add_labels(labels, )
        z = 0
        xmin,xmax = 0,3
        ymin,ymax = 2.5/128, (128-2.5)/128


        regular = FontManager.get("OpenSans-Regular.ttf")
        bold    = FontManager.get("OpenSans-Bold.ttf")
        N_MAJOR_LINES = 30
        scale = 0.001
        for x in np.linspace(xmin,xmax, N_MAJOR_LINES+1):
            text = "{:.2f}".format(x)

            labels.append(text, regular, origin = (x, -0.05, z),
                    scale= scale, direction = (0,1,0),
                    anchor_x = "center", anchor_y = "top")


        if title is not None and len(title) != 0:
            labels.append(title, bold, origin = (1.5, 1.1, z),
                    scale=scale, direction = (1,0,0),
                    anchor_x = "center", anchor_y = "center")


        

    if arrows is not None:
        z = np.linspace(0.1, 0.99, arrow_centers.shape[1])

        P0 = np.zeros((arrow_centers.shape[1],3))
        P1 = np.ones((arrow_centers.shape[1],3))

        # a =  arrow_centers[current_frame]
        # b = arrow_centers[current_frame]+arrow_dts
        # pts = np.dstack([a, b])
        # pmin = pts.min(axis=1)
        # pmax = pts.max(axis=1)

        # P0[:, :2] = pmin
        # P1[:, :2] = pmax

        P0[:, 2] = z
        P1[:, 2] = z

        if arrow_widths is None:
            arrow_widths = [2] * arrow_centers.shape[1]
        if arrow_colors is None:
            arrow_colors = [[0,0,0, 1]] * arrow_centers.shape[1]
        arrows.append(P0, P1, linewidth=arrow_widths, color=arrow_colors)

    trackball.aspect = 1
    trackball2.aspect = 1
    trackball.pan = (-0.9, -0.75)
    trackball2.pan = (-0.9, -0.75)
    # trackball.pan = (-0.9, -1.5)
    # trackball2.pan = (-0.9, -1.5)
    trackball.zoom = 1
    trackball2.zoom = 1

    # rotation around the X axis
    # trackball.phi = 0
    # trackball2.phi = 0
    # rotation around the Y axis
    # trackball.theta = 0
    # trackball2.theta = 0
    trackball.zoom = 4
    trackball2.zoom = 4
    trackball.view_x = -0.9
    trackball2.view_x = -0.9
    trackball.view_y = -0.85
    trackball2.view_y = -0.85
    radius_multiplier = 1.0

    @window.event
    def on_draw(dt):
        global current_frame, num_frames, step_size, _title, radius_multiplier
        current_frame += step_size
        if current_frame >= num_frames and gui_auto_close:
            window.close()
            return 
        current_frame %= num_frames
        program['position'][:, :pos_num_dims] = pos[current_frame] # load first frame

        if radius_num_dims == 2:
            program['radius'] = radius[current_frame] * radius_multiplier

        if colors_num_dims == 3:
            program['bg_color'][:, :colors_num_channels_fill] = colors[current_frame]

        if arrows is not None:
            # pa =  np.tile(arrow_centers[current_frame], [4, 1])
            # pb = np.tile(arrow_centers[current_frame]+arrow_dts, [4, 1])
            pa =  arrow_centers[current_frame]
            pb = (arrow_centers[current_frame]+arrow_dts[current_frame]) #*(10/trackball.zoom) )

            # pts = np.dstack([pa,pb])
            # P0 = pts.min(axis=2)
            # P1 = pts.max(axis=2)
            # IPython.embed()
            arrows[0]['P0'][:, :2] = pa.repeat(4, axis=0)
            arrows[0]['P1'][:, :2] = pb.repeat(4, axis=0)

        window.clear()
        program.draw(gl.GL_POINTS)
        if enable_ticks_and_text:
            floorticks.draw()
            ticks.draw()
            labels.draw()
        if arrows is not None:
            # IPython.embed()
            arrows.draw()

        if using_pygame:
            pygame.display.set_caption(f"{_title} | step: {current_frame} of {num_frames} (size: {step_size})")
        else:
            window.set_title(f"{_title} | step: {current_frame} of {num_frames} (size: {step_size})")
        

    @window.event
    def on_key_release(symbol, modifiers):
        global current_frame, step_size, radius_backup, radius_multiplier
        if (symbol == app.window.key.RIGHT):
            step_size += 1
        elif (symbol == app.window.key.LEFT):
            step_size -= 1
        if using_pygame:
        
            try:
                character = chr(symbol)
                
                if (character in '+='):
                    program['radius'] *= 1.1
                    radius_multiplier *= 1.1
                elif (character in "-_"):
                    program['radius']  /= 1.1
                    radius_multiplier /= 1.1
                elif (character in "rR"):
                    current_frame = 0
                    step_size= 0
                    program['radius'] = radius_backup if radius_num_dims == 1 else radius_backup[current_frame]
                    radius_multiplier = 1.0

                elif (character in 'aA'):
                   current_frame -= 1
                elif (character in "dD"):
                   current_frame += 1
                elif character in "zZ":
                  trackball.zoom *= 1.1
                  trackball2.zoom *= 1.1
                elif character in "xX":
                  trackball.zoom /= 1.1
                  trackball2.zoom /= 1.1
                   
            except:
                pass

    @window.event
    def on_character(character):
        global radius_multiplier
        if (character in '+='):
            program['radius'] *= 1.1
            radius_multiplier *= 1.1
        elif (character in "-_"):
            program['radius']  /= 1.1
            radius_multiplier /= 1.1
        elif (character in "rR"):
            global current_frame, step_size, radius_backup
            current_frame = 0
            step_size= 0
            program['radius'] = radius_backup if radius_num_dims == 1 else radius_backup[current_frame]
            radius_multiplier = 1.0
        elif (character in 'aA'):
           current_frame -= 1
        elif (character in "dD"):
           current_frame += 1

    window.attach(program["transform"])
    if enable_ticks_and_text:
        window.attach(ticks["transform"])
        window.attach(ticks["viewport"])
        # window.attach(labels["transform"])
        # window.attach(labels["viewport"])

    if not enable_ticks_and_text and arrows is not None:
        window.attach(arrows["transform"])
        window.attach(arrows["viewport"])

    gl.glEnable(gl.GL_DEPTH_TEST)
    app.run()

if __name__ == "__main__":
    pos = (np.random.random((100, 1<<10, 1)) - 0.5)
    pos[1:] *= 0.01
    pos = np.cumsum(pos, axis=0)
    visualize(pos = pos)
    pos = (np.random.random((100, 1<<15, 2)) - 0.5)
    pos[1:] *= 0.01
    pos = np.cumsum(pos, axis=0)
    visualize(pos = pos)

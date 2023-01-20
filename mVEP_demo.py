# -*- coding: utf-8 -*-
"""
mVEP test demo

Author: Brynhildr W
Refer: 
    Swolf (NeuroScanPort)
update: 2022/9/11

"""

# %% load modules
import string

import numpy as np

from psychopy import (core, data, event)
from psychopy.visual import (Window, shape, line, TextStim, TextBox2)
from ex_base import (NeuroScanPort, sinusoidal_sample)

# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# config window object
win = Window([1920, 1080], color=(-1,-1,-1), fullscr=False, monitor='testmonitor',
                    screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False
event.globalKeys.add(key='escape', func=win.close)

# config basic parameters of stimuli
n_elements = 15                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_len = 150                         # side length of a single square | unit: pix
square_size = np.array([square_len, square_len])
stim_sizes[:] = square_size

win_size = np.array(win.size)
rows, columns = 3, 5
distribution = np.array([columns, rows])

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0]/columns, win_size[1]/rows]) / 2
if (origin_pos[0]<(square_len/2)) or (origin_pos[1]<(square_len/2)):
    raise Exception('Too much blocks or too big the single square!')
for i in range(distribution[0]):  # loop in columns
    for j in range(distribution[1]):  # loop in rows
        stim_pos[i*distribution[1]+j] = origin_pos + [i,j]*origin_pos*2

# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:,1] *= -1     # invert the y-axis

# %%
# config ssvep
refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
# refresh_rate = 60
display_time = 1.  # keyboard display time before 1st stimulus
index_time = 0.5   # indicator display time
rest_time = 0.5    # rest-state time
blink_time = 0.5
stim_time= 5
stim_frames = int(stim_time*refresh_rate)

# config colors
stim_colors = (1,-1,-1)

# config line elements
# method 1: Line object
mvep_stimuli = []
pos_x = np.linspace(stim_pos[0][0], stim_pos[3][0], stim_frames)
pos_y = [stim_pos[0][1], stim_pos[2][1]]
for frame in range(stim_frames):
    mvep_stimuli.append(line.Line(win=win, start=[pos_x[frame], pos_y[0]], end=[pos_x[frame], pos_y[1]],
        units='pix', lineWidth=5, opacity=1.0, color=stim_colors, colorSpace='rgb'))

# config index stimuli: downward triangle
# method 1: TextStim
index_stimuli = TextStim(win=win, text='\u2BC6', font='Arial', color=(1.,1.,0.), colorSpace='rgb',
                         units='pix', height=75, bold=True, autoLog=False)

# config experiment parameters
mvep_conditions = [{'id': i} for i in range(n_elements)]
mvep_nrep = 1
trials = data.TrialHandler(mvep_conditions, mvep_nrep, name='mvep', method='random')

# initialise experiment
# if port_available:
#     port.sendLabel(0)

routine_timer = core.CountdownTimer()

# start routine
# display speller interface
routine_timer.reset(0)
routine_timer.add(display_time)
while routine_timer.getTime() > 0:
    for mvep_stimulus in mvep_stimuli:
        mvep_stimulus.draw()
        win.flip()

win.close()
core.quit()
# %%

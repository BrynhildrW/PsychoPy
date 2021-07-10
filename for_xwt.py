# -*- coding: utf-8 -*-
"""
A SSVEP experiment script.
Remember to draw the image of each frame in advance instead of calculating the color in real time for each frame

Author: Brynhildr W
Refer: Swolf (NeuroScanPort)
update: 10/1/2020
"""
# %% load modules
import os
import string
import sys

import numpy as np
from numpy import (sin, tan, pi)
import scipy.io as io

from psychopy import (core, event, data, visual, monitors)
from psychopy.core import quit
from psychopy.preferences import prefs

from ex_base import (NeuroScanPort, sinusoidal_sample, square_sample)

# %%
# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# config window
win = visual.Window(size=[1920, 1080], color=(-1,-1,-1), fullscr=True, monitor='testmonitor',
                    screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False

event.globalKeys.add(key='escape', func=win.close)

# config SSVEP stimuli
n_elements = 1                           # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

win_size = np.array(win.size)
pix_length, pix_width = win_size[0], win_size[1]  # unit: pix
win_length, win_width = 54.35, 30.2               # unit: centimeter
view_angle = 4                                    # unit: degree
distance = 50                                     # unit: centimeter
field = tan(view_angle*pi/180)*distance*2         # unit: centimeter
stim_length = field*pix_length/win_length         # unit: pix
stim_width = field*pix_width/win_width            # unit: pix

stim_size = np.array([stim_length, stim_width])
stim_sizes[:] = stim_size

rows, columns = 1, 1
distribution = np.array([columns, rows])

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0]/columns, win_size[1]/rows]) / 2
if (origin_pos[0]<(stim_length/2)) or (origin_pos[1]<(stim_width/2)):
    raise Exception('Too much blocks or too big the single square!')
else:
    for i in range(distribution[0]):      # loop in colums
        for j in range(distribution[1]):  # loop in rows
            stim_pos[i*distribution[1]+j] = origin_pos + [i,j]*origin_pos*2

# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:,1] *= -1     # invert the y-axis

# config time template (using frames)
# refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
refresh_rate = 120
display_time = 1   # show the character or any symbol
index_time = 1     # show the index triangle indicating which character you need to look at
lag_time = 1       # the interval from the disappearance of the index to the start of the flashing 
flash_time= 20        # total duration of stimulus flashing
blink_time = 1     # for subject to blink/rest their eyes

flash_frames = int(flash_time*refresh_rate)

# config colors
# freqs = [x+10 for x in range(12)]  # avoid 50Hz
freqs = [15]
# phases = [x*0.35 for x in range(12)]  # 0 & pi
phases = [0]
stim_colors = sinusoidal_sample(freqs, phases, refresh_rate, flash_frames, mode='zip')

ssvep_stimuli = []
for i in range(flash_frames):
    ssvep_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
    sizes=stim_sizes, xys=stim_pos, colors=stim_colors[i,...], opacities=stim_opacities,
    oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases, elementTex=np.ones((256,256)),
    elementMask=None, texRes=48))

# config text stimuli
symbols = ''.join([string.ascii_uppercase, '/12345'])
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos, color=(1.,1.,1.), colorSpace='rgb',
                                        units='pix', height=stim_width/2, bold=True, name=symbol, autoLog=False))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(win=win, text='\u2BC6', font='Arial', color=(1.,1.,0.), colorSpace='rgb',
                                units='pix', height=stim_width/2, bold=True, name=symbol, autoLog=False)

# config experiment parameters
ssvep_conditions = [{'id': i} for i in range(n_elements)]
ssvep_nrep = 1
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, name='ssvep', method='random')

paradigm_clock = core.Clock()
routine_timer = core.CountdownTimer()

t = 0
paradigm_clock.reset(0)

# start routine
# display speller
routine_timer.reset(0)
routine_timer.add(display_time)
while routine_timer.getTime() > 0:
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    win.flip()

# begin to flash
for trial in trials:
    # initialise index position
    id = int(trial['id'])
    index_stimuli.setPos(stim_pos[id] + np.array([0, stim_width/2]))

    # Phase 1: speller & index
    routine_timer.reset(0)
    routine_timer.add(index_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        index_stimuli.draw()
        win.flip()
    
    # Phase 2: eye shifting
    routine_timer.reset(0)
    routine_timer.add(lag_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()
 
    # Phase 3: code-VEP flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    for i in range(flash_frames):
        ssvep_stimuli[i].draw()
        win.flip()
    
    # Phase 4: blink
    routine_timer.reset(0)
    routine_timer.add(blink_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

# t = paradigm_clock.getTime()
win.close()
core.quit()


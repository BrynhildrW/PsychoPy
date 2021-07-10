# -*- coding: utf-8 -*-
"""
For Little Ding

Common SSVEP stimulus program (JFPM)
(1) Change the method to config stimulus position array
(2) Added the function of customizing multiple blinking conditions

Author: Brynhildr W
Refer: Swolf (NeuroScanPort)
update: 2021.1.19
"""

# load modules
import os
import string
import sys

import numpy as np
from numpy import (sin, pi)
from numpy import newaxis as NA
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)

from ex_base import (NeuroScanPort, sinusoidal_sample)

# %% global variables
global code_series, n_codes

event.clearEvents()
event.globalKeys.add(key='escape', func=core.quit)  # if 'esc' detected, quit the program

# port_available = True
# port_address = 0xDEFC  # hex value
# port = NeuroScanPort(port_address=port_address)

# check the official API for more information
win = visual.Window([1920, 1080], color=(-1,-1,-1), fullscr=True, monitor='testmonitor',
                    unit='pix', screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False

# config SSVEP stimuli
n_elements = 32                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_len = 180                         # side length of a single square | unit: pix
square_size = np.array([square_len, square_len])
stim_sizes[:] = square_size 

win_size = np.array(win.size)
rows, columns = 4, 8
distribution = np.array([columns, rows])

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0]/columns, win_size[1]/rows]) / 2
if (origin_pos[0]<(square_len/2)) or (origin_pos[1]<(square_len/2)):
    raise Exception('Too much blocks or too big the single square!')
else:
    for i in range(distribution[0]):      # loop in columns
        for j in range(distribution[1]):  # loop in rows
            stim_pos[i*distribution[1]+j] = origin_pos + [i,j]*origin_pos*2

# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:,1] *= -1     # invert the y-axis

# config time (using frames(precisely control) and real time(commonly control))
refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
# refresh_rate = 240
display_time = 0.5   # show the character or any symbol
index_time = 0.5     # show the index triangle indicating which character you need to look at
lag_time = 0.5       # the interval from the disappearance of the index to the start of the flashing 
flash_time= 0.5      # total duration of stimulus flashing
blink_time = 0.5     # for subject to blink/rest their eyes

flash_frames = int(flash_time*refresh_rate)

# config colors
freqs = [x+15 for x in range(16)]  # 15-30Hz, d=1Hz
phases = [0,1]  # 0 & pi
stim_colors = sinusoidal_sample(freqs, phases, refresh_rate, flash_frames, mode='combine')

ssvep_stimuli = []
for i in range(flash_frames):  # add your simuli for each frame
    # Sometimes it will show a red underline under 'visual', don’t worry, it’s just a weird bug
    ssvep_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
    sizes=stim_sizes, xys=stim_pos, colors=stim_colors[i,...], opacities=stim_opacities,
    oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases, elementTex=np.ones((128,128)),
    elementMask=None, texRes=48))

# config text simuli
symbols = ''.join([string.ascii_uppercase, '/12345'])  # if you want more stimulus, just add more symbols
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos, color=(1.,1.,1.), colorSpace='rgb',
                                        units='pix', height=square_len/2, bold=True, name=symbol, autoLog=False))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(win=win, text='\u2BC6', font='Arial', color=(1.,1.,0.), colorSpace='rgb',
                                units='pix', height=square_len/2, bold=True, name=symbol, autoLog=False)

# config experiment parameters
ssvep_conditions = [{'id': i} for i in range(n_elements)]
ssvep_nrep = 1
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, name='ssvep', method='random')

paradigm_clock = core.Clock()
routine_timer = core.CountdownTimer()

t = 0
paradigm_clock.reset(0)

# start routine
# display speller interface
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
    index_stimuli.setPos(stim_pos[id] + np.array([0, square_len/2]))

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
 
    # Phase 3: SSVEP flashing: precise control of time, use frames
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

t = paradigm_clock.getTime()
win.close()
core.quit()
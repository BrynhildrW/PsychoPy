# -*- coding: utf-8 -*-
"""
Common SSVEP stimulus program (JFPM)

Author: Brynhildr W
Refer: Swolf (NeuroScanPort)
update: 2021/6/29

"""

# %% load modules
import os
import string
import sys

import numpy as np
from numpy import (sin, pi)
from numpy import newaxis as NA
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)
from psychopy.visual import line
from ex_base import (NeuroScanPort, sinusoidal_sample)

# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# %% config window object
win = visual.Window(
    size=[2560, 1440],
    color=(-1, -1, -1),
    fullscr=True,
    monitor='testmonitor',
    screen=0,
    waitBlanking=False,
    allowGUI=True
)
win.mouseVisible = False
event.globalKeys.add(key='escape', func=win.close)

# config basic parameters of stimuli
n_elements = 320                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_len = 80                         # side length of a single square | unit: pix
square_size = np.array([square_len, square_len])
stim_sizes[:] = square_size

win_size = np.array(win.size)
rows, columns = 16, 20
distribution = np.array([columns, rows])

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0] / columns, win_size[1] / rows]) / 2
if (origin_pos[0] < (square_len / 2)) or (origin_pos[1] < (square_len / 2)):
    raise Exception('Too much blocks or too big the single square!')
for i in range(distribution[0]):  # loop in columns
    for j in range(distribution[1]):  # loop in rows
        stim_pos[i * distribution[1] + j] = origin_pos + [i, j] * origin_pos * 2

# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:, 1] *= -1     # invert the y-axis

# config ssvep
refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
# refresh_rate = 165
display_time = 1.  # keyboard display time before 1st stimulus
index_time = 0.5   # indicator display time
rest_time = 0.5    # rest-state time
blink_time = 0.5
flash_time = 1
flash_frames = int(flash_time*refresh_rate)

# config colors
freqs = [8 + 0.1 * x for x in range(320)]  # 15-30Hz, d=1Hz
phases = [0.35 * x for x in range(320)]  # 0 & pi
stim_colors = sinusoidal_sample(freqs, phases, refresh_rate, flash_frames, mode='zip')

# config flashing elements
# pic_path = r'C:\Users\Administrator\Desktop\圆环\0.png'
pic_path = r'C:\Users\Administrator\Desktop\square_with_cross.png'
# pic_path = r'C:\Users\Administrator\Desktop\圆环\120.png'
# pic_path = r'C:\Users\Administrator\Desktop\圆环\180.png'
# pic_path = r'C:\Users\Administrator\Desktop\圆环\240.png'
ssvep_stimuli = []
for i in range(flash_frames):  # add your simuli for each frame
    ssvep_stimuli.append(visual.ElementArrayStim(
        win=win,
        units='pix',
        nElements=n_elements,
        sizes=stim_sizes,
        xys=stim_pos,
        colors=stim_colors[i],
        opacities=stim_opacities,
        oris=stim_oris,
        sfs=stim_sfs,
        contrs=stim_contrs,
        phases=stim_phases,
        elementTex=pic_path,
        elementMask=None
    ))

# config text simuli
# symbols = ''.join([string.ascii_uppercase, '1234567890+-*/'])  # if you want more stimulus, just add more symbols
symbols = [str(i) for i in range(320)]
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(
        win=win,
        text=symbol,
        font='Arial',
        pos=pos,
        color=(1., 1., 1.),
        colorSpace='rgb',
        units='pix',
        height=square_len / 2,
        bold=True,
        name=symbol,
        autoLog=False
    ))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(
    win=win,
    text='\u2BC6',
    font='Arial',
    color=(1., 1., 0.),
    colorSpace='rgb',
    units='pix',
    height=square_len / 2,
    bold=True,
    name=symbol,
    autoLog=False
)

# config experiment parameters
ssvep_conditions = [{'id': i} for i in range(n_elements)]
ssvep_nrep = 5
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, name='ssvep', method='random')

# initialise experiment
# if port_available:
#     port.sendLabel(0)

routine_timer = core.CountdownTimer()

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

    # Phase 1: speller & index (eye shifting)
    routine_timer.reset(0)
    routine_timer.add(index_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        index_stimuli.draw()
        win.flip()

    # Phase 2: rest state
    routine_timer.reset(0)
    routine_timer.add(rest_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

    # Phase 3: SSVEP flashing
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

win.close()
core.quit()

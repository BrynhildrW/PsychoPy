# -*- coding: utf-8 -*-
"""
A code-VEP experiment script. (based on high frequency SSVEP)
    using update() function to compute color for each frame
Encountered some weird bugs. In some processes that obviously do not need to take up
    a lot of graphics card computing resources, there are still frame drops.
I'm trying to reduce the use of frames for precise timing operations.
For some processes that do not require precise timing, I'll use regular cpu timing to accomplish.

Author: Brynhildr W
update: 19/12/2020
"""
# %% load modules
import os
import string
import sys

import numpy as np
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)

from ex_base import (Experiment, NeuroScanPort, CodeVEP)

# %% Config code-VEP
# load in code infomation
global code_series, n_codes

event.clearEvents()
event.globalKeys.add(key='escape', func=core.quit)

# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

data_path = r'D:\SSVEP\program\code_1bits.mat'
code_data = io.loadmat(data_path)
code_series = code_data['VEPSeries_1bits']  # (n_codes, n_elements)
n_codes = code_series.shape[0]

win = visual.Window([1920, 1080], color=(-1,-1,-1), fullscr=True, monitor='testmonitor',
                    unit='pix', screen=0, allowGUI=True)
win.mouseVisible = False

# config code-VEP stimuli
n_elements = 32                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_colors = np.zeros((n_elements, 3))  # color array (RGB)
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_len = 160                         # side length of a single square | unit: pix
square_size = np.array([square_len, square_len])
stim_sizes[:] = square_size 

win_size = np.array(win.size)         # the window size: [1920, 1080]
rows, columns = 4, 8
distribution = np.array([columns, rows])

# padding: the distance from the outermost square to the edge of the screen
# interval: the distance between squares
# win_size[0](1920) - columns*square_len = (columns-1)*interval + 2*padding
# win_size[1](1080) - rows*square_len = (rows-1)*interval + 2*padding
interval = ((win_size[0]-win_size[1]) + square_len*(rows-columns)) / (columns-rows)
padding = (win_size[1] - rows*square_len - (rows-1)*interval) / 2
# or: padding = (win_size[0] - columns*square_len - (columns-1)*interval) / 2

# the coordinates of the center point of the first square
origin_pos = np.array([padding + 0.5*square_len, padding + 0.5*square_len])

# set the position coordinates of each square 
for i in range(distribution[0]):  # loop in columns
    for j in range(distribution[1]):  # loop in rows
        # (0,0)->(1,0)->(2,0)->(3,0) --> (0,1)->(1,1)->(2,1)->(3,1) --> ...
        stim_pos[i*distribution[1]+j] = origin_pos + (square_size+interval) * [i,j]
# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:,1] *= -1     # invert the y-axis

stim_colors[:] = np.array([0,0,0])  # initialise colors

cvep_stimuli = CodeVEP(win)
cvep_stimuli.setElements(texture='square', mask=None, nElements=n_elements, sizes=stim_sizes,
                        xys=stim_pos, oris=stim_oris, sfs=stim_sfs, phases=stim_phases,
                        colors=stim_colors, opacities=stim_opacities, contrs=stim_contrs)

#  config text simuli
symbols = ''.join([string.ascii_uppercase, '_12345'])
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos,
                                        color=(1.,1.,1.), units='pix', height=square_len/2,
                                        bold=True, name=symbol))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(win=win, text='\u2BC6', color='yellow', units='pix',
                                height=square_len/2, name=symbol)

# config experiment parameters
cvep_conditions = [{'id': i} for i in range(n_elements)]
cvep_nrep = 1
trials = data.TrialHandler(cvep_conditions, cvep_nrep, name='code-vep', method='random')
cvep_freq = 15


# config time template (using frames)
display_time, index_time, lag_time, blink_time = 1., 1., 1., 1.
flash_time, gap_time = 0.3, 0.1
refresh_rate = cvep_stimuli.refreshRate
code_frames = int(flash_time*refresh_rate)
gap_frames = int(gap_time*refresh_rate)

# initialise experiment
# if port_available:
#     port.sendLabel(0)

paradigm_clock = core.Clock()
routine_timer = core.CountdownTimer()

t = 0
frame_flash = np.zeros((32,5))
frame_gap = np.zeros_like(frame_flash)
paradigm_clock.reset(0)

# start routine
# display speller
routine_timer.reset(0)
routine_timer.add(display_time)
while routine_timer.getTime() > 0:
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    win.flip()

j = 0
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
 
    # Phase 3: code-VEP flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    for nc in range(n_codes):
        for i in range(code_frames):
            cvep_stimuli.update(code_series[nc,:], i, refresh_rate, cvep_freq)
            # for text_stimulus in text_stimuli:
            #     text_stimulus.draw()
            win.flip()
            frame_flash[j,nc] += 1
        # routine_timer.reset(0)
        # routine_timer.add(gap_time)
        # while routine_timer.getTime() > 0:
        for g in range(gap_frames):
            for text_stimulus in text_stimuli:
                text_stimulus.draw()
            win.flip()
            frame_gap[j,nc] += 1
    
    # Phase 4: blink
    routine_timer.reset(0)
    routine_timer.add(blink_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()
    j += 1

t = paradigm_clock.getTime()
win.close()
core.quit()


# %%

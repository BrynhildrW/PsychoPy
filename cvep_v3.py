# -*- coding: utf-8 -*-
"""
A code-VEP experiment script.
    (1) based on 60Hz SSVEP
    (2) 6 codes: 
        [1,0,-1,0,0,0]; [0,1,0,-1,0,0]; [0,0,1,0,-1,0]; [0,0,0,1,0,-1]; [-1,0,0,0,1,0]; [0,-1,0,0,0,1]
    (3) 1 for 0 phase, 0 for empty, -1 for pi phase

Author: Brynhildr W
update: 30/3/2021
"""

# %% load modules
import os
import string
import sys

import numpy as np
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)

from ex_base import (NeuroScanPort, sinusoidal_sample, square_sample)

# Config code-VEP
# event.clearEvents()

# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# load in code infomation
code_series = np.array(([1,0,1,0,-1,0,-1,0], [1,0,-1,0,1,0,-1,0],
                        [-1,0,1,0,-1,0,1,0], [1,0,1,0,1,0,-1,0],
                        [-1,0,1,0,1,0,-1,0], [-1,0,-1,0,-1,0,-1,0],
                        [1,0,-1,0,-1,0,1,0], [-1,0,1,0,-1,0,-1,0],
                        [-1,0,-1,0,1,0,1,0], [1,0,1,0,-1,0,1,0],
                        [-1,0,1,0,1,0,1,0], [-1,0,-1,0,-1,0,1,0],
                        [1,0,1,0,1,0,1,0], [-1,0,-1,0,1,0,-1,0],
                        [1,0,-1,0,-1,0,-1,0], [1,0,-1,0,1,0,1,0]))
data_path = r'D:\SSVEP\program\ncw_16codes.mat'
code_data = io.loadmat(data_path)
code_series = code_data['CodeSeries']  # (n_codes, n_elements)

win = visual.Window([1920, 1080], color=(-1,-1,-1), fullscr=False, monitor='testmonitor',
                    screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False

event.globalKeys.add(key='escape', func=win.close)

# config code-VEP stimuli
n_elements = 16                          # number of the objects
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
rows, columns = 4, 4
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

# config time template (using frames)
# refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=10, nWarmUpFrames=10))
refresh_rate = 60
display_time = 0.5  # keyboard display time before 1st stimulus
index_time = 0.5  # indicator display time
rest_time = 1  # rest-state time
code_time = 0.5  # time for each code
blink_time = 0.5  # blink time
code_frames = int(code_time*refresh_rate)

# config code colors
len_code = 8
color_b = np.zeros((6,3))
color_p, color_n = np.zeros((code_frames,3)), np.zeros((code_frames,3))
for i in range(int(code_frames)):
    if i%4==0 or i%4==2:  # 1st and 3rd frame (in each period): grey
        color_p[i,:] = np.array((0,0,0))
    elif i%4==1:  # 2nd frame (in each period): white for 0
        color_p[i,:] = np.array((1,1,1))
    elif i%4==3:  # 4th frame (in each period): black for 0
        color_p[i,:] = -1*np.array((1,1,1))
color_n = -1*color_p

stim_frames = 4*(code_frames+6)
color_frame = np.zeros((stim_frames, n_elements, 3))

n = 0
for code in code_series:
    temp = np.zeros((1,3))
    for i in range(len_code):
        if code[i] == 0:
            temp = np.concatenate((temp, color_b), axis=0)
        elif code[i] == 1:
            temp = np.concatenate((temp, color_p), axis=0)
        elif code[i] == -1:
            temp = np.concatenate((temp, color_n), axis=0)
    temp = np.delete(temp, 0, axis=0)
    color_frame[:,n,:] = temp
    n += 1
del n, i

cw_stimuli = []
for i in range(stim_frames):
    cw_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
    sizes=stim_sizes, xys=stim_pos, colors=color_frame[i,...], opacities=stim_opacities,
    oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases, elementTex=np.ones((128,128)),
    elementMask=None, texRes=48))

# config text simuli
symbols = ''.join([string.ascii_uppercase, '/12345'])
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
cw_conditions = [{'id': i} for i in range(n_elements)]
cw_nrep = 1
trials = data.TrialHandler(cw_conditions, cw_nrep, name='code-words', method='random')

# initialise experiment
# if port_available:
#     port.sendLabel(0)

routine_timer = core.CountdownTimer()

# start routine
# display speller
routine_timer.reset(0)
routine_timer.add(display_time)
while routine_timer.getTime() > 0:
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    win.flip()

# %% begin to flash
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
    routine_timer.add(rest_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()
 
    # Phase 3: code-VEP flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    # win.callOnFlip(port_serial.serialWrite, id+1)
    for i in range(stim_frames):
        cw_stimuli[i].draw()
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

# %%

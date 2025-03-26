# -*- coding: utf-8 -*-
"""
A Code Words experiment script based on medium-high frequency SSVEP.
1 freq --> 40 targets
4 freqs in total.

Author: Brynhildr W
Refer: Swolf (NeuroScanPort)
update: 2023/9/8

"""

# %% load modules
import os
import string
import sys

import numpy as np
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)

from ex_base import NeuroScanPort

# config port transformation
# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# load in code infomation
data_path = r'D:\BaiduSyncdisk\程序\PsychoPy\cw_40.pkl'
code_data = np.load(data_path, allow_pickle=True)
code_series = code_data['code words']  # (n_codes, n_elements)

# config window object
win = visual.Window([1920, 1080], color=(-1,-1,-1), fullscr=True, monitor='testmonitor',
                    screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False
event.globalKeys.add(key='escape', func=win.close)

# config basic parameters of stimuli
n_elements = 160                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_len = 90                         # side length of a single square | unit: pix
square_size = np.array([square_len*1.15, square_len])
stim_sizes[:] = square_size

win_size = np.array(win.size)
rows, columns = 5, 8
distribution = np.array([columns, rows])

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
unit_x_dist = win_size[0]/(columns*2)/2
unit_y_dist = win_size[1]/(rows*2)/2
if (unit_x_dist < (square_len / 2)) or (unit_y_dist < (square_len / 2)):
    raise Exception('Too much blocks or too big the single square!')

origin_pos = np.array([[unit_x_dist, unit_y_dist],
                       [unit_x_dist, unit_y_dist+win_size[1]/2],
                       [unit_x_dist+win_size[0]/2, unit_y_dist],
                       [unit_x_dist+win_size[0]/2, unit_y_dist+win_size[1]/2]])
                    #    [unit_x_dist, unit_y_dist+win_size[1]/2]])
                    #    [unit_x_dist+win_size[0]/3, unit_y_dist],
                    #    [unit_x_dist+win_size[0]/3, unit_y_dist+win_size[1]/2],
                    #    [unit_x_dist+win_size[0]/3*2, unit_y_dist],
                    #    [unit_x_dist+win_size[0]/3*2, unit_y_dist+win_size[1]/2],])
for npart in range(origin_pos.shape[0]):
    for ncol in range(distribution[0]):      # loop in columns
        for nrow in range(distribution[1]):  # loop in rows
            stim_pos[ncol*distribution[1] + nrow + 40*npart] = origin_pos[npart] + [ncol,nrow]*np.array([unit_x_dist, unit_y_dist])*2

# Note that those coordinates are not the real ones that need to be set on the screen.
# Actually the origin point of the coordinates axis is right on the center of the screen
# (not the upper left corner), and the larger the coordinate value, the farther the actual
# position is from the center. So next we need to transform the position matrix.
stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
stim_pos[:,1] *= -1     # invert the y-axis

# %% config code words
# refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
refresh_rate = 60
display_time = 1.  # keyboard display time before 1st stimulus
index_time = 0.5   # indicator display time
rest_time = 0.5     # rest-state time
blink_time = 0.5
code_time = 0.3    # time for each code
code_frames = int(code_time*refresh_rate)

# config colors
n_codes = 6
stim_frames = n_codes*code_frames
stim_colors = np.zeros((stim_frames, n_elements, 3))
time_point = np.linspace(0, (code_frames-1)/refresh_rate, code_frames)
freqs = [20,22,24,26]
phases = [0,0,0,0]

ne = 0
for freq,phase in zip(freqs, phases):
    sinw = np.sin(2*np.pi*freq*time_point + np.pi*phase)
    unit_code = np.vstack((sinw, sinw, sinw)).T

    for code in code_series:
        temp = np.zeros((1,3))
        for nc in range(n_codes):
            # temp = np.concatenate((temp, code[nc]*unit_code, blank_code), axis=0)
            temp = np.concatenate((temp, code[nc]*unit_code), axis=0)
        temp = np.delete(temp, 0, axis=0)
        stim_colors[:,ne,:] = temp
        ne += 1

# config flashing elements
pic_path = r'C:\Users\Administrator\Desktop\square_with_cross.png'
cw_stimuli = []
for sf in range(stim_frames):
    cw_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
    sizes=stim_sizes, xys=stim_pos, colors=stim_colors[sf,...], opacities=stim_opacities,
    oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases, elementTex=pic_path,
    elementMask=None, texRes=48))

# config text stimuli
# symbols = ''.join([string.ascii_uppercase, '/12345'])
symbols = [str(ne) for ne in range(n_elements)]
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos, color=(1.,1.,1.), colorSpace='rgb',
                                        units='pix', height=square_len/2.5, bold=True, name=symbol, autoLog=False))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(win=win, text='\u2BC6', font='Arial', color=(1.,1.,0.), colorSpace='rgb',
                                units='pix', height=square_len, bold=True, name=symbol, autoLog=False)

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
routine_timer.addTime(display_time)
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
    routine_timer.addTime(index_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        index_stimuli.draw()
        win.flip()
    
    # Phase 2: rest state
    routine_timer.reset(0)
    routine_timer.addTime(rest_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()
 
    # Phase 3: Code Words flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    for i in range(stim_frames):
        cw_stimuli[i].draw()
        win.flip()
    
    # Phase 4: blink
    routine_timer.reset(0)
    routine_timer.addTime(blink_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

win.close()
core.quit()
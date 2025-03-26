# -*- coding: utf-8 -*-
"""
An SSVEP experiment script。
freqs & phases:
    (1) 10.4-17.2Hz, df=0.4Hz, dp=0.7pi
    (2) 10.4-17.2Hz, df=0.4Hz, dp=0.35pi
    (3) 10.4-13.8Hz, df=0.2Hz, dp=0.35pi

Author: Brynhildr W
Refer: Swolf (NeuroScanPort)
update: 2024/3/9

"""

# %% load modules
import os
import string
import sys

import numpy as np
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)
from psychopy.visual import line

from ex_base import (NeuroScanPort, sinusoidal_sample)
import pickle

# %% config port transformation
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

# %% config basic parameters of stimuli
n_elements = 120                         # number of the objects
group_num = 6
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

square_size = np.array([74, 64])
stim_sizes[:] = square_size
# for ne in range(n_elements):
#     stim_sizes[ne, 0] += (5 - (ne) % 6) * 12

pkl_path = r'D:\position.pkl'
with open(pkl_path, 'rb') as f:
    stim_pos = pickle.load(f)

# config ssvep
refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
# refresh_rate = 165
display_time = 1.  # keyboard display time before 1st stimulus
index_time = 0.5   # indicator display time
rest_time = 0.5    # rest-state time
blink_time = 0.5
flash_time = 1.4
flash_frames = int(flash_time * refresh_rate)

# config colors
freqs = np.array([10.4 + 0.2 * x for x in range(20)])  # 15-30Hz, d=1Hz
phases = np.array([0.35 * x for x in range(20)])  # 0 & pi
stim_idx = [0, 16, 13, 7, 10, 6, 12, 9, 1, 17, 15, 8, 18, 2, 5, 11, 14, 4, 19, 3]
stim_freqs, stim_phases = [], []
for freq, phase in zip(freqs[stim_idx], phases[stim_idx]):
    stim_freqs += [freq for gn in range(group_num)]
    stim_phases += [phase for gn in range(group_num)]
stim_colors = sinusoidal_sample(
    freqs=stim_freqs,
    phases=stim_phases,
    refresh_rate=refresh_rate,
    flash_frames=flash_frames,
    mode='zip'
)

# %% config flashing elements
pic_path = r'C:\Users\Administrator\Desktop\square_with_cross.png'
ssvep_stimuli = []
for i in range(flash_frames):
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
        elementTex=np.ones((16, 16)),
        elementMask=None
    ))

# config text simuli
# symbols = ''.join([string.ascii_uppercase, '1234567890+-*/'])  # if you want more stimulus, just add more symbols
symbols = [i + str(j) for i in 'ABCDEFGHIJKLMNOPQRST' for j in range(6)]
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
        height=40,
        bold=True,
        name=symbol,
        autoLog=False
    ))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# buffer_stimuli = []
# for i in range(flash_frames):
#     buffer_stimuli.append(visual.BufferImageStim(
#         win=win,
#         stim=[
#             text_stimuli,
#             visual.ElementArrayStim(
#                 win=win,
#                 units='pix',
#                 nElements=n_elements,
#                 sizes=stim_sizes,
#                 xys=stim_pos,
#                 colors=stim_colors[i],
#                 opacities=stim_opacities,
#                 oris=stim_oris,
#                 sfs=stim_sfs,
#                 contrs=stim_contrs,
#                 phases=stim_phases,
#                 elementTex=np.ones((16, 16)),
#                 elementMask=None)
#         ]))

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(
    win=win,
    text='\u2BC6',
    font='Arial',
    color=(1., 1., 0.),
    colorSpace='rgb',
    units='pix',
    height=60,
    bold=True,
    name=symbol,
    autoLog=False
)

# %% config experiment parameters
ssvep_conditions = [{'id': i} for i in range(54)]  # 1-54
# ssvep_conditions = [{'id': i + 54} for i in range(54)]  # 55-108
ssvep_nrep = 1
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, name='ssvep', method='sequential')

# initialise experiment
# if port_available:
#     port.sendLabel(0)

routine_timer = core.CountdownTimer()

# start routine
# warm up
for i in range(flash_frames):
    ssvep_stimuli[i].draw()
    win.flip()

# display speller interface
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
    index_stimuli.setPos(stim_pos[id] + np.array([0, 30]))

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

    # Phase 3: SSVEP flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    for i in range(flash_frames):
        ssvep_stimuli[i].draw()
        # buffer_stimuli[i].draw()
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

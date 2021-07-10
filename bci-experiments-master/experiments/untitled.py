# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:49:52 2020

40 blocks chess-paradigm SSVEP platform

@author: Brynhildr
"""

#%% import basic modules
import os
import string

import numpy as np
import scipy.io as sio
from psychopy import (monitors, visual, core, data, event)

from bci_ex_base import Experiment
from bci_ex_amlifiers import NeuroScanPort
from bci_ex_paradigms import SSVEP

#%% config ssvep stimulis
def SSVEP(win, port_available=False):
    # when official start, set port_available to True
    # create neuroscan label interface
    if port_available:
        port_address = 0x378
        port = NeuroScanPort(port_address=port_address)
    
    # initialize SSVEP stimulus parameters
    n_stim = 40                      # number of the stimulus (5*8)
    sizes = np.zeros((n_stim, 2))    # the size array of objects (length*width)
    
    pos = np.zeros((n_stim, 2))      # the position array of stimulus (x,y)
    oris = np.zeros((n_stim,))       # the orientation array
    sfs = np.ones((n_stim,))         # the spatial frequency array
    phases = np.zeros((n_stim,))     # the phase array
    colors = np.zeros((n_stim, 3))   # the color array (RGB)
    opacities = np.ones((n_stim,))   # the opacity array (default 1)
    contrs = np.ones((n_stim,))      # the contrast array
    
    winSize = np.array(win.size)     # the window size
    size = np.array([140, 140])      # the size of one square (pixes)
    
    # config SSVEP stimulus parameters
    sizes[:] = size
    # [a,b] means the total number of blocks is a*b
    n_blocks = np.array([8, 5])  # a for columns, b for rows
    # vertical & horizontal distances between 2 neighboring stims
    pad_stim = 50
    # the actual size of the stimulation area
    area = size*n_blocks + (n_blocks - 1)*pad_stim
    # the area for paddings between the stimuli and the screen border
    pad_screen = (winSize - area)/2
    # the position for each block, taking paddings into consideration
    fst_pos = pad_screen + size/2
    for i in range(n_blocks[1]):
        for j in range(n_blocks[0]):
            pos[i*n_blocks[0]+j] = fst_pos + (size+pad_stim)*[j,i]
    pos -= winSize/2  # reset position for center
    pos[:,-1] *= -1
            
    # config SSVEP other parameters
    colors[:] = np.array([0, 0, 0])  # default background color is black
    
    ssvep_stim = SSVEP(win)
    ssvep_stim.setElements(texture='square', mask=None, nElements=n_stim,
        sizes=sizes, xys=pos, oris=oris, sfs=sfs, phases=phases, colors=colors,
        opacities=opacities, contrs=contrs)
    
    # config text stimuli
    texts = ''.join([string.ascii_uppercase, '_?,.0123456789'])
    text_stimulis = []
    for text, xy in zip(texts, pos):
        text_stimulis.append(visual.TextStim(win=win, text=text, pos=xy, color=(0.5,0.5,0.5),
            opacity=1.0, contrast=1.0, units='pix', height=size[1]/2,
            bold=False, wrapWidth=winSize[1], name=text))
    
    # config index stimuli
    index_stim = visual.TextStim(win=win, text='\u2bc6', pos=[0,0], color='yellow',
    contrast=1.0, units='pix', height=size[1]/3, wrapWidth=winSize[1], name=text)
    
    for text_stimuli in text_stimulis:
        text_stimuli.draw()
    win.flip()
    
    # config block stimuli
    block_stim = visual.TextStim(win=win, units='pix', text='', pos=[0,0],
        height=size[1]/2, wrapWidth=winSize[1], color=(0,0,0), contrast=1.0, bold=False)
    
    # declare experiment conditions
    ssvep_condi = [{'id':i} for i in range(n_stim)]
    nrep = 5  # number of repeatation for each group
    # handle trial sequencing and data storage
    trials = data.TrialHandler(trialList=ssvep_condi, nReps=nrep, method='random', name='SSVEP')
    ssvep_freqs = np.arange(8, 16, 0.2)
    ssvep_phases = np.arange(0, 0.35*40, 0.35)
    for i in range(ssvep_phases):
        while ssvep_phases[i] > 2:
            ssvep_phases[i] -= 2
    
    # seconds for each period
    index_time = 1.5
    lagging_time = 1
    flash_time = 1
    flash_frames = int(flash_time*ssvep_stim.refreshRate)
    
    # experiment initialization
    if port_available:
        port.sendLabel(0)
        
    paradigmClock = core.Clock()
    routineTimer = core.CountdownTimer()
    t = 0
    frameN = 0
    paradigmClock.reset(0)
    
    # experiment start
    for trial in trials:
        # initialize:
        ID = int(trial['id'])
        index_stim.setPos(pos[ID] + np.array([0,size[1]/2]))
        
        # phase1: index period
        routineTimer.reset(0)
        routineTimer.add(index_time)
        while routineTimer.getTime() > 0:
            for text_stimli in text_stimulis:
                text_stimli.draw()
            index_stim.draw()
            win.flip()
            frameN += 1
        
        # phase2: lag before the flash
        routineTimer.reset(0)
        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            for text_stimli in text_stimulis:
                text_stimli.draw()
            win.flip()
            frameN += 1
        
        # phase3: flashing
        for i in range(flash_frames):
            if port_available:
                win.callOnFlip(port.sendLabel, ID+1)
            ssvep_stim.update(i, ssvep_freqs, ssvep_phases)
            for text_stimli in text_stimulis:
                text_stimli.draw()
            win.flip()
            frameN += 1
            
        # phase4: lag before the next flash
        routineTimer.reset(0)
        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            pass
        
        # quit the experiment in need
        if event.getKeys('escape'):
            print('Force stopped by user')
            break

    t = paradigmClock.getTime()
    return frameN, t

#%% run experiment
if __name__ == '__main__':
    # clarify monitor information
    mon = monitors.Monitor(name='brynhildr', width=53.704, distance=45,
        gamma=None, verbose=False, autoLog=False)
    mon.setSizePix([1920, 1080])
    mon.save()
    
    # register self-defined paradigm and run experiment
    ex = Experiment()
    ex.setMonitor(mon)
    ex.registerParadigm('SSVEP', SSVEP, port_available=False)
    ex.run()
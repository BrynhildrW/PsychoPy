# -*- coding: utf-8 -*-
"""
A SSVEP experiment demo.

Author: Swolf
Version: 0.1.0
"""
import os
import string

import numpy as np
import scipy.io as sio
from psychopy import (monitors, visual, core, data, event)

from experiments.base import Experiment
from experiments.amlifiers import NeuroScanPort
from experiments.paradigms import SSVEP

def ssvep(win, port_available=False):
    # create neuroscan label interface
    if port_available:
        port_address = 0x378
        port = NeuroScanPort(port_address=port_address)

    # config ssvep stimulis
    nElements = 32
    sizes = np.zeros((nElements, 2))

    xys = np.zeros((nElements, 2))
    oris = np.zeros((nElements,))
    sfs = np.ones((nElements,))
    phases = np.zeros((nElements,))
    colors = np.zeros((nElements, 3))
    opacities = np.ones((nElements,))
    contrs = np.ones((nElements,))

    winSize = np.array(win.size)
    size = np.array([180, 180])
    shape = np.array([8, 4])
    paddings = (winSize - size*shape)/2
    sizes[:] = size

    origin_pos = paddings + size/2
    for i in range(0, shape[1]):
        for j in range(0, shape[0]):
            xys[i*shape[0]+j] = origin_pos+size*[j, i]
    xys = xys - winSize/2
    xys[:, 1] *= -1

    colors[:] = np.array([0, 0, 0])

    ssvep_stimuli = SSVEP(win)
    ssvep_stimuli.setElements(texture='square', mask=None,
        nElements=nElements, sizes=sizes, xys=xys, oris=oris, sfs=sfs, phases=phases,
        colors=colors, opacities=opacities, contrs=contrs)

    # config text stimulis
    texts = ''.join([string.ascii_uppercase, '_12345'])
    text_stimulis = []
    for text,xy in zip(texts, xys):
        text_stimulis.append(visual.TextStim(win=win, units='pix',
            text=text, pos=xy, height=size[1]/2, wrapWidth=winSize[1],
            color=(0.5, 0.5, 0.5), contrast=1, bold=False, name=text))
    
    # config index stimuli
    index_stimuli = visual.TextStim(win=win, units='pix',
        text='\u2BC6', pos=[0, 0], height=size[1]/2, wrapWidth=winSize[1],
        color='yellow', contrast=1, bold=False, name=text)

    for text_stimuli in text_stimulis:
        text_stimuli.draw()
    win.flip()

    # declare specific ssvep parameters
    ssvep_conditions = [{'id': i} for i in range(nElements)]
    ssvep_nrep = 1
    trials = data.TrialHandler(ssvep_conditions, ssvep_nrep,
        name='ssvep', method='random')
    ssvep_freqs = np.linspace(8, 15, nElements) # 8 -15 HZ
    ssvep_phases = np.linspace(0, 2, nElements) # different phase

    # seconds
    index_time = 1.5
    lagging_time = 0.25
    flash_time = 1
    flash_frames = int(flash_time*ssvep_stimuli.refreshRate)
    
    # initialze
    if port_available:
        port.sendLabel(0)

    paradigmClock = core.Clock()
    routineTimer = core.CountdownTimer()
    
    t = 0
    frameN = 0
    paradigmClock.reset(0)

    # start
    for trial in trials:
        # initialize
        id = int(trial['id'])
        index_stimuli.setPos(xys[id]+np.array([0, size[1]/2]))
        # phase1:
        routineTimer.reset(0)
        routineTimer.add(index_time)
        while routineTimer.getTime() > 0:
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            index_stimuli.draw()
            win.flip()
            frameN += 1
        # phase2:
        routineTimer.reset(0)
        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            win.flip()
            frameN += 1
        # phase2:
        for i in range(flash_frames):
            if port_available:
                win.callOnFlip(port.sendLabel, id + 1)
            ssvep_stimuli.update(i, ssvep_freqs, ssvep_phases)
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            win.flip()
            frameN += 1
        # phase4:
        routineTimer.reset(0)
        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            pass
        
        if event.getKeys('backspace'):
            print('Force stopped by user.')
            break

    t = paradigmClock.getTime()
    return frameN, t

if __name__ == '__main__':
    # clarify monitor information
    mon = monitors.Monitor(name='swolf_mon', width=53.704, distance=45,
    gamma=None, verbose=False, autoLog=False)
    mon.setSizePix([1920, 1080])
    mon.save()

    # register self-defined paradigm and run experiment
    ex = Experiment()
    ex.setMonitor(mon)
    ex.registerParadigm("SSVEP", ssvep, port_available=False)
    ex.run()
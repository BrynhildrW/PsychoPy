# -*- coding: utf-8 -*-
"""
A RVEP experiment based on paper[1].
Author: Swolf
Version: 0.1.0

[1]: Random Visual Evoked Potentials for Brain-Computer Interface Control
"""
import os
import string

import numpy as np
import scipy.io as sio
from psychopy import (monitors, visual, core, data, event)

from experiments.base import Experiment
from experiments.amlifiers import NeuroScanPort
from experiments.paradigms import CodeVEP

def generateRandomCodes(n_stimuli, n_sample):
    codes = np.random.randint(0, high=2, size=(n_stimuli, n_sample))
    return codes

def rvep(win, experiment_id=0, port_available=True):
    if port_available:
        port_address = 0x378
        port = NeuroScanPort(port_address=port_address)

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

    cvep_stimuli = CodeVEP(win)
    cvep_stimuli.setElements(texture='square', mask=None,
        nElements=nElements, sizes=sizes, xys=xys, oris=oris, sfs=sfs, phases=phases,
        colors=colors, opacities=opacities, contrs=contrs)

    texts = ''.join([string.ascii_uppercase, '_12345'])
    text_stimulis = []
    for text,xy in zip(texts, xys):
        text_stimulis.append(visual.TextStim(win=win, units='pix',
            text=text, pos=xy, height=size[1]/2, wrapWidth=winSize[1],
            color=(0.5, 0.5, 0.5), contrast=1, bold=False, name=text))
    
    index_stimuli = visual.TextStim(win=win, units='pix',
        text='\u2BC6', pos=[0, 0], height=size[1]/2, wrapWidth=winSize[1],
        color='yellow', contrast=1, bold=False, name=text)

    for text_stimuli in text_stimulis:
        text_stimuli.draw()
    win.flip()

    rvep_conditions = [{'id': i} for i in range(nElements)]
    rvep_nrep = 3
    trials = data.TrialHandler(rvep_conditions, rvep_nrep,
        name='rvep', method='random')
    code_length = 120 # 5s * 60Hz
    codes = generateRandomCodes(nElements, 300)
    
    index_time = 1.5
    lagging_time = 0.25
    
    if port_available:
        port.sendLabel(0)

    paradigmClock = core.Clock()
    routineTimer = core.CountdownTimer()
    
    t = 0
    frameN = 0
    paradigmClock.reset(0)
    ids = []
    for trial in trials:
        # initialize
        id = int(trial['id'])
        ids.append(id)
        index_stimuli.setPos(xys[id]+np.array([0, size[1]/2]))
        # phase1:
        routineTimer.reset(0)
        routineTimer.add(index_time)
        if port_available:
            port.sendLabel(int(id+3))
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
        for i in range(code_length):
            if port_available:
                win.callOnFlip(port.sendLabel, int(codes[id, i]+1))
            cvep_stimuli.update(codes[:, i])
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
    
    os.makedirs('codes', exist_ok=True)
    sio.savemat(os.path.join('codes', '{}.mat'.format(experiment_id)), {'codes': codes, 'ids': ids})
    t = paradigmClock.getTime()
    return frameN, t

if __name__ == '__main__':
    # clarify monitor information
    mon = monitors.Monitor(name='swolf_mon', width=53.704, distance=45, gamma=None, verbose=False, autoLog=False)
    mon.setSizePix([1920, 1080])
    mon.save()

    # register self-defined paradigm and run experiment
    ex = Experiment()
    ex.setMonitor(mon)
    ex.registerParadigm("Randowm Visual Evoked Potential", rvep, experiment_id=0, port_available=False)
    ex.run()
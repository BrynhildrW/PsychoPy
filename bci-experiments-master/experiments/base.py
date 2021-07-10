# -*- coding: utf-8 -*-
"""
An experiment framework based on psychopy, also includes lots of useful tools in BCI field.
Author: Swolf
Version: 1.0.0
"""
import time

import numpy as np
from psychopy import (monitors, core, visual, event, parallel)


class Experiment:
    """Experiment framework."""
    def __init__(self, startup=True, restore_gamma=False):
        """
        Experiment framework for human.
        :param startup: whether to show welcome startup interface;if not, jump to first paradigm directly
        :param restore_gamma: whether to restore gamma map after experiment, only use when you find your screen luminance changed after experiment.
        """
        # global keys to exit experiment
        # only works in pyglet backend,the num lock key should be released first
        self.startup = startup
        self.restore_gamma = restore_gamma
        self.monitor = None
        self.currentWin = None
        self.currentParadigm = None
        self.continueStartup = True
        self.continueParadigm = False
        # self.routineTimer = core.CountdownTimer()
        # self.paradigmClock = core.Clock()

        self.winSize = None
        self.winFrameRate = None
        self.winFramePeriod = None

        # register paradigms
        self.stimulis = {}
        self.paradigm_names = []
        self.paradigms = {}
        self.paradigms_paras = {}

    def initEvent(self):
        """Init operations before run."""
        event.clearEvents()
        event.globalKeys.add(key='escape', func=self.closeEvent)

    def closeEvent(self):
        """Close operation after run."""
        event.globalKeys.remove('escape')
        # restore gamma map
        if self.restore_gamma:
            origLUT = np.round(self.currentWin.backend._origGammaRamp * 65535.0).astype("uint16")
            origLUT = origLUT.byteswap() / 255.0
            self.currentWin.backend._origGammaRamp = origLUT
        core.quit()

    def setMonitor(self, mon):
        """
        Set current monitor info.
        :param mon: psychopy monitor object
        :return:
        """
        self.monitor = mon

    def registerParadigm(self, paradigm_name, paradigm_func, *args, **kwargs):
        """
        Register paradigm function.
        :param paradigm_name: the name of your paradigm
        :param paradigm_func: the function handle of your paradigm, the first argument must be psychopy window object
        :param args: other arguments you need for your paradigm, like a label port
        :param kwargs: other keyword arguments you need for your paradigm
        :return:
        """
        self.paradigm_names.append(paradigm_name)
        self.paradigms[paradigm_name] = paradigm_func
        self.paradigms_paras[paradigm_name] = (args, kwargs)

    def unregisterParadigm(self, paradigm_name):
        """
        Unregister your paradigm by name.
        :param paradigm_name: the name of your paradigm
        :return:
        """
        self.paradigm_names.remove(paradigm_name)
        self.paradigms[paradigm_name] = None
        del self.paradigms[paradigm_name]

    def updateStartup(self):
        '''Initialize startup window.'''
        height = 0.02*self.winSize[1]
        startPix = (len(self.paradigm_names)) * height
        stepPix = 2 * height
        posList = 2 * np.arange(startPix, -startPix - 1, -stepPix) / self.winSize[1]
        height = 2 * height / self.winSize[1]

        self.currentWin.setColor((0, 0, 0))

        if self.paradigm_names:
            if (self.currentParadigm not in self.paradigm_names or
                    not self.currentParadigm):
                self.currentParadigm = self.paradigm_names[0]

            for i, paradigm in enumerate(self.paradigm_names):
                try:
                    self.stimulis[paradigm].setPos((0, posList[i]))
                    self.stimulis[paradigm].setColor('#909090')
                except KeyError:
                    textStim = visual.TextStim(win=self.currentWin, text=paradigm, units='norm', contrast=1,
                                               pos=(0, posList[i]), height=height, name=paradigm, color='#909090', bold=False,
                                               wrapWidth=self.winSize[0])
                    self.stimulis[paradigm] = textStim

            self.stimulis[self.currentParadigm].setColor('#e6e6e6')

            for paradigm in self.paradigm_names:
                self.stimulis[paradigm].draw()

        try:
            self.stimulis['welcome'].setPos((0, posList[0] + 0.2))
        except KeyError:
            welcomeTips = 'Welcome,choose one of the following paradigms'
            textStim = visual.TextStim(win=self.currentWin, text=welcomeTips, units='norm', contrast=1,
                                       pos=(0, posList[0] + 0.2), height=2 * height, name=welcomeTips, color='#02243c', bold=True,
                                       wrapWidth=self.winSize[0])
            self.stimulis['welcome'] = textStim
        self.stimulis['welcome'].draw()

        self.currentWin.flip()

    def run(self):
        '''Run the main loop.'''
        self.initEvent()
        self.currentWin = visual.Window(monitor=self.monitor, screen=0, fullscr=True, color=(0, 0, 0), winType='pyglet')
        self.currentWin.setMouseVisible(False)
        self.winSize = np.array(self.currentWin.size)
        self.winFrameRate = self.currentWin.getActualFrameRate()
        self.winFramePeriod = self.currentWin.monitorFramePeriod
        try:
            if self.startup:
                while self.continueStartup:
                    self.updateStartup()
                    keys = event.waitKeys()
                    if self.paradigm_names:
                        if 'up' in keys:
                            index = self.paradigm_names.index(self.currentParadigm) - 1
                            self.currentParadigm = self.paradigm_names[index]
                        elif 'down' in keys:
                            index = self.paradigm_names.index(self.currentParadigm) + 1
                            self.currentParadigm = self.paradigm_names[index if index < len(self.paradigm_names) else 0]

                        if 'return' in keys:
                            self.continueStartup = False
                            args, kwargs = self.paradigms_paras[self.currentParadigm]
                            self.paradigms[self.currentParadigm](self.currentWin, *args, **kwargs)
                            self.continueStartup = True
                            self.updateStartup() # avoid background color change latency
            else:
                if self.paradigm_names:
                    args, kwargs = self.paradigms_paras[self.paradigm_names[0]]
                    self.paradigms[self.paradigm_names[0]](self.currentWin, *args, **kwargs)
        except Exception as e:
            print(e)
        finally:
            self.closeEvent()


class BaseDynamicStimuli:
    """Base stimuli of dynamic changing attributes."""

    _SHAPES = {
        'square': np.ones((128, 128)),
        'rectangle': np.ones((128, 64))
    }

    def __init__(self, win):
        self.win = win
        # ssvep requires exact timing infomation
        self.refreshPeriod = win.monitorFramePeriod
        self.refreshRate = np.floor(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
        self.winSize = np.array(win.size)
        # els objects
        self.els = {}

    def setElements(self,
            name='default', texture='square', mask=None,
            nElements=None, sizes=None, xys=None, oris=None, sfs=None, phases=None,
            colors=None, opacities=None, contrs=None):
        """
        Private wrapper for Psychopy ElementArrayStim.
        :param name: element array name
        :param texture: texture to render, could be string or numpy array. If string, right now support 'square' or 'rectangle'
        :param mask: mask to mask texture, should be numpy array or None.
        :param nElements: the number of elements to render.
        :param sizes: the size of elements, (nElements, 2) in pixel unit.
        :param xys: the position of elements, (nElements, 2) in pixel unit.
        :param oris: the orientation of elements, (nElements,)
        :param sfs: reserved
        :param phases: reserverd
        :param colors: the color of elements, (nElements, 3)
        :param opacities: the opacity of elements, (nElements,)
        :param contrs: the contrast of elements, (nElements,)
        :return:
        """
        if type(texture) is str:
            tex = self._SHAPES[texture]
            mask = None
        self.els[name] = visual.ElementArrayStim(self.win, units='pix',
            elementTex=tex, elementMask=mask, texRes=48,
            nElements=nElements, sizes=sizes, xys=xys, oris=oris, sfs=sfs, phases=phases,
            colors=colors, opacities=opacities, contrs=contrs)


class NeuroPort:
    def sendLabel(self, label):
        raise NotImplementedError
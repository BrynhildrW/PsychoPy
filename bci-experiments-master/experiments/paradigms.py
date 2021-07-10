# -*- coding:utf-8 -*-
import time

import numpy as np
from psychopy import (monitors, core, visual, event, parallel)

from .base import BaseDynamicStimuli


class CodeVEP(BaseDynamicStimuli):
    def computeColors(self, code):
        colors = np.tile(2*code-1, (3, 1)).T
        return colors

    def update(self, code, name='default'):
        els = self.els[name]
        els.colors = self.computeColors(code)
        els.draw()
        

class SSVEP(BaseDynamicStimuli):

    def computeColors(self, frameN, srate, freqs, phases, flash_func=np.sin):
        c_val = flash_func(2 * np.pi * freqs * frameN / srate + np.pi * phases)
        colors = np.tile(c_val, (3, 1)).T
        return colors

    def update(self, frameN, freqs, phases, flash_func=np.sin, name='default'):
        els = self.els[name]
        els.colors =  self.computeColors(frameN, self.refreshRate, freqs,
                            phases, flash_func=flash_func)
        els.draw()

class SSVEP_deprecated:
    """An SSVEP class to simplify stimuli design."""

    _SHAPES = {
        'square': np.ones((128, 128)),
        'rectangle': np.ones((128, 64))
    }

    _FLASH_FUNCS = {
        'sin': np.sin,
        'cos': np.cos
    }

    def __init__(self, win):
        """
        SSVEP for humans.
        :param win: Psychopy Window object.
        """
        self.win = win
        # ssvep requires exact timing infomation
        self.refreshPeriod = win.monitorFramePeriod
        self.refreshRate = np.floor(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
        self.winSize = np.array(win.size)
        # group parameters
        self.els = {}
        self.groups = {}

    def _setElements(self,
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

    def computeColorValue(self, frameN, paras, flash_func=np.cos, color_lim=np.array(((1, 1, 1), (-1, -1, -1)))):
        """
        Compute new color based on current frame.
        :param frameN: current frame number
        :param paras: frequency and phase parameters, (nGroups, nPieces, 2)
        :param flash_func: cos or sin function to update color
        :param color_lim: an numpy array indicate color range
        :return:
        """
        deta_c = (color_lim[0] - color_lim[1])/2
        new_paras = np.reshape(paras, (-1, 2), order='C')
        c_val = flash_func(2*np.pi*new_paras[:, 0]*self.refreshPeriod*frameN + np.pi*new_paras[:, 1])
        new_colors = (c_val[:, np.newaxis] + 1) * deta_c[np.newaxis, :]
        new_colors += color_lim[1]
        return new_colors

    def update(self, name='default', i_frame=0):
        """
        Update SSVEP object and draw based on current frame
        :param name: the name of elementArray to use
        :param i_frame: current frame number
        :return:
        """
        els = self.els[name]
        el_paras = self.groups[name]['el_paras']
        flash_func = self.groups[name]['flash_func']
        color_lim = self.groups[name]['color_lim']
        els.colors = self.computeColorValue(i_frame, paras=el_paras, flash_func=self._FLASH_FUNCS[flash_func], color_lim=color_lim)
        els.draw()

    def setGroup(self,
            name='default', texture='square', mask=None,
            el_locs=None, el_sizes=None, el_oris=None, el_sfs=None, el_phases=None,
            el_opacities=None, el_contrs=None,
            el_paras=None, flash_func='cos',color_lim=((1, 1, 1), (-1, -1, -1))):
        """
        Using group design pattern to create ssvep stimuli.
        :param name: the name of elementArray to use
        :param texture: texture to render, could be string or numpy array. If string, right now support 'square' or 'rectangle'
        :param mask: mask to mask texture, should be numpy array or None.
        :param el_locs: the position of all elements, (nGroups, nPieces, 2) in norm unit
        :param el_sizes: the size of all elements, (nGroups, nPieces 2) in norm unit
        :param el_oris: the orientation of all elements, (nGroups, nPieces, 1)
        :param el_sfs: reserved, pass ones array of shape (nGroups, nPieces, 1)
        :param el_phases: reserved, pass zeros array of shape (nGroups, nPieces, 1)
        :param el_opacities: the opacity of all elements, (nGroups, nPieces, 1)
        :param el_contrs: the contrast of all elements, (nGroups, nPieces, 1)
        :param el_paras: the update parameters(frequency, phase) of all elements, (nGroups, nPieces, 2)
        :param flash_func: the flash function to use in ssvep update, could be 'cos' or 'sin'
        :param color_lim: an numpy array indicate color range
        :return:
        """
        color_lim = np.array(color_lim)
        new_el_locs = np.reshape(el_locs, (-1, 2), order='C')*self.winSize/2
        new_el_sizes = np.reshape(el_sizes, (-1, 2), order='C')*self.winSize/2

        new_el_oris = np.squeeze(np.reshape(el_oris, (-1, 1), order='C'))
        new_el_sfs = np.squeeze(np.reshape(el_sfs, (-1, 1), order='C'))
        new_el_phases = np.squeeze(np.reshape(el_phases, (-1, 1), order='C'))
        new_el_opacities = np.squeeze(np.reshape(el_opacities, (-1, 1), order='C'))
        new_el_contrs = np.squeeze(np.reshape(el_contrs, (-1, 1), order='C'))

        new_el_colors = self.computeColorValue(0, el_paras,
            flash_func=self._FLASH_FUNCS[flash_func], color_lim=color_lim)

        nElements = new_el_locs.shape[0]

        self.groups[name] = {
            'nElements': nElements,
            'el_locs': el_locs,
            'el_paras': el_paras,
            'el_sizes': el_sizes,
            'el_oris': el_oris,
            'el_sfs': el_sfs,
            'el_phases': el_phases,
            'el_opacities': el_opacities,
            'el_contrs': el_contrs,
            'flash_func': flash_func,
            'color_lim': color_lim
        }

        self._setElements(name=name, texture=texture, mask=mask, nElements=nElements,
            sizes=new_el_sizes, xys=new_el_locs, oris=new_el_oris,
            colors=new_el_colors, opacities=new_el_opacities, contrs=new_el_contrs,
            sfs=new_el_sfs, phases=new_el_phases)

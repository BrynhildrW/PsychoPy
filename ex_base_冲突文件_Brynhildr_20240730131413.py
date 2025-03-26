# -*- coding: utf-8 -*-
"""
An experiment framework based on psychopy

Author: Swolf (Experiment, BaseDynamicStimuli, NeuroPort, Amplifier, NoConnectionError,
                Neuroscan, NeuroScanPort)
Heir: Brynhildr W (CodeVEP, inf_combine, sinusoidal_sample)

update: 2021.1.19
"""

import socket
import struct
import threading
import time
import queue
from collections import deque
from copy import deepcopy

import numpy as np
from numpy import (sin, pi)
from numpy import newaxis as NA
from numpy import ndarray
from typing import Optional, List, Tuple, Dict, Union, Callable, Any

import scipy.io as sio
import functools

from psychopy import (monitors, core, visual, event, parallel)


class Experiment:
    """
    Experiment framework.
    """
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
            if (self.currentParadigm not in self.paradigm_names or not self.currentParadigm):
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
                            self.updateStartup()  # avoid background color change latency
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
        # else objects
        self.els = {}
        pass

    def setElements(
            self,
            name='default',
            texture='square',
            mask=None,
            nElements=None,
            sizes=None,
            xys=None,
            oris=None,
            sfs=None,
            phases=None,
            colors=None,
            opacities=None,
            contrs=None):
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
        self.els[name] = visual.ElementArrayStim(
            self.win,
            units='pix',
            elementTex=tex,
            elementMask=mask,
            texRes=48,
            nElements=nElements,
            sizes=sizes,
            xys=xys,
            oris=oris,
            sfs=sfs,
            phases=phases,
            colors=colors,
            opacities=opacities,
            contrs=contrs
        )


class NeuroPort:
    def sendLabel(self, label):
        raise NotImplementedError
    pass


class CodeVEP(BaseDynamicStimuli):
    """
    0-1 encoding visual evoked potential based on short-time
        steady state visual evoked potantial(SSVEP).
    5 unit length, total 32 instructions: (each code last for 300ms)
        '0' stands for 60Hz 0 phase
        '1' stands for 60Hz pi phase
    """
    def compute_color(self, code, frameN, sfreq, freq):
        """

        Parameters
        ----------
        code : 0 or 1
            0 for 0 phase, 1 for pi phase (initial phase)
        frameN : int
            current refresh frames
        sfreq : int/float
            sampling frequency (screen refresh rate)
        freq : int/float
            stimulus frequency

        Returns
        -------
        colors : (1,3)
            RGB color codes
        """
        color_value = sin(2*pi*freq*frameN/sfreq + pi*code)
        colors = np.tile(color_value, (3, 1)).T
        return colors

    def update(self, code, frameN, sfreq, freq, name='default'):
        els = self.els[name]
        els.colors = self.compute_color(code, frameN, self.refreshRate, freq)
        els.draw()


class Amplifier:
    """
    An abstract class for eeg amplifiers.
    """
    class _Buff:
        """
        Inner buffer object to store buffer data.
        """
        def __init__(self, lim_buff):
            self.lim_buff = lim_buff
            self.buff = deque(maxlen=self.lim_buff)
            for _ in range(self.lim_buff):
                self.buff.append(None)
            self.buff_size = 0

        def buffering(self, data):
            for d in data:
                # value = self.buff.popleft()
                self.buff.append(d)
                if self.buff_size < self.lim_buff-1:
                    self.buff_size += 1

        def access_buffer(self):
            c_buffer = []
            for _ in range(self.lim_buff):
                d = self.buff.popleft()
                if d:
                    c_buffer.append(d)  # only access those are not None
                self.buff.append(d)
            return c_buffer

        def is_full(self):
            return self.buff_size == self.lim_buff-1

    class _MarkerRuler:
        def __init__(self, marker=None, latency=0):
            self.marker = marker
            self.latency = latency
            self.countdowns = {}

        def __call__(self, data):
            if self.marker:  # int marker
                if data[-1] == self.marker:
                    self.countdowns[str(self.marker) + str(len(self.countdowns))] = self.latency if self.latency > 0 else 0
                drop_keys = []
                for key in self.countdowns.keys():
                    if self.countdowns[key] == 0:
                        drop_keys.append(key)
                    self.countdowns[key] -= 1
                if drop_keys:
                    for key in drop_keys:
                        del self.countdowns[key]
                    return True
            else:  # no marker)
                if 'fixed' not in self.countdowns.keys():
                    self.countdowns['fixed'] = self.latency if self.latency > 0 else 0
                if self.countdowns['fixed'] == 0:
                    self.countdowns['fixed'] = self.latency if self.latency > 0 else 0
                    return True
                self.countdowns['fixed'] -= 1
            return False

    def __init__(self):
        self.registered_handlers = {}
        self._input_queues = {}
        self._output_queues = {}
        self._ts = {}
        self._t_data_pipeline = None
        self._t_save_pipeline = None
        self.register_lock = threading.Lock()
        self.srate = 1000

    def recv(self):
        r_data = []
        extras = ()
        return r_data, extras

    def send(self, message):
        pass

    def _data_pipeline(self):
        while True:
            try:
                package = self.recv()
            except:
                self.register_lock.acquire()
                for key in self._input_queues:
                    self._input_queues[key].put(None)
                self.register_lock.release()
                break
            # Should there be a lock?
            self.register_lock.acquire()
            for key in self._input_queues:
                self._input_queues[key].put(package)
            self.register_lock.release()

    def establish_data_pipeline(self):
        self._t_data_pipeline = threading.Thread(target=self._data_pipeline, daemon=True, name='data_pipeline')
        self._t_data_pipeline.start()

    def _save_handler(self, input_queue=None, file_name='default', info={}):
        save_data = None
        while True:
            package = input_queue.get()
            if package is None:
                break
            r_data, extras = package
            r_data = np.array(r_data).T
            save_data = np.append(save_data, r_data, axis=1) if save_data is not None else r_data
            input_queue.task_done()
        if save_data is not None and info is not None:
            print('saving data...')
            sio.savemat(file_name, {'data': save_data, 'info': info})
        # np.save(file_name, save_data)

    def use_save_hook(self, file_name='default', info=None):
        self.register_lock.acquire()
        input_queue = queue.Queue()
        self._input_queues['save'] = input_queue
        self.register_lock.release()
        kwargs = {'file_name': file_name, 'input_queue': input_queue, 'info': info}
        self._t_save_pipeline = threading.Thread(
            target=self._save_handler,
            args=(),
            kwargs=kwargs,
            daemon=True,
            name='save'
        )
        self._t_save_pipeline.start()

    def unuse_save_hook(self):
        self.register_lock.acquire()
        if 'save' in self._input_queues:
            self._input_queues['save'].put(None)
            del self._input_queues['save']
        self.register_lock.release()

    def register_hook(self, name=None, data_type=None, handler=None, args=(), kwargs={}):
        if name is None:
            name = handler.__name__

        if data_type is None:
            data_type = {
                'process_type': 'realtime',
                'label': None,
                'tlim': None
            }

        self.registered_handlers[name] = (handler, args, kwargs, data_type)

    def use_hooks(self, names=None):
        if names is None:
            names = self.registered_handlers.keys()
        self.register_lock.acquire()
        for name in names:
            handler, args, kwargs, data_type = self.registered_handlers[name]
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            self._input_queues[name] = input_queue
            self._output_queues[name] = output_queue

            wrapper_kwargs = {}
            wrapper_kwargs['args'] = args
            wrapper_kwargs['kwargs'] = kwargs
            wrapper_kwargs['input_queue'] = input_queue
            wrapper_kwargs['output_queue'] = output_queue
            wrapper_kwargs['handler'] = handler
            wrapper_kwargs['data_type'] = data_type

            self._ts[name] = threading.Thread(target=self._wrapper_handler, args=(), kwargs=wrapper_kwargs, name=name, daemon=True)
            self._ts[name].start()
        self.register_lock.release()

    def unuse_hooks(self, names=None):
        self.register_lock.acquire()
        if names is None:
            names = list(self._input_queues.keys())
        for name in names:
            self._input_queues[name].put(None)
            del self._input_queues[name]
            self._output_queues[name].put(None)
            del self._output_queues[name]
        self.register_lock.release()

    def unregister_hook(self, handler=None):
        del self.registered_handlers[handler.__name__]

    def _wrapper_handler(self, data_type=None, input_queue=None, output_queue=None, handler=None, args=(), kwargs={}):
        process_type = data_type['process_type']
        label = data_type['label']  # int
        tlim = data_type['tlim']  # tuple

        if process_type == 'fixed':
            lim_buff, latency = tlim
            lim_buff = np.int(lim_buff*self.srate)
            latency = np.int(latency*self.srate)
            buff = self._Buff(lim_buff)
            is_trigger = self._MarkerRuler(marker=label, latency=latency)
        elif process_type == 'label':
            f_p, r_p = tlim
            f_p = np.int(f_p*self.srate)
            r_p = np.int(r_p*self.srate)
            lim_buff = max(r_p, 0) - f_p
            latency = max(f_p, r_p, 0)
            r_p -= f_p
            f_p -= f_p
            buff = self._Buff(lim_buff)
            is_trigger = self._MarkerRuler(marker=label, latency=latency)
        waiting_queue = deque()

        while True:
            try:
                package = input_queue.get()
                if package is None:
                    output_queue.put(None)
                    break
                # initialization
                r_data, extras = package
                if process_type == 'realtime':
                    eeg_data = deepcopy(r_data)  # avoid reference trouble
                    waiting_queue.append(eeg_data)
                elif process_type == 'fixed':
                    for d in r_data:
                        if is_trigger(d) and buff.is_full():
                            eeg_data = deepcopy(buff.access_buffer())
                            waiting_queue.append(eeg_data)
                        else:
                            buff.buffering([d])
                else:
                    for d in r_data:
                        if is_trigger(d) and buff.is_full():
                            eeg_data = deepcopy(buff.access_buffer())
                            waiting_queue.append(eeg_data[f_p:r_p])
                        else:
                            buff.buffering([d])

                for _ in range(len(waiting_queue)):
                    eeg_data = waiting_queue.popleft()
                    output = handler(eeg_data, extras, *args, **kwargs)
                    if output is not None:
                        output_queue.put(output)

            except Exception as e:
                print(e)
                break

    def get_output_queue(self, name):
        return self._output_queues[name]

    def _detect_label(self, labels):
        idxs = np.nonzero(labels)[0]
        if idxs.size != 0:
            return idxs[0], labels[idxs[0]]
        else:
            return None, None


class NoConnectionError(Exception):
    """No connection established."""
    pass


class Neuroscan(Amplifier):

    _COMMANDS = {
        'stop_connect': b'CTRL\x00\x01\x00\x02\x00\x00\x00\x00',
        'start_acq': b'CTRL\x00\x02\x00\x01\x00\x00\x00\x00',
        'stop_acq': b'CTRL\x00\x02\x00\x02\x00\x00\x00\x00',
        'start_trans': b'CTRL\x00\x03\x00\x03\x00\x00\x00\x00',
        'stop_trans': b'CTRL\x00\x03\x00\x04\x00\x00\x00\x00',
        'show_ver': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'show_edf': b'CTRL\x00\x03\x00\x01\x00\x00\x00\x00',
        'start_imp': b'CTRL\x00\x02\x00\x03\x00\x00\x00\x00',
        'req_version': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'correct_dc': b'CTRL\x00\x02\x00\x05\x00\x00\x00\x00',
        'change_setup': b'CTRL\x00\x02\x00\x04\x00\x00\x00\x00'
    }

    def __init__(self, address=None, srate=1000, num_chans=68):
        Amplifier.__init__()
        self.address = address
        self.neuro_link = None
        self.num_chans = num_chans
        # the size of package in neuroscan data is srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate/25*(num_chans+1)*4
        self.timeout = 0.0625
        self.connected = False
        self.started = False

    def _unpack_header(self, b_header):
        ch_id = struct.unpack('>4s', b_header[:4])

        w_code = struct.unpack('>H', b_header[4:6])
        w_request = struct.unpack('>H', b_header[6:8])
        pkg_size = struct.unpack('>I', b_header[8:])

        return (ch_id[0].decode('utf-8'), w_code[0], w_request[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        fmt = '>' + str(num_chans + 1) + 'i'
        data = np.array(list(struct.iter_unpack(fmt, b_data)), dtype=np.float)
        data[:, :-1] *= 0.0298
        data[:, -1] -= 65280
        return data

    def _pack_header(self, ch_id='CTRL', w_code=None, w_request=None, pkg_size=0):
        ch_id = ch_id.encode('utf-8')
        w_code = struct.pack('>H', w_code)
        w_request = struct.pack('>H', w_request)
        pkg_size = struct.pack('>I', pkg_size)
        b_header = b''.join([ch_id, w_code, w_request, pkg_size])
        return b_header

    def _pack_data(self, data):
        data = np.array(data).astype('>i')
        return data.tobytes()

    def _recv(self, num_bytes):
        fragments = []
        done = False
        b_count = 0
        while not done:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout:
                return None

            if b_count == num_bytes or not chunk:
                done = True
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b''.join(fragments)
        return b_data

    def recv(self):
        extras = None
        header = None
        r_data = None
        b_header = self._recv(12)
        if b_header:
            header = self._unpack_header(b_header)

        if b_header and header[-1] != 0:
            b_data = self._recv(header[-1])
            if b_data:
                r_data = self._unpack_data(self.num_chans, b_data)
        return r_data, extras

    def send(self, message):
        self.neuro_link.sendall(message)

    def command(self, method):
        if method == 'connect':
            self.neuro_link.connect(self.address)
            self.connected = True
        elif method == 'start_acq':
            self.neuro_link.send(self._COMMANDS['start_acq'])
            r_data, extras = self.neuro_link.recv()
            r_data, extras = self.neuro_link.recv()
        elif method == 'stop_acq':
            self.neuro_link.send(self._COMMANDS['stop_acq'])
            time.sleep(self.timeout*2)
        elif method == 'start_transport':
            self.neuro_link.send(self._COMMANDS['start_trans'])
            self.establish_data_pipeline()
            self.started = True
        elif method == 'stop_transport':
            self.neuro_link.send(self._COMMANDS['stop_trans'])
            self.started = False
            self._cleanup()
        elif method == 'disconnect':
            self.neuro_link.send(self._COMMANDS['stop_connect'])
            if self.neuro_link:
                self.neuro_link.close()
                self.connected = False
                self.neuro_link = None

    def _cleanup(self):
        self.unuse_hooks()
        self.unuse_save_hook()


class NeuroScanPort(NeuroPort):
    def __init__(self, port_address=0xdefc):
        super().__init__()
        self.port_address = port_address
        self.port = parallel.ParallelPort(address=port_address)
        if self.port is None:
            raise ValueError("No avaliable parrallel port")
        self.sendLabel(label=0)

    def sendLabel(self, label=0):
        self.port.setData(label)
        time.sleep(0.001)
        self.port.setData(0)


def inf_combine(*args):
    """
    Infinite combination iteration
    Parameters
    ----------
    *args : list

    Returns
    -------
    result : list of all possible combinations
    """
    results = []
    total_times = functools.reduce(lambda x, y: x*y, [len(item) for item in args])
    n = 0
    while n < total_times:
        temp = n
        temp_results = []
        for i in range(len(args)):
            temp, cur = divmod(temp, len(args[i]))
            temp_results.append(args[i][cur])
        results.append(temp_results)
        n += 1

    return results


def sinusoidal_sample(
        freqs: List[Union[int, float]],
        phases: List[Union[int, float]],
        rrate: int,
        flash_frames: int) -> ndarray:
    """Sampled sinusoidal stimulation method.

    Args:
        freqs (List[Union[int, float]]): Stim frequencies.
            n_elements = len(freqs).
        phases (List[Union[int, float]]): Initial phases.
            n_elements = len(phases).
        rrate (int): Refresh rate.
        flash_frames (int).

    Returns:
        stim_colors (ndarray): (flash_frames, n_elements, 3)
            Color values in RGB space of each frame.
    """
    time_point = np.linspace(
        start=0,
        stop=(flash_frames - 1) / rrate,
        num=flash_frames
    )
    stim_colors = np.zeros((flash_frames, 3))[:, None, :]  # (flash_frames, 1, 3)
    for freq, phase in zip(freqs, phases):
        sinw = np.sin(2 * np.pi * freq * time_point + pi * phase)
        stim_colors = np.concatenate(
            (stim_colors, np.vstack((sinw, sinw, sinw)).T[:, None, :]),
            axis=1
        )  # add (flash_frames, 1, 3) on 2nd axis for each loop
    stim_colors = np.delete(stim_colors, 0, axis=1)
    return stim_colors


def square_sample(freqs, phases, refresh_rate, flash_frames, mode='combine'):
    """
    Sampled square stimulation method
    Parameters
    ----------
    freqs : list of int/float
        frequency sequence.
    phases : list of int/float
        phase sequence.
    refresh_rate : int
    flash_frames : int
    mode : str, optional
        design stimulus conditions by 'combine' or 'zip' method.
        The default is 'combine'.

    Returns
    -------
    stim_colors : ndarray, (flash_frames, n_elements, 3)
        color values in RGB space of each frame.
    """
    stim_colors = sinusoidal_sample(freqs, phases, refresh_rate, flash_frames, mode)
    for nf in range(stim_colors.shape[0]):
        for ns in range(stim_colors.shape[1]):
            if round(stim_colors[nf, ns, 0], 5) > 0:
                stim_colors[nf, ns, :] = 1
            elif round(stim_colors[nf, ns, 0], 5) == 0.0:
                stim_colors[nf, ns, :] = 0
            elif round(stim_colors[nf, ns, 0], 5) < 0:
                stim_colors[nf, ns, :] = -1
    return stim_colors

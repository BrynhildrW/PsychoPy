# -*- coding: utf-8 -*-

import socket, struct, threading, time, queue
from collections import deque
from copy import deepcopy

import numpy as np
import scipy.io as sio

from psychopy import parallel

from .base import NeuroPort


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
                value = self.buff.popleft()
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
            if self.marker: # int marker
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
            else: # no marker)
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
        kwargs={'file_name': file_name, 'input_queue': input_queue, 'info': info}
        self._t_save_pipeline = threading.Thread(target=self._save_handler, args=(), kwargs=kwargs, daemon=True, name='save')
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
        label = data_type['label'] # int
        tlim = data_type['tlim'] # tuple

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
                    eeg_data = deepcopy(r_data) # avoid reference trouble
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




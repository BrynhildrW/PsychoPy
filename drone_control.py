# -*- coding: utf-8 -*-
"""
A SSVEP experiment script for drone control interface.

Author: Brynhildr W
Refer: 
(1) Swolf (NeuroScanPort)
(2) Jie Mei (SerialPort)
update: 10/1/2020
"""
# %% load modules
import os
import string
import sys

import numpy as np
from numpy import (sin, tan, pi)
import scipy.io as io

from psychopy import (core, data, visual, monitors, event)

from ex_base import (NeuroScanPort, sinusoidal_sample)

# %% global variables
global code_series, n_codes

event.clearEvents()
event.globalKeys.add(key='escape', func=core.quit)

# %% Serial communication
import serial

def checkSerial_ports():
	ports = ['COM%s'%(i+1)for i in range(256)]
	result = []
	for port in ports:
		try:
			s = serial.Serial(port)
			s.close()
			result.append(port)
		except (OSError, serial.SerialException):
			pass
	return result

class serialPort():
	'''
	Trigger write class for Neuracle devices, details see the instruction of trigger box.
	To use, please figure out which "COM" port is your device's port. For example, if the port
	is COM3, then when initial the class, you need to set the value of portx to "COM3"
	'''

	def __init__(self,portx,bps=115200,timex=1):
		self.portx = portx
		self.bps = bps
		self.timex = timex
		self.mySerial = serial.Serial(port=self.portx,baudrate=self.bps,timeout=self.timex)
		#self.mySerial.bytesize = 8
		#self.mySerial.stopbits = 1
	
	def serialWrite(self,command):
		toSendString = '01E10100'
		b = hex(command)
		hexcommand = str(hex(command))
		if len(hexcommand) == 3:
			hexvalue = hexcommand[2]
			hexcommand = '0'+hexvalue.upper()
		else:
			#hexcommand = hexcommand[-1:-2].upper()
			a = hexcommand[2:].upper()
			hexcommand = a
		toSendString += hexcommand
		#self.mySerial.write(str(toSendString).encode("utf-8"))
		#HexArray = [toSendString[i:i+2] for i in range(0, len(toSendString),2)]
		toSendStringHex2Bytes = [int(toSendString[i:i+2],16) for i in range(0, len(toSendString),2)]
		self.mySerial.write(toSendStringHex2Bytes)
	
	def closePort(self):
		if self.mySerial.isOpen():
			self.mySerial.close()

# %% config transformation port
port_serial = serialPort('COM4')

# port_available = False
# port_address = 0xDEFC
# port = NeuroScanPort(port_address=port_address)

# config window
win = visual.Window([1920, 1080], color=(-1,-1,-1), fullscr=True, monitor='alienware',
                    unit='pix', screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False

# config SSVEP stimuli
n_elements = 12                          # number of the objects
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

win_size = np.array(win.size)
pix_length, pix_width = win_size[0], win_size[1]  # unit: pix
win_length, win_width = 54.35, 30.2               # unit: centimeter
view_angle = 2                                    # unit: degree
distance = 50                                     # unit: centimeter
field = tan(view_angle*pi/180)*distance*2         # unit: centimeter
stim_length = field*pix_length/win_length         # unit: pix
stim_width = field*pix_width/win_width            # unit: pix

stim_size = np.array([stim_length, stim_width])
stim_sizes[:] = stim_size

rows, columns = 3, 4
distribution = np.array([columns, rows])

# # divide the whole screen into rows*columns blocks, and pick the center of each block as the position
# origin_pos = np.array([win_size[0]/columns, win_size[1]/rows]) / 2
# if (origin_pos[0]<(stim_length/2)) or (origin_pos[1]<(stim_width/2)):
#     raise Exception('Too much blocks or too big the single square!')
# else:
#     for i in range(distribution[0]):      # loop in columns
#         for j in range(distribution[1]):  # loop in rows
#             stim_pos[i*distribution[1]+j] = origin_pos + [i,j]*origin_pos*2

# # Note that those coordinates are not the real ones that need to be set on the screen.
# # Actually the origin point of the coordinates axis is right on the center of the screen
# # (not the upper left corner), and the larger the coordinate value, the farther the actual
# # position is from the center. So next we need to transform the position matrix.
# stim_pos -= win_size/2  # from Quadrant 1 to Quadrant 3
# stim_pos[:,1] *= -1     # invert the y-axis

xs = [-566, -188, 188, 566]
# xs = [-702, -234, 234, 702]
ys = [322, 107, -107, -322]
# ys = [402, 134, -134, -402]
stim_pos = np.array([[xs[0],ys[0]], [xs[0],ys[1]], [xs[0],ys[2]], [xs[0],ys[-1]],
                     [xs[1],ys[0]], [xs[1],ys[-1]], [xs[2],ys[0]], [xs[2],ys[-1]],
                     [xs[-1],ys[0]], [xs[-1],ys[1]], [xs[-1],ys[2]], [xs[-1],ys[-1]]])

# config time template (using frames)
refresh_rate = np.ceil(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))
# refresh_rate = 120
display_time = 0.5   # show the character or any symbol
index_time = 0.3     # show the index triangle indicating which character you need to look at
lag_time = 0.2       # the interval from the disappearance of the index to the start of the flashing 
flash_time= 0.3      # total duration of stimulus flashing
blink_time = 0.3     # for subject to blink/rest their eyes

flash_frames = int(flash_time*refresh_rate)

# config colors
freqs = [x+48 for x in range(12)]
phases = [x*0.35 for x in range(12)]  # 0 & pi
stim_colors = sinusoidal_sample(freqs, phases, refresh_rate, flash_frames, mode='zip')

ssvep_stimuli = []
for i in range(flash_frames):
    ssvep_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
    sizes=stim_sizes, xys=stim_pos, colors=stim_colors[i,...], opacities=stim_opacities,
    oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases, elementTex=np.ones((256,256)),
    elementMask=None, texRes=48))

# config text stimuli
symbols = ''.join([string.ascii_uppercase, '/12345'])
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos, color=(1.,1.,1.), colorSpace='rgb',
                                        units='pix', height=stim_width/2, bold=True, name=symbol, autoLog=False))
for text_stimulus in text_stimuli:
    text_stimulus.draw()
win.flip()

# config index stimuli: downward triangle
index_stimuli = visual.TextStim(win=win, text='\u2BC6', font='Arial', color=(1.,1.,0.), colorSpace='rgb',
                                units='pix', height=stim_width/2, bold=True, name=symbol, autoLog=False)

# config experiment parameters
ssvep_conditions = [{'id': i} for i in range(n_elements)]
ssvep_nrep = 5
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, name='ssvep', method='random')

paradigm_clock = core.Clock()
routine_timer = core.CountdownTimer()

t = 0
paradigm_clock.reset(0)

# start routine
# display speller
routine_timer.reset(0)
routine_timer.add(display_time)
while routine_timer.getTime() > 0:
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    win.flip()

# begin to flash
for trial in trials:
    # initialise index position
    id = int(trial['id'])
    index_stimuli.setPos(stim_pos[id] + np.array([0, stim_width/2]))

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
    routine_timer.add(lag_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()
 
    # Phase 3: code-VEP flashing
    # if port_available:
    #     win.callOnFlip(port.sendLabel, id+1)
    win.callOnFlip(port_serial.serialWrite, id+1)
    for i in range(flash_frames):
        ssvep_stimuli[i].draw()
        win.flip()
    
    # Phase 4: blink
    routine_timer.reset(0)
    routine_timer.add(blink_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

t = paradigm_clock.getTime()
win.close()
core.quit()


import os
import string
import sys
import numpy as np
from numpy import (sin, pi)
from numpy import newaxis as NA
import scipy.io as io
from psychopy import (core, data, visual)
from psychopy.hardware import keyboard
from psychopy.visual import line, rect, shape

win = visual.Window([1000, 600], color=(-1,-1,-1), fullscr=False, monitor='testmonitor',screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False  #鼠标可视与否

n_elements = 1                           # number of the objects 项目数
stim_sizes = np.zeros((n_elements, 2))   # size array | unit: pix  数组大小|单位:pix
stim_pos = np.zeros((n_elements, 2))     # position array
stim_oris = np.zeros((n_elements,))      # orientation array (default 0)
stim_sfs = np.ones((n_elements,))        # spatial frequency array (default 1)
stim_phases = np.zeros((n_elements,))    # phase array
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)
stim_contrs = np.ones((n_elements,))     # contrast array (default 1)

#设置各种时间参数
refresh_rate = 60
display_time = 3   # keyboard display time before 1st stimulus 第一次刺激前的键盘显示时间
index_time = 1     # indicator display time 指示器显示时间
rest_time = 2      # rest-state time 休息时间
blink_time = 1.5
flash_time_line = 0.25   #mVEP刺激框250ms一个循环
flash_frames_line = int(flash_time_line*refresh_rate)

routine_timer = core.CountdownTimer()    # 倒计时

## 设置开头文字部分
#text_stimuli = []
#text_stimuli.append(visual.TextStim(win=win, text='请按照提示观察相应刺激框', font='Arial', pos=(0, 50), color=(0, 0, 0), colorSpace='rgb',
#                                        units='pix', height=50, bold=True, autoLog=False, alignText='center'))
#text_stimuli.append(visual.TextStim(win=win, text='    准备好后按空格键开始', font='Arial', pos=(0, -50), color=(0, 0, 0), colorSpace='rgb',
#                        units='pix', height=50, bold=True, autoLog=False, alignText='center'))
#routine_timer.reset(0)
#routine_timer.add(display_time)
#while routine_timer.getTime() > 0:
#    for text_stimulus in text_stimuli:
#        text_stimulus.draw()
#    win.flip()


# 设置实验参数
mvep_conditions = [{'id': i} for i in range(n_elements)]
mvep_nrep = 3    # 所有条件重复次数
trials = data.TrialHandler(mvep_conditions, mvep_nrep, name='mvep', method='random')

# 设置mvep刺激框
line_stimuli = []
x_line = np.linspace(30, 60, flash_frames_line)
for i in range(flash_frames_line):
    line_stimuli.append(visual.line.Line(win=win, start=(x_line[i], 10), units='pix', end=(x_line[i], -10),
                               lineWidth=1, lineColor=(1, 0, 0), ori=0, pos=(0, 0)))

# rect1_stimuli = []
# rect1_pos = (45, 0)
# rect1_width = 30
# rect1_height = 20
# rect1_stimuli.append(visual.rect.Rect(win=win, width=rect1_width, height=rect1_height, units='pix',
#                                       lineWidth=2, lineColor=(0,1,0), fillColor=None, ori=0, pos=(0, 0)))

shape_stimuli = []
shape_vertices = np.array([[15,10],[15,-10],[-15,10],[-15,-10]])
shape_stimuli.append(visual.shape.ShapeStim(win=win, units='pix', lineWidth=2, lineColor=(1,0,0), fillColor=None,
                                pos=(45, 0), vertices= [[-0.5, 0], [0, 0.5], [0.5, 0]]))

# 开始闪烁
for trial in trials:
    for i in range(flash_frames_line):
        line_stimuli[i].draw()
        # rect1_stimuli.draw()
        win.flip()

    routine_timer.reset(0)
    routine_timer.add(3)
    while routine_timer.getTime() > 0:
        print(shape_stimuli)
        shape_stimuli[0].draw()
        win.flip()
# print(1)

win.close()
core.quit()
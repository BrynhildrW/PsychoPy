from psychopy import (core, data, visual, monitors, event)
from psychopy.visual import line

win = visual.Window([1600, 900], color=(-1,-1,-1), fullscr=False, monitor='testmonitor',
                    screen=0, waitBlanking=False, allowGUI=True)
win.mouseVisible = False
event.globalKeys.add(key='escape', func=win.close)

flash_frames = 600  # 10 s
ring_stimuli = []
for fr in range(flash_frames):
    ring_stimuli.append(visual.circle.Circle(
        win=win, radius=0.5, edges='circle', 
    ))
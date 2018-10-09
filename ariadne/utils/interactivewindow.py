import cv2
import numpy as np
import sys


class InteractiveWindowKeys(object):
    KEY_ARROW_LEFT = 81
    KEY_ARROW_RIGHT = 83


class InteractiveWindow(object):
    EVENT_DRAWING = "EVENT_DRAWING"
    EVENT_CLEARING = "EVENT_CLEARING"
    EVENT_MOUSEDOWN = "EVENT_MOUSEDOWN"
    EVENT_MOUSEUP = "EVENT_MOUSEUP"
    EVENT_MOUSEMOVE = "EVENT_MOUSEMOVE"
    EVENT_QUIT = "EVENT_QUIT"
    EVENT_KEYDOWN = "EVENT_KEYDOWN"

    def __init__(self, name, autoexit=False):
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.mouseCallback)

        self.drawing = False
        self.clearing = False
        self.callbacks = []
        self.callbacks_map = {}
        self.autoexit = autoexit

    def mouseCallback(self, event, x, y, flags, param):
        point = np.array([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.fireEvent(InteractiveWindow.EVENT_MOUSEDOWN, (0, point))

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.fireEvent(InteractiveWindow.EVENT_DRAWING, (0, point))
            elif self.clearing == True:
                self.fireEvent(InteractiveWindow.EVENT_CLEARING, (1, point))
            else:
                self.fireEvent(InteractiveWindow.EVENT_MOUSEMOVE, (1, point))

        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fireEvent(InteractiveWindow.EVENT_MOUSEUP, (0, point))

        if event == cv2.EVENT_MBUTTONDOWN:
            self.clearing = True
            self.fireEvent(InteractiveWindow.EVENT_MOUSEDOWN, (2, point))

        if event == cv2.EVENT_MBUTTONUP:
            self.clearing = False
            self.fireEvent(InteractiveWindow.EVENT_MOUSEUP, (2, point))

    def showImg(self, img=np.zeros((500, 500)), time=0, disable_keys=False):
        #res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(self.name, img)
        if time >= 0:
            c = cv2.waitKey(time)
            if disable_keys:
                return
            # print "CH", c
            c = c & 255
            if c != 255:
                self.fireEvent(InteractiveWindow.EVENT_KEYDOWN, (chr(c), c))
                if c == 113:
                    self.fireEvent(InteractiveWindow.EVENT_QUIT, None)
                    if self.autoexit:
                        sys.exit(0)
            return c
        return -1

    def fireEvent(self, evt, data):
        for c in self.callbacks:
            c(evt, data)
        for event, cbs in self.callbacks_map.items():
            if event == evt:
                for cb in cbs:
                    cb(data)

    def registerCallback(self, callback, event=None):
        if event is None:
            self.callbacks.append(callback)
        else:
            if event not in self.callbacks_map:
                self.callbacks_map[event] = []
            self.callbacks_map[event].append(callback)

    def removeCallback(self, callback, event=None):
        if event is None:
            self.callbacks.remove(callback)
        else:
            if event not in self.callbacks_map:
                self.callbacks_map[event].remove(callback)

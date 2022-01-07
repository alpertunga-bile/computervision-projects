import cv2
import numpy as np
import wx

from wxGUI import BaseLayout
from tools import convert2pencilSketch, spline2LookupTable, applyRGBFilters, applyHUEFilter, cartoonize

INCREASE_LOOKUP_TABLE = spline2LookupTable([0, 64, 128, 192, 256],
                                               [0, 70, 140, 210, 256])
DECREASE_LOOKUP_TABLE = spline2LookupTable([0, 64, 128, 192, 256],
                                               [0, 30, 80, 120, 192])

class FilterLayout(BaseLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def augment_layout(self):
        panel = wx.Panel(self, -1)

        self.mode_warm = wx.RadioButton(panel, -1, "Warming Filter", (10, 10), style=wx.RB_GROUP)
        self.mode_cool = wx.RadioButton(panel, -1, "Cooling Filter", (10, 10))
        self.mode_sketch = wx.RadioButton(panel, -1, "Pencil Sketch", (10, 10))
        self.mode_cartoon = wx.RadioButton(panel, -1, "Cartoon", (10, 10))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.mode_warm, 1)
        hbox.Add(self.mode_cool, 1)
        hbox.Add(self.mode_sketch, 1)
        hbox.Add(self.mode_cartoon, 1)
        
        panel.SetSizer(hbox)

        self.panels_vertical.Add(panel, flag = wx.EXPAND | wx.BOTTOM | wx.TOP, border = 1)

    @staticmethod
    def renderWarm(reelImg: np.ndarray) -> np.ndarray:
        interImg = applyRGBFilters(reelImg,
                                    redFilter=INCREASE_LOOKUP_TABLE,
                                    blueFilter=DECREASE_LOOKUP_TABLE)

        return applyHUEFilter(interImg, INCREASE_LOOKUP_TABLE)

    @staticmethod
    def renderCool(reelImg: np.ndarray) -> np.ndarray:
        interImg = applyRGBFilters(reelImg,
                                    redFilter=DECREASE_LOOKUP_TABLE,
                                    blueFilter=INCREASE_LOOKUP_TABLE)

        return applyHUEFilter(interImg, DECREASE_LOOKUP_TABLE)

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.mode_warm.GetValue():
            return self.renderWarm(frame_rgb)
        elif self.mode_cool.GetValue():
            return self.renderCool(frame_rgb)
        elif self.mode_sketch.GetValue():
            return convert2pencilSketch(frame_rgb)
        elif self.mode_cartoon.GetValue():
            return cartoonize(frame_rgb)
        else:
            raise NotImplementedError()

def main():
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    layout = FilterLayout(capture, title = "FilterGUI")
    layout.Center()
    layout.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
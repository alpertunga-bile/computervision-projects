import cv2
import numpy as np
import wx

class BaseLayout(wx.Frame):
    def __init__(self, 
                capture: cv2.VideoCapture,
                title: str = None,
                parent = None,
                window_id: int = -1,
                fps: int = 10):
        self.capture = capture
        _, frame = self._acquire_frame()
        self.imgHeight, self.imgWidth = frame.shape[:2]

        super().__init__(parent, window_id, title, size=(self.imgWidth, self.imgHeight + 20))

        self.fps = fps
        self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./self.fps)

        self.Bind(wx.EVT_TIMER, self._on_next_frame)

        self.video_pnl = wx.Panel(self, size=(self.imgWidth, self.imgHeight))
        self.video_pnl.SetBackgroundColour(wx.BLACK)
        self.video_pnl.Bind(wx.EVT_PAINT, self._on_paint)

        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.video_pnl, 1, flag = wx.EXPAND | wx.TOP, border = 1)

        self.augment_layout()

        self.SetMinSize((self.imgWidth, self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _on_next_frame(self, event):
        success, frame = self._acquire_frame()

        if success:
            frame = self.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.bmp.CopyFromBuffer(frame)
            self.Refresh(eraseBackground = False)

    def _on_paint(self, event):
        wx.BufferedPaintDC(self.video_pnl).DrawBitmap(self.bmp, 0, 0)

    def _acquire_frame(self) -> (bool, np.ndarray):
        return self.capture.read()

    def augment_layout(self):
        raise NotImplementedError
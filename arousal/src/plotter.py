#import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

timespan = 60  # secondi sul grafico

class Plotter:
    def __init__(self):
        self.canvas = GraphicsLayoutWidget()
        self.plot = self.canvas.addPlot(title='Arousal')
        self.curve = self.plot.plot(pen='y')
        self.data = []

    def update(self, value):
        self.data.append(value)
        if len(self.data) > timespan:
            self.data.pop(0)
        self.curve.setData(self.data)
from .isogrid import IsoGrid

class MIST(IsoGrid):
    EEP_TICKS = [1, 202, 353, 454, 605, 631, 707, 808, 1409, 1710]
    EEP_LABELS = ['pre-MS', 'ZAMS', 'IAMS', 'TAMS', 'RGBTip', 'ZACHeB', 'TACHeB', 'TPAGB/CBurn', 'PostAGB', 'WDCS']

    def __init__(self):
        super(MIST, self).__init__()
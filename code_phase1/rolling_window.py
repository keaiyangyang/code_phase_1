"""
Rolling Time Window Mechanism
"""

import numpy as np


class RollingWindowProcessor:
    """Rolling Time Window Processor"""

    def __init__(self, n_windows=3):
        self.n_windows = n_windows

    def create_windows(self, time_series):
        """Create rolling time windows"""
        n = len(time_series)
        n1 = int(n * 0.5)
        remaining = n - n1
        n2 = int(remaining / 3)
        n3 = int(remaining / 3)
        n4 = remaining - n2 - n3

        segments = [
            time_series[:n1],
            time_series[n1:n1 + n2],
            time_series[n1 + n2:n1 + n2 + n3],
            time_series[n1 + n2 + n3:]
        ]

        windows = []

        # Window 1: Train on Segment 1, Test on Segment 2
        windows.append({
            'window_id': 1,
            'train_data': segments[0],
            'test_data': segments[1]
        })

        # Window 2: Train on Segments 1+2, Test on Segment 3
        windows.append({
            'window_id': 2,
            'train_data': np.concatenate([segments[0], segments[1]]),
            'test_data': segments[2]
        })

        # Window 3: Train on Segments 1+2+3, Test on Segment 4
        windows.append({
            'window_id': 3,
            'train_data': np.concatenate([segments[0], segments[1], segments[2]]),
            'test_data': segments[3]
        })

        return windows
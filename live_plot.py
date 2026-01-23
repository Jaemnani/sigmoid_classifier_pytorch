"""
Authors : inzapp

Github url : https://github.com/inzapp/sigmoid-classifier

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from matplotlib import pyplot as plt

class LivePlot:
    def __init__(self, iterations, interval=20, mean=10, y_min=0.0, y_max=0.2, legends=['loss'], output_file=None):
        plt.style.use(['dark_background'])
        self.fig, self.ax = plt.subplots()
        pad = ((y_max - y_min) * 0.05)
        self.interval = interval
        self.mean = mean
        self.y_min = y_min - pad
        self.y_max = y_max + pad
        self.output_file = output_file
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlim(0, iterations)
        
        self.legends = legends if isinstance(legends, list) else [legends]
        self.data = {}
        self.lines = {}
        self.recent_values = {}
        
        for legend in self.legends:
            self.data[legend] = np.full(iterations, np.nan, dtype=np.float32)
            self.lines[legend], = self.ax.plot(np.full(iterations, np.nan), label=legend)
            self.recent_values[legend] = []

        self.interval_count = 0
        self.index = 0
        plt.xlabel('Iteration')
        plt.legend()
        plt.tight_layout(pad=0.5)

    def update(self, **kwargs):
        # Allow single argument update for backward compatibility or simple usage if only 1 legend
        if not kwargs and len(self.legends) == 1:
            # If called like update(val) but getting it might be tricky without *args.
            # Let's rely on kwargs.
            pass

        for legend in self.legends:
            if legend in kwargs:
                val = kwargs[legend]
                
                if not np.isnan(val):
                    # Dynamic Y-axis adjustment
                    if val > self.y_max:
                        self.y_max = val * 1.1 + 0.01
                        self.ax.set_ylim(self.y_min, self.y_max)
                    
                    if val < self.y_min:
                        # Prevent going below 0 for loss/accuracy
                        if val < 0.0:
                            val = 0.0 # Just clamp value or allow negative? Plan said 'no negative axis'. 
                                      # If value is truly negative, we might need to show it, but typically loss >= 0.
                                      # Let's update min but lower bound at 0.
                            pass
                        
                        self.y_min = val if val >= 0.0 else 0.0
                        self.ax.set_ylim(self.y_min, self.y_max)

                self.data[legend][self.index] = self.get_recent_avg_value(legend, val)
        
        self.index += 1
        self.interval_count += 1
        if self.interval_count == self.interval:
            self.interval_count = 0
            for legend in self.legends:
                self.lines[legend].set_ydata(self.data[legend])
            
            if self.output_file is None:
                plt.pause(1e-9)
            else:
                plt.savefig(self.output_file)

    def get_recent_avg_value(self, legend, val):
        if len(self.recent_values[legend]) > self.mean:
            self.recent_values[legend].pop(0)
        self.recent_values[legend].append(val)
        return np.mean(self.recent_values[legend])


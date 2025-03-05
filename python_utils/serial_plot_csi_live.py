import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if using PyQt

import math
import numpy as np
import collections
from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line
from inference import inference
# Set subcarrier to plot
subcarrier = 44

# Wait Timers. Change these values to increase or decrease the rate of `print_stats` and `render_plot`.
print_stats_wait_timer = WaitTimer(1.0)
render_plot_wait_timer = WaitTimer(0.2)

# Deque definition
perm_amp = collections.deque(maxlen=100)
perm_phase = collections.deque(maxlen=100)

# Variables to store CSI statistics
packet_count = 0
total_packet_counts = 0

# Create figure for plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
fig.canvas.draw()
plt.show(block=False)


def carrier_plot(amp):
    plt.clf()
    df = np.asarray(amp, dtype=np.int32)
    # Can be changed to df[x] to plot sub-carrier x only (set color='r' also)
    plt.plot(range(100 - len(amp), 100), df[:, subcarrier], color='r')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.xlim(0, 100)
    plt.title(f"Amplitude plot of Subcarrier {subcarrier}")
    # TODO use blit instead of flush_events for more fastness
    # to flush the GUI events
    fig.canvas.flush_events()
    #plt.show()
    plt.draw()
    plt.pause(0.001)


def process(res):
    # Parser
    all_data = res.split(',')
    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    csi_data = csi_data[12:-10]
    del csi_data[52:54]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    phases = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            phase_calc = math.atan2(imaginary[j], real[j])
            amplitudes.append(amplitude_calc)
            phases.append(phase_calc)

        perm_phase.append(phases)
        perm_amp.append(amplitudes)
        # print(perm_amp)

print_until_first_csi_line()

import time
from collections import deque

packet_count = 0
total_packet_counts = 0
last_process_time = time.time()
process_interval = 1 /4# Allow only 4 rows per second (every 0.25s)
# buffer = deque(maxlen=20)  # Buffer to store the last 20 lines (5 seconds worth)

while True:
    line = readline()
    # print("line",line)
    if "CSI_DATA" in line:
        current_time = time.time()

        # Process only if at least 0.25s has passed since the last processed row
        if current_time - last_process_time >= process_interval:
            process(line)
            # print("hshsh",perm_amp)
            # print(f"Before Append: Buffer length = {len(buffer)}")
            # # buffer.append(list(perm_amp))
            # print(f"After Append: Buffer length = {len(buffer)}")

            packet_count += 1
            total_packet_counts += 1
            last_process_time = current_time  # Reset timer
        # If 20 lines have been collected (5 seconds worth of data)
        if len(perm_amp) == 20:
            # print("buffer",buffer)
            inference(list(perm_amp))  # Pass a copy of the buffer to inference function
            perm_amp.clear()
            
            # if result is not None:
            #     print("Inference result:", result)

            # buffer.clear()   # Optionally clear the buffer after inference
        if print_stats_wait_timer.check():
            print_stats_wait_timer.update()
            print("Packet Count:", packet_count, "per second.", "Total Count:", total_packet_counts)
            packet_count = 0

        if render_plot_wait_timer.check() and len(perm_amp) > 2:
            render_plot_wait_timer.update()
            # print(perm_amp)
            # carrier_plot(perm_amp)

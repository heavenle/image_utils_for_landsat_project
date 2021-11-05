import os

import numpy
import glob
import matplotlib.pyplot as plt
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img_name = os.listdir(r'D:\project\crop\GF1_data_512_25\images')
    mak_name = os.listdir(r'D:\project\crop\GF1_data_512_25\masks')
    # show_name = os.listdir(r'D:\project\crop\GF1_data_320_25\show_mask')
    count = 0
    for name in img_name:
        if name not in mak_name:
            count += 1
            print(os.path.join('D:\\project\\crop\\GF1_data_512_25\\images', name))
            os.remove(os.path.join('D:\\project\\crop\\GF1_data_512_25\\images', name))

    print(count)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

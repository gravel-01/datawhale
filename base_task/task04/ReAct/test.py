import torch
import matplotlib.pyplot as plt
import numpy as np


def frequency_demo():
    max_len = 500  # 假设句子长 500 个词
    d_model = 128
    positions = np.arange(max_len)

    # 维度 0 (i=0): 频率最高
    dim_0_freq = 1.0 / (10000 ** (0 / d_model))
    # 维度 100 (i=50): 频率较低
    dim_100_freq = 1.0 / (10000 ** (100 / d_model))

    pe_0 = np.sin(positions * dim_0_freq)
    pe_100 = np.sin(positions * dim_100_freq)

    plt.figure(figsize=(10, 4))
    plt.plot(positions, pe_0, label="Dimension 0 (High Freq)", color="blue")
    plt.plot(positions, pe_100, label="Dimension 100 (Low Freq)", color="red")
    plt.title("Frequency difference within a single word vector")
    plt.xlabel("Position in sentence")
    plt.ylabel("Value added to embedding")
    plt.legend()
    plt.show()


frequency_demo()

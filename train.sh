#!/bin/bash

uv run --with torch --with torchvision --with soundfile --with numpy --with scipy --with pillow --with pypng --with phase-spectrogram --reinstall-package torch --reinstall-package phase-spectrogram --with pygoruut --with tqdm python3 train.py

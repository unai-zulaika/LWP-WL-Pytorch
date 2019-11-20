#!/bin/bash
python3 -u lwp-wl/main.py --no-cnn >> remake2.out 
python3 -u lwp-wl/main.py --random >> remake3.out
python3 -u lwp-wl/main.py >> remake4.out
# pyuzawamg

FEniCS and cbc.block based library containing components to build a multigrid for saddle-point problems based on uzawa smoothers.

## Install

Install with `python3 -m pip install .` or `python3 -m pip install -e .` for dev mode.

## Run

Run for instance a demo with
```
python3 -m pyuzawamg.demos.stokes --degree-velocity 1 --stabilization --num-levels 6 --w-cycles 2 --smoothing-steps 2
```
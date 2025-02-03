# DLIMP

Dataloading is my passion.

## Installation
Requires Python >= 3.8.
```bash
git clone https://github.com/kvablack/dlimp
cd dlimp
pip install -e .
```

## Usage
Core usage is through the `DLataset` class, defined in `dlimp/dlimp/dataset.py`. It is a thin wrapper around `tf.data.Dataset` designed for working with datasets of trajectories; it has two creation methods, `from_tfrecords` and `from_rlds`. This library additionally provides a suite of *frame-level* and *trajectory-level* transforms designed to be used with `DLataset.frame_map` and `DLataset.traj_map`, respectively.

Scripts for converting various datasets to the dlimp TFRecord format (compatible with `DLataset.from_tfrecords`) can be found in `legacy_converters/`. This should be considered deprecated in favor of the RLDS format, converters for which can be found in `rlds_converters/` and will be expanded from now on.

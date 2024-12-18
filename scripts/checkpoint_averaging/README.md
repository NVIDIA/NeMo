Checkpoint Averaging Script
============================

Overview
--------
The checkpoint averaging script is used to compute the average of multiple distributed checkpoints in a framework. This can be useful for improving model performance by combining multiple training states.

When executed, the script processes checkpoints stored in a specified directory, averages them, and generates new directories containing the averaged checkpoints.

How It Works
------------
1. **Input Checkpoints**: The script takes checkpoints from a directory containing subfolders (e.g., `mp_rank_X`) that store distributed checkpoints.
2. **Averaging Process**:
   - If a list of specific checkpoint steps is provided, the script averages only those.
   - If no steps are specified, the script averages all available checkpoints in the directory.
3. **Output Checkpoints**: The script generates new directories within each distributed checkpoint subfolder. The output directories are named `<checkpoint name>-averaged`.

How to Run the Script
---------------------
Use the following command to run the checkpoint averaging script:

.. code-block:: shell

   python scripts/checkpoint_averaging/distributed_checkpoint_averaging.py \
       --name_prefix=<checkpoint name> \
       --checkpoint_dir=<folder with mp_rank_X subfolders containing checkpoints> \
       --steps <optionally a list of checkpoint steps to average, if not provided, it will average all the checkpoints>

**Arguments**:
- `--name_prefix`: Specifies the prefix for the generated averaged checkpoint.
- `--checkpoint_dir`: Specifies the folder containing the distributed checkpoints.
- `--steps`: (Optional) A comma-separated list of checkpoint steps to average. If not provided, the script will average all the checkpoints in the directory.

**Example**:
.. code-block:: shell

   python scripts/checkpoint_averaging/distributed_checkpoint_averaging.py \
       --name_prefix=final_model \
       --checkpoint_dir=/path/to/checkpoints \
       --steps 1000,2000,3000

This will create new averaged checkpoint directoriy named `final_model-averaged`.

Output
------
After execution, the script generates averaged checkpoint files in subdirectories named `<checkpoint name>-averaged`. These directories are created within the respective distributed checkpoint subfolders.

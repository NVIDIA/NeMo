from typing import Any, Dict

import tensorflow as tf


def add_next_obs(traj: Dict[str, Any], pad: bool = True) -> Dict[str, Any]:
    """
    Given a trajectory with a key "observations", add the key "next_observations". If pad is False, discards the last
    value of all other keys. Otherwise, the last transition will have "observations" == "next_observations".
    """
    if not pad:
        traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
        traj_truncated["next_observations"] = tf.nest.map_structure(
            lambda x: x[1:], traj["observations"]
        )
        return traj_truncated
    else:
        traj["next_observations"] = tf.nest.map_structure(
            lambda x: tf.concat((x[1:], x[-1:]), axis=0), traj["observations"]
        )
        return traj

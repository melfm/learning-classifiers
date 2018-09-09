# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python 3
 * Numpy
 * TensorFlow
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python 3 is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

**Mel Note**: The current version of the code, as far as I know, only works with `Python3.6`.You also need the followings in your `bashrc` in order to run the code:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin:/usr/lib/nvidia-384:/usr/local/cuda-8.0/lib64"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so"
```

## Behavioral Cloning Experiments

### Steps
1. Generate expert roll-outs
```
python3.6 run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 20
```

2. Train behavioral cloning policy
```
python3.6 train_bc_policy.py Humanoid-v2 --num_epochs 200
```

3. Run behavioral cloning policy
```
python3.6 run_bc_policy.py --envname Humanoid-v2 --render --num_rollouts 20
```

## DAgger Experiments

Some notes about the experiment result data.

- Returns: This is the total rewrad. The scale varies between tasks but overall the goal is to increase this value.
This is visualized as a list of rewards for each trajectory rollout.
- Std: Shows the amount of variation between each rollout reward.
A low standard deviation indicates that the data points tend to be close to the mean of the set, while a high standarddeviation indicates that the data points are spread out over a wider range of values.

### Steps
These steps assume the expert rollsouts have already been generated.
1. Run DAgger
```
python3.6 run_dagger.py --envname Humanoid-v2 --render --num_rollouts 20
```

2. Visualize results
```
python plot_dagger_results.py
```

## TODO
- Add experiment results
- Try adding Dropout and BatchNorm

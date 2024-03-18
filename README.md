# large-scale-DRL-exploration
[RAL 2024] Deep Reinforcement Learning-based Large-scale Robot Exploration - - Public code and model

## Run

#### Dependencies
* `python == 3.10.8`
* `pytorch == 1.12.0`
* `ray == 2.1.0`
* `scikit-image == 0.19.3`
* `scikit-learn == 1.2.0`
* `scipy == 1.9.3`
* `matplotlib == 3.6.2`
* `tensorboard == 2.11.0`


#### Training
1. Set training parameters in `parameters.py`.
2. Run `python driver.py`

#### Evaluation
1. Set parameters in `test_parameters.py`.
2. Run `test_driver.py`

## Files
* `parameters.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the local network.
* `worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `graph_generator.py` Generate and update the collision-free graph.
* `ground_truth_graph.py` Generate and update the ground truth graph.
* `node.py` Initialize and update nodes in the coliision-free graph.
* `sensor.py` Simulate the sensor model of Lidar.
* `/model` Trained model.
* `/DungeonMaps` Maps of training environments provided by <a href="https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration">Chen et al.</a>.


### Authors
[Yuhong Cao](https://github.com/caoyuhong001)\
Rui Zhao\
[Yizhuo Wang](https://github.com/wyzh98)\
Bairan Xiang\
[Guillaume Sartoretti](https://github.com/gsartoretti)

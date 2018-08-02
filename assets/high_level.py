#  high level api running experiments from config files

cd energy_py/experiments

python experiment.py example dqn

tensorboard --logdir='./results/example/tensorboard'

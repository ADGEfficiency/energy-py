# copies of usefulbash commands

#  start tensorboard server with only the rewards
tensorboard --logdir=run1:./results/new_flex/tensorboard/dqn1/rl,run2:./results/new_flex/tensorboard/dqn2/rl,run3:./results/new_flex/tensorboard/dqn3/rl,autoflex:./results/new_flex/tensorboard/autoflex/rl

#  copy results to local machine
rsync -chavzP --exclude 'tensorboard' --exclude 'env_histories' --exclude '*.log'
adam@ec2-34-242-11-163.eu-west-1.compute.amazonaws.com:/home/adam/git/energy_py/energy_py/experiments/results/new_flex
/Users/adam/Downloads

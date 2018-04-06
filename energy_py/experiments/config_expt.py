"""
test - can test by reading the log file!

python config_expt.py example example --run_name DQN_1

"""
import os

from energy_py import make_expt_parser, run_config_expt


if __name__ == '__main__':
    args = make_expt_parser()

    #  expt path is set here to get the current working directory
    expt_path = os.path.join(os.getcwd(),
                             'results',
                             args.expt_name)

    run_config_expt(args.expt_name, args.run_name, expt_path)

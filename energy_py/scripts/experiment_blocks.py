import argparse
import csv
import logging
import logging.config
import time

from energy_py.scripts.utils import ensure_dir

logger = logging.getLogger(__name__)

def experiment():
    """
    A decorator used to wrap experiments
    """
    def outer_wrapper(my_expt_func):

        def wrapper(agent, 
                    env, 
                    data_path, 
                    base_path, 
                    opt_parser_args=None,
                    opt_agent_args=None):

            parser, args = expt_args(opt_parser_args)

            paths = make_paths(base_path)

            logger = make_logger(paths['logs'], args.log)

            env = env(data_path, 
                      episode_length=args.len,
                      episode_random=args.rand)
            
            agent_outputs, env_outputs = my_expt_func(agent,
                                                      args,
                                                      paths,
                                                      env,
                                                      opt_agent_args)

            return agent_outputs, env_outputs
        
        return wrapper
    
    return outer_wrapper


def expt_args(optional_args=None):
    """
    args
        optional_args (list) one dict per optional parser arg required
    """
    parser = argparse.ArgumentParser(description='energy_py expt arg parser')

    args_list = [{'name': '--ep',
                  'type': int,
                  'default': 10,
                  'help': 'number of episodes to run (default: 10)'},
                 {'name': '--len',
                  'type': int,
                  'default': 48,
                  'help': 'length of a single episode (default: 48)'},
                 {'name': '--rand',
                  'type': bool,
                  'default': False,
                  'help': 'randomize start of each ep (default: False)'},
                 {'name': '--gamma',
                  'type': float,
                  'default': 0.9,
                  'help': 'discount rate (default: 0.9)'},
                 {'name': '--out',
                  'type': int,
                  'default': 500,
                  'help': 'output results every n episodes (default: n=500)'},
                 {'name': '--log',
                  'type': str,
                  'default': 'INFO',
                  'help': 'logging status (default: info)'}]

    if optional_args:
        args_list.append(optional_args)

    for arg in args_list:
        parser.add_argument(arg['name'],
                            type=arg['type'],
                            default=arg['default'],
                            help=arg['help'])

    args = parser.parse_args()

    return parser, args


def save_args(argparse, path, optional={}):
    """
    Saves args from an argparse object and from an optional
    dictionary

    args
        argparse (object)
        path (str)        : path to save too
        optional (dict)   : optional dictionary of additional arguments

    returns
        writer (object) : csv Writer object
    """
    with open(path, 'w') as outfile:
        writer = csv.writer(outfile)
        for k, v in vars(argparse).items():
            print('{} : {}'.format(k, v))
            writer.writerow([k]+[v])

        if optional:
            for k, v in optional.items():
                print('{} : {}'.format(k, v))
                writer.writerow([k]+[v])
    return writer


def make_paths(name):
    results = name + '/'
    paths = {'results': results,
             'brain': results + 'brain/',
             'logs': results + 'logs.log',
             'args': results + 'args.txt'}
    for k, path in paths.items():
        ensure_dir(path)
    return paths


def make_logger(log_path, log_status):

    logger = logging.getLogger(__name__)
    logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,  # this fixes the problem
            'formatters': {'standard': {'format': '%(asctime)s [%(levelname)s]%(name)s: %(message)s'}},

            'handlers': {'console': {'level': log_status,
                                     'class': 'logging.StreamHandler',
                                     'formatter': 'standard'},

                         'file': {'class': 'logging.FileHandler',
                                  'level': 'DEBUG',
                                  'filename': log_path,
                                  'mode': 'w',
                                  'formatter': 'standard'}, },

            'loggers': {'': {'handlers': ['console', 'file', ],
                             'level': 'DEBUG',
                             'propagate': True}}})

    return logger


def run_single_episode(episode_number,
                       agent,
                       env,
                       sess=None,
                       normalize_return=True):
    """
    Helper function to run through a single episode
    """

    #  initialize before starting episode
    done, step = False, 0
    observation = env.reset(episode_number)
    #  while loop runs through a single episode
    while done is False:
        #  select an action
        action = agent.act(observation=observation, session=sess)
        #  take one step through the environment
        next_observation, reward, done, info = env.step(action)
        #  store the experience
        agent.memory.add_experience(observation, action, reward,
                                    next_observation, done,
                                    step, episode_number)
        step += 1
        observation = next_observation

    #  now episode is done - process the episode in the agent memory
    return agent, env, sess


class Timer(object):
    """
    Logs useful infomation and times experiments
    """
    def __init__(self):
        self.start_time = time.time()
        self.logger_timer = logging.getLogger('Timer')

    def report(self, output_dict):
        """
        The main functionality of this class
        """
        output_dict['run time'] = self.calc_time()

        log = ['{} : {}'.format(k, v) for k,v in output_dict.items()]
        self.logger_timer.info(log)

    def calc_time(self):
        return (time.time() - self.start_time) / 60

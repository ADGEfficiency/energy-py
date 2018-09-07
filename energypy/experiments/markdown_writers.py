import os
from os.path import join


def run_markdown_writer(
        run,
        path
):
    with open(join(path, 'run_results.md'), 'w+') as text_file:
            text_file.write(
                '## {} run of the {} experiment'.format(
                    run.name, run.expt) + os.linesep)

            text_file.write(
                '### delta versus the no_op case' + os.linesep)

            text_file.write(
                '$/day {:2.2f}'.format(
                    run.summary['delta_reward_per_day']) + os.linesep)

            text_file.write(
                '$/yr {:2.0f}'.format(
                    run.summary['delta_reward_per_day'] * 365) + os.linesep)

            text_file.write('![img](fig1.png)' + os.linesep)


def expt_markdown_writer(
        runs,
        path
):
    with open(join(path, 'expt_results.md'), 'w+') as text_file:

        for run_name, run in runs.items():
            text_file.write('## ' + run_name + os.linesep)

            text_file.write('![img](total_episode_rewards.png)')

            #  TODO adding stuff to summary


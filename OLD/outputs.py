import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def df_to_graph(df, names, xlim):
    start = xlim[0]
    stop = xlim[1]
    length = df.iloc[0, 0].shape[0]
    f, axes = plt.subplots(length, 1)
    for i, ax in enumerate(axes):
        df.loc[:, names[i]] = [item_list[i] for item_list in df.iloc[:, 0]]
        ax.plot(df.index, df.loc[:, names[i]])
        ax.set_xlim([start, stop])
        ax.set_title(names[i])
        if i != length:
            ax.get_xaxis().set_ticks([])
    return f, df


def create_setup(agent, env, run_path):
    setup = {}
    # doing the agent setup first
    setup['Q function'] = agent.Q.summary()
    setup['Batch size'] = agent.batch_size
    setup['Q function'] = agent.epochs

    # now the env, starting with base env setup
    setup['State names'] = env.state_names
    setup['Action names'] = env.action_names

    # and now our specific env
    setup['Episode length'] = env.episode_length
    setup['Episode length'] = env.lag
    setup['Random ts'] = env.random_ts
    return setup

def create_outputs(agent, training_session_length):
    print('Generating outputs')
    plt.style.use('seaborn-deep')

    memory = pd.DataFrame(agent.memory,
                          columns=['State', 'Action', 'State Action',
                                   'Reward', 'Next State', 'Episode',
                                   'Step', 'Epsilon', 'Choice', 'Done', 'Age'])

    agent_info = pd.DataFrame(agent.info,
                              columns=['Run Time', 'Q1 Test', 'Q2 Test'])

    network_memory = pd.DataFrame(agent.network_memory,
                                  columns=['State Action (normalized)',
                                           'Reward', 'Next State & Actions', 'Done'])

    env_info = pd.DataFrame(agent.env.info,
                            columns=['Settlement period',
                                     'Power generated [MWe]',
                                     'Import electricity price [£/MWh]',
                                     'Total heat demand [MW]',
                                     'Timestamp'])
    # set index, set type to datetime
    env_info.loc[:, 'Timestamp'] = env_info.loc[:, 'Timestamp'].apply(pd.to_datetime)
    env_info.set_index(keys='Timestamp',
                       inplace=True,
                       drop=True)
    sums = memory.groupby(['Age']).sum().add_prefix('Total ')
    means = memory.groupby(['Age']).mean().add_prefix('Average ')
    summary = pd.concat([sums, means], axis=1)

    memory = pd.concat([memory, agent_info], axis=1)
    f1 = figure_1(memory, training_session_length)
    f2 = figure_2(memory, summary, training_session_length)
    # f3 = figure_3(memory, agent)
    # f4 = figure_4(memory, agent)
    f5 = figure_5(env_info)
    # plt.close('all')

    if agent.save_csv:
        print('Saving csvs')
        memory.to_csv('results/memory.csv')
        network_memory.to_csv('results/network_memory.csv')

    final_results = pd.DataFrame(
        data=[summary.loc[1, 'Total Reward'],
              summary.loc[agent.age, 'Total Reward'],
              summary.loc[agent.age, 'Total Reward']-summary.loc[1, 'Total Reward'],
              agent.timer.get_time(),
              agent.age],
        index=['Naive reward [£/episode]',
               'Last run reward [£/episode]',
               'Value [£/episode]',
               'Run time',
               'Age'],
        columns=['Final results'])

    final_results.to_csv('results/final_results.csv')
    print(final_results)
    return final_results

def figure_1(memory, training_session_length):
    f1, ax1 = plt.subplots(2, 2, sharex=False)
    f1.set_size_inches(8*2, 6*2)

    memory.plot(y='Q Test',
                kind='line',
                ax=ax1[0,0],
                use_index=True)
    ax1[0,0].set_title('Q Test All Time',  fontsize=18)

    memory.plot(y='Training History',
                kind='line',
                ax=ax1[1,0],
                use_index=True)
    ax1[1,0].set_title('Training History All Time',  fontsize=18)


    memory.loc[training_session_length:, :].plot(y='Q Test',
                                                 kind='line',
                                                 ax=ax1[0,1],
                                                 use_index=True)
    ax1[0,1].set_title('Q Test This Session',  fontsize=18)

    memory.loc[training_session_length:, :].plot(y='Training History',
                                                 kind='line',
                                                 ax=ax1[1,1],
                                                 use_index=True)
    ax1[1,1].set_title('Training History This Session',  fontsize=18)

    f1.savefig('results/figures/f1.png')
    return f1

def figure_2(memory, summary, training_session_length):
    f2, ax2 = plt.subplots(2, 1, sharex=False)
    f2.set_size_inches(8*2, 6*2)

    sums = memory.groupby(['Age']).sum().add_prefix('Total ')
    means = memory.groupby(['Age']).mean().add_prefix('Average ')
    summary = pd.concat([sums, means], axis=1)

    ax2[0].plot(summary.index,
                  summary.loc[:, 'Total Reward'],
                  label='Total Reward',
                  marker='o')
    ax2[0].set_title('Total Reward All Time')

    ax2[1].plot(summary.index[training_session_length:],
                  summary.loc[:, 'Total Reward'].values[training_session_length:],
                  label='Total Reward',
                  marker='o')
    ax2[1].set_title('Total Reward This Training Session')
    ax2[1].set_xlabel('Episode', fontsize=14)
    f2.savefig('results/figures/f2.png')
    return f2

def figure_3(memory, agent):
    actions = memory.loc[:, 'Action'].to_frame(name='Actions')
    action_names = agent.env.action_names
    f3, actions = df_to_graph(actions, action_names, xlim)
    f3.set_size_inches(8*2, 6*2)
    f3.savefig('results/figures/f3.png')
    return f3


def figure_4(memory, agent):
    states = memory.loc[:, 'State'].to_frame(name='States')
    state_names = agent.env.state_names
    f4, states = df_to_graph(states, state_names, xlim)
    f4.set_size_inches(8*2, 6*2)
    f4.savefig('results/figures/f4.png')
    return f4

def figure_5(env_info):
    f5, ax5 = plt.subplots(1, 1)
    f5.set_size_inches(8*2, 6*2)
    print(env_info.shape)
    print(env_info.index)
    env_info.plot(y=['Power generated [MWe]',
                     'Import electricity price [£/MWh]',
                     'Total heat demand [MW]'],
                      subplots=False,
                      kind='line',
                      use_index=True,
                      ax=ax5)

    ax5.legend(loc='best', fontsize=18)
    ax5.set_xlabel('Steps (Last Episode)')
    ax5.set_title('Operation for Last Episode')
    f5.savefig('results/figures/f5.png')
    return f5

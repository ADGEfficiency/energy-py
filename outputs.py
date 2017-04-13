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


def gen(agent):
    print('Generating outputs')
    plt.style.use('seaborn-deep')

    idx = np.linspace(1, len(agent.memory), len(agent.memory), endpoint=True)
    index = pd.Index(idx, dtype=int, name='Step')

    memory = pd.DataFrame(
        agent.memory,
        index=index,
        columns=['State', 'Action', 'State Action',
                 'Reward', 'Next State', 'Episode',
                 'Step', 'Epsilon', 'Choice', 'Done']
        )

    agent_info = pd.DataFrame(
        agent.info,
        index=index,
        columns=['Run Time', 'Q Test',
                 'Training History']
        )
    agent_info.loc[:, 'Hist Roll Avg.'] = agent_info.loc[:,
                                                         'Training History'].rolling(window=1000,
                                                                                     min_periods=None,
                                                                                     center=False).mean()

    network_memory = pd.DataFrame(
        agent.network_memory,
        index=index,
        columns=['State Action (normalized)',
                 'Reward', 'Next State & Actions', 'Done']
        )

    memory = pd.concat([memory, agent_info], axis=1)
    sums = memory.groupby(['Episode']).sum().add_prefix('Total ')
    means = memory.groupby(['Episode']).mean().add_prefix('Mean ')
    count = memory.groupby('Episode')['Step'].count().rename('Step')
    summary = pd.concat([sums, means, count], axis=1)

    steps = summary.loc[:, 'Step'].values
    tick_locations = np.cumsum(steps)
    last_step_start = int(tick_locations[-2])
    last_step_stop = int(tick_locations[-1])
    xlim = [last_step_start, last_step_stop]

    env_info = pd.DataFrame(
        agent.env.info,
        index=range(last_step_start, last_step_stop),
        columns=['Settlement period',
                 'Power generated [MWe]',
                 'Import electricity price [£/MWh]',
                 'Total heat demand [MW]']
        )
    env_info.loc[:, 'Steps'] = np.arange(1, env_info.shape[0] + 1)

    # Figure 1
    f1, ax1 = plt.subplots(2, 2, sharex=True)
    f1.set_size_inches(8*2, 6*2)
    f1_ = [{'Name': 'Q Test'},
           {'Name': 'Training History'},
           {'Name': 'Q Test'},
           {'Name': 'Training History'}]

    for k, axes1 in enumerate(f1.axes):
        name = f1_[k]['Name']
        axes1.plot(memory.index,
                   memory.loc[:, name],
                   label=name)
        axes1.set_xlim(tick_locations[1], tick_locations[-1])

        if name == 'Training History':  # TODO
            axes1.plot(memory.index,
                       memory.loc[:, 'Hist Roll Avg.'],
                       label='Hist Roll Avg.',
                       color='r')

        if k == 2 or k == 3:
            axes1.set_xlim(tick_locations[-2], tick_locations[-1])
            axes1.set_xlabel('Steps', fontsize=14)
        else:
            axes1.set_title(name, y=1.11, fontsize=18)

    ax1_ = ax1[0, 0].twiny()
    ax1_.set_xlim(ax1[0, 0].get_xlim())
    ax1_.set_xticks(tick_locations[0::5])  # TODO
    ax1_.set_xticklabels(np.arange(1, len(steps)+1, 5))
    ax1_.set_xlabel('Episode', fontsize=14)

    ax1_ = ax1[0, 1].twiny()
    ax1_.set_xlim(ax1[0, 0].get_xlim())
    ax1_.set_xticks(tick_locations[0::5])
    ax1_.set_xticklabels(np.arange(1, len(steps)+1, 5))
    ax1_.set_xlabel('Episode', fontsize=14)
    f1.savefig('results/figures/f1.png')

    # Figure 2
    f2, ax2 = plt.subplots(2, 1, sharex=True)
    f2.set_size_inches(8*2, 6*2)
    f2_ = [{'Name': 'Mean Epsilon'},
           {'Name': 'Total Reward'}]

    for j, axes2 in enumerate(ax2):
        name = f2_[j]['Name']
        axes2.plot(summary.index,
                   summary.loc[:, name],
                   label=name,
                   marker='o')
        axes2.set_title(name, fontsize=18)
    ax2[1].set_xlabel('Episode', fontsize=14)
    f2.savefig('results/figures/f2.png')

    # Figure 3
    actions = memory.loc[:, 'Action'].to_frame(name='Actions')
    action_names = agent.env.action_names
    f3, actions = df_to_graph(actions, action_names, xlim)
    f3.set_size_inches(8*2, 6*2)
    f3.savefig('results/figures/f3.png')

    # Figure 4
    states = memory.loc[:, 'State'].to_frame(name='States')
    state_names = agent.env.state_names
    f4, states = df_to_graph(states, state_names, xlim)
    f4.set_size_inches(8*2, 6*2)
    f4.savefig('results/figures/f4.png')

    # Figure 5
    f5, ax5 = plt.subplots(1, 1)
    f5.set_size_inches(8*2, 6*2)
    x_range = env_info['Steps'].as_matrix()
    fig5_vars = ['Power generated [MWe]',
                 'Import electricity price [£/MWh]',
                 'Total heat demand [MW]']
    for var in fig5_vars:
        ax5.plot(x_range, env_info.loc[:, var], label=var)
    ax5.legend(loc='best', fontsize=18)
    ax5.set_xlabel('Steps (Episode ' + str(memory['Episode'].iloc[-1]) +')',
                   fontsize=14)
    ax5.set_xlim(0, x_range[-1])
    ticks = x_range[0::50]
    ax5.set_xticks(ticks)
    f5.savefig('results/figures/f5.png')

    plt.close('all')

    if agent.save_csv:
        print('Saving csvs')
        memory.to_csv('results/memory.csv')
        network_memory.to_csv('results/network_memory.csv')
        summary.to_csv('results/summary.csv')
        states.to_csv('results/states.csv')
        actions.to_csv('results/actions.csv')

    print('Finished generating outputs')
    return memory, network_memory, summary, states, actions

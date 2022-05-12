def update_target_network(online, target, rho):
    """copy from qfunc - TODO move to use only one"""
    for on, ta in zip(online.net.parameters(), target.net.parameters()):
        ta.data.copy_(rho * ta.data + on.data * (1.0 - rho))


def update(
    onlines,
    targets,
    hyp,
    counters
):
    for onl, tar in zip(onlines, targets):
        update_target_network(onl, tar, hyp['rho'])

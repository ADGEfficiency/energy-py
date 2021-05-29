def update(
    onlines,
    targets,
    hyp,
    counters
):
    for onl, tar in zip(onlines, targets):
        update_target_network(onl, tar, hyp['rho'])


def update_target_network(online, target, rho, step=None):
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        t.assign(rho * t.value() + (1 - rho) * o.value())

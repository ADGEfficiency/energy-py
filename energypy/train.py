from energypy import utils, qfunc, policy, target, alpha


def train(*args, **kwargs):
    if 'network' in kwargs.keys():
        raise NotImplementedError()
    else:
        return train_one_head_network(*args, **kwargs)


def train_one_head_network(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizers,
    counters,
    hyp
):
    st = utils.now()
    qfunc.update(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
        writer,
        [optimizers['online-1'], optimizers['online-2']],
        counters,
        hyp
    )
    counters['q-func-update-seconds'] += utils.now() - st

    st = utils.now()
    policy.update(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
        writer,
        optimizers['actor'],
        counters
    )
    counters['pol-func-update-seconds'] += utils.now() - st

    st = utils.now()
    target.update(
        onlines,
        targets,
        hyp,
        counters
    )
    counters['target-update-seconds'] += utils.now() - st

    st = utils.now()
    alpha.update(
        batch,
        actor,
        log_alpha,
        hyp,
        optimizers['alpha'],
        counters,
        writer
    )
    counters['alpha-update-seconds'] += utils.now() - st
    counters['train-seconds'] += utils.now() - st
    counters['train-steps'] += 1


def train_multi_head_network(
    batch,
    network,
    log_alpha,
    writer,
    optimizers,
    counters,
    hyp
):
    st = utils.now()
    #  train net



    #  maybe do the actor fwd pass here...
    st = utils.now()
    alpha.update(
        batch,
        actor,
        log_alpha,
        hyp,
        optimizers['alpha'],
        counters,
        writer
    )
    counters['alpha-update-seconds'] += utils.now() - st

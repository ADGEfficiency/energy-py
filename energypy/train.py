from energypy import utils, qfunc, actor, target, alpha


def train(*args, **kwargs):
    return train_one_head_network(*args, **kwargs)


def train_one_head_network(
    batch,
    policy,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizers,
    counters,
    hyp,
    **kwargs
):
    st = utils.now()
    qloss = qfunc.update(
        batch,
        policy,
        onlines,
        targets,
        log_alpha,
        writer,
        [optimizers["online-1"], optimizers["online-2"]],
        counters,
        hyp,
    )
    counters["q-func-update-seconds"] += utils.now() - st

    st = utils.now()
    ploss = actor.update(
        batch, policy, onlines, targets, log_alpha, writer, optimizers["actor"], counters, hyp
    )
    counters["pol-func-update-seconds"] += utils.now() - st

    st = utils.now()
    target.update(onlines, targets, hyp, counters)
    counters["target-update-seconds"] += utils.now() - st

    st = utils.now()
    alpha.update(batch, policy, log_alpha, hyp, optimizers["alpha"], counters, writer)
    counters["alpha-update-seconds"] += utils.now() - st
    counters["train-seconds"] += utils.now() - st
    counters["train-steps"] += 1

    return {'qfunc-loss': qloss, 'policy-loss': ploss}


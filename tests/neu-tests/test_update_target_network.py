from energypy.agent.qfunc import update_target_network, DenseQFunc


def test_update_target_network():

    obs_shape = (3, 6)
    n_actions = 10
    online = DenseQFunc(obs_shape, n_actions)
    target = DenseQFunc(obs_shape, n_actions)

    #  check weights diff
    for on, ta in zip(online.parameters(), target.parameters()):
        assert not (on.data.numpy() == ta.data.numpy()).all()

    #  update with wrong, check not same
    update_target_network(online, target, 1.0)
    for on, ta in zip(online.parameters(), target.parameters()):
        assert not (on.data.numpy() == ta.data.numpy()).all()

    #  update with correct rho, check same
    update_target_network(online, target, 0.0)
    for on, ta in zip(online.parameters(), target.parameters()):
        assert (on.data.numpy() == ta.data.numpy()).all()

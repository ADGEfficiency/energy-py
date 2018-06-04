import tensorflow as tf


def get_tf_params(scope):
    params = [p for p in tf.trainable_variables()
              if p.name.startswith(scope)]

    #  sort parameters list by the variable name
    params = sorted(params, key=lambda var: var.name)

    return params


def make_copy_ops(parent, child, scope='copy_ops'):
    copy_ops = []

    with tf.variable_scope(scope):
        tau = tf.Variable(1.0, name='tau')

        for p, c in zip(parent, child):
            assert p.name.split('/')[1:] == c.name.split('/')[1:]

            new_value = tf.add(tf.multiply(p, tau),
                               tf.multiply(c, 1 - tau))
            op = c.assign(new_value)
            copy_ops.append(op)

    return copy_ops, tau

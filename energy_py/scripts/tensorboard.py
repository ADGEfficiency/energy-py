import tensorflow as tf


class TensorboardHepler(object):

    def __init__(self, logdir, scalars):
        self.writer = tf.summary.FileWriter(logdir)
        self.summaries = []
        self.steps = 0
        for tag, variable in scalars.items():
            self.add_scalar(tag, variable)

    def add_scalar(self, tag, variable):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=variable)])
        self.summaries.append(summary)

    def write_summaries(self):
        _ = [self.writer.add_summary(s, self.steps) for s in self.summaries]
        self.writer.flush()

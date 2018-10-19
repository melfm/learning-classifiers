import numpy as np
import tensorflow as tf

import train_pg_f18 as PG


class TrainPGTest(tf.test.TestCase):

    def test_build_mlp(self):

        x = tf.placeholder(shape=[None, 3], dtype=tf.float32)

        net_output = PG.build_mlp(
            x,
            output_size=1,
            scope='mlp_test',
            n_layers=1,
            size=20)

        init = tf.global_variables_initializer()

        with self.test_session() as sess:
            sess.run(init)
            x_vals = np.random.normal(0, 1, (10, 3))
            net_out = sess.run([net_output],
                               feed_dict={x: x_vals})
            # Check output size
            self.assertEqual(net_out[0].shape,
                             (10, 20))


if __name__ == '__main__':
    tf.test.main()

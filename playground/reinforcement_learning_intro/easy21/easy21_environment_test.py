import unittest
from unittest.mock import patch

import easy21_environment as easyEnv


class Easy21EnvTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = easyEnv.Easy21Env()

    def test_step(self):
        self.env.reset(dealer=4, player=1)

        state = self.env.observe()
        self.assertEqual(self.env.player_sum,
                         state[1])
        self.env.step(easyEnv.ACTION_HIT)

        # New state
        self.assertNotEqual(self.env.player_sum,
                            state[1])


if __name__ == '__main__':
    unittest.main()

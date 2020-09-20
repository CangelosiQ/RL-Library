import unittest

from rl_library.agents import DQAgent


class TestDQAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DQAgent(8, 4)
        self.sequential_agent = DQAgent(8,4)

    def test_setup(self):
        self.assertEqual(self.agent.state_size, 8)
        self.assertEqual(self.agent.action_size, 4)


if __name__ == '__main__':
    unittest.main()

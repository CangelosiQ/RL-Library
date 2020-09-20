import unittest

from rl_library.agents import DQAgent
from torch import nn
from collections import OrderedDict

class TestDQAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DQAgent(8, 4)
        self.sequential_agent = DQAgent(8, 4, hidden_layer_sizes=[10, 12, 13])
        architecture = OrderedDict([
            ('fc1', nn.Linear(8, 22)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(22, 44)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(44, 4)),
            ('softmax', nn.Softmax(dim=1))])
        model = nn.Sequential(architecture)
        self.model_agent = DQAgent(8, 4, model=model)

    def test_setup(self):
        self.assertEqual(self.agent.state_size, 8)
        self.assertEqual(self.agent.action_size, 4)
        self.assertEqual(len(self.agent.qnetwork_local.state_dict()), 2)
        self.assertEqual(tuple(self.agent.qnetwork_local.state_dict()["fc1.weight"].size()), (4, 8))

        self.assertEqual(self.sequential_agent.state_size, 8)
        self.assertEqual(self.sequential_agent.action_size, 4)
        self.assertEqual(len(self.sequential_agent.qnetwork_local.state_dict()), 4*2)
        self.assertEqual(tuple(self.sequential_agent.qnetwork_local.state_dict()["fc0.weight"].size()), (10, 8))

        self.assertEqual(self.model_agent.state_size, 8)
        self.assertEqual(self.model_agent.action_size, 4)
        self.assertEqual(len(self.model_agent.qnetwork_local.state_dict()), 3 * 2)
        self.assertEqual(tuple(self.model_agent.qnetwork_local.state_dict()["fc1.weight"].size()), (22, 8))

    def test_sequential_agent(self):
        agent = DQAgent(8, 4, hidden_layer_sizes=[10])
        self.assertEqual(agent.state_size, 8)
        self.assertEqual(agent.action_size, 4)
        self.assertEqual(len(agent.qnetwork_local.state_dict()), 2*2)
        self.assertEqual(tuple(agent.qnetwork_local.state_dict()["fc0.weight"].size()), (10, 8))

        agent = DQAgent(10, 1, hidden_layer_sizes=[30, 20])
        self.assertEqual(agent.state_size, 10)
        self.assertEqual(agent.action_size, 1)
        self.assertEqual(len(agent.qnetwork_local.state_dict()), 2*3)
        self.assertEqual(tuple(agent.qnetwork_local.state_dict()["fc0.weight"].size()), (30, 10))


if __name__ == '__main__':
    unittest.main()

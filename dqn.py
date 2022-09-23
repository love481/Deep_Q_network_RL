import torch
import os
from model import QNetwork
from replay_buffer import Buffer
import torch.nn.functional as F
class DQN:
    def __init__(self, args,agent_id):  
        self.args = args
        self.train_step = 0
        # create the network
        self.q_network = QNetwork(args)
        self.agent_id =agent_id
        # build up the target network
        self.q_target_network = QNetwork(args)
        self.huber_loss = torch.nn.SmoothL1Loss()
        # load the weights into the target networks
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        # create the optimizer
        self.q_optim = torch.optim.Adam(self.q_network.parameters(), lr=self.args.lr_actor)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.q_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
        for target_param, param in zip(self.q_target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.q_target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    # update the network
    def train(self, experiences):
        for key in experiences.keys():
            experiences[key] = torch.tensor(experiences[key], dtype=torch.float32).to(self.args.device)
        r = experiences['r'] 
        done=experiences['done']
        o=experiences['o']
        a=experiences['a']
        o_next=experiences['o_next']

        # calculate the target Q value function
        #for greedy approach
        q_next=self.q_target_network.forward(o_next).detach().max(1)[0].unsqueeze(1)
        target_q = (r.unsqueeze(-1) + (self.args.gamma *q_next*(1-done.unsqueeze(-1))))
        expected_q=self.q_network.forward(o).gather(1, a.type(torch.int64)).max(1)[0].unsqueeze(1)

        loss = F.mse_loss(expected_q, target_q )
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        self._soft_update_target_network()

    def save_model(self):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.q_network.state_dict(), model_path + '/actor_params.pkl')




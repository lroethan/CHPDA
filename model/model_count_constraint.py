import math
import os
import pickle
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import environment.env_count_constraint as env
import environment.env_storage_constraint as env2
from model import prioritized_replay_buffer as BufferX
from model import replay_buffer as Buffer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

script_name = os.path.basename(__file__)
directory = "./exp_" + script_name + "monitor" + "/"


class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        actions = self.layers(state)
        return actions


class DNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        # self.l3 = nn.Linear(256, 128)
        self.adv1 = nn.Linear(256, 256)
        self.adv2 = nn.Linear(256, action_dim)
        self.val1 = nn.Linear(256, 64)
        self.val2 = nn.Linear(64, 1)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)
        # self.l3.weight.data.normal_(0.0, 1e-2)
        # self.l3.weight.data.uniform_(-0.1, 0.1)
        self.adv1.weight.data.normal_(0.0, 1e-2)
        self.adv1.weight.data.uniform_(-0.1, 0.1)
        self.adv2.weight.data.normal_(0.0, 1e-2)
        self.adv2.weight.data.uniform_(-0.1, 0.1)
        self.val1.weight.data.normal_(0.0, 1e-2)
        self.val1.weight.data.uniform_(-0.1, 0.1)
        self.val2.weight.data.normal_(0.0, 1e-2)
        self.val2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        # actions = self.layers(state)
        x = self.relu(self.l1(state))
        x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))
        adv = self.relu(self.adv1(x))
        val = self.relu(self.val1(x))
        adv = self.relu(self.adv2(adv))
        val = self.relu(self.val2(val))
        qvals = val + (adv - adv.mean())
        return qvals


class DQN:
    def __init__(self, workload, action, index_mode, args, is_dnn, is_ps, is_double, a):
        self.args = args
        self.workload = workload
        self.action = action
        self.index_mode = index_mode
        self.state_dim = len(workload) + len(action)
        # We do not need another flag to indicate 'deletion/creation'
        self.action_dim = len(action)
        self.is_ps = is_ps
        self.is_double = is_double

        # Initialize actor network
        if is_dnn:
            self.actor = DNN(self.state_dim, self.action_dim).to(device)
            self.actor_target = DNN(self.state_dim, self.action_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            self.actor = NN(self.state_dim, self.action_dim).to(device)
            self.actor_target = NN(self.state_dim, self.action_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize actor optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), self.args.lr
        )  # optim.SGD(self.actor.parameters(), lr=self.conf['LR'], momentum=0.9)

        # Initialize replay buffer and monitoring variables
        self.replay_buffer = None
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(self.workload, self.action, self.index_mode, a)

        # Initialize summary writer for storing parameters (Tensorboard)
        self.writer = SummaryWriter(directory)

        # Initialize learn step counter
        self.learn_step_counter = 0

    def select_action(self, t, state):
        if not self.replay_buffer.can_update():
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= self.args.epsilon:  # *(1 - math.pow(0.5, t/50)):  #*(t/MAX_STEP):  # greedy policy
            action_value = self.actor.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            return action
        else:  # random policy
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action

    def _sample(self):
        batch, idx = self.replay_buffer.sample(self.args.batch_size)
        # state, next_state, action, reward, np.float(done))
        # batch = self.replay_memory.sample(self.batch_size)
        x, y, u, r, d = [], [], [], [], []
        for _b in batch:
            x.append(np.array(_b[0], copy=False))
            y.append(np.array(_b[1], copy=False))
            u.append(np.array(_b[2], copy=False))
            r.append(np.array(_b[3], copy=False))
            d.append(np.array(_b[4], copy=False))
        return (
            idx,
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1),
        )

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, ep):
        if self.learn_step_counter % self.args.q_iterations == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1
        # self.adjust_learning_rate(self.actor_optimizer, ep)
        for it in range(self.args.u_iterations):
            idxs = None
            if self.is_ps:
                idxs, x, y, u, r, d = self._sample()
            else:
                x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.LongTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            q_eval = self.actor(state).gather(1, action)

            if self.is_double:
                next_batch = self.actor(next_state)
                nx = next_batch.max(1)[1][:, None]
                # max_act4next = np.argmax(q_eval_next, axis=1)
                q_next = self.actor_target(next_state)
                qx = q_next.gather(1, nx)
                # q_target = reward + (1 - done) * self.conf['GAMMA'] * qx.max(1)[0].view(self.conf['BATCH_SIZE'], 1)
                q_target = reward + (1 - done) * self.args.discount_factor * qx
            else:
                q_next = self.actor_target(next_state).detach()
                q_target = reward + (1 - done) * self.args.discount_factor * q_next.max(1)[0].view(self.args.batch_size, 1)
            actor_loss = F.mse_loss(q_eval, q_target)
            error = torch.abs(q_eval - q_target).data.numpy()
            if self.is_ps:
                for i in range(self.args.batch_size):
                    idx = idxs[i]
                    self.replay_buffer.update(idx, error[i][0])

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.actor_loss_trace.append(actor_loss.data.item())
            # for item in self.actor.named_parameters():
            # h = item[1].register_hook(lambda grad: print(grad))

    def save(self):
        torch.save(self.actor_target.state_dict(), directory + "dqn.pth")
        print("====== model Saved ======")

    def load(self):
        print("====== model Loaded ======")
        self.actor.load_state_dict(torch.load(directory + "dqn.pth"))

    def train(self, is_load, max_index):
        """
        This function trains the reinforcement learning agent.

        Args:
        is_load (bool): Flag to determine if we want to load from a saved checkpoint.
        max_index (int): Maximum number of indexes to use for training.
        """
        # Load a saved checkpoint if the flag is set.
        if is_load:
            self.load()

        # Set maximum number of indexes for the environment.
        self.envx.max_count = max_index

        # Get pre-created indexes if they exist.
        # pre_created_indexes = self.envx.checkout
        pre_created_indexes = []

        # If there are enough pre-created indexes, we use them and exit early.
        if pre_created_indexes and len(pre_created_indexes) >= max_index:
            print(f"Using pre-created indexes: {pre_created_indexes}")
            return pre_created_indexes[:max_index]

        # Decide which type of replay buffer to use based on the `is_ps` flag.
        learning_start = min(self.args.replay_start, 200 * self.envx.max_count)
        if self.is_ps:
            self.replay_buffer = BufferX.PrioritizedReplayMemory(self.args.memory_capacity, learning_start)
        else:
            self.replay_buffer = Buffer.ReplayBuffer(self.args.memory_capacity, learning_start)

        current_best_reward = 0
        current_best_index = None
        rewards = []

        # Start training over multiple episodes.
        for episode in range(self.args.num_episodes):
            print(f"==== Episode: {episode} ====")

            state = self.envx.reset()
            total_reward = 0

            # Variables to store states, actions and rewards for the current episode.
            states = []
            next_states = []
            actions = []
            rewards = []
            dones = []

            # Run the episode.
            for t in count():
                action = self.select_action(episode, state)
                next_state, reward, done = self.envx.step(action)

                total_reward += reward
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(np.float64(done))

                # Add the experience to the replay buffer.
                experience = (state, next_state, action, reward, np.float64(done))
                if self.is_ps:
                    self.replay_buffer.add(1.0, experience)
                else:
                    self.replay_buffer.push(experience)

                if done:
                    # Update the rewards in the replay buffer and keep track of the best reward.
                    for i in range(len(states)):
                        experience = (
                            states[i],
                            next_states[i],
                            actions[i],
                            rewards[i] + total_reward / max_index,
                            dones[i],
                        )
                        if self.is_ps:
                            self.replay_buffer.add(1.0, experience)
                        else:
                            self.replay_buffer.push(experience)

                    if episode > (self.args.num_episodes - 100) and total_reward > current_best_reward:
                        current_best_reward = total_reward
                        current_best_index = self.envx.index_trace_overall[-1]
                        print(f"New best index: {current_best_index}")

                    # Update the model every 5 episodes.
                    if self.replay_buffer.can_update() and episode % 5 == 0:
                        self.update(episode)
                    break

                state = next_state
            rewards.append(total_reward)
        self.save()

        # Plot and save the cost trace.
        plt.figure(max_index)
        x = range(len(self.envx.cost_trace_overall))
        y2 = [math.log10(a) for a in self.envx.cost_trace_overall]
        plt.plot(x, y2, marker="x")
        plt.xlabel("Iterations")
        plt.ylabel("Log Cost")
        plt.title("Cost Frequency")
        plt.savefig(self.args.exp_name + "freq.png", dpi=120)
        plt.clf()
        plt.close()

        # Plot and save the rewards trace.
        plt.figure(max_index + 1)
        x = range(len(rewards))
        plt.plot(x, rewards, marker="x")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Reward Frequency")
        plt.savefig(self.args.exp_name + "rewardfreq.png", dpi=120)
        plt.clf()
        plt.close()

        # Save the cost trace overall as a pickle file.
        with open("{}.pickles".format(self.args.exp_name), "wb") as f:
            pickle.dump(self.envx.cost_trace_overall, f, protocol=0)

        print("Current Best Reward: ", current_best_reward)

        return current_best_index

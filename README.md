{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, I will go through a quick implementation (about 130 lines) of DQN (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). For starters, DQN, or Deep Q Network, is Q Learning (see https://dylanamiller.github.io/ to get spun up on the details) with extra bells and whistles; the main bell and/or whistle being the use of a neural network as the function approximator - interestibly enough, doing this for Q Learning actuall causes the algorithm to have no gaurentee of convergence (https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf) despite the algorithm's success. Along with using a neural network of course, come certain alterations to the Q Learning algorithm that are required in order to make it manageable. But do not let these things obscure what is really going on: Q Learning.\n",
    "\n",
    "Note: I will be using Pytorch for this example. I will not however, discuss implementation details involved with using Pytorch."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.nn import functional as F\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from random import sample\n",
    "from statistics import mean, stdev\n",
    "import copy\n",
    "\n",
    "import gym"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To begin, we define a class for the Experience Replay Buffer. This is the fancy name for the user defined data structure that will hold the transition tuples our algorithm will collect as it attempts to navigate the environment. DQN (even though it is really just Q Learning, I will revert to saying DQN as it saves space) is an off-policy algorithm. This means, that the policy being used to define the agent's behavior is not the one we are improving; the reason for this will be apparent in a bit. This means that we do not need to use current, by which I mean from the agent's history within the current update period, experience in order to train our agent. We can instead choose to use experience from the entire history or training. This is what makes off-policy algorithms more sample efficient that on policy algorithms.\n",
    "\n",
    "In the __init__() function, we define maxlen, batch_size, and buffer. \n",
    "\n",
    ":maxlen defines how many experience (transition tuples of state, action, reward, next_state, next_action) we want to include in our buffer. After the buffer hits this limit, it will start removing data points, starting with the oldest; we may as well leave in the most recent for our agent to use. \n",
    "\n",
    ":batch_size determines how many data points we will use for each update.\n",
    "\n",
    ":buffer of course stores the experience tuples.\n",
    "\n",
    "In the add_exp() function, we first check to see if the buffer is full. If it is, we eject the oldest data point before adding in the new one.\n",
    "\n",
    "In the prime() function, we add to the buffer randomly generated experience tuples until we a batch size worth of them. We do this as poadding for when the algorithm starts. Just in case there are not data points collected in the first episode to perform an update, we are still able to sample a full batch.\n",
    "\n",
    "In the sample() function, we randomly sample from the buffer batch_size data points to use in our update."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Replay:\n",
    "    def __init__(self, maxlen=10000, batch_size=128):\n",
    "        self.maxlen = maxlen\n",
    "        self.batch_size = batch_size\n",
    "        \n",
    "        self.buffer = []\n",
    "        \n",
    "    def add_exp(self, exp):\n",
    "        def buffer_full():\n",
    "            return True if len(self.buffer) == self.maxlen else False\n",
    "        \n",
    "        if buffer_full():\n",
    "            self.buffer.pop()\n",
    "        self.buffer.append(exp)\n",
    "        \n",
    "    def prime(self, env):\n",
    "        for exp in range(self.batch_size):\n",
    "            obs = env.reset()\n",
    "            action = env.action_space.sample()\n",
    "            next_obs, reward, _, _ = env.step(action)\n",
    "            next_action = env.action_space.sample()\n",
    "            \n",
    "            obs = torch.from_numpy(obs).float().unsqueeze(0)\n",
    "            next_obs = torch.from_numpy(next_obs).float().unsqueeze(0)\n",
    "            \n",
    "            self.add_exp((obs, action, reward, next_obs, next_action))\n",
    "    \n",
    "    def sample(self):\n",
    "        return sample(self.buffer, self.batch_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The two functions below are just to help build the neural network. As we will be using the CartPole environment in this example, the network is simple."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_layer(dim0, dim1):\n",
    "    return nn.Sequential(\n",
    "        nn.Linear(dim0, dim1),\n",
    "        nn.ReLU()\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_network(dims):\n",
    "    return [get_layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we have the class the builds defines our network. The part worth looking at here, is the forward() function. We run the input, X, through the network to produce the Q values. What these are, are the expected return given that we are in some state and deviate from the policy for one step by taking some action, after which we return to following the policy - this tells us what the difference is if our agent takes a different action action than what our policy dictates with the current time step. \n",
    "\n",
    "You will notice that forward() has the argument action with a default of None. When collecting experience, no action will be passed to the function and it will return q_values. When performing our update however, we already know the actions we want, because they are in our experience tuples. Here, we will pass them to forward() in order to get the Q values only corresponding to the actions we want (i.e. the ones we have already taken) so that we can perform our update."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DQN(nn.Module):\n",
    "    def __init__(self, input_dim, n_actions, hidden_dim=128, n_hidden=2):\n",
    "        super(DQN, self).__init__()\n",
    "        self.network = nn.Sequential(*build_network([input_dim] + [hidden_dim]*n_hidden))\n",
    "        self.q_values = nn.Linear(hidden_dim, n_actions)\n",
    "        \n",
    "    def forward(self, X, action=None):\n",
    "        X = self.network(X)\n",
    "        q_values = self.q_values(X)\n",
    "        \n",
    "        if action:\n",
    "            q_values = q_values[torch.arange(len(action)), action].view(-1,1)\n",
    "        \n",
    "        return q_values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To help our agent choose actions, we will use an epsilon greedy policy. This means, that most of the time we will greedily choose the action that the highest Q value, but epsilon percent of the time, we will choose an action randomly. It is common to decay the epsilon over the duration of training, much like you would the learning rate.\n",
    "\n",
    "This is probably the most common way for DQN to handle the exploration vs exploitation dilema. In case you are not familiar with this concept (first I would go learn some Reinforcement Learning (RL) basics before trying to learn RL with neural networks, but I won't judge), exploration is how much our agent should look for new experience, and exploitation is how much our agent should prioritize behavior it already knows to be good (at least compared to its experience so far, which may or may not be optimal given the underlying environment dynamics); any good RL agent must find a way to balance these."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def epsilon_greedy(q_values, episode, epsilon=0.3):\n",
    "    r = np.random.rand()\n",
    "    return torch.argmax(q_values).item() if r > (epsilon/episode) else np.random.randint(len(q_values.tolist()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, we have the training loop. First, I will walk through the Q Learning algorithm, then I will point out something about why the algorithm does what it does and the result of doing this, and finally I will explain the changes made in order to accomodate the neural network.\n",
    "\n",
    "The first thing we have to do, is prime out buffer as described above. We only have to do this once for the entire run of the algorithm, because the agent will accumulate experience.\n",
    "\n",
    "At the start of the episode, we get a new state. We use this, to get our Q values that correspond to the value of taking action in that state. With our Q values in hand, we use the epsilon greedy policy to choose an action. We use the chosen action to step the environment forward and get the reward and next state. This next state is run through the network to get its Q values. This time, instead of using our epsilon greedy policy to choose the action, we will simply choose the action with the highest Q value. \n",
    "\n",
    "This is our entire experience tuple. So we add (state, action, reward, next state and next action) to our buffer. At this point, we will set the state to be what is currently the next state and repeat the experience tuple collection process. \n",
    "\n",
    "At the end of the episode, we will randomly sample a batch from our buffer and perform our update. Now, the Q learning update is $R + \\gamma max_{a \\in A}Q_{\\theta}(S_{t+1}, a) - Q_{\\theta}(S_{t}, A_{t})$. This is very similar to the TD error, and it intuitively tells us that the update is the difference between what happened and what we expected to happen. \n",
    "\n",
    "And that is the algorithm. Or at least that is the Q learning algorithm. I will explain necessary changes in a moment. First however, I want to make a note of the second term in our update, $\\gamma max_{a \\in A}Q_{\\theta}(S_{t+1}, a)$:\n",
    "\n",
    "$\\gamma$ is the discount factor. This tells us by how much we wish to say that future rewards are not as important as the current reward. 1 corresponds to no discount and says that we want all rewards considered equally and 0 says that only the current reward matter. The choice depends on the problem, but for toy problems such as CartPole, a value like 0.9 or 0.99 are fairly standard choices. \n",
    "\n",
    "The more important part of this term, is $max_{a \\in A}Q_{\\theta}(S_{t+1}, a)$. $R$ plus this term could describe a possible Q value from $S_{t}$ for a different policy. So, given a different policy, say with parameters $\\theta'$, this could be written as $Q_{\\theta'}(S_{t}, A_{t})$, which looks an awful lot like the final term of the update. It is this fact that makes Q Learning an off-policy algorithm. Rather than use the agent's behavior defining policy, epsilon greedy, to choose the action, we are using the max action. The result is that the agent is attempting the estimate the optimal policy. Cool strategy. \n",
    "\n",
    "This is also why we can use experience from any point in training. In RL, the state distribution underlying the agent's behavior is not stationary. As the agent learns, its probability of ending up in states changes as it action preference changes. For on-policy algorithms, where the agent is updating the same policy it is using to choose actions, that means that its policy is moving around underneath it quite a bit, so it can only use experience from its current update period. In Q Learning however, we are trying to estimate the optimal policy, which does not change. So, all data points converge toward it and will help with our updates.\n",
    "\n",
    "Now, for the required (not really, but they make it work better) neural network changes. Since there is a considerable amount of noise that results from using a nonlinear function approximator and, in this case, constantly comparing Q values against a changing set of Q values, we will actually want to define two neural networks rather than one. We will not train our second network, but freeze it with the weights of the first network from a given update period. While this network will be kept frozen, we will occasionally set it to match the current weights of our training network. We will use this second network as our target (i.e. the term added to the reward). By keeping its weights, and therefore Q values, largely stationary, we are able to remove some noise from our updates. Imagine trying to throw pebbles in a jar, but the jar were to move every time you threw. It would be very difficult. By keeping it still, you will get better at aiming at the jar.\n",
    "\n",
    "We will call the parameters of this second network $\\theta'$. This makes our update $R + \\gamma max_{a \\in A}Q_{\\theta'}(S_{t+1}, a) - Q_{\\theta}(S_{t}, A_{t})$.\n",
    "\n",
    "And that's it. DQN in about 130 lines."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(env, q_network, optim, buffer, q_target, copy_network=200, n_episodes=250, max_steps=500, epsilon=0.3, gamma=0.9):\n",
    "    buffer.prime(env)\n",
    "    \n",
    "    returns = []\n",
    "    \n",
    "    for episode in range(1, n_episodes+1):\n",
    "        r = 0\n",
    "        \n",
    "        obs = env.reset()\n",
    "        obs = torch.from_numpy(obs).float().unsqueeze(0)\n",
    "        \n",
    "        for step in range(1, max_steps+1):\n",
    "            q_values = q_network(obs)\n",
    "            action = epsilon_greedy(q_values, episode, epsilon=epsilon)\n",
    "            next_obs, reward, done, _ = env.step(action)\n",
    "            next_obs = torch.from_numpy(next_obs).float().unsqueeze(0)\n",
    "            next_q_values = q_network(next_obs)\n",
    "            next_action = torch.argmax(next_q_values)\n",
    "            \n",
    "            buffer.add_exp((obs, action, reward, next_obs, next_action))\n",
    "            \n",
    "            r += reward\n",
    "            \n",
    "            obs = next_obs\n",
    "            \n",
    "            if done:\n",
    "                returns.append(r)\n",
    "                break\n",
    "                \n",
    "        batch = buffer.sample()\n",
    "        \n",
    "        obss, actions, rewards, next_obss, next_actions = [exp for exp in zip(*batch)]\n",
    "        \n",
    "        obss = torch.stack(obss).squeeze()\n",
    "        next_obss = torch.stack(next_obss).squeeze()\n",
    "        rewards = torch.tensor(rewards).view(-1,1)\n",
    "\n",
    "        td_loss = F.smooth_l1_loss(rewards + gamma*q_target(next_obss, next_actions), q_network(obss, actions))\n",
    "        \n",
    "        optim.zero_grad()\n",
    "        td_loss.backward()\n",
    "        optim.step()\n",
    "        \n",
    "        if episode % copy_network == 0:\n",
    "            q_target.load_state_dict(q_network.state_dict())\n",
    "        \n",
    "        if episode % 10 == 0:\n",
    "            print('Episode: {} - TD Loss: {:.4f} - Return: {} - Avg Return: {:.4f} - Std Return: {:.4f}'.format(episode, td_loss.item(), r, mean(returns), stdev(returns)))\n",
    "            \n",
    "    plt.plot(returns)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this cell, I am just setting up the problem. Not very exciting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(environment='CartPole-v0', seed=0, n_hidden=2, hidden_dim=128, epsilon=0.3, gamma=0.9, buffer_size=10000, batch_size=128):\n",
    "    env = gym.make(environment)\n",
    "    obs_dim = env.observation_space.shape[0]\n",
    "    n_actions = env.action_space.n\n",
    "    \n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    env.seed(seed)\n",
    "    \n",
    "    q_network = DQN(obs_dim, n_actions, hidden_dim=hidden_dim)\n",
    "    q_target = DQN(obs_dim, n_actions, hidden_dim=hidden_dim)\n",
    "    q_target.load_state_dict(q_network.state_dict())\n",
    "    q_target.eval()\n",
    "                             \n",
    "    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)\n",
    "    buffer = Replay()\n",
    "    \n",
    "    train(env, q_network, optimizer, buffer, q_target, epsilon=epsilon, gamma=epsilon)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Episode: 10 - TD Loss: 0.4864 - Return: 95.0 - Avg Return: 84.9000 - Std Return: 9.3862\n",
      "Episode: 20 - TD Loss: 0.4415 - Return: 72.0 - Avg Return: 83.6500 - Std Return: 9.0045\n",
      "Episode: 30 - TD Loss: 0.3929 - Return: 73.0 - Avg Return: 81.5333 - Std Return: 8.8463\n",
      "Episode: 40 - TD Loss: 0.3497 - Return: 72.0 - Avg Return: 81.0250 - Std Return: 8.6780\n",
      "Episode: 50 - TD Loss: 0.3025 - Return: 93.0 - Avg Return: 79.4800 - Std Return: 9.3814\n",
      "Episode: 60 - TD Loss: 0.2706 - Return: 91.0 - Avg Return: 79.7000 - Std Return: 9.2137\n",
      "Episode: 70 - TD Loss: 0.2247 - Return: 79.0 - Avg Return: 80.2429 - Std Return: 10.0397\n",
      "Episode: 80 - TD Loss: 0.1909 - Return: 120.0 - Avg Return: 81.5875 - Std Return: 11.4656\n",
      "Episode: 90 - TD Loss: 0.1626 - Return: 100.0 - Avg Return: 83.8222 - Std Return: 14.4895\n",
      "Episode: 100 - TD Loss: 0.1311 - Return: 127.0 - Avg Return: 87.0600 - Std Return: 17.3455\n",
      "Episode: 110 - TD Loss: 0.1060 - Return: 108.0 - Avg Return: 90.1091 - Std Return: 20.4950\n",
      "Episode: 120 - TD Loss: 0.0814 - Return: 119.0 - Avg Return: 94.3667 - Std Return: 26.4502\n",
      "Episode: 130 - TD Loss: 0.0634 - Return: 108.0 - Avg Return: 95.2692 - Std Return: 25.8963\n",
      "Episode: 140 - TD Loss: 0.0489 - Return: 200.0 - Avg Return: 97.4429 - Std Return: 27.2117\n",
      "Episode: 150 - TD Loss: 0.0337 - Return: 151.0 - Avg Return: 99.9867 - Std Return: 29.0410\n",
      "Episode: 160 - TD Loss: 0.0282 - Return: 200.0 - Avg Return: 103.1625 - Std Return: 31.6826\n",
      "Episode: 170 - TD Loss: 0.0232 - Return: 200.0 - Avg Return: 107.0588 - Std Return: 35.5193\n",
      "Episode: 180 - TD Loss: 0.0187 - Return: 200.0 - Avg Return: 111.4389 - Std Return: 39.5494\n",
      "Episode: 190 - TD Loss: 0.0137 - Return: 182.0 - Avg Return: 114.5895 - Std Return: 41.4650\n",
      "Episode: 200 - TD Loss: 0.0129 - Return: 200.0 - Avg Return: 117.6850 - Std Return: 43.1688\n",
      "Episode: 210 - TD Loss: 0.0563 - Return: 200.0 - Avg Return: 120.2905 - Std Return: 44.2518\n",
      "Episode: 220 - TD Loss: 0.0376 - Return: 112.0 - Avg Return: 122.3545 - Std Return: 45.0675\n",
      "Episode: 230 - TD Loss: 0.0247 - Return: 114.0 - Avg Return: 124.1435 - Std Return: 45.6129\n",
      "Episode: 240 - TD Loss: 0.0174 - Return: 200.0 - Avg Return: 125.8208 - Std Return: 46.0917\n",
      "Episode: 250 - TD Loss: 0.0117 - Return: 141.0 - Avg Return: 128.1240 - Std Return: 46.8284\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
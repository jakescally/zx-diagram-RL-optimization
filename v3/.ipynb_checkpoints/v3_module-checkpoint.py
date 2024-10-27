# imports
import torch
import os
import pyzx as zx
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import numpy as np
from torch_geometric.nn import GCNConv
from statsmodels.nonparametric.smoothers_lowess import lowess
from torch_geometric.data import Data
import time
import datetime
import sys

# Set the number of threads PyTorch can use for CPU operations
torch.set_num_threads(16)  # Set this to the number of threads you want to use

# Optionally, set the number of interop threads (for inter-op parallelism)
# torch.set_num_interop_threads(4)

# misc checking functions

def is_hadamard_edge(g, edge):
    return g.edge_type(edge) == zx.EdgeType.HADAMARD

def is_hadamard_edgeV(g, v1, v2):
    if g.connected(v1, v2):
        edge = g.edge(v1, v2)
        return g.edge_type(edge) == zx.EdgeType.HADAMARD
    else:
        return False

def can_fuse(g, edge):
    v1, v2 = g.edge_st(edge)
    return g.type(v1) == g.type(v2)

def get_pivot_verts(g):
    vertex_set = set()
    matches = zx.rules.match_pivot_parallel(g)
    for match in matches:
        v1, v2, _, _ = match
        vertex_set.add(v1)
        vertex_set.add(v2)

    return vertex_set

def get_fusion_verts(g):
    fusions = set()
    # loop through all EDGES in the graph
    for edge in g.edges():
        # get the vertices of the current edge
        v1, v2 = edge
        type1 = g.type(v1)
        type2 = g.type(v2) # note the types of vertex
        # make sure the edge isn't hadamard
        if g.edge_type(edge) != zx.EdgeType.HADAMARD:
            # make sure neither vertex is a boundary
            if type1 != zx.VertexType.BOUNDARY and type2 != zx.VertexType.BOUNDARY:
                # check if they can fuse (same spider type)
                if type1 == type2:
                    fusions.add(v1)
                    fusions.add(v2)

    return fusions
    
def get_lcomp_verts(g):
    matches = zx.rules.match_lcomp_parallel(g)
    vertex_set = set()

    for match in matches:
        v1, neighbors = match 
        vertex_set.add(v1)
        for vert in neighbors:
            vertex_set.add(vert)

    return list(vertex_set)

def get_bialg_verts(g):
    matches = zx.rules.match_bialg_parallel(g)
    vertex_set = set()
    for match in matches:
        v1, v2, _, _ = match
        vertex_set.add(v1)
        vertex_set.add(v2)

    return list(vertex_set)

def get_supp_verts(g):
    matches = zx.rules.match_supplementarity(g)
    vertex_set = set()
    for match in matches:
        v1, v2, _, _ = match
        vertex_set.add(v1)
        vertex_set.add(v2)
    return vertex_set

def get_fusion_vert_pairs(g: zx.Graph):
    fusions = set()
    # loop through all EDGES in the graph
    for edge in g.edges():
        # get the vertices of the current edge
        v1, v2 = edge
        type1 = g.type(v1)
        type2 = g.type(v2) # note the types of vertex
        # make sure the edge isn't hadamard
        if g.edge_type(edge) != zx.EdgeType.HADAMARD:
            # make sure neither vertex is a boundary
            if type1 != zx.VertexType.BOUNDARY and type2 != zx.VertexType.BOUNDARY:
                # check if they can fuse (same spider type)
                if type1 == type2:
                    fusions.add((v1,v2))
    
    return fusions

def color_change(g, v):
    vtype = g.type(v)
    if vtype == zx.VertexType.BOUNDARY:
        return False
    neighbors = g.neighbors(v)
    for neighbor in neighbors:
        edge = g.edge(v, neighbor)
        etype = g.edge_type(edge)
        if etype == zx.EdgeType.HADAMARD:
            g.set_edge_type(edge, zx.EdgeType.SIMPLE)
        elif etype == zx.EdgeType.SIMPLE:
            g.set_edge_type(edge, zx.EdgeType.HADAMARD)
    if vtype == zx.VertexType.X:
        g.set_type(v, zx.VertexType.Z)
        return True
    elif vtype == zx.VertexType.Z:
        g.set_type(v, zx.VertexType.X)
        return True
    else:
        return False


# graph representation class
class ZXGraphRepresentation:
    def __init__(self, g: zx.Graph):
        # super(ZXGraphRepresentation, self).__init__()
        self.g = g
        self.node_features = self.get_node_features()
        self.edge_features, self.edge_index = self.get_edge_features()
        self.action_features = self.get_action_features()

    def get_node_features(self):
        node_features = []

        pivots = get_pivot_verts(self.g)
        lcomps = get_lcomp_verts(self.g)
        bialgs = get_bialg_verts(self.g)
        fusions = get_fusion_verts(self.g)
        supps = get_supp_verts(self.g)

        # debugging
        # print(f"length of g.verices {len(self.g.vertices())}")
        # print(f"g.vertices {self.g.vertices}")

        length = 0
        vertices_test = self.g.vertices()
        # print(f"vertices right before looping: {vertices_test}")
        for v in self.g.vertices():
            feature_vect = [0]*12 # [node_type, degree, phase, hadamard_count, fuse_count, pivot_possible, lcomp_possible, bialg_possible, fusion_possible, supp_involved, color_changeable, stop]
    
            # set node type 0 for Z, 1 for X, 2 for boundary
            if self.g.type(v) == zx.VertexType.Z:
                feature_vect[0] = 0
            elif self.g.type(v) == zx.VertexType.X:
                feature_vect[0] = 1
            elif self.g.type(v) == zx.VertexType.BOUNDARY:
                feature_vect[0] = 2
            else:
                feature_vect[0] = 9
    
            feature_vect[1] = len(self.g.neighbors(v))
            feature_vect[2] = self.g.phase(v)

            hadamard_count = sum(is_hadamard_edgeV(self.g, v, neighbor) for neighbor in self.g.neighbors(v))
            feature_vect[3] = hadamard_count

            fuse_count = 0
            for neighbor in self.g.neighbors(v):
                if self.g.type(v) == self.g.type(neighbor):
                    fuse_count += 1

            feature_vect[4] = fuse_count

            feature_vect[5] = 1 if v in pivots else 0
            feature_vect[6] = 1 if v in lcomps else 0
            feature_vect[7] = 1 if v in bialgs else 0
            feature_vect[8] = 1 if v in fusions else 0
            feature_vect[9] = 1 if v in supps else 0
            feature_vect[10] = 1
            feature_vect[11] = 1
    
            node_features.append(feature_vect)
            length+=1

        # print(f"length: {length}")
        out = torch.tensor(node_features, dtype=torch.float)
        # print(f"shape of ZXGR feature tensor: {out.shape}")
        # print(f"node feature tensor in ZXGR: {out}")
        return out

    def get_edge_features(self):
        edge_features = []
        edge_indices = []
    
        for edge in self.g.edges():
            source, target = self.g.edge_st(edge)
    
            feature_vect = [0]*2 # [Hadamard, Fuseable]
            feature_vect[0] = 1 if is_hadamard_edge(self.g, edge) else 0
            feature_vect[1] = 1 if can_fuse(self.g, edge) else 0
    
            edge_features.append(feature_vect)
            edge_indices.append([source,target])
    
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # print(f"EDGE INDICES TENSOR: {edge_indices_tensor}")
        # print(f"EDGE FEATURES TENSOR: {edge_features_tensor}")
    
        return edge_features_tensor, edge_indices_tensor

    def get_action_features(self):
        action_features = []

        pivot_set = get_pivot_verts(self.g)
        lcomp_set = get_lcomp_verts(self.g)
        bialg_set = get_bialg_verts(self.g)
        
        for v in self.g.vertices():
            feature_vect = [0]*3 # [pivot, lcomp, bialgebra]
            feature_vect[0] = 1 if v in pivot_set else 0
            feature_vect[1] = 1 if v in lcomp_set else 0
            feature_vect[2] = 1 if v in bialg_set else 0
            
            action_features.append(feature_vect)
    
        return torch.tensor(action_features, dtype=torch.float)

    def get_graph_data(self):
        return Data(x=self.node_features, edge_index=self.edge_index)


# define the environment
class ZXEnv():
    def __init__(self, g: zx.Graph, num_qubits, num_gates, MAX_STEPS=100):
        self.g = g
        self.MAX_STEPS = MAX_STEPS
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        self.current_step = 0
        self.done = False
        self.possible_actions = self._get_possible_actions()
        self.num_actions = len(self.possible_actions)

    def reset(self):
        # resets the environment to the initial state, which in this case will be a random zx graph
        self.current_step = 0
        self.done = False
        self.g = zx.generate.CNOT_HAD_PHASE_circuit(self.num_qubits, self.num_gates).to_graph()
        self.possible_actions = self._get_possible_actions()
        self._update_action_space()
        
        return self.g

    def _get_possible_actions(self):
        actions = []
        # put together all possible actions and matches for each step
        # pivot, lcomp, bialgebra, fusion, supp, color change, [unfuse?]
        
        p_matches = zx.rules.match_pivot_parallel(self.g)
        for match in p_matches:
            actions += [('pivot', match)]

        lc_matches = zx.rules.match_lcomp_parallel(self.g)
        for match in lc_matches:
            actions += [('lcomp', match)]

        bialg_matches = zx.rules.match_bialg_parallel(self.g)
        for match in bialg_matches:
            actions += [('bialg', match)]

        fusion_matches = get_fusion_vert_pairs(self.g)
        for match in fusion_matches:
            actions += [('fusion', match)]

        supp_matches = zx.rules.match_supplementarity(self.g)
        for match in supp_matches:
            actions += [('supp', match)]

        for v in self.g.vertices():
            actions += [('color', v)]
            # actions += [('unfuse', v)]

        actions += [('stop', None)]

        return actions

    def step(self, action):
        past_node_count = len(self.g.vertices())
        action_type, match = action

        if action_type == 'pivot':
            zx.rules.apply_rule(self.g, zx.rules.pivot, [match])
        elif action_type == 'lcomp':
            zx.rules.apply_rule(self.g, zx.rules.lcomp, [match])
        elif action_type == 'bialg':
            zx.rules.apply_rule(self.g, zx.rules.bialg, [match])
        elif action_type == 'fusion':
            zx.rules.apply_rule(self.g, zx.rules.spider, [match])
        elif action_type == 'supp':
            zx.rules.apply_rule(self.g, zx.rules.apply_supplementarity, [match])
        elif action_type == 'color':
            color_change(self.g, match)
    

        zx.rules.apply_rule(self.g, zx.rules.remove_ids, zx.rules.match_ids_parallel(self.g))
        zx.rules.apply_rule(self.g, zx.rules.remove_ids, zx.rules.match_ids_parallel(self.g))
        
        reward = past_node_count - len(self.g.vertices())
        self.current_step += 1

        if self.current_step >= self.MAX_STEPS:
            self.done = True

        return self.g, reward, self.done, {}

    def _update_action_space(self):
        self.num_actions = len(self.possible_actions)

    def render(self):
        labels = {v: f"Vertex {i}" for i, v in enumerate(self.g.vertices())}
        zx.draw(self.g, labels=labels)
        
    def close(self):
        pass        

# single gcn layer class
class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        # define a gcn convolution layer
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x is a node feature matrix, of dimensions [num_nodes, node_features]
        # edge_index is a tensor that tells the GCN which nodes are connected
        # and has dimensions [2, num_edges]

        # a first layer aggregates features from its direct neighbors
        # a second layer would be aggregating features from neighbors of neighbors
        # and so on
        
        # perform a convolution
        # print(f"Shape of x: {x.shape}")
        # print(f"x itself: {x}:")
        # print(f"Shape of edge_index: {edge_index.shape}")
        # print(f"edge_index itself: {edge_index}")
        x = self.conv(x, edge_index)
        # apply a non-linear activation function
        x = F.relu(x)
        return x


# gcn model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # create list of GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNLayer(input_dim, hidden_dim))
        # add middle (hidden) layers
        for _ in range(1, num_layers-1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
        # add final layer
        self.convs.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # convolve the nodes and edges
            if i != self.num_layers - 1:
                x = F.relu(x) # activation function on the last layer

        return x


class GCNPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCNPolicyNetwork, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim, output_dim, num_layers)
        self.fc = torch.nn.Linear(output_dim, output_dim) # we want 6 logits (one for each possible action)

    def forward(self, graph_data):
        # print(f"Graph data: {graph_data.x}, graph_data.edge_index: {graph_data.edge_index}")
        node_embeddings = self.gcn(graph_data.x, graph_data.edge_index)
        action_logits = self.fc(node_embeddings)
        action_logits = torch.softmax(action_logits, dim=1)
        return action_logits


def mask_logits(logits, zx_graph_rep: ZXGraphRepresentation):
    action_features = zx_graph_rep.node_features[:, 5:12] # action features should be [pivot?, lcomp?, bialgebra?, fusion?, supp?, color?, stop?] color is always possible ofc
    masked_logits = logits.masked_fill(action_features == 0, 0)
    return masked_logits

def get_random_reward(env):
    # get initial node count
    start_node_count = len(env.g.vertices())
    for step in env.MAX_STEPS:
        valid_actions = env.possible_actions
        random_action = random.choice(valid_actions)
        env.step(random_action)
    end_node_count = len(env.g.vertices())
    return(start_node_count-end_node_count)

def reinforce(env, policy_net, optimizer, num_episodes, name, gamma=0.99, verbose=False, save_results=True, comparative=False):
    # check if the folder exists first
    if save_results:
        directory = os.path.dirname(f"results/{name}/figs/")
        os.makedirs(directory)
    
    num_qubits = env.num_qubits
    num_gates = env.num_gates

    start_time = time.time()
    
    results = []
    pyzx_results = []

    rl_wins = 0
    rl_ties = 0
    rl_losses = 0
    
    for episode in range(num_episodes):
        state = ZXGraphRepresentation(env.reset())

        if comparative:
            original_graph = state.g.copy()
            original_verts = len(original_graph.vertices())
            zx.full_reduce(original_graph)
            pyzx_reward = original_verts - len(original_graph.vertices())
        
        log_probs = []
        rewards = []

        done = False
        epsilon = 1e-8
        while not done:
            action_probs = policy_net(state.get_graph_data())
            # print(f"action_probs: {action_probs}")
            # env.render()
            # mask impossible actions
            masked_logits = mask_logits(action_probs, state)
            # print(f"masked_logits: {masked_logits}")
            # here we want to flatten the tensor into 1D
            flattened_probs = torch.flatten(masked_logits)
            # then normalize all of the probabilities
            # check to see if there are no valid actions
            probsum = torch.sum(flattened_probs)
            if probsum != 0:
                normalized_probs = flattened_probs / torch.sum(flattened_probs)

                normalized_probs = torch.clamp(normalized_probs, min=0.0)  # Clamp negative values to 0
                normalized_probs[torch.isnan(normalized_probs)] = epsilon  # Replace NaNs with epsilon
                normalized_probs[torch.isinf(normalized_probs)] = epsilon  # Replace infs with epsilon
                
                # Re-normalize after the corrections to ensure valid probabilities
                normalized_probs = normalized_probs / torch.sum(normalized_probs + epsilon)
                # print(f"normalized_probs: {normalized_probs}")
                action_index = torch.multinomial(normalized_probs, 1).item()
                node_index = action_index // 7
                action_type = action_index % 7 # [0: pivot, 1: lcomp, 2: bialg, 3: fusion, 4: supp, 5: color, 6: stop]
    
                # map actions to action data
                action_map = {0: 'pivot', 1: 'lcomp', 2: 'bialg', 3: 'fusion', 4: 'supp', 5: 'color', 6: 'stop'}
                valid_actions = env.possible_actions
                selected_action = None
                for action, action_data in valid_actions:
                    if action == action_map[action_type]:
                        if action == 'pivot':
                            # a pivot match is a 4-tuple
                            if node_index in action_data:
                                selected_action = (action, action_data)
                                break
                        elif action == 'lcomp':
                            # an lcomp match is (vertex, neighbors)
                            temp_vert, _ = action_data
                            if temp_vert == node_index:
                                selected_action = (action, action_data)
                                break
                        elif action == 'bialg':
                            # a bialg match is a 4-tuple (v1, v2, neighbors_of_v1,neighbors_of_v2)
                            v1, v2, _, _ = action_data
                            if v1 == node_index or v2 == node_index:
                                selected_action = (action, action_data)
                                break
                        elif action == 'fusion':
                            # a fusion match is just a tuple
                            if node_index in action_data:
                                selected_action = (action, action_data)
                                break
                        elif action == 'supp':
                            # a supp match is a 4-tuple
                            v1, v2, _, _ = action_data
                            if v1 == node_index or v2 == node_index:
                                selected_action = (action, action_data)
                                break
                        elif action == 'color':
                            # color match is a single vertex
                            if node_index == action_data:
                                selected_action = (action, action_data)
                                break
                        elif action == 'stop':
                            selected_action = 'stop'
                            break

                # print(f"SELECTED ACTION: {selected_action}")
                if selected_action != None and selected_action != 'stop':
                    # env.render()
                    new_g, reward, done, _ = env.step(selected_action)
                    # copy the current step as well
                    current_step = env.current_step
                    env = ZXEnv(new_g.copy(),num_qubits,num_gates)
                    env.current_step = current_step
        
                    log_prob = torch.log(normalized_probs[action_index])
                    log_probs.append(log_prob)
                    rewards.append(reward)
        
                    state = ZXGraphRepresentation(new_g.copy())
                    #print("Updated state")
                elif selected_action == 'stop':
                    reward = 0
                    log_prob = torch.log(normalized_probs[action_index])
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    done = True
                else:
                    done = True
            else:
                done = True
                #print(f"Done with episode {episode}")

        if comparative:
            rl_final_nodes = len(env.g.vertices())
            rl_total_node_difference = original_verts - rl_final_nodes
            if rl_total_node_difference > pyzx_reward:
                rl_wins += 1
            elif rl_total_node_difference == pyzx_reward:
                rl_ties += 1
            else:
                rl_losses += 1
        
        if sum(rewards) == 0:
            rewards.append(-1)
        
        # compute the discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r+gamma*R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards)

        if discounted_rewards.numel() > 1:  # Ensure there are enough elements
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        else:
            # If there's only one or zero elements, just skip normalization
            discounted_rewards = discounted_rewards - discounted_rewards.mean()  # Center the rewards but no division by std
        
        
        # compute the policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append((-log_prob * R).reshape(1))
        if len(policy_loss) > 0:
            policy_loss = torch.cat(policy_loss)
            # update the policy network
            optimizer.zero_grad()           # Clears the gradients from the previous iteration
            total_loss = policy_loss.sum()   # Sums all the individual losses
            total_loss.backward()            # Backpropagates the loss to compute gradients
            optimizer.step()                 # Updates the policy network's weights using the optimizer        
            
            # monitor
            total_reward = sum(rewards)
            if verbose:
                print(f"EPISODE: {episode}, TOTAL REWARD: {total_reward}, TOTAL LOSS: {total_loss.item()}")
            results.append({
                'episode': episode,
                'rewards': rewards,
                'log_probs': log_probs,
                'policy_loss': policy_loss,
                'total_reward': total_reward,
                'total_loss': total_loss.item()
            })
            if comparative:
                pyzx_results.append({
                    'episode': episode,
                    'reward': pyzx_reward
                })
                
        else:
            if verbose:
                print("Stopped too early. Skipping...")
        

        # handle outputs
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Calculate the average time per episode
        avg_time_per_episode = elapsed_time / episode if episode > 0 else elapsed_time
        
        # Estimate remaining time
        episodes_remaining = num_episodes - episode
        estimated_time_remaining = avg_time_per_episode * episodes_remaining
        
        # Convert elapsed time and estimated remaining time to readable format
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        remaining_str = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
        
        # Print the progress on the same line
        if not verbose:
            sys.stdout.write(f"\rEpisode: {episode}/{num_episodes} | "
                         f"Elapsed time: {elapsed_str} | "
                         f"Estimated time to completion: {remaining_str}")
        
    
        # Flush the output to ensure it is displayed immediately
        sys.stdout.flush()

    # note to self: add chosen action, possible actions, action probs...
    
    # save results
    if save_results:
        torch.save(policy_net.state_dict(), f"results/{name}/{name}.pth")
        with open(f"results/{name}/training_progression.txt", 'w') as f:
            for entry in results:
                f.write(f"Episode: {entry['episode']}, Rewards: {entry['rewards']}, Log Probs: {entry['log_probs']}, Total Reward: {entry['total_reward']}, Total Loss: {entry['total_loss']}\n")

        print(f"\nModel saved at results/{name}.")
    print("\n")
    if comparative:
        print(f"Wins: {rl_wins}, Ties: {rl_ties}, Losses: {rl_losses}, Win rate: {rl_wins / num_episodes}, Tie rate: {rl_ties / num_episodes}")
        return results, pyzx_results
    else:
        return results


# analyze the data
def create_plots(results, experiment_name, verbose=False, save_plots=True, comparative=False):
    plt.ioff()
    # Extract the data
    episodes = [result['episode'] for result in results]
    total_rewards = [result['total_reward'] for result in results]
    total_losses = [result['total_loss'] for result in results]
    
    # Create a figure for plotting
    plt.figure(figsize=(12, 6))
    
    # Plot total rewards over episodes
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    plt.plot(episodes, total_rewards, label='Total Reward', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Plot total losses over episodes
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.plot(episodes, total_losses, label='Total Loss', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Total Loss over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/raw_plot.png")
    if verbose:
        plt.show()

    # smoothed plots

    # Define a simple moving average function
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Apply moving average to smooth the data (window_size can be adjusted)
    smoothed_rewards = moving_average(total_rewards, window_size=10)
    smoothed_losses = moving_average(total_losses, window_size=10)
    
    # Adjust the episode range after smoothing (because the moving average reduces data points)
    smoothed_episodes = episodes[:len(smoothed_rewards)]
    
    # Plot the smoothed results
    plt.figure(figsize=(12, 6))
    
    # Smoothed rewards
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_episodes, smoothed_rewards, label='Smoothed Total Reward', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Smoothed Total Reward over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Smoothed losses
    plt.subplot(1, 2, 2)
    plt.plot(smoothed_episodes, smoothed_losses, label='Smoothed Total Loss', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Smoothed Total Loss over Episodes')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/smoothed_plot.png")
    if verbose:
        plt.show()


    # Assuming episodes, total_rewards, and total_losses are arrays
    episodes = np.array(episodes)
    rewards = np.array(total_rewards)
    losses = np.array(total_losses)
    
    # Lowess smoothing for rewards
    smoothed_rewards = lowess(rewards, episodes, frac=0.05)  # frac controls the smoothing span (0.05 = 5%)
    
    # Lowess smoothing for losses
    smoothed_losses = lowess(losses, episodes, frac=0.05)
    
    # Plotting the data with Lowess centerline
    plt.figure(figsize=(14, 8))
    
    # Plot for rewards with Lowess
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, label='Total Reward', color='blue', alpha=0.3)
    plt.plot(smoothed_rewards[:, 0], smoothed_rewards[:, 1], label='Lowess Centerline', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Lowess Centerline')
    plt.legend()
    plt.grid(True)
    
    # Plot for losses with Lowess
    plt.subplot(2, 1, 2)
    plt.plot(episodes, losses, label='Total Loss', color='orange', alpha=0.3)
    plt.plot(smoothed_losses[:, 0], smoothed_losses[:, 1], label='Lowess Centerline', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Total Loss with Lowess Centerline')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/lowess_centerline.png")
    if verbose:
        plt.show()

def create_plots_comp(results, pyzx_results, experiment_name, verbose=False, save_plots=True):
    plt.ioff()
    
    # Extract the data from results
    episodes = [result['episode'] for result in results]
    total_rewards = [result['total_reward'] for result in results]
    total_losses = [result['total_loss'] for result in results]
    
    # Extract the data from pyzx_results
    pyzx_episodes = [result['episode'] for result in pyzx_results]
    pyzx_rewards = [result['reward'] for result in pyzx_results]
    
    # Create a figure for plotting rewards and losses
    plt.figure(figsize=(12, 6))
    
    # Plot total rewards over episodes (both results and pyzx_results)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    plt.plot(episodes, total_rewards, label='Total Reward (RL)', color='blue')
    plt.plot(pyzx_episodes, pyzx_rewards, label='Total Reward (PyZX)', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Plot total losses over episodes (only for results, as pyzx_results has no loss)
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.plot(episodes, total_losses, label='Total Loss', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Total Loss over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/raw_plot_comparison.png")
    if verbose:
        plt.show()

    # Smoothed plots
    
    # Define a simple moving average function
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Apply moving average to smooth the data (window_size can be adjusted)
    smoothed_rewards = moving_average(total_rewards, window_size=10)
    smoothed_pyzx_rewards = moving_average(pyzx_rewards, window_size=10)
    smoothed_losses = moving_average(total_losses, window_size=10)
    
    # Adjust the episode range after smoothing
    smoothed_episodes = episodes[:len(smoothed_rewards)]
    smoothed_pyzx_episodes = pyzx_episodes[:len(smoothed_pyzx_rewards)]
    
    # Plot the smoothed results
    plt.figure(figsize=(12, 6))
    
    # Smoothed rewards (both RL and PyZX)
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_episodes, smoothed_rewards, label='Smoothed Total Reward (RL)', color='blue')
    plt.plot(smoothed_pyzx_episodes, smoothed_pyzx_rewards, label='Smoothed Total Reward (PyZX)', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Smoothed Total Reward over Episodes')
    plt.grid(True)
    plt.legend()
    
    # Smoothed losses (only for results, as pyzx_results has no loss)
    plt.subplot(1, 2, 2)
    plt.plot(smoothed_episodes, smoothed_losses, label='Smoothed Total Loss', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Smoothed Total Loss over Episodes')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/smoothed_plot_comparison.png")
    if verbose:
        plt.show()

    # LOWESS smoothing for rewards
    
    # Lowess smoothing for rewards in results and pyzx_results
    rewards = np.array(total_rewards)
    pyzx_rewards_array = np.array(pyzx_rewards)
    smoothed_rewards_lowess = lowess(rewards, episodes, frac=0.05)
    smoothed_pyzx_rewards_lowess = lowess(pyzx_rewards_array, pyzx_episodes, frac=0.05)
    
    # Lowess smoothing for losses in results
    losses = np.array(total_losses)
    smoothed_losses_lowess = lowess(losses, episodes, frac=0.05)
    
    # Plotting the data with Lowess centerline
    plt.figure(figsize=(14, 8))
    
    # Plot for rewards with Lowess (both RL and PyZX)
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, label='Total Reward (RL)', color='blue', alpha=0.3)
    plt.plot(pyzx_episodes, pyzx_rewards, label='Total Reward (PyZX)', color='green', alpha=0.3)
    plt.plot(smoothed_rewards_lowess[:, 0], smoothed_rewards_lowess[:, 1], label='Lowess Centerline (RL)', color='red')
    plt.plot(smoothed_pyzx_rewards_lowess[:, 0], smoothed_pyzx_rewards_lowess[:, 1], label='Lowess Centerline (PyZX)', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Lowess Centerline')
    plt.legend()
    plt.grid(True)
    
    # Plot for losses with Lowess (only for results)
    plt.subplot(2, 1, 2)
    plt.plot(episodes, losses, label='Total Loss', color='orange', alpha=0.3)
    plt.plot(smoothed_losses_lowess[:, 0], smoothed_losses_lowess[:, 1], label='Lowess Centerline (Loss)', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.title('Total Loss with Lowess Centerline')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"results/{experiment_name}/figs/lowess_centerline_comparison.png")
    if verbose:
        plt.show()


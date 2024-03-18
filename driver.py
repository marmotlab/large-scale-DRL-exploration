import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def writeToTensorBoard(writer, tensorboardData, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, value, policyLoss, qValueLoss, entropy, policyGradNorm, qValueGradNorm, log_alpha, alphaLoss, travel_dist, success_rate, explored_rate = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alphaLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=qValueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policyGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=qValueGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha'] 
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
        global_policy_net.to(device)
        global_q_net1.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
    weights_set.append(policy_weights)
    weights_set.append(q_net1_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(27):
        experience_buffer.append([])
    
    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)
            
            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            
            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    node_inputs_batch = torch.stack(rollouts[0]).to(device)
                    edge_inputs_batch = torch.stack(rollouts[1]).to(device)
                    current_inputs_batch = torch.stack(rollouts[2]).to(device)
                    node_padding_mask_batch = torch.stack(rollouts[3]).to(device)
                    edge_padding_mask_batch = torch.stack(rollouts[4]).to(device)
                    edge_mask_batch = torch.stack(rollouts[5]).to(device)
                    action_batch = torch.stack(rollouts[6]).to(device)
                    reward_batch = torch.stack(rollouts[7]).to(device)
                    done_batch = torch.stack(rollouts[8]).to(device).to(device)
                    next_node_inputs_batch = torch.stack(rollouts[9]).to(device)
                    next_edge_inputs_batch = torch.stack(rollouts[10]).to(device)
                    next_current_inputs_batch = torch.stack(rollouts[11]).to(device)
                    next_node_padding_mask_batch = torch.stack(rollouts[12]).to(device)
                    next_edge_padding_mask_batch = torch.stack(rollouts[13]).to(device)
                    next_edge_mask_batch = torch.stack(rollouts[14]).to(device)

                    critic_node_inputs_batch = torch.stack(rollouts[15]).to(device)
                    critic_edge_inputs_batch = torch.stack(rollouts[16]).to(device)
                    critic_current_inputs_batch = torch.stack(rollouts[17]).to(device)
                    critic_node_padding_mask_batch = torch.stack(rollouts[18]).to(device)
                    critic_edge_padding_mask_batch = torch.stack(rollouts[19]).to(device)
                    critic_edge_mask_batch = torch.stack(rollouts[20]).to(device)
                    critic_next_node_inputs_batch = torch.stack(rollouts[21]).to(device)
                    critic_next_edge_inputs_batch = torch.stack(rollouts[22]).to(device)
                    critic_next_current_inputs_batch = torch.stack(rollouts[23]).to(device)
                    critic_next_node_padding_mask_batch = torch.stack(rollouts[24]).to(device)
                    critic_next_edge_padding_mask_batch = torch.stack(rollouts[25]).to(device)
                    critic_next_edge_mask_batch = torch.stack(rollouts[26]).to(device)

                    # SAC
                    with torch.no_grad():
                        q_values1, _ = dp_q_net1(critic_node_inputs_batch, critic_edge_inputs_batch, critic_current_inputs_batch, critic_node_padding_mask_batch, critic_edge_padding_mask_batch, critic_edge_mask_batch)
                        q_values2, _ = dp_q_net2(critic_node_inputs_batch, critic_edge_inputs_batch, critic_current_inputs_batch, critic_node_padding_mask_batch, critic_edge_padding_mask_batch, critic_edge_mask_batch)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                    policy_loss = torch.sum((logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())), dim=1).mean()
                    
                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values1, _ = dp_target_q_net1(critic_next_node_inputs_batch, critic_next_edge_inputs_batch, critic_next_current_inputs_batch, critic_next_node_padding_mask_batch, critic_next_edge_padding_mask_batch, critic_next_edge_mask_batch)
                        next_q_values2, _ = dp_target_q_net2(critic_next_node_inputs_batch, critic_next_edge_inputs_batch, critic_next_current_inputs_batch, critic_next_node_padding_mask_batch, critic_next_edge_padding_mask_batch, critic_next_edge_mask_batch)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime_batch = torch.sum(next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_q_batch = reward_batch + GAMMA * (1 - done_batch) * value_prime_batch

                    mse_loss = nn.MSELoss()
                    
                    q_values1, _ = dp_q_net1(critic_node_inputs_batch, critic_edge_inputs_batch, critic_current_inputs_batch, critic_node_padding_mask_batch, critic_edge_padding_mask_batch, critic_edge_mask_batch)
                    q1 = torch.gather(q_values1, 1, action_batch)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000, norm_type=2)
                    global_q_net1_optimizer.step()         
                    
                    q_values2, _ = dp_q_net2(critic_node_inputs_batch, critic_edge_inputs_batch, critic_current_inputs_batch, critic_node_padding_mask_batch, critic_edge_padding_mask_batch, critic_edge_mask_batch)
                    q2 = torch.gather(q_values2, 1, action_batch)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000, norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1


                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_prime_batch.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(), alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
                global_policy_net.to(device)
                global_q_net1.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
            weights_set.append(policy_weights)
            weights_set.append(q_net1_weights)
            
            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()

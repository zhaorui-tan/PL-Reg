import os
import yaml
import argparse
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from feature_maps_manip import feature_maps_manip as fm
from metrics import partitioning_metrics as p_metrics
from sklearn.cluster import KMeans
from models.clustering_methods import torch_semi_supervised_kmeans as ssKM
from configs import config_K
from params_estim.ssKM_protos_initialization import ssKM_protos_init
from params_estim.automatic_lambda_search import lambda_search
from models.PIM import PIM_partitioner


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)


def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()


def partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, dset_name, path_k_strat):
    # GCD metrics (used in GCD arxiv v1)
    accs_v1 = p_metrics.cluster_acc_v1(unlab_gt_labs, unlab_preds, seen_mask)

    # GCD metrics (used in GCD CVPR-22 and GCD arxiv v2)
    accs_v2 = p_metrics.cluster_acc_v2(unlab_gt_labs, unlab_preds, seen_mask)

    # ORCA metrics (used in ORCA ICLR-22)
    orca_accs = p_metrics.orca_all_old_new_ACCs(unlab_preds, unlab_gt_labs, seen_mask)

    print("Classes:                      (All) & (Old) & (New)")

    print("PIM ACC (v1):                 ", np.round(100. * accs_v1[0], 1),
          "\t", np.round(100. * accs_v1[1], 1),
          "\t", np.round(100. * accs_v1[2], 1))

    print("PIM ACC (ORCA metric):        ", np.round(100. * orca_accs[0], 1),
          "\t ", np.round(100. * orca_accs[1], 1),
          "\t", np.round(100. * orca_accs[2], 1))

    print("PIM ACC (Official GCD metric):", np.round(100. * accs_v2[0], 1),
          "\t", np.round(100. * accs_v2[1], 1),
          "\t", np.round(100. * accs_v2[2], 1))

    path_accs_v2 = 'params_estim/' + path_k_strat + '/' + dset_name + '/scores'
    if not os.path.exists(path_accs_v2):
        os.makedirs(path_accs_v2)

    accs_v2_file_name = path_accs_v2 + '/ACCs_v2.npy'
    with open(accs_v2_file_name, 'wb') as f:
        np.save(f, accs_v2)
    return 1


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for GCD challenge")

    parser.add_argument('--dataset', type=str, default='cub',
                        choices=['cifar10', 'cifar100', 'imagenet_100',
                                 'cub', 'scars', 'herbarium'])

    parser.add_argument('--device_name', type=str, default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    parser.add_argument('--centroids_init_nbr', type=int, default=100,
                        help="Number of separate centroids initialization.")

    parser.add_argument('--cfg', type=str, default="./configs/config_fm_paths.yml",
                        help="feature maps roots config file")

    parser.add_argument('--epochs', type=int, default=1000,
                        help="Total number of training epoch iterations for PIM partitioner.")

    parser.add_argument('--perform_init_protos', type=bool, default=False,
                        help="Set True if you want to (re)perform prototypes initiliaztion using ssKM.")

    parser.add_argument('--perform_lambda_search', type=bool, default=False,
                        help="Set True if you want to (re)perform automatic lambda search.")

    parser.add_argument('--seed', type=int, default=2022,
                        help="Seed for RandomState.")

    parser.add_argument('--k_strat', type=str, default='ground_truth_K',
                        choices=['ground_truth_K',
                                 'Max_ACC_PIM_Brent'],
                        help="Select a strategy to set the assumed number of classes " +
                             "(ground truth or precomputed using Max-ACC).")


    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
    parser.add_argument('--alpha', default=0.75, type=float)
    return parser.parse_args()


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    print('nu', nu)
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    print(xy)
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    for v in xy:
        print(len(v))
    return [torch.cat(v, dim=0) for v in xy]


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk
        self.imb_factor = args.imb_factor

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # Samples

        if self.imb_factor > 1:
            # obtain permutation/order from the marginals
            marginals_argsort = torch.argsort(Q.sum(1))
            marginals_argsort = marginals_argsort.detach()
            r = []
            for i in range(Q.shape[0]):  # Classes
                r.append((1 / self.imb_factor) ** (i / (Q.shape[0] - 1.0)))
            r = np.array(r)
            r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
            r = torch.from_numpy(r).cuda(non_blocking=True)
            r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
            r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
            r /= r.sum()  # Scaling to make it prob
        else:
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)


def main():
    args = get_arguments()
    with open(args.cfg, "r") as stream:
        try:
            cfg_paths = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    torch_device = torch.device(args.device_name)
    np.random.seed(args.seed)

    ### Get feature map set, ground truth labels, labeled set mask, old-new mask
    print("Dataset:", args.dataset)
    gcd_feature_map_datasets_path = cfg_paths["gcd_feature_map_datasets_path"]
    feature_maps_folder_name = cfg_paths[args.dataset]
    set_path = os.path.join(gcd_feature_map_datasets_path, feature_maps_folder_name, 'train')
    feature_map_preds, gt_labels, nbr_of_classes, mask_old_new, mask_lab = fm.get_fm_preds_and_gt_labels(set_path)
    ###

    ### Get assumed K
    assumed_k, path_k_strat = config_K.get_assumed_nbr_of_classes(args.k_strat, args.dataset)
    print('K strategy:', args.k_strat, '\nAssumed K:', assumed_k, '\n')
    path_params = '/Data_PHD/phd22_zhaorui_tan/ssl/pim-generalized-category-discovery-main/params_estim/' + path_k_strat + '/' + args.dataset
    if not os.path.exists(path_params):
        os.makedirs(path_params)
    ###

    ### Get ssKM centroids to initialize PIM prototypes
    protos_file_name = path_params + '/' + 'protos.npy'
    if ((not os.path.exists(protos_file_name)) or (args.perform_init_protos == True)) and (
            args.k_strat != 'ground_truth_K'):
        # Estimate ssKM centroids
        print("PIM init prototypes estimation using ssKM...")
        prototypes = ssKM_protos_init(feature_map_preds,
                                      assumed_k,
                                      gt_labels[mask_lab],
                                      mask_lab,
                                      mask_old_new,
                                      args.centroids_init_nbr,
                                      args.device_name,
                                      protos_file_name)
    else:
        print("PIM init prototypes already generated \n")
        with open(protos_file_name, 'rb') as f:
            prototypes = np.load(f)
    prototypes = from_numpy_to_torch(np.asarray(prototypes), torch_device)
    ###

    ### Get lambda (automatically estimated)
    path_auto_lambda = path_params + '/auto_lambda_search'
    lambda_search_lab_acc_file_name = path_auto_lambda + '/' + "all_lab_Accs_" + str(assumed_k) + ".npy"
    if (not os.path.exists(lambda_search_lab_acc_file_name)) or (args.perform_lambda_search == True):
        # Auto lambda search
        print("Start automatic lambda search...")
        auto_lambda_val = lambda_search(path_auto_lambda,
                                        feature_map_preds,
                                        args.epochs,
                                        assumed_k,
                                        mask_lab,
                                        args.device_name,
                                        args.dataset,
                                        torch_device,
                                        gt_labels[mask_lab])
    else:
        print("lambda already estimated")
        with open(path_auto_lambda + '/' + 'lambda_vals_list_' + str(assumed_k) + '.npy', 'rb') as f:
            lambda_vals_list = np.load(f)
        with open(path_auto_lambda + '/' + 'all_lab_Accs_' + str(assumed_k) + '.npy', 'rb') as f:
            all_lab_Accs = np.load(f)
        auto_lambda_val = lambda_vals_list[1:][np.argmax(all_lab_Accs[1:])]
    print("Obtained lambda:", auto_lambda_val, "\n")
    ###

    ### PIM partitioner
    # Init model and prototypes
    pim = PIM_partitioner(num_features=len(feature_map_preds[0]), num_classes=assumed_k).to(args.device_name)
    for name, param in pim.named_parameters():
        if name == "partitioner.weight":
            pim.partitioner.weight.data = prototypes.type_as(param)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pim.parameters(), lr=0.001, weight_decay=1e-2)

    # PIM training
    print('PIM training...')
    mb_size = len(feature_map_preds)

    mb_gt_labels = gt_labels
    mb_lab_mask = mask_lab
    mb_lab_points = mb_gt_labels[mb_lab_mask]
    no_seen_N = assumed_k -  mb_lab_points.max() -1
    print('no_seen_N', no_seen_N)
    sinkhorn = SinkhornKnopp(args)
    for epoch in range(args.epochs):  # loop over the feature map set multiple times
        running_loss = 0.0

        for mb_id in range(0, int(len(feature_map_preds) / mb_size)):

            # with torch.no_grad():
            #     if torch.cuda.device_count() > 1:
            #         w = pim.module.partitioner.weight.data.clone()
            #         w = F.normalize(w, dim=1, p=2)
            #         pim.module.partitioner.weight.copy_(w)
            #     else:
            #         w = pim.partitioner.weight.data.clone()
            #         w = F.normalize(w, dim=1, p=2)
            #         pim.partitioner.weight.copy_(w)

            # get the batch
            mb_inputs = from_numpy_to_torch(feature_map_preds[mb_id * mb_size:(mb_id + 1) * mb_size],
                                            torch_device).float()
            mb_gt_labels = from_numpy_to_torch(gt_labels[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device)
            mb_lab_mask = from_numpy_to_torch(mask_lab[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device)
            mb_lab_points = mb_gt_labels[mb_lab_mask]  # We only use labels information for Z_L subset
            y = F.one_hot(mb_lab_points.to(torch.int64), num_classes=assumed_k)
            optimizer.zero_grad()

            # with torch.no_grad():
            #     inputs_u = mb_inputs[~mb_lab_mask]
            #     outputs_u = pim(inputs_u)
            #     outputs_u = F.softmax(outputs_u, dim=1)
            #     targets_u = sinkhorn(outputs_u)
            #
            # targets_u = torch.argmax(targets_u, dim=1)
            # loss = 0.
            # logits = pim(mb_inputs)
            # preds = F.log_softmax(logits, dim=1)
            #
            # seen_preds = preds[mb_lab_mask]
            # unseen_preds = preds[~mb_lab_mask]
            # loss += - ((unseen_preds + 2.220446049250313e-16) * torch.log(
            #     unseen_preds + 2.220446049250313e-16)).sum(1).mean() * auto_lambda_val
            # loss += criterion(seen_preds, mb_lab_points) + criterion(unseen_preds, targets_u)
            # loss += (preds.mean(0) * torch.log(preds.mean(0) + 1e-12)).sum()
            #
            #
            #
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item()

            # with torch.no_grad():
            #     # outputs_u = soft_mb_logits_outputs[~mb_lab_mask]
            #     outputs_u = pim(mb_inputs[~mb_lab_mask])
            #
            #     targets_u = sinkhorn(outputs_u)
            #     print(targets_u)
            #     targets_u_novel = targets_u[:, no_seen_N:]
            #     max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
            #     hard_novel_idx1 = torch.where(max_pred_novel >= args.threshold)[0]
            #     targets_u[hard_novel_idx1] = targets_u[hard_novel_idx1].ge(args.threshold).float()
            #     targets_u = torch.softmax(targets_u, dim=1)

            mb_logits_outputs = pim(mb_inputs)
            soft_mb_logits_outputs = F.softmax(mb_logits_outputs, dim=1)
            loss = 0.
            with torch.no_grad():
                outputs_u = soft_mb_logits_outputs[~mb_lab_mask]
                # outputs_u = pim(mb_inputs[~mb_lab_mask])
                # outputs_u = torch.softmax(outputs_u, dim=1)
                targets_u = sinkhorn(outputs_u)

                # # print(targets_u)
                # targets_u = torch.softmax(targets_u, dim=1)
                targets_u_novel = targets_u[:, no_seen_N:]
                max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
                hard_novel_idx1 = torch.where(max_pred_novel >= args.threshold)[0]
                targets_u[hard_novel_idx1] = targets_u[hard_novel_idx1].ge(args.threshold).float()
                targets_u = torch.softmax(targets_u, dim=1)
                # print(targets_u)

            # l = np.random.beta(args.alpha, args.alpha)
            # idx = torch.randperm(all_inputs.size(0))

            soft_mb_logits_outputs_seen_preds = soft_mb_logits_outputs[mb_lab_mask]
            soft_mb_logits_outputs_unseen_preds = soft_mb_logits_outputs[~mb_lab_mask]
            mb_logits_outputs_seen_preds = mb_logits_outputs[mb_lab_mask]
            mb_logits_outputs_unseen_preds = mb_logits_outputs[~mb_lab_mask]

            # loss += - ((unseen_preds + 2.220446049250313e-16) * torch.log(
            #     unseen_preds + 2.220446049250313e-16)).sum(1).mean() * auto_lambda_val
            # loss += (soft_mb_logits_outputs.mean(0) * torch.log(soft_mb_logits_outputs.mean(0) + 1e-12)).sum()
            # loss += criterion(seen_preds, mb_lab_points.to(torch.int64)) * len(seen_preds)/len(soft_mb_logits_outputs)
            # loss += criterion(unseen_preds, torch.argmax(targets_u, dim=1).to(torch.int64)) * len(unseen_preds)/len(soft_mb_logits_outputs)
            # loss += criterion(seen_preds, mb_lab_points.to(torch.int64))

            loss = - ((soft_mb_logits_outputs_unseen_preds + 2.220446049250313e-16) * torch.log(
                soft_mb_logits_outputs_unseen_preds + 2.220446049250313e-16)).sum(1).mean() * auto_lambda_val
            loss += (soft_mb_logits_outputs.mean(0) * torch.log(soft_mb_logits_outputs.mean(0) + 1e-12)).sum()
            loss += criterion(mb_logits_outputs_seen_preds, mb_lab_points.to(torch.int64))
            loss += criterion(mb_logits_outputs_unseen_preds,  torch.argmax(targets_u, dim=1).to(torch.int64)) * len(mb_logits_outputs_seen_preds)/len(mb_logits_outputs_unseen_preds) * 0.1

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 100 == 0:
            print("  epoch:", epoch, ", loss:", running_loss)
            with torch.no_grad():
                outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
                _, predicted = torch.max(outputs.data, 1)
                unlab_preds = np.asarray(from_torch_to_numpy(predicted)[~mask_lab], dtype=int)
            unlab_gt_labs = np.asarray(gt_labels[~mask_lab], dtype=int)
            seen_mask = mask_old_new[~mask_lab].astype(bool)
            partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat)

    print('PIM training is completed! \n')

    # PIM partitioning evaluation
    with torch.no_grad():
        outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
        _, predicted = torch.max(outputs.data, 1)
        unlab_preds = np.asarray(from_torch_to_numpy(predicted)[~mask_lab], dtype=int)
    unlab_gt_labs = np.asarray(gt_labels[~mask_lab], dtype=int)
    seen_mask = mask_old_new[~mask_lab].astype(bool)
    partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat)
    ###


if __name__ == '__main__':
    main()
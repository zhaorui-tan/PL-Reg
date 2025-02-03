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

from feature_maps_manip import feature_maps_manip_sampling as fm
# from feature_maps_manip_sampling import feature_maps_manip as fm

from metrics import partitioning_metrics as p_metrics
from sklearn.cluster import KMeans
from models.clustering_methods import torch_semi_supervised_kmeans as ssKM
from configs import config_K
from params_estim.ssKM_protos_initialization import ssKM_protos_init
from params_estim.automatic_lambda_search import lambda_search
from models.PIM import PIM_partitioner
# from axiom.utils.metrics import save


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def sample_covariance(a, b, invert=False):
    """
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    """
    # assert (a.shape[0] == b.shape[0])
    # assert (a.shape[1] == b.shape[1])
    # m = a.shape[1]
    N = a.shape[0]
    C = torch.matmul(a.T, b) / N
    if invert:
        return torch.linalg.pinv(C)
    else:
        return C


def get_cond_shift(X1, Y1, estimator=sample_covariance):
    # print(matrix1.shape, matrix2.shape)
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    c_x1_y = estimator(x1, y1)
    c_y_x1 = estimator(y1, x1)
    # c_x1_x1 = estimator(x1, x1)

    inv_c_y_y = estimator(y1, y1, invert=True)
    # c_x1_given_y1 = c_x1_x1 - torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    # return nn.MSELoss()((c_x1_given_y1) , (c_x2_given_y2))
    shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    # return torch.trace(aa) + torch.trace(bb)
    return nn.MSELoss()(shift, torch.zeros_like(shift))


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

    print(
        "PIM ACC (v1):                 ",
        np.round(100.0 * accs_v1[0], 1),
        "\t",
        np.round(100.0 * accs_v1[1], 1),
        "\t",
        np.round(100.0 * accs_v1[2], 1),
    )

    print(
        "PIM ACC (ORCA metric):        ",
        np.round(100.0 * orca_accs[0], 1),
        "\t",
        np.round(100.0 * orca_accs[1], 1),
        "\t",
        np.round(100.0 * orca_accs[2], 1),
    )

    print(
        "PIM ACC (Official GCD metric):",
        np.round(100.0 * accs_v2[0], 1),
        "\t",
        np.round(100.0 * accs_v2[1], 1),
        "\t",
        np.round(100.0 * accs_v2[2], 1),
    )

    # path_accs_v2 = "params_estim/" + path_k_strat + "/" + dset_name + "/scores"
    # if not os.path.exists(path_accs_v2):
    #     os.makedirs(path_accs_v2)

    # accs_v2_file_name = path_accs_v2 + "/ACCs_v2.npy"
    # with open(accs_v2_file_name, "wb") as f:
    #     np.save(f, accs_v2)
    # return 1
    return {
        "all": np.round(100.0 * accs_v2[0], 1),
        "old": np.round(100.0 * accs_v2[1], 1),
        "new": np.round(100.0 * accs_v2[2], 1),
    }


def cluster_acc(y_true, y_pred, return_ind=False):
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for GCD challenge")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cub",
        choices=["cifar10", "cifar100", "imagenet_100", "cub", "scars", "herbarium"],
    )

    parser.add_argument(
        "--device_name", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"]
    )

    parser.add_argument(
        "--centroids_init_nbr",
        type=int,
        default=100,
        help="Number of separate centroids initialization.",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/config_fm_paths.yml",
        help="feature maps roots config file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Total number of training epoch iterations for PIM partitioner.",
    )

    parser.add_argument(
        "--perform_init_protos",
        type=bool,
        default=False,
        help="Set True if you want to (re)perform prototypes initiliaztion using ssKM.",
    )

    parser.add_argument(
        "--perform_lambda_search",
        type=bool,
        default=False,
        help="Set True if you want to (re)perform automatic lambda search.",
    )

    # parser.add_argument("--seed", type=int, default=2022, help="Seed for RandomState.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for RandomState.")
    parser.add_argument(
        "--k_strat",
        type=str,
        default="ground_truth_K",
        choices=["ground_truth_K", "Max_ACC_PIM_Brent"],
        help="Select a strategy to set the assumed number of classes "
        + "(ground truth or precomputed using Max-ACC).",
    )

    parser.add_argument(
        "--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn"
    )
    parser.add_argument(
        "--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn"
    )
    parser.add_argument(
        "--imb-factor",
        default=1,
        type=float,
        help="imbalance factor of the data, default 1",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="threshold for hard pseudo-labeling",
    )
    parser.add_argument("--alpha", default=0.75, type=float)
    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--ratio", default=0.1, type=float)
    parser.add_argument("--ema", default=None)
    parser.add_argument("--save-logits", default=False, type=float)
    parser.add_argument("--H_unlabelled", default=True, action="store_true")
    parser.add_argument("--H_labelled", default=True, action="store_true")
    parser.add_argument("--save-metrics", default="./metrics_test.csv", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)

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
    print("nu", nu)
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
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
            r = r * (
                Q.shape[1] / Q.shape[0]
            )  # Per-class distribution in the mini-batch
            r = torch.from_numpy(r).cuda(non_blocking=True)
            r[marginals_argsort] = torch.sort(r)[
                0
            ]  # Sort/permute based on the data order
            r = torch.clamp(
                r, min=1
            )  # Clamp the min to have a balance distribution for the tail classes
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


def main(wight_decay, lr = None):
    args = get_arguments()
    if lr:
        args.lr = lr
    if args.ema == "None":
        args.ema = None
    if args.ema is not None:
        args.ema = float(args.ema)
        print(f"EMA: {args.ema}")
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
    set_path = os.path.join(
        gcd_feature_map_datasets_path, feature_maps_folder_name, "train"
    )
    (
        feature_map_preds,
        gt_labels,
        nbr_of_classes,
        mask_old_new,
        mask_lab,
    ) = fm.get_fm_preds_and_gt_labels(set_path)
    ###

    ### Get assumed K
    assumed_k, path_k_strat = config_K.get_assumed_nbr_of_classes(
        args.k_strat, args.dataset
    )
    print("K strategy:", args.k_strat, "\nAssumed K:", assumed_k, "\n")
    path_params = "./params_estim/" + path_k_strat + "/" + args.dataset
    if not os.path.exists(path_params):
        os.makedirs(path_params)
    ###

    ### Get ssKM centroids to initialize PIM prototypes
    protos_file_name = path_params + "/" + "protos.npy"
    if (
        (not os.path.exists(protos_file_name)) or (args.perform_init_protos == True)
    ) and (args.k_strat != "ground_truth_K"):
        # Estimate ssKM centroids
        print("PIM init prototypes estimation using ssKM...")
        prototypes = ssKM_protos_init(
            feature_map_preds,
            assumed_k,
            gt_labels[mask_lab],
            mask_lab,
            mask_old_new,
            args.centroids_init_nbr,
            args.device_name,
            protos_file_name,
        )
    else:
        print("PIM init prototypes already generated \n")
        with open(protos_file_name, "rb") as f:
            prototypes = np.load(f)
    prototypes = from_numpy_to_torch(np.asarray(prototypes), torch_device)
    ###

    ### Get lambda (automatically estimated)
    path_auto_lambda = path_params + "/auto_lambda_search"
    lambda_search_lab_acc_file_name = (
        path_auto_lambda + "/" + "all_lab_Accs_" + str(assumed_k) + ".npy"
    )
    if (not os.path.exists(lambda_search_lab_acc_file_name)) or (
        args.perform_lambda_search == True
    ):
        # Auto lambda search
        print("Start automatic lambda search...")
        auto_lambda_val = lambda_search(
            path_auto_lambda,
            feature_map_preds,
            args.epochs,
            assumed_k,
            mask_lab,
            args.device_name,
            args.dataset,
            torch_device,
            gt_labels[mask_lab],
        )
    else:
        print("lambda already estimated")
        with open(
            path_auto_lambda + "/" + "lambda_vals_list_" + str(assumed_k) + ".npy", "rb"
        ) as f:
            lambda_vals_list = np.load(f)
        with open(
            path_auto_lambda + "/" + "all_lab_Accs_" + str(assumed_k) + ".npy", "rb"
        ) as f:
            all_lab_Accs = np.load(f)
        auto_lambda_val = lambda_vals_list[1:][np.argmax(all_lab_Accs[1:])]
    print("Obtained lambda:", auto_lambda_val, "\n")
    ###

    ### PIM partitioner
    # Init model and prototypes
    # if args.dataset == 'imagenet_100':
    #     r = 0.
    # elif args.dataset == 'cub':
    #     r = 0.5 # 0.5
    # elif args.dataset == 'herbarium':
    #     r = 1.
    # else:
    #     r = 1.

    # if args.dataset == "scars":
    #     r = 1.0
    # elif args.dataset == "cub":
    #     # r = 0.5
    #     r = 1.0
    # elif args.dataset == "herbarium":
    #     r = 1.0
    # elif args.dataset == "cifar100":
    #     r = 1.0
    # elif args.dataset == "cifar10":
    #     r = 1.0
    # elif args.dataset == "imagenet_100":
    #     r = 1.0
    # else:
    #     r = 1.0
    r = 1.0

    pim = PIM_partitioner(
        num_features=len(feature_map_preds[0]), num_classes=assumed_k, r=r
    ).to(
        args.device_name
    )  # imagenet 0.0, cub, 0.5,   others 1
    print(f"using r={r}")
    ###################################
    # image_net100
    # pim = PIM_partitioner(num_features=len(feature_map_preds[0]),
    #                       num_classes=assumed_k,
    #                       r=0).to(args.device_name)
    ###################################
    if args.ema is not None:
        ema = EMA(pim, decay=args.ema)

    for name, param in pim.named_parameters():
        if name == "partitioner.weight":
            pim.partitioner.weight.data = prototypes.type_as(param)

    # Optimizer
    criterion = nn.CrossEntropyLoss()

    # if args.weight_decay is not None:
    #     r = args.weight_decay

    # else:
    #     if args.dataset == "scars":
    #         r = 1e-2
    #     elif args.dataset == "cub":
    #         # r = 0.5
    #         r = 1e-2
    #     elif args.dataset == "herbarium":
    #         r = 2e-2  # 0.015
    #     elif args.dataset == "cifar100":
    #         r = 2e-3
    #     elif args.dataset == "cifar10":
    #         r = 1e-2
    #     elif args.dataset == "imagenet_100":
    #         r = 2e-3
    #     else:
    #         r = 1e-2
    #

    optimizer = optim.Adam(pim.parameters(), lr=args.lr, weight_decay=wight_decay)

    # PIM training
    print("PIM training...")
    mb_size = len(feature_map_preds)

    all_labels = gt_labels
    seen_mask = mask_lab
    seen_labels = all_labels[seen_mask]
    # TODO selecting validation data set from seen data
    # print(f'seen sum: {seen_mask.astype(int).sum()}')
    all_filtered_seen_mask, all_valid_mask,  valid_mask_seen_mask, valid_mask_unseen_mask, labeled_targets  = fm.get_valid_mask_and_filtered_seen_mask(
        set_path
    )
    # feature_map_preds = labeled_targets
    # print(f'all_filtered_seen_mask all_valid_mask: {all_filtered_seen_mask.astype(int).sum() + all_valid_mask.astype(int).sum()}')


    seen_N = seen_labels.max() + 1
    print("no_seen_N", seen_N)
    sinkhorn = SinkhornKnopp(args)

    for epoch in range(args.epochs):  # loop over the feature map set multiple times
        running_loss = 0.0

        for mb_id in range(0, int(len(feature_map_preds) / mb_size)):
            mb_inputs = from_numpy_to_torch(
                feature_map_preds[mb_id * mb_size : (mb_id + 1) * mb_size], torch_device
            ).float()  # z
            all_labels = from_numpy_to_torch(
                gt_labels[mb_id * mb_size : (mb_id + 1) * mb_size], torch_device
            )
            seen_mask = from_numpy_to_torch(
                mask_lab[mb_id * mb_size : (mb_id + 1) * mb_size], torch_device
            )

            valid_mask = from_numpy_to_torch(
                all_valid_mask[mb_id * mb_size : (mb_id + 1) * mb_size], torch_device
            )
            filtered_seen_mask = from_numpy_to_torch(
                all_filtered_seen_mask[mb_id * mb_size : (mb_id + 1) * mb_size],
                torch_device,
            )
            # valid_mask = from_numpy_to_torch(
            #     valid_mask[mb_id * mb_size : (mb_id + 1) * mb_size], torch_device
            # )

            seen_labels = all_labels[
                seen_mask
            ]  # We only use labels information for Z_L subset, y^L

            valid_labels = all_labels[valid_mask]  # constructed unlabelled data

            filtered_seen_labels = all_labels[
                filtered_seen_mask
            ]  # constructed seen labelled data
            optimizer.zero_grad()

            mb_logits_outputs = pim(mb_inputs)
            soft_mb_logits_outputs = F.softmax(mb_logits_outputs, dim=1)
            loss = 0.0

            soft_mb_logits_outputs_seen_preds = soft_mb_logits_outputs[
                filtered_seen_mask
            ]  # bar(logits)^L
            soft_mb_logits_outputs_unseen_preds = soft_mb_logits_outputs[
                valid_mask
            ]  # bar(logits)^U
            mb_logits_outputs_seen_preds = mb_logits_outputs[
                filtered_seen_mask
            ]  # bar(y)^L
            mb_logits_outputs_unseen_preds = mb_logits_outputs[
                valid_mask
            ]  # bar(y)^U







            with torch.no_grad():
                # TODO: 这里好像有点问题
                # TODO: 这里筛选了unseen的label计算了他们的confidence

                targets_u = sinkhorn(soft_mb_logits_outputs_unseen_preds)
                targets_u_novel = targets_u[:, :]
                max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
                hard_idx = torch.where(max_pred_novel >= args.threshold)[0]
                hard_novel_idx = hard_idx[torch.where(hard_idx >= seen_N)]
                hard_u_cond_l_index = hard_idx[torch.where(hard_idx < seen_N)]
                # targets_u[hard_idx] = targets_u[hard_idx].ge(args.threshold).float()
                targets_u = torch.softmax(targets_u, dim=1)

                smoothed_ratio = 0.05  # 0.05
                l = F.one_hot(
                    filtered_seen_labels.to(torch.int64), num_classes=assumed_k
                )
                targets_l = sinkhorn(soft_mb_logits_outputs_seen_preds)
                targets_l = torch.softmax(targets_l, dim=1)
                targets_l = l * (1 - smoothed_ratio) + targets_l * smoothed_ratio

            # h_u_loss = - ((soft_mb_logits_outputs_unseen_preds + 2.220446049250313e-16) * torch.log(
            # soft_mb_logits_outputs_unseen_preds  + 2.220446049250313e-16)).sum(1).mean() * auto_lambda_val
            # h_u_loss = 0.

            h_u_loss = (
                -(
                    (soft_mb_logits_outputs_unseen_preds + 2.220446049250313e-16)
                    * torch.log(
                        soft_mb_logits_outputs_unseen_preds + 2.220446049250313e-16
                    )
                )
                .sum(1)
                .mean()
                * auto_lambda_val
            )

            ratio = args.ratio

            h_u_cond_l_loss = (
                -(
                    (
                        soft_mb_logits_outputs_unseen_preds[hard_u_cond_l_index]
                        + 2.220446049250313e-16
                    )
                    * torch.log(
                        soft_mb_logits_outputs_unseen_preds[hard_u_cond_l_index]
                        + 2.220446049250313e-16
                    )
                )
                .sum(1)
                .mean()
                * auto_lambda_val
                * ratio
            )

            ce_l_loss = criterion(
                mb_logits_outputs_seen_preds, filtered_seen_labels.to(torch.int64)
            )

            h_u_l_cond_y = (
                soft_mb_logits_outputs.mean(0)
                * torch.log(soft_mb_logits_outputs.mean(0) + 1e-12)
            ).sum()

            loss = h_u_loss + h_u_cond_l_loss + ce_l_loss + h_u_l_cond_y
            # print( h_u_loss.item(), h_u_cond_l_loss.item(),ce_l_loss.item(), h_u_l_cond_y.item(),)
            # if args.H_labelled:
            H_labelled = (
                    -(
                        (targets_l + 1e-12)
                        * torch.log(soft_mb_logits_outputs_seen_preds + 1e-12)
                    )
                    .sum(1)
                    .mean()
                )
            loss += H_labelled

            # if args.H_unlabelled:
            H_unlabelled = (
                    -(
                        (targets_u + 1e-12)
                        * torch.log(soft_mb_logits_outputs_unseen_preds + 1e-12)
                    )
                    .sum(1)
                    .mean()
                    * auto_lambda_val
                    * ratio
                )
            loss += H_unlabelled

            loss.backward()
            optimizer.step()
            if args.ema is not None:
                ema.update(pim)

            running_loss = loss.item()

            # H_unseen_score = -((targets_u[:, seen_N+1 :] + 1e-12).mean(0) * torch.log(soft_mb_logits_outputs_unseen_preds[:, seen_N+1 :] + 1e-12).mean(0)).sum().item()
        # if epoch % 100 == 0:
#     print("  epoch:", epoch, ", loss:", running_loss)
#
#     with torch.no_grad():
#         outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
#         _, predicted = torch.max(outputs.data, 1)
#         unlab_preds = np.asarray(from_torch_to_numpy(predicted)[~mask_lab], dtype=int)
#     unlab_gt_labs = np.asarray(gt_labels[~mask_lab], dtype=int)
#     seen_mask = mask_old_new[~mask_lab].astype(bool)
#     partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat)
#     ###
#     # print(ratio)
    if args.ema is not None:
        ema.apply_shadow()
    # print(
    #     f"PIM training is completed! H_unseen_score {h_u_cond_l_loss.item(), ce_l_loss.item(), h_u_l_cond_y.item(), loss.item(), H_labelled.item()}\n "
    # )
    print(running_loss)
    # PIM partitioning evaluation
    with torch.no_grad():
        outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
        _, predicted = torch.max(outputs.data, 1)
        all_pred = np.asarray(from_torch_to_numpy(predicted[all_valid_mask]), dtype=int)
        unlab_preds = np.asarray(from_torch_to_numpy(predicted[valid_mask_unseen_mask]), dtype=int)
        lab_preds = np.asarray(
            from_torch_to_numpy(predicted[valid_mask_seen_mask]), dtype=int
        )




    all_valid_label = (gt_labels[all_valid_mask]).astype(int)
    unlab_gt_labs = (gt_labels[valid_mask_unseen_mask]).astype(int)
    lab_gt_labs =(gt_labels[valid_mask_seen_mask]).astype(int)


    check = np.isin(lab_gt_labs, unlab_gt_labs)
    # print(check.sum(), 'overlap')

    # filtered_seen_mask = from_torch_to_numpy(filtered_seen_mask).astype(bool)
    # seen_mask = mask_old_new[filtered_seen_mask].astype(bool)
    # metrics = partitioning_eval(
    #     unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat
    # )
    # acc = partitioning_eval(unlab_gt_labs, unlab_preds, valid_mask_seen_mask,  args.dataset, path_k_strat)
    # unseen_valid_acc = cluster_acc(unlab_gt_labs, unlab_preds)
    # seen_valid_acc = cluster_acc(lab_gt_labs, lab_preds)
    acc_unlabel = p_metrics.orca_cluster_acc(unlab_preds, unlab_gt_labs)
    acc_label = p_metrics.orca_accuracy(lab_preds, lab_gt_labs)

    # print(lab_preds, lab_gt_labs)

    acc_all = p_metrics.orca_cluster_acc(all_pred, all_valid_label)

    # print(from_torch_to_numpy(predicted).astype(int)[-100:],  from_torch_to_numpy(all_labels).astype(int)[-100:],)
    # acc_all = p_metrics.orca_cluster_acc(from_torch_to_numpy(predicted).astype(int), from_torch_to_numpy(all_labels).astype(int), )

    # print(f"valid acc {valid_acc}")
    # print(f"ratio: {ratio}")
    # print(f"weight decay: {args.weight_decay}")
    # save(
    #     parameters={
    #         "ratio": args.ratio,
    #         "weight_decay": args.weight_decay,
    #         "dataset": args.dataset,
    #         "ema": args.ema,
    #         "H_unlabelled": args.H_unlabelled,
    #         "H_labelled": args.H_labelled,
    #         "lr": args.lr,
    #     },
    #     metrics={"valid_acc": valid_acc},
    #     output=args.save_metrics,
    # )
    printout = {
        'parameters': {
            "ratio": args.ratio,
            "weight_decay": wight_decay,
            "dataset": args.dataset,
            "ema": args.ema,
            "H_unlabelled": args.H_unlabelled,
            "H_labelled": args.H_labelled,
            "lr": args.lr,
        },
        'metrics':{"unseen_valid_acc": acc_unlabel,
                   'seen_valid_acc': acc_label,
                   'valid_acc': acc_all},
        'output':args.save_metrics,
    }
    for k in printout:
        print(f'{k}: {printout[k]}')
    return printout
    ###


if __name__ == "__main__":
    # r_list = [ 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,]
    r_list = [1e-3,2e-3,5e-3, 1e-2, 2e-2,  5e-2, ]
    best_acc = 0.
    best_paras = {}
    for r in r_list:
        # for lr in [0.001, 0.0001 , 0.01]:
        res = main(wight_decay = r, )
        if res['metrics']['valid_acc'] >= best_acc:
            best_acc = res['metrics']['valid_acc']
            best_paras = res
            print('find new BEST~~~~~~~~')
            print(best_paras)
    print('BEST~~~~~~~~')
    print(best_paras)

    # for r in r_list:
    #     # for lr in [0.001, 0.0001 , 0.01]:
    #     res = main(wight_decay=r, )
    #     if res['metrics']['valid_acc'] >= best_acc:
    #         best_acc = res['metrics']['seen_valid_acc']
    #         best_paras = res
    #         print('find new BEST~~~~~~~~')
    #         print(best_paras)
    # print('BEST~~~~~~~~')
    # print(best_paras)
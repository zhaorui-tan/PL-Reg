import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
# import copy
from collections import OrderedDict

class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

        self.partial_logic_mask = nn.ModuleList()
        self.partial_logic_classifier = nn.ModuleList()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

        partial_mask_pipe = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.out_size, self.out_size, bias=False)),
            ('sig', nn.Sigmoid()),
            # ('linear2', nn.Linear(self.out_size, self.out_size, bias=False)),
            # ('sig2', nn.Sigmoid())
        ])
        )
        # if len(self.partial_logic_mask) > 0:
        #     averaged_model = deepcopy(self.partial_logic_mask[-1])  # 创建第一个模型的深拷贝
        #     for param1, param2, param_avg in zip(partial_mask_pipe.parameters(), self.partial_logic_mask[-1].parameters(), averaged_model.parameters()):
        #         param_avg.data = (param1.data + param2.data) / 2  # 对每个权重进行平均
        #
        #     self.partial_logic_mask.append(averaged_model)
        #
        # else:
        # nn.init.uniform_(partial_mask_pipe.linear2.weight.data, a=0.8, b=1.0)
        # nn.init.uniform_(partial_mask_pipe.linear1.weight.data, a=0.4, b=0.5)
        # nn.init.uniform_(partial_mask_pipe.linear1.weight.data, a=0.2, b=0.7)


        self.partial_logic_mask.append(partial_mask_pipe)
        self.partial_logic_classifier.append(nn.Linear(self.out_size, 2))
        self.mask = None


    def forward(self, x, return_features=False,
                    pl_classification_w = 0.,
                    pl_mask_entropy_w = 0.,
                    # l_reg_w = 0.,
                ):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """


        assert (len(self.heads) > 0), "Cannot access any head"
        y = []

        PL_loss = 0.

        x = self.model(x)
        for head_idx in range(len(self.heads)):
            if pl_classification_w>0:
                mask = self.partial_logic_mask[head_idx](x)
                tmp_feat = x * mask
                p_feat = x * (1. - mask)

                PL_true = self.partial_logic_classifier[head_idx](tmp_feat)
                PL_False = self.partial_logic_classifier[head_idx](p_feat)
                pl = torch.cat([PL_true, PL_False])

                pl_target = torch.cat(
                    [torch.ones(PL_true.size(0)).to(torch.int64).cuda(),
                     torch.zeros(PL_False.size(0)).to(torch.int64).cuda()])
                PL_loss += nn.CrossEntropyLoss()(pl, pl_target) * pl_classification_w  #0.0001

                mask_entropy = F.softmax(mask, dim=1)
                mask_entropy = (
                    -((mask_entropy + 1e-12)
                      * torch.log(
                                mask_entropy + 1e-12
                            )
                      )
                    .mean()
                )
                PL_loss += -mask_entropy * pl_mask_entropy_w  #0.000001  #* self.use_partial

            else:
                tmp_feat = x

            tmp_x = self.heads[head_idx](tmp_feat)
            y.append(tmp_x)

        if return_features:
            return y, x, PL_loss
        else:
            return y, PL_loss

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass

from copy import deepcopy

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

model = some torch model
avg_para = copy_params(model)
for epoch in range(all_epoch):
    for step in range(all_step):
        forward the model.....
        get loss....
        backward...

        for p, avg_p in zip(model.parameters(), avg_para):
            avg_p.mul_(0.999).add_(0.001, p.data)

    for eval or save: # only using the avg_para
        backup_para = copy_params(model)
        load_params(model, avg_para)
        do eval or save

        load_params(model, backup_para)
# at the end of training, remember only use the avg_para
load_params(model, avg_para)
save model
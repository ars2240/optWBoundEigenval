# Asymmetric Valley Optimizer
# Modified from https://github.com/962086838/code-for-Asymmetric-Valley

import os
import sys
import time
import torch
import torch.utils.data as utils_data
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from opt import *


class AsymmetricValley(OptWBoundEignVal):

    def __init__(self, model, loss, optimizer, scheduler=None, mu=0, K=0, eps=-1, pow_iter_eps=1e-3,
                 use_gpu=False, batch_size=128, min_iter=10, max_iter=250, max_pow_iter=1000, pow_iter=True,
                 max_samples=512, ignore_bad_vals=True, verbose=False, mem_track=False, header='', num_workers=0,
                 test_func='maxacc', swa=True, swa_start=161, sgd_start=201, swa_c_epochs=1, swa_lr=0.05, eval_freq=5,
                 save_freq=5, division_part=40, distances=20):
        super().__init__(model, loss, optimizer, scheduler, mu, K, eps, pow_iter_eps, use_gpu, batch_size, min_iter,
                         max_iter, max_pow_iter, pow_iter, max_samples, ignore_bad_vals, verbose, mem_track, header,
                         num_workers, test_func)
        self.swa = swa
        self.swa_start = swa_start
        self.sgd_start = sgd_start
        self.swa_c_epochs = swa_c_epochs
        self.swa_n = 0
        self.swa_lr = swa_lr
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.dir = './models/'
        self.swa_model = deepcopy(self.model).to(self.device)
        self.lr_init = get_lr(optimizer)
        self.train_res_swa = None
        self.test_res_swa = None
        self.swa_path = None
        self.sgd_path = None
        self.division_part = division_part
        self.distances = distances

    def schedule(self):
        t = self.i / (self.swa_start if self.swa else self.max_iter)
        lr_ratio = self.swa_lr / self.lr_init if self.swa else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.lr_init * factor

    def iter(self):
        lr = self.schedule()
        adjust_learning_rate(self.optimizer, lr)
        _ = self.train_epoch(self.dataloader, self.model)

        if self.swa and (self.i + 1) >= self.swa_start and (self.i + 1 - self.swa_start) % self.swa_c_epochs == 0:
            moving_average(self.swa_model, self.model, 1.0 / (self.swa_n + 1))
            self.swa_n += 1
            if self.i == 0 or self.i % self.eval_freq == self.eval_freq - 1 or self.i == self.sgd_start - 2:
                bn_update(self.dataloader, self.swa_model)

        if (self.i + 1) % self.save_freq == 0:
            self.swa_path = save_checkpoint(self.dir, self.i + 1, state_dict=self.model.state_dict(),
                                            swa_state_dict=self.swa_model.state_dict() if self.swa else None,
                                            swa_n=self.swa_n if self.swa else None,
                                            optimizer=self.optimizer.state_dict())

    def iter2(self, valid_loader):

        if self.train_res_swa is None or self.test_res_swa is None:
            self.train_res_swa = self.eval(self.dataloader, self.model)
            self.test_res_swa = self.eval(valid_loader, self.model)

            # load model
            state = self.load_state(self.swa_path)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            bn_update(self.dataloader, self.model)

        adjust_learning_rate(self.optimizer, self.lr_init)
        train_res = self.train_epoch(self.dataloader, self.model)
        test_res = self.eval(valid_loader, self.model)

        if train_res['loss'] < self.train_res_swa['loss'] and test_res['loss'] > self.test_res_swa['loss']:
            self.sgd_path = save_checkpoint(self.dir, self.i + 1, state_dict=self.model.state_dict(),
                                            optimizer=self.optimizer.state_dict())

    def interpolation(self, valid_loader):

        # load models
        state = self.load_state(self.sgd_path)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        bn_update(self.dataloader, self.model)
        vec_1 = parameters_to_vector(self.model.parameters())

        state = self.load_state(self.swa_path, 'swa_state_dict')
        self.swa_model.load_state_dict(state)
        self.swa_model.to(self.device)
        bn_update(self.dataloader, self.swa_model)
        vec_2 = parameters_to_vector(self.swa_model.parameters())

        state = self.load_state(self.swa_path)
        model_temp = deepcopy(self.model)
        model_temp.load_state_dict(state)
        model_temp.to(self.device)

        vec_inter = vec_1 - vec_2
        vec_inter = vec_inter / self.division_part

        dis_counter = 0
        result_shape = self.distances * 2 + self.division_part + 1

        train_loss_results_bnupdate = np.zeros(result_shape)
        test_loss_results_bnupdate = np.zeros(result_shape)
        train_acc_results_bnupdate = np.zeros(result_shape)
        test_acc_results_bnupdate = np.zeros(result_shape)

        for i in range(0, int(result_shape), 1):
            vec_temp = vec_2 + (i - self.distances) * vec_inter
            vector_to_parameters(vec_temp, model_temp.parameters())
            bn_update(self.dataloader, model_temp)

            train_temp = self.eval(self.dataloader, model_temp)
            test_temp = self.eval(valid_loader, model_temp)

            train_loss_results_bnupdate[dis_counter] = train_temp['loss']
            train_acc_results_bnupdate[dis_counter] = train_temp['accuracy']
            test_loss_results_bnupdate[dis_counter] = test_temp['loss']
            test_acc_results_bnupdate[dis_counter] = test_temp['accuracy']

            np.savetxt(os.path.join('./logs/', "asymmetric_valley_train_loss_results.txt"), train_loss_results_bnupdate)
            np.savetxt(os.path.join('./logs/', "asymmetric_valley_test_loss_results.txt"), test_loss_results_bnupdate)
            np.savetxt(os.path.join('./logs/', "asymmetric_valley_train_acc_results.txt"), train_acc_results_bnupdate)
            np.savetxt(os.path.join('./logs/', "asymmetric_valley_test_acc_results.txt"), test_acc_results_bnupdate)
            dis_counter += 1

        plt.cla()
        plt.plot(train_loss_results_bnupdate)
        plt.savefig(os.path.join('./plots/', 'asymmetric_valley_train_loss_results.png'))
        plt.cla()
        plt.plot(test_loss_results_bnupdate)
        plt.savefig(os.path.join('./plots/', 'asymmetric_valley_test_loss_results.png'))
        plt.cla()
        plt.plot(train_acc_results_bnupdate)
        plt.savefig(os.path.join('./plots/', 'asymmetric_valley_train_acc_results.png'))
        plt.cla()
        plt.plot(test_acc_results_bnupdate)
        plt.savefig(os.path.join('./plots/', 'asymmetric_valley_test_acc_results.png'))

    def train(self, inputs=None, target=None, inputs_valid=None, target_valid=None, train_loader=None,
              valid_loader=None, train_loader_na=None):

        start = time.time()  # start timer

        if train_loader is not None:
            self.dataloader = train_loader
        elif inputs is not None and target is not None:
            self.x = inputs  # input data
            self.y = target  # output data

            # create dataloader
            train_data = utils_data.TensorDataset(self.x, self.y)
            if self.use_gpu:
                self.dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size,
                                                        num_workers=self.num_workers, pin_memory=True)
            else:
                self.dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size)
        else:
            raise Exception('No input data')

        # make sure logs & models folders exist
        check_folder('./logs')
        check_folder('./models')
        check_folder('./plots')

        old_stdout = sys.stdout  # save old output

        f_hist = []  # tracks function value after each epoch

        log_file = open(self.log_file, "w")  # open log file
        sys.stdout = log_file  # write to log file

        # header of log file
        if (inputs_valid is None or target_valid is None) and (valid_loader is None):
            print('epoch\t f\t rho\t h\t norm')
        else:
            print('epoch\t f\t rho\t h\t norm\t val_acc\t val_f1')

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        for self.i in range(0, self.max_iter):
            # take step
            if (self.i + 1) >= self.sgd_start:
                self.iter2(valid_loader)
            else:
                self.iter()

            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file

            self.save()

            # add values to log file
            if (inputs_valid is None or target_valid is None) and (valid_loader is None):
                print('%d\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm))
            else:
                with torch.no_grad():
                    _, self.val_acc, val_f1 = self.test_model(inputs_valid, target_valid, valid_loader)
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
                    self.best_rho = self.rho
                    self.best_val_iter = self.i
                    self.save('_trained_model_best.pt')
                print('%d\t %f\t %f\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm,
                                                            self.val_acc, val_f1))

            # add function value to history log
            # check if h is tensor
            if torch.is_tensor(self.h):
                self.h = self.h.item()
            f_hist.append(self.h)

            # check if convergence criteria met
            if self.i >= (self.min_iter - 1):
                coef_var = np.std(f_hist[-10:]) / np.abs(np.mean(f_hist[-10:]))
                if coef_var <= self.eps:
                    print(coef_var)
                    break

            if self.i < (self.max_iter - 1):
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

        # compute time elapsed
        end = time.time()
        tTime = end - start
        timeHMS(tTime)

        # best validation accuracy
        print('Best Validation Iterate:', self.best_val_iter)
        print('Best Validation Accuracy:', self.best_val_acc)
        print('Rho:', self.best_rho)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        self.interpolation(valid_loader)

        # compute loss & accuracy on training set
        if train_loader_na is not None:
            self.test_set(inputs, target, train_loader_na)
        else:
            self.test_set(inputs, target, train_loader)

    def train_epoch(self, loader, model):
        loss_sum = 0.0
        correct = 0.0

        model.train()

        for i, (input, target) in enumerate(loader):
            input = input.to(self.device)
            target = target.to(self.device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = self.loss(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(list(model.parameters())[0].grad[0][0])
            # print('i',i)

            loss_sum += loss.item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_var.data.view_as(pred)).sum().item()
            # print('len(loader.dataset)', len(loader.sampler))

        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }

    def eval(self, loader, model):
        loss_sum = 0.0
        correct = 0.0

        model.eval()

        for i, (input, target) in enumerate(loader):
            input = input.to(self.device)
            target = target.to(self.device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = self.loss(output, target_var)

            loss_sum += loss.item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_var.data.view_as(pred)).sum().item()

        # print(len(loader.dataset))
        return {
            'loss': loss_sum / len(loader.dataset),
            # 'loss': loss_sum,
            'accuracy': correct / len(loader.dataset) * 100.0,
        }


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)
    return filepath


def moving_average(net1, net2, alpha=1.0):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def adding_weight(net1, net2, distances, distance_scale, epoch):
    if epoch == 0:
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            param1.data = param1.data - distances * distance_scale * param2.data
    else:
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            param1.data = param1.data + distance_scale * param2.data


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

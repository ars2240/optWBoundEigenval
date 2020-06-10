# opt.py
#
# Author: Adam Sandler
# Date: 5/17/19
#
# Contains methods & classes for optimizing NNs with spectral radius regulation
#
#
# Dependencies:
#   Packages: requests, numpy, scipy, sklearn, torch

import inspect
import random
import re
import numpy as np
import os
import shutil
import sys
from scipy.stats import skewnorm
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score
import time
import torch
from torch.autograd import Variable
import torch.utils.data as utils_data
import torch.nn.functional as F


class HVPOperator(object):
    """
    Modified from: https://github.com/noahgolmant/pytorch-hessian-eigenthings

    Use PyTorch autograd for Hessian Vec product calculation
    """

    def __init__(self, model, data, criterion, use_gpu=True, mem_track=False):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.data = data  # data (inputs, target)
        self.criterion = criterion  # loss function
        self.use_gpu = use_gpu  # whether or not GPUs are used
        self.stored_grad = None  # stored gradient (on CPU)
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage

    def mem_check(self):
        # checks max memory used
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

    def Hv(self, vec, storedGrad=False):
        # Returns H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        vec = vec.to(self.device)
        vec = vec.double()  # convert to double

        # compute original gradient, tracking computation graph
        if storedGrad and (self.stored_grad is not None):
            grad_vec = self.stored_grad.to(self.device)
        else:
            self.zero_grad()
            grad_vec = self.prepare_grad()
            self.stored_grad = grad_vec.to('cpu')

        # check memory usage
        self.mem_check()

        self.zero_grad()
        # compute gradient of vector product
        grad_grad = torch.autograd.grad(grad_vec, self.model.parameters(), grad_outputs=vec, retain_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad]))

        # check memory usage
        self.mem_check()

        # hessian_vec_prod = hessian_vec_prod.to('cpu')
        return hessian_vec_prod.data.double()

    def vGHv(self, vec, storedGrad=False):
        # Returns vec*grad H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        vec = vec.to(self.device)
        vec = vec.double()  # convert to double

        # compute original gradient, tracking computation graph
        if storedGrad and (self.stored_grad is not None):
            grad_vec = self.stored_grad.to(self.device)
        else:
            self.zero_grad()
            grad_vec = self.prepare_grad()
            self.stored_grad = grad_vec.to('cpu')

        # check memory usage
        self.mem_check()

        self.zero_grad()
        # compute gradient of vector product
        grad_grad = torch.autograd.grad(grad_vec, self.model.parameters(), grad_outputs=vec, create_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad]))

        # check memory usage
        self.mem_check()

        self.zero_grad()
        # compute second gradient of vector product
        grad_grad = torch.autograd.grad(hessian_vec_prod.double(), self.model.parameters(), grad_outputs=vec)
        # concatenate the results over the different components of the network
        vec_grad_hessian_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad]))

        # check memory usage
        self.mem_check()

        # vec_grad_hessian_vec = vec_grad_hessian_vec.to('cpu')
        return vec_grad_hessian_vec.data.double()

    def zero_grad(self):
        # Zeros out the gradient info for each parameter in the model
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        # Compute gradient w.r.t loss over all parameters and vectorize
        if type(self.data) == list:
            inputs, target = self.data
        elif type(self.data) == dict:
            inputs, target = Variable(self.data['image']), Variable(self.data['label'])
        else:
            raise Exception('Data type not supported')

        inputs = inputs.to(self.device)
        target = target.to(self.device)

        output = self.model(inputs)
        if self.criterion.__class__.__name__ == 'KLDivLoss':
            target_onehot = torch.zeros(np.shape(output))
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            loss = self.criterion(output.float(), target_onehot.float())
        else:
            loss = self.criterion(output, target)
        grad_dict = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_dict]))
        return grad_vec.double()


# check if folder exists; if not, create it
def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


# Download and parse the dataset
def download(url):
    import requests
    root = './data'
    check_folder(root)  # make sure data folder exists
    filename = root + '/' + url.split("/")[-1]
    exists = os.path.isfile(filename)
    if not exists:
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    ftype = os.path.splitext(filename)[1]
    fname = filename[:-len(ftype)] + '.csv'
    exists = os.path.isfile(fname)
    if not exists:
        if ftype == '.gz':
            import gzip
            with gzip.open(filename, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif ftype == '.bz2':
            import bz2
            with bz2.open(filename, 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    return fname


class OptWBoundEignVal(object):
    def __init__(self, model, loss, optimizer, scheduler=None, mu=0, K=0, eps=-1, pow_iter_eps=1e-3,
                 use_gpu=False, batch_size=128, min_iter=10, max_iter=100, max_pow_iter=1000, pow_iter=True,
                 max_samples=512, ignore_bad_vals=True, verbose=False, mem_track=False, header='', num_workers=0,
                 test_func='maxacc'):

        # set default device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.ndim = sum(p.numel() for p in model.parameters())  # number of dimensions
        self.f = 0  # loss function value
        self.gradf = torch.zeros(self.ndim).to(self.device)  # gradient of f
        self.rho = 0  # spectral radius (maximal absolute value eignevalue)
        self.v = torch.from_numpy(1.0/np.sqrt(self.ndim)*np.ones(self.ndim)).to(self.device)  # eigenvector
        self.gradrho = torch.zeros(self.ndim).to(self.device)  # gradient of rho
        self.g = 0  # regularizer function
        self.gradg = torch.zeros(self.ndim).to(self.device)  # gradient of g
        self.h = 0  # objective function f+mu*g
        self.mu = mu  # coefficient in front of regularizer
        self.K = float(K)  # constant, spectral radius < K
        self.batch_size = batch_size  # batch size
        self.eps = eps  # convergence
        self.pow_iter_eps = pow_iter_eps  # convergence
        self.model = model.to(self.device)  # model (from torch)
        self.dataloader = None  # dataloader for training set (from torch)
        self.loss = loss  # loss function (from torch)
        self.optimizer = optimizer  # optimizer function, optional (from torch)
        self.scheduler = scheduler  # learning rate scheduler, optional (from torch)
        self.use_gpu = use_gpu  # whether or not cuda GPUs used
        self.min_iter = min_iter  # minimum number of iterations
        self.max_iter = max_iter  # maximum number of iterations
        self.max_pow_iter = max_pow_iter  # maximum number of power iterations
        self.max_samples = max_samples  # maximum batch size
        self.pow_iter = pow_iter  # whether or not it performs power iteration
        self.hvp_op = None  # Hessian-vector operation object
        self.i = 0  # iteration count
        self.norm = 0  # norm of H*v-lambda*v
        self.val_acc = 0  # validation accuracy (only used if validation set is provided)
        self.best_val_acc = 0  # best validation accuracy (only used if validation set is provided)
        self.best_val_iter = 0  # best validation iterate
        self.best_rho = 0  # spectral radius at best validation accuracy
        self.verbose = verbose  # more extensive read-out
        self.x = None  # input data
        self.y = None  # output data
        name = self.optimizer.__class__.__name__
        if callable(mu):
            mname = 'Func'
        else:
            mname = str(mu)
        # log files
        self.header = header  # header to files
        self.header2 = header + "_" + name + "_mu" + mname + "_K" + str(K)
        self.log_file = "./logs/" + self.header2 + ".log"
        self.verbose_log_file = "./logs/" + self.header2 + "_verbose.log"
        self.ignore_bad_vals = ignore_bad_vals  # whether or not to ignore bad power iteration values
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage
        self.num_workers = num_workers  # number of GPUs
        self.test_func = test_func  # test function

    def mem_check(self):
        # checks & prints max memory used
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])
            print('Running Max GPU Memory used (in bytes): %d' % self.mem_max)

    def comp_rho(self, p=False):
        # computes rho, v

        self.model.train()

        v = self.v  # initial guess for eigenvector (prior eigenvector)

        # initialize lambda and the norm
        lam = 0
        n = 0
        r_old = 0
        n_old = 0

        if self.verbose:
            old_stdout = sys.stdout  # save old output
            log_file = open(self.verbose_log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('iter\t lam\t norm\t delRes')

        # power iteration
        for i in range(0, np.min([self.ndim, self.max_pow_iter])):
            vnew = self.hvp_op.Hv(v, storedGrad=True)  # compute H*v

            # if converged, break
            lam = torch.dot(vnew, v)  # update eigenvalue
            r = vnew-lam*v  # residual
            n = torch.norm(r)  # norm of H*v-lambda*v
            rn = np.min([torch.norm(r-r_old), torch.norm(r+r_old)])
            if self.verbose:
                print('%d\t %f\t %f\t %f' % (i, lam, n, rn))

            if n < self.pow_iter_eps or rn/n_old < self.pow_iter_eps:
                break

            v = 1.0/torch.norm(vnew)*vnew  # update vector and normalize
            if i < (np.min([self.ndim, self.max_pow_iter])-1):
                r_old = r
                n_old = n

        if self.verbose:
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

        self.v = v  # update eigenvector
        self.rho = torch.abs(lam)  # update spectral radius
        self.norm = n  # update norm

        # check if rho is tensor
        if torch.is_tensor(self.rho):
            self.rho = self.rho.item()

        if n > self.pow_iter_eps and rn/n_old > self.pow_iter_eps:
            print('Warning: power iteration has not fully converged')
            if self.ignore_bad_vals:
                print('Ignoring rho.')
                self.rho = -1  # if value is discarded due to poor convergence, set to -1
                # as negative values of rho work in the other algorithms and are nonsensical

        if lam == 0:
            print('Warning: rho = 0')

        if p:
            old_stdout = sys.stdout  # save old output
            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('Rho:', self.rho)
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

    def comp_gradrho(self):
        # computes grad rho

        self.gradrho = self.hvp_op.vGHv(self.v, storedGrad=True)  # compute v*gradH*v

    def comp_f(self, inputs, target, classes=None, model_classes=None):
        # computes f

        self.model.eval()  # set model to evaluation mode

        # if using gpu, move data to it
        with torch.no_grad():
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            output = self.model(inputs)  # compute prediction

            # subset classes
            if classes is not None:
                if model_classes is None:
                    model_classes = classes
                if target.shape[1] == 1:
                    print('"Classes" argument only implemented for one-hot encoding')
                else:
                    target = target[:, classes]
                    output = output[:, model_classes]

            # compute loss
            if self.loss.__class__.__name__ == 'KLDivLoss':
                target_onehot = torch.zeros(output.shape)
                target_onehot.scatter_(1, target.view(-1, 1), 1)
                f = self.loss(output.float(), target_onehot.float()).item()
            else:
                f = self.loss(output, target).item()
            return f, output

    def comp_g(self):
        # computes g

        self.comp_rho()
        self.g = np.max([0.0, self.rho - self.K])

    def iter(self):
        # performs one gradient descent iteration

        self.model.train()  # set model to training mode

        # if verbose, make header for file
        if self.verbose:
            old_stdout = sys.stdout  # save old output
            if self.i == 0:
                log_file = open(self.verbose_log_file, "w")  # open log file
            else:
                log_file = open(self.verbose_log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('batch\t rho\t norm\t gradf\t gradg')
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

        # compute mu
        if callable(self.mu):
            mu = self.mu(self.i)
        else:
            mu = self.mu

        # pick random batch for estimation of spectral radius at end of epoch
        rbatch = random.randint(0, len(self.dataloader)-1)

        for j, data in enumerate(self.dataloader):

            # store random batch
            if j == rbatch:
                rdata = data

            if self.pow_iter:
                # initialize hessian vector operation class
                self.hvp_op = HVPOperator(self.model, data, self.loss, use_gpu=self.use_gpu)

                self.comp_g()  # compute g

                self.optimizer.zero_grad()  # zero gradient

                # compute grad f
                if self.hvp_op.stored_grad is not None:
                    self.gradf = self.hvp_op.stored_grad.data.to(self.device)
                else:
                    self.gradf = torch.zeros(self.ndim).double().to(self.device)  # set gradient to zero

                # compute grad g
                if self.g > 0:
                    self.comp_gradrho()  # compute gradient of rho
                    self.gradg = self.gradrho  # compute g
                else:
                    self.gradg = torch.zeros(self.ndim).double().to(self.device)  # set gradient to zero

                p = self.gradf + mu * self.gradg  # gradient step

                i = 0
                for param in self.model.parameters():
                    s = param.data.size()  # number of input & output nodes
                    n = torch.prod(torch.tensor(s))  # total number of parameters
                    param.grad = p[i:(i + n)].view(s).float()  # adjust gradient
                    i += n  # increment
            else:
                # for testing purposes
                self.optimizer.zero_grad()  # zero gradient
                if type(data) == list:
                    inputs, target = data
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                elif type(data) == dict:
                    inputs, target = Variable(data['image'].to(self.device)), Variable(data['label'].to(self.device))
                else:
                    raise Exception('Data type not supported')
                output = self.model(inputs)
                loss = self.loss(output, target)  # loss function
                loss.backward()  # back prop

            # optimizer step
            self.optimizer.step()

            # if verbose, add values to file
            if self.verbose:
                log_file = open(self.verbose_log_file, "a")  # open log file
                sys.stdout = log_file  # write to log file
                if self.pow_iter:
                    print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm, torch.norm(self.gradf.detach()),
                                                      torch.norm(self.gradg.detach())))
                else:
                    print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm, torch.norm(self.gradf),
                                                      torch.norm(self.gradg)))
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

            # if using GPU, memory cleanup & tracking
            if self.use_gpu:
                torch.cuda.empty_cache()
                # check max memory usage
                self.mem_check()

        # compute overall estimates
        f_list = []
        size = []
        # compute f on each batch (to avoid memory issues)
        for _, data in enumerate(self.dataloader):
            if type(data) == list:
                inputs, target = data
            elif type(data) == dict:
                inputs, target = Variable(data['image']), Variable(data['label'])
            else:
                raise Exception('Data type not supported')
            size.append(len(target))
            f, _ = self.comp_f(inputs, target)
            f_list.append(f)  # compute f on each batch
        self.f = np.average(f_list, weights=size)  # weighted mean of f values
        # initialize hessian vector operation class for random batch
        self.hvp_op = HVPOperator(self.model, rdata, self.loss, use_gpu=self.use_gpu)
        self.comp_g()  # compute g
        self.h = self.f + mu * self.g  # compute objective function

        # adjust learning rate
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, inputs=None, target=None, inputs_valid=None, target_valid=None, loader=None, valid_loader=None,
              train_loader=None):

        start = time.time()  # start timer

        if loader is not None:
            self.dataloader = loader
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
            self.iter()  # take step

            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file

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
                    self.model.to('cpu')
                    torch.save(self.model.state_dict(), './models/' + self.header2 + '_trained_model_best.pt')
                    self.model.to(self.device)
                print('%d\t %f\t %f\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm,
                                                            self.val_acc, val_f1))

            # add function value to history log
            # check if h is tensor
            if torch.is_tensor(self.h):
                self.h = self.h.item()
            f_hist.append(self.h)

            # Save model weights
            self.model.to('cpu')
            torch.save(self.model.state_dict(), './models/' + self.header2 + '_trained_model.pt')
            self.model.to(self.device)

            # check if convergence criteria met
            if self.i >= (self.min_iter - 1):
                coef_var = np.std(f_hist[-10:])/np.abs(np.mean(f_hist[-10:]))
                if coef_var <= self.eps:
                    print(coef_var)
                    break

            if self.i < (self.max_iter - 1):
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

        # compute time elapsed
        end = time.time()
        tTime = end - start
        hrs = np.floor(tTime / 3600)
        tTime = tTime - hrs * 3600
        mins = np.floor(tTime / 60)
        secs = tTime - mins * 60
        print('Time elapsed: %2i hrs, %2i min, %4.2f sec ' % (hrs, mins, secs))

        # best validation accuracy
        print('Best Validation Iterate:', self.best_val_iter)
        print('Best Validation Accuracy:', self.best_val_acc)
        print('Rho:', self.best_rho)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        # compute loss & accuracy on training set
        if train_loader is not None:
            self.test_train_set(inputs, target, train_loader)
        else:
            self.test_train_set(inputs, target, loader)

    def test_model(self, x=None, y=None, loader=None, classes=None, model_classes=None):
        # Computes the loss and accuracy of model on given dataset

        self.model.eval()  # set model to evaluation mode

        with torch.no_grad():
            if loader is not None:
                dataloader = loader
            elif x is not None and y is not None:
                # create dataloader
                train_data = utils_data.TensorDataset(x, y)
                if self.use_gpu:
                    dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size,
                                                       num_workers=self.num_workers, pin_memory=True)
                else:
                    dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size)
            else:
                raise Exception('No test data')

            f_list = []
            acc_list = []
            f1_list = []
            size = []
            outputs = []
            labels = []
            for _, data in enumerate(dataloader):

                if type(data) == list:
                    inputs, target = data
                elif type(data) == dict:
                    inputs, target = Variable(data['image']), Variable(data['label'])
                else:
                    raise Exception('Data type not supported')

                # compute loss
                f, ops = self.comp_f(inputs, target, classes, model_classes)
                f_list.append(f)

                # subset classes
                if classes is not None:
                    if target.shape[1] == 1:
                        print('"Classes" argument only implemented for one-hot encoding')
                    else:
                        target = target[:, classes]

                if any(x in self.test_func for x in ['sigmoid', 'logit']):
                    ops = torch.sigmoid(ops)

                # size of dataset
                size.append(len(target))

                # compute accuracy
                if 'max' in self.test_func:
                    _, predicted = torch.max(ops.data, 1)
                else:
                    predicted = (ops.data > 0.5).float()
                target = target.to(self.device)
                if 'acc' in self.test_func:
                    acc = torch.mean((predicted == target).float()).item() * 100
                    acc_list.append(acc)

                target = target.to('cpu')
                predicted = predicted.to('cpu')
                ops = ops.to('cpu')
                if 'auc' in self.test_func:
                    outputs.append(ops)
                    labels.append(target)
                else:
                    f1 = f1_score(target, predicted, average='micro')
                    f1_list.append(f1)

            if 'auc' in self.test_func:
                labels = torch.cat(labels)
                outputs = torch.cat(outputs)
                classes = outputs.size()[1]
                roc = np.zeros(classes)
                f1 = np.zeros(classes)
                for i in range(classes):
                    # remove NaN labels
                    outputs2 = outputs[:, i]
                    labels2 = labels[:, i]
                    good = labels2 == labels2
                    outputs2 = outputs2[good]
                    labels2 = labels2[good]

                    roc[i] = roc_auc_score(labels2, outputs2, average=None)  # compute AUC of ROC curves
                    f1[i] = f1_score(labels2, (outputs2 > 0.5).float(), average='micro')
                test_acc = roc.mean()  # mean AUCs
                # weighted mean of f1 scores
                test_f1 = f1.mean()  # mean AUCs
            else:
                test_acc = np.average(acc_list, weights=size)  # weighted mean of accuracy
                test_f1 = np.average(f1_list, weights=size)  # weighted mean of f1 scores
            test_loss = np.average(f_list, weights=size)  # weighted mean of f values

        return test_loss, test_acc, test_f1

    def model_load(self, fname=None):
        # load model from file

        if fname is None:
            fname = './models/' + self.header2 + '_trained_model_best.pt'

        state = torch.load(fname)
        if 'state_dict' in state.keys():
            state2 = state['state_dict']

            from collections import OrderedDict
            state = OrderedDict()

            for k, v in state2.items():
                k = k.replace('encoder.', 'features.')
                k = k.replace('module.', '')
                p = re.compile("(norm|conv)\.([0-9+])")
                k = p.sub(r'\1\2', k)
                state[k] = v

        self.model.load_state_dict(state)

        self.model.to(self.device)

    def test_model_best(self, x=None, y=None, loader=None, classes=None, model_classes=None, fname=None):
        # tests best model, loaded from file

        self.model_load(fname)
        return self.test_model(x, y, loader, classes, model_classes)

    def test_train_set(self, x=None, y=None, loader=None, classes=None, model_classes=None, fname=None):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc, f1 = self.test_model_best(x, y, loader, classes, model_classes, fname)  # test best model

        print('Train Loss:', loss)
        print('Train Accuracy:', acc)
        print('Train F1:', f1)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def test_test_set(self, x=None, y=None, loader=None, classes=None, model_classes=None, fname=None):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc, f1 = self.test_model_best(x, y, loader, classes, model_classes, fname)  # test best model

        print('Test Loss:', loss)
        print('Test Accuracy:', acc)
        print('Test F1:', f1)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def test_model_cov(self, x, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                       train_skew=[0]):
        # Computes the loss and accuracy of model on given dataset

        test_data = utils_data.TensorDataset(x, y)
        dataloader = utils_data.DataLoader(test_data, batch_size=self.batch_size)

        f_list = []
        acc_list = []
        f1_list = []
        size = []
        wm_list = []
        min_weight = 1
        max_weight = 1

        modes = np.logical_or(np.subtract(test_mean, train_mean) != 0, np.subtract(test_sd, train_sd) != 0)
        modes = np.logical_or(modes, np.subtract(test_skew, train_skew) != 0)
        modes = np.where(modes)[0]

        for _, data in enumerate(dataloader):

            if type(data) == list:
                inputs, target = data
            elif type(data) == dict:
                inputs, target = Variable(data['image']), Variable(data['label'])
            else:
                raise Exception('Data type not supported')

            feats = inputs.shape[1]
            if len(test_mean) == 1:
                test_mean = test_mean * feats
            if len(test_sd) == 1:
                test_sd = test_sd * feats
            if len(test_skew) == 1:
                test_skew = test_skew * feats
            if len(train_mean) == 1:
                train_mean = train_mean * feats
            if len(train_sd) == 1:
                train_sd = train_sd * feats
            if len(train_skew) == 1:
                train_skew = train_skew * feats

            # compute loss
            f, ops = self.comp_f(inputs, target)
            f_list.append(f)

            # size of dataset
            size.append(len(target))

            # compute accuracy
            _, predicted = torch.max(ops.data, 1)

            w = np.exp(self.get_prob(inputs[:, modes], [test_mean[i] for i in modes], [test_sd[i] for i in modes],
                                     [test_skew[i] for i in modes]) -
                       self.get_prob(inputs[:, modes], [train_mean[i] for i in modes], [train_sd[i] for i in modes],
                                     [train_skew[i] for i in modes]))
            weights = torch.from_numpy(w)
            wm = torch.mean(weights).item()
            if wm == 0:
                print(weights)
            wm_list.append(wm)
            min_weight = np.min([min_weight, np.min(wm)])
            max_weight = np.max([max_weight, np.max(wm)])
            weights /= wm * len(target)
            target = target.to(self.device)
            weights = weights.to(self.device)
            acc = torch.sum(weights.float() * (predicted == target).float()).item() * 100
            acc_list.append(acc)

            target = target.to('cpu')
            predicted = predicted.to('cpu')
            weights = weights.to('cpu')
            f1 = f1_score(target, predicted, average='micro', sample_weight=weights)
            f1_list.append(f1)

        test_loss = np.average(f_list, weights=size)  # weighted mean of f values
        acc_w = np.array(size) * np.array(wm_list)
        if np.sum(acc_w) == 0 or np.isinf(np.sum(acc_w)):
            print(acc_w)
            print(wm_list)
        acc_w = acc_w/np.sum(acc_w)
        test_acc = np.average(acc_list, weights=acc_w)  # weighted mean of accuracy
        test_f1 = np.average(f1_list, weights=acc_w)  # weighted mean of f1 scores

        return test_loss, test_acc, test_f1, min_weight, max_weight

    def test_model_best_cov(self, x, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                            train_skew=[0]):
        # tests best model, loaded from file

        self.model.load_state_dict(torch.load('./models/' + self.header2 + '_trained_model_best.pt'))
        self.model.to(self.device)

        return self.test_model_cov(x, y, test_mean, test_sd, test_skew, train_mean, train_sd, train_skew)

    def test_cov_shift(self, x, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                       train_skew=[0]):

        # test best model with covariate shifts
        loss, acc, f1, min_weight, max_weight = self.test_model_best_cov(x, y, test_mean, test_sd, test_skew,
                                                                         train_mean, train_sd, train_skew)

        print('Test Accuracy:', acc)
        print('Test F1:', f1)
        print('Min-weight:', min_weight)
        print('Max-weight:', max_weight)

    # comparison test (requires list of data loaders)
    def comp_test(self, loaders):
        # test loader for model must be 0 index in list
        classes = [loader.classes.keys() for loader in loaders if not isinstance(loader, utils_data.DataLoader)]
        if len(classes) > 1:
            overlap = classes[0]
            for c in classes[1:]:
                overlap = [x for x in overlap if x in c]

            # print overlap
            old_stdout = sys.stdout  # save old output
            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print(overlap)
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

            # model classes
            mc = [x for x in range(len(classes[0])) if list(classes[0])[x] in overlap]
        i = 0
        for loader in loaders:
            # print header
            old_stdout = sys.stdout  # save old output
            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('Comparison Test - Data Set {0}'.format(i))
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

            # test model
            if len(classes) > 1:
                c = [x for x in range(len(classes[i])) if list(classes[i])[x] in overlap]
                self.test_test_set(loader=assert_dl(loader, self.batch_size), classes=c, model_classes=mc)
            else:
                self.test_test_set(loader=assert_dl(loader, self.batch_size))
            i += 1

    def parse(self):
        with open(self.log_file, "r") as log_file:  # open log file
            data = log_file.readlines()
            res = []
            if len(data) < 8:
                raise Exception('Not enough log data.')
            for string in data[-8:]:
                string = string.split()
                res.append(string[-1])

        order = [0, 2, 3, 4, 5, 6, 7, 1]
        res = [res[i] for i in order]
        print('Best_Val_Acc  Train_Loss  Train_Acc  Train_F1  Test_Loss  Test_Acc  Test_F1  Rho')
        print('  '.join(res))


def get_prob(inputs,  m=[0], sd=[1], skew=[0]):
    # computes log pdf of inputs given mean (m), standard deviation (sd), and skewness (skew)
    if len(m) != 1 or len(sd) != 1 or len(skew) != 1:
        if len(m) == 1:
            if len(sd) > 1:
                m = m*np.ones(len(sd))
            else:
                m = m*np.ones(len(skew))
        if len(sd) == 1:
            sd = sd*np.ones(len(m))
        if len(skew) == 1:
            skew = skew*np.ones(len(m))

    if not np.any(skew):
        w = norm.logpdf(inputs, m, sd)
    else:
        w = skewnorm.logpdf(inputs, skew, m, sd)
    bad = np.where(np.isinf(w))[0]
    if len(bad) > 0:
        w[bad] = norm.logpdf(inputs[bad, :], m, sd)
    w = np.sum(w, axis=1)

    return w


def cov_shift_tester(models, x, y, iters=1000, bad_modes=[], header='', mult=.1, prob=0.5, mean_diff=0, sd_diff=0,
                     skew_diff=0, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                     train_skew=[0]):
    # make sure logs folder exists
    check_folder('./logs')

    feats = x.shape[1]
    modes = range(0, feats)
    good_modes = np.setdiff1d(modes, bad_modes)
    good_feats = len(good_modes)
    nmod = len(models)

    if len(test_mean) == 1:
        test_mean = test_mean * feats
    if len(test_sd) == 1:
        test_sd = test_sd * feats
    if len(test_skew) == 1:
        test_skew = test_skew * feats

    acc = np.zeros((nmod, iters))
    f1 = np.zeros((nmod, iters))
    indices = np.zeros((feats, iters))
    indices[good_modes, :] = mult*np.random.normal(size=(good_feats, iters))

    for i in range(0, iters):

        mean = test_mean + indices[:, i] * mean_diff
        sd = test_sd + indices[:, i] * sd_diff
        skew = test_skew + indices[:, i] * skew_diff

        for j in range(0, nmod):
            model = models[j]
            _, acc[j, i], f1[j, i], _, _ = model.test_model_best_cov(x, y, test_mean=mean, test_sd=sd, test_skew=skew,
                                                                     train_mean=train_mean, train_sd=train_sd,
                                                                     train_skew=train_skew)

    np.savetxt("./logs/" + header + "_cov_shift_acc.csv", acc, delimiter=",")
    np.savetxt("./logs/" + header + "_cov_shift_f1.csv", f1, delimiter=",")
    np.savetxt("./logs/" + header + "_cov_shift_indices.csv", indices, delimiter=",")


# fill in missing parameters (with their defaults) for function in options dictionary
def missing_params(func, options, replace={}):
    # func = function
    # options = options dictionary
    # preplace = dictionary of options to replace
    parameters = inspect.getfullargspec(func)
    npar = len(parameters.args)  # number of arguments
    ndef = len(parameters.defaults)  # number of defaults
    diff = npar - ndef
    for i in range(0, npar):
        if parameters.args[i] in replace.keys():
            opt = replace[parameters.args[i]]
        else:
            opt = parameters.args[i]
        if parameters.args[i] != 'self' and opt not in options.keys():
            if i < diff:
                raise Exception('Missing ' + opt)
            else:
                options[opt] = parameters.defaults[i - diff]

    return options


# Assert DataLoader class
def assert_dl(x, batch_size):
    if isinstance(x, utils_data.DataLoader):
        return x
    else:
        return utils_data.DataLoader(x, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)


# Prints CPU%, # of cores, and Memory %
def check_cpu():
    import psutil
    print('CPU %: ' + str(psutil.cpu_percent()) + ', CPU Cores: ' + str(torch.get_num_threads()) + ', Mem %: ' +
          str(psutil.virtual_memory()[2]))


# main method
def main(pfile):
    # check if folders exist
    check_folder('./data')
    check_folder('./params')

    # add params folder to path
    sys.path.insert(0, './params')

    # load params file and options
    params = __import__(pfile)
    options = params.options()

    # get missing options
    options = missing_params(OptWBoundEignVal, options)

    # init class
    opt = OptWBoundEignVal(model=options['model'], loss=options['loss'], optimizer=options['optimizer'],
                           scheduler=options['scheduler'], mu=options['mu'], K=options['K'], eps=options['eps'],
                           pow_iter_eps=options['pow_iter_eps'], use_gpu=options['use_gpu'],
                           batch_size=options['batch_size'], min_iter=options['min_iter'], max_iter=options['max_iter'],
                           max_pow_iter=options['max_pow_iter'], pow_iter=options['pow_iter'],
                           max_samples=options['max_samples'], ignore_bad_vals=options['ignore_bad_vals'],
                           verbose=options['verbose'], mem_track=options['mem_track'], header=options['header'],
                           num_workers=options['num_workers'], test_func=options['test_func'])

    # get missing options for train & test
    options = missing_params(opt.train, options, replace={'loader': 'train_loader', 'train_loader': 'train_loader_na'})
    options = missing_params(opt.test_test_set, options, replace={'loader': 'test_loader'})
    bs = options['batch_size']

    # Train model
    if ('train' in options.keys() and options['train']) or 'train' not in options.keys():
        opt.train(inputs=options['inputs'], target=options['target'], inputs_valid=options['inputs_valid'],
                  target_valid=options['target_valid'], loader=assert_dl(options['train_loader'], bs),
                  valid_loader=assert_dl(options['valid_loader'], bs),
                  train_loader=assert_dl(options['train_loader_na'], bs))
    """
    try:
        opt.train(inputs=options['inputs'], target=options['target'], inputs_valid=options['inputs_valid'],
                  target_valid=options['target_valid'], loader=options['train_loader'],
                  valid_loader=options['valid_loader'], train_loader=options['train_loader_na'])
    except RuntimeError as e:
        from nvsmi import NVLog
        log = NVLog()
        print(log.as_table())
        print(str(os.system("top -n 1")))
    """
    if ('test' in options.keys() and options['test']) or 'test' not in options.keys():
        if type(options['test_loader']) is list:
            loader = options['test_loader'][0]
        else:
            loader = options['test_loader']
        if 'train' in options.keys() and not options['train']:
            if options['train_loader_na'] is None:
                loader = options['train_loader']
            else:
                loader = options['train_loader_na']
            #opt.test_train_set(options['inputs'], options['target'], loader, fname=options['fname'])
            data = iter(loader).next()
            opt.hvp_op = HVPOperator(opt.model, data, opt.loss, use_gpu=opt.use_gpu)
            opt.comp_rho(p=True)
        # test model on test set
        opt.test_test_set(x=options['x'], y=options['y'], loader=assert_dl(loader, bs), fname=options['fname'])

    # Parse log file
    if (('train' in options.keys() and options['train']) or 'train' not in options.keys()) and\
            (('test' in options.keys() and options['test']) or 'test' not in options.keys()):
        opt.parse()

    # Augmented Testing
    if 'aug_test' in options.keys() and options['aug_test']:
        _, acc, f1 = opt.test_model_best(loader=options['test_loader_aug'], fname=options['fname'])
        print('Aug_Test_Acc\tAug_Test_F1')
        print(str(acc) + '\t' + str(f1))

    # Comparison Test (requires data loader)
    if 'comp_test' in options.keys() and options['comp_test'] and type(options['test_loader']) is list:
        opt.comp_test(options['test_loader'], fname=options['fname'])

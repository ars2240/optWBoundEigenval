# opt.py
#
# Author: Adam Sandler
# Date: 5/7/19
#
# Contains methods & classes for optimizing NNs with spectral radius regulation
#
#
# Dependencies:
#   Packages: random, numpy, torch


import random
import numpy as np
import os
import sys
import torch
import torch.utils.data as utils_data


class HVPOperator(object):
    """
    Modified from: https://github.com/noahgolmant/pytorch-hessian-eigenthings

    Use PyTorch autograd for Hessian Vec product calculation
    """

    def __init__(self, model, data, criterion, use_gpu=True, mem_track=False):
        self.model = model  # NN model
        if use_gpu:
            self.model = self.model.cuda()
        self.data = data  # data (inputs, target)
        self.criterion = criterion  # loss function
        self.use_gpu = use_gpu  # whether or not GPUs are used
        self.stored_grad = None  # stored gradient (on CPU)
        self.stored_grad_gpu = None  # stored gradient (on GPU)
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage

    def Hv(self, vec, storedGrad=False):
        # Returns H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        if self.use_gpu:
            vec = vec.cuda()

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            if self.use_gpu:
                grad_vec = self.stored_grad_gpu
            else:
                grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad().double()
            if self.use_gpu:
                self.stored_grad = grad_vec.cpu()
                self.stored_grad_gpu = grad_vec
            else:
                self.stored_grad = grad_vec

        # compute the product
        grad_product = torch.sum(grad_vec * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        if self.use_gpu:
            hessian_vec_prod = hessian_vec_prod.cpu()

        return hessian_vec_prod.data

    def vGHv(self, vec, storedGrad=False):
        # Returns vec*grad H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        if self.use_gpu:
            vec = vec.cuda()

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            if self.use_gpu:
                grad_vec = self.stored_grad_gpu
            else:
                grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad().double()
            if self.use_gpu:
                self.stored_grad = grad_vec.cpu()
                self.stored_grad_gpu = grad_vec
            else:
                self.stored_grad = grad_vec

        # compute the product
        grad_product = torch.sum(grad_vec * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True, create_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()
        # compute the product
        grad_product = torch.sum(hessian_vec_prod * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        vec_grad_hessian_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        if self.use_gpu:
            vec_grad_hessian_vec = vec_grad_hessian_vec.cpu()
        return vec_grad_hessian_vec.data

    def zero_grad(self):
        # Zeros out the gradient info for each parameter in the model
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        # Compute gradient w.r.t loss over all parameters and vectorize
        inputs, target = self.data

        if self.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        output = self.model(inputs)
        if self.criterion.__class__.__name__ == 'KLDivLoss':
            target_onehot = torch.zeros(np.shape(output))
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            loss = self.criterion(output.float(), target_onehot.float())
        else:
            loss = self.criterion(output, target)
        grad_dict = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_dict]))
        grad_vec = grad_vec.double()  # / len(target)  # convert to double if float
        return grad_vec


class OptWBoundEignVal(object):
    def __init__(self, model, loss, optimizer, scheduler=None, mu=0, K=0, eps=1e-3, pow_iter_eps=1e-3,
                 use_gpu=False, batch_size=128, min_iter=10, max_iter=100, max_pow_iter=1000, max_samples=512,
                 ignore_bad_vals=True, verbose=False, mem_track=False, header=''):
        self.ndim = sum(p.numel() for p in model.parameters())  # number of dimensions
        self.x = 1.0/np.sqrt(self.ndim)*np.ones(self.ndim)  # initial point
        self.f = 0  # loss function value
        self.gradf = np.zeros(self.ndim)  # gradient of f
        self.rho = 0  # spectral radius (maximal absolute value eignevalue)
        self.v = torch.from_numpy(1.0/np.sqrt(self.ndim)*np.ones(self.ndim))  # eigenvector
        self.gradrho = np.zeros(self.ndim)  # gradient of rho
        self.g = 0  # regularizer function
        self.gradg = np.zeros(self.ndim)  # gradient of g
        self.h = 0  # objective function f+mu*g
        self.mu = mu  # coefficient in front of regularizer
        self.K = float(K)  # constant, spectral radius < K
        self.batch_size = batch_size  # batch size
        self.eps = eps  # convergence
        self.pow_iter_eps = pow_iter_eps  # convergence
        self.model = model  # model (from torch)
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = None  # dataloader for training set (from torch)
        self.loss = loss  # loss function (from torch)
        self.optimizer = optimizer  # optimizer function, optional (from torch)
        self.scheduler = scheduler  # learning rate scheduler, optional (from torch)
        self.use_gpu = use_gpu  # whether or not cuda GPUs used
        self.min_iter = min_iter  # minimum number of iterations
        self.max_iter = max_iter  # maximum number of iterations
        self.max_pow_iter = max_pow_iter  # maximum number of power iterations
        self.max_samples = max_samples  # maximum batch size
        self.hvp_op = None  # Hessian-vector operation object
        self.i = 0  # iteration count
        self.norm = 0  # norm of H*v-lambda*v
        self.val_acc = 0  # validation accuracy (only used if validation set is provided)
        self.best_val_acc = 0  # best validation accuracy (only used if validation set is provided)
        self.verbose = verbose  # more extensive read-out
        self.x = None  # input data
        self.y = None  # output data
        name = self.optimizer.__class__.__name__
        if callable(mu):
            mname = 'Func'
        else:
            mname = str(mu)
        # log files
        self.log_file = "./logs/" + header + "_" + name + "_mu" + mname + "_K" + str(K) + ".log"
        self.verbose_log_file = "./logs/" + header + "_" + name + "_mu" + mname + "_K" + str(K) + "_verbose.log"
        self.ignore_bad_vals = ignore_bad_vals  # whether or not to ignore bad power iteration values
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage

    def comp_rho(self):
        # computes rho, v

        v = self.v  # initial guess for eigenvector (prior eigenvector)

        # initialize lambda and the norm
        lam = 0
        norm = 0

        # power iteration
        for i in range(0, np.min([self.ndim, self.max_pow_iter])):
            vnew = self.hvp_op.Hv(v, storedGrad=True)  # compute H*v

            # if converged, break
            lam = np.dot(vnew, v)  # update eigenvalue
            norm = np.linalg.norm(vnew-lam*v)  # norm of H*v-lambda*v

            if norm < self.pow_iter_eps:
                break

            v = 1.0/np.linalg.norm(vnew)*vnew.double()  # update vector and normalize

        self.v = v  # update eigenvector
        self.rho = np.abs(lam)  # update spectral radius
        self.norm = norm  # update norm

        if norm > self.pow_iter_eps:
            print('Warning: power iteration has not fully converged')
            if self.ignore_bad_vals:
                print('Ignoring rho.')
                self.rho = -1  # if value is discarded due to poor convergence, set to -1
                # as negative values of rho work in the other algorithms and are nonsensical

        if lam == 0:
            print('Warning: rho = 0')

    def comp_gradrho(self):
        # computes grad rho

        self.gradrho = self.hvp_op.vGHv(self.v, storedGrad=True)  # compute v*gradH*v

    def comp_f(self, inputs, target):
        # computes f

        # if using gpu, move data to it
        if self.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()
        output = self.model(inputs)  # compute prediction

        # compute loss
        if self.loss.__class__.__name__ == 'KLDivLoss':
            target_onehot = torch.zeros(np.shape(output))
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

        # adjust learning rate
        if self.scheduler is not None:
            self.scheduler.step()

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
            """
            # initialize hessian vector operation class
            self.hvp_op = HVPOperator(self.model, data, self.loss, use_gpu=self.use_gpu)

            self.comp_g()  # compute g

            self.optimizer.zero_grad()  # zero gradient

            # compute grad f
            if self.hvp_op.stored_grad is not None:
                self.gradf = self.hvp_op.stored_grad.data
            else:
                self.gradf = torch.zeros(self.ndim).double()  # set gradient to zero

            # compute grad g
            if self.g > 0:
                self.comp_gradrho()  # compute gradient of rho
                self.gradg = self.gradrho  # compute g
            else:
                self.gradg = torch.zeros(self.ndim).double()  # set gradient to zero

            p = self.gradf + mu * self.gradg  # gradient step
            if self.use_gpu:
                p = p.cuda()  # move to GPU

            i = 0
            for param in self.model.parameters():
                s = param.data.size()  # number of input & output nodes
                l = np.product(s)  # total number of parameters
                param.grad = p[i:(i + l)].view(s).float()  # adjust gradient
                i += l  # increment

            """
            # for testing purposes
            inputs, target = data
            if self.use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()
            output = self.model(inputs)
            loss = self.loss(output, target)  # loss function
            loss.backward()  # back prop


            # optimizer step
            self.optimizer.step()

            # if verbose, add values to file
            if self.verbose:
                log_file = open(self.verbose_log_file, "a")  # open log file
                sys.stdout = log_file  # write to log file
                print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm, np.linalg.norm(self.gradf.detach().numpy()),
                                                  np.linalg.norm(self.gradg.detach().numpy())))
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

            # if using GPU, memory cleanup & tracking
            if self.use_gpu:
                torch.cuda.empty_cache()
                # check max memory usage
                if self.mem_track:
                    self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])
                    print('Running Max GPU Memory used (in bytes): %d' % self.mem_max)

        # compute overall estimates
        f_list = []
        size = []
        # compute f on each batch (to avoid memory issues)
        for _, data in enumerate(self.dataloader):
            inputs, target = data
            f, _ = self.comp_f(inputs, target)
            f_list.append(f)  # compute f on each batch
            size.append(len(target))
        self.f = np.average(f_list, weights=size)  # weighted mean of f values
        # initialize hessian vector operation class for random batch
        self.hvp_op = HVPOperator(self.model, rdata, self.loss, use_gpu=self.use_gpu)
        self.comp_g()  # compute g
        self.h = self.f + mu * self.g  # compute objective function

    def train(self, inputs, target, inputs_valid=None, target_valid=None):

        self.x = inputs  # input data
        self.y = target  # output data

        # make sure logs file exists
        if not os.path.exists('./logs'):
            os.mkdir('./logs')

        old_stdout = sys.stdout  # save old output

        f_hist = []  # tracks function value after each epoch

        # create dataloader
        train_data = utils_data.TensorDataset(self.x, self.y)
        self.dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size)

        log_file = open(self.log_file, "w")  # open log file
        sys.stdout = log_file  # write to log file

        # header of log file
        if (inputs_valid is None) or (target_valid is None):
            print('epoch\t f\t rho\t h\t norm')
        else:
            print('epoch\t f\t rho\t h\t norm\t val_acc')

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        for self.i in range(0, self.max_iter):
            self.iter()  # take step

            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file

            # add values to log file
            if (inputs_valid is None) or (target_valid is None):
                print('%d\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm))
            else:
                _, self.val_acc = self.test_model(inputs_valid, target_valid)
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
                    torch.save(self.model.state_dict(), 'trained_model_best.pt')
                print('%d\t %f\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm, self.val_acc))

            # add function value to history log
            f_hist.append(self.h)

            # check if convergence criteria met
            if self.i >= (self.min_iter - 1):
                coef_var = np.std(f_hist[-10:])/np.abs(np.mean(f_hist[-10:]))
                if coef_var <= self.eps:
                    print(coef_var)
                    break

            if self.i < (self.max_iter - 1):
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

        # Save model weights
        torch.save(self.model.state_dict(), 'trained_model.pt')

        # best validation accuracy
        print('Best Validation Accuracy:', self.best_val_acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        # compute loss & accuracy on training set
        self.test_train_set(inputs, target)

    def test_model(self, X, y):
        # Computes the loss and accuracy of model on given dataset

        test_data = utils_data.TensorDataset(X, y)
        dataloader = utils_data.DataLoader(test_data, batch_size=self.batch_size)

        f_list = []
        acc_list = []
        size = []
        for _, data in enumerate(dataloader):

            inputs, target = data

            # compute loss
            f, ops = self.comp_f(inputs, target)
            f_list.append(f)

            # compute accuracy
            _, predicted = torch.max(ops.data, 1)
            acc = torch.mean((predicted == target).float()).item() * 100
            acc_list.append(acc)

            # size of dataset
            size.append(len(target))

        test_loss = np.average(f_list, weights=size)  # weighted mean of f values
        test_acc = np.average(acc_list, weights=size)  # weighted mean of f values

        return test_loss, test_acc

    def test_model_best(self, X, y):
        # tests best model, loaded from file

        self.model.load_state_dict(torch.load('trained_model_best.pt'))

        return self.test_model(X, y)

    def test_train_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc = self.test_model_best(X, y)  # test best model

        print('Train Loss:', loss)
        print('Train Accuracy:', acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def test_test_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc = self.test_model_best(X, y)  # test best model

        print('Test Loss:', loss)
        print('Test Accuracy:', acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output




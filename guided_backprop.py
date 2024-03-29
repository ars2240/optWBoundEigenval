# Guided Backprop
# Modified from: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py

import torch
from torch.nn import ReLU


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.modules())[0]  # [1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, inputs, target_class, mc):
        # Forward pass
        inputs.requires_grad_()
        output = self.model(inputs)
        output = output[:, mc]
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1
        # Backward pass
        # model_output.backward(gradient=one_hot_output)
        output.backward(gradient=target_class)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        # gradients_as_arr = self.gradients.data.numpy()[0]
        # gradients_as_arr = self.gradients.data
        gradients_as_arr = inputs.grad.data
        return gradients_as_arr, output

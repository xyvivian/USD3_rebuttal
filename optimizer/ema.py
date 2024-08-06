# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py
import torch
import lightning as L
import logging

logger = logging.getLogger(__name__)

class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self,decay):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.shadow_params = None
        self.collected_params = []

    def update(self, parameters):
        if self.shadow_params is None:
            self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        else:    
            one_minus_decay = 1.0 - self.decay
            with torch.no_grad():
                parameters = [p.clone().detach() for p in parameters if p.requires_grad]
                for idx in range(len(self.shadow_params)):
                    self.shadow_params[idx] = self.shadow_params[idx].to(parameters[0].device)
                for s_param, param in zip(self.shadow_params, parameters):
                    s_param.sub_(one_minus_decay * (s_param - param))
                

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for idx in range(len(self.shadow_params)):
            self.shadow_params[idx] = self.shadow_params[idx].to(parameters[0].device)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


    def state_dict(self):
        return dict(decay=self.decay, shadow_params=self.shadow_params)


    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']
        print("EMA parameters are successfully loaded...")
        
        
        
class EMACallback(L.pytorch.callbacks.Callback):
    def __init__(self,parameters=None,wait_steps=1,decay=0.999):
        super().__init__()
        self.decay = decay
        self.wait_steps = wait_steps
        self.ema_model = ExponentialMovingAverage(self.decay)
        
    def load_ema_checkpoints_to_params(self,parameters):
        self.ema_model.copy_to(parameters)
        
    def on_train_batch_end(self,trainer,pl_module, *args, **kwargs):
        if trainer.global_step >= self.wait_steps:
            self.ema_model.update(pl_module.module.parameters())
            
            
    def load_state_dict(self,state_dict):
        logger.info('Loading EMA into the model')
        self.ema_model.load_state_dict(state_dict)
    def state_dict(self):
        return self.ema_model.state_dict()
        
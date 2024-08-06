import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.diffusion = diffusion = ml_collections.ConfigDict()
    config.training = training = ml_collections.ConfigDict()
    config.model = model = ml_collections.ConfigDict()
    config.data = data = ml_collections.ConfigDict()
    config.sampler = sampler = ml_collections.ConfigDict()
    config.dataset_name="cifar10"
    config.simplified_vlb = False

    #--------------Important HPs for training ------------------
    diffusion.num_steps = 0 # 0 for continuous, 1000 for discrete
    diffusion.noise_type = 'uniform'
    diffusion.noise_schedule_type = 'loglinear' #'cosine' #'loglinear'
    diffusion.noise_schedule_args = {'eps': 1e-4} #{'alpha': 0.08}
    diffusion.condition_dim = 0 #unconditional text generation
    diffusion.num_classes = 512  #26+1 for text generation
    diffusion.min_time = 1e-2
    diffusion.nll_weight = 0.001 #weight on CE
    diffusion.simplified_max_val=1

    ######### --------------------- change here ----------------------
    sampler.num_intermediates = 1
    sampler.mcmc_step_size = 0.0
    sampler.mcmc_num_steps = 0
    sampler.mcmc_start_ratio = 0.1
    

    #---------- Distributed or not --------------------------------
    training.distributed = True
    training.num_gpus = 1
    training.lr = 3e-4
    training.beta1 = 0.9
    training.beta2 = 0.98
    training.weight_decay = 1e-1
    training.num_warmup_steps = 5000
    training.num_training_steps = 800_000
    training.ema_wait_steps = 5000
    training.ema_decay=0.999
    training.enable_16_precision=False
    
    #---------------- Architecture Variables -----------------
    model.hidden_size= 768
    model.length = 64
    model.n_blocks=  12
    model.n_heads = 12
    model.dropout = 0.1
    model.scale_by_sigma = False
    model.ffn_unit = 'glu'
    model.prenorm = False
    model.name = 'DDitTransformer' #'DDitTransformer' # 
 
    #---------------- Data -----------------------------
    data.name = 'cifar10'
    data.train_num_workers =0
    data.train_batch_size = 512
    data.test_num_workers = 0
    data.test_batch_size = 512
    data.val_num_workers = 0
    data.val_batch_size = 512

    #-------------------Other training variables---------------
    

    config.exp_name = f'{model.name}_{data.name}_lr_{training.lr}_wd_{training.weight_decay}_nll_{diffusion.nll_weight}_l2_{config.simplified_vlb}' #change discrete_diffusion when using ctmc only or ctmcplus
    return config

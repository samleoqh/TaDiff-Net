from munch import DefaultMunch

# -----------------------------------------------
# model config 
network = 'TaDiff_Net' 
data_pool = ['sailor', 'lumiere']
# ms3
data_dir = {'sailor': '/home/brian/project/pl_ddpm/src/data/sailor_npy', 
            # 'sailor': '/home/brian/project/pl_ddpm/data_cp/sailor_npy',
            'lumiere': '/home/brian/project/pl_ddpm/src/data/lumiere_npy'}
# # ms 2
# data_dir = {'sailor': '/mnt/CRAI-NAS/all/brian/dataset/sailor_npy', 
#             'lumiere': '/mnt/CRAI-NAS/all/brian/dataset/lumiere_npy'}

image_size = 192
in_channels = 13 
out_channels = 7
num_intv_time = 3
model_channels = 32
num_res_blocks = 2
channel_mult = (1,2,3,4)
attention_resolutions = [8,4]
num_heads = 4
num_classes = 81  # treat_code

max_T = 1000 # diffusion steps
ddpm_schedule = 'linear' # 'linear', 'cosine', 'log'

# -----------------------------------------------
# optimizer, lr, loss, train config 
opt = 'adamw' # adam, adamw, sgd, adan
# momentum = 0.99
# betas=(0.9, 0.999) # for adam

lr = 5.e-3
max_epochs = 3000 # total number of training epochs 
max_steps = 60000 # total number of training iterations
weight_decay = 3e-5
lrdecay_cosine = True
lr_gamma=0.585 # 0.5, 0.575, 0.65, 0.585
warmup_steps = 100

loss_type = 'ssim' # or mse
batch_size = 1
sw_batch = 16 # total bach = sw_batch * batch_size 
num_workers = 8 # 4, 8, 16 up to 32 normally num of CPU cores

grad_clip = 1.5
accumulate_grad_batches = 4 # simulate larger batch size to save GPU mem
n_repeat_tr = 10   # simulate larger train dataset by repeating it
n_repeat_val = 5  # simulate larger val data by repeating 

cache_rate = 0.  # cache all data in memory  # 1.


# -----------------------------------------------
# I/O, system and log config for trainer (e.g. lightning)
# wandb_project = network + '_' + dataset_name
# exp_name = network + '_' + dataset_name + '_' + kfold
wandb_entity = "qhliu"
logdir = './ckpt'
log_interval = 1
seed = 114514 # 5000 # 114514 #5000,114514,3407
gpu_devices = '0, 1' # str or int e.g. 0, 1, 2, 3
gpu_strategy = "ddp"
gpu_accelerator = "gpu"
precision = 32 # '32' # 16-mixed, 32

val_interval_epoch = 10 # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 not support val_interval_step now
# val_interval_step = 50
resume_from_ckpt = False
ckpt_best_or_last = None # if not none, will load ckpt for val or resume training
ckpt_save_top_k = 3
ckpt_save_last = True
ckpt_monitor ="val_loss" # val_loss/val_dice need match ckpt_mode and ckpt_filename
ckpt_filename = "ckpt-{epoch}-{step}-{val_loss:.6f}" # for checkpoint callback
ckpt_mode = "min" # for checkpoint callback

do_train_only = False
do_test_only = False


# -----------------------------------------------
# -----------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and 
               isinstance(v, (int, float, bool, str, list, tuple, dict))]
config = {k: globals()[k] for k in config_keys}
config = DefaultMunch.fromDict(config)

# # print config for debug
# for key, value in config.items():
#     print(key, value)

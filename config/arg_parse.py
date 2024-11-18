import argparse


def load_args(cfg):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--gpu_devices", type=str, default=cfg.gpu_devices) # default = -1

    ## MODE
    # parser.add_argument("--do_train_only", type=str2bool, default=False)
    # parser.add_argument("--do_test_only", type=str2bool, default=False)
    
    # python 3.9 and above support BooleanOptionalAction 
    # parser.add_argument('--do_train_only', default=False, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--do_test_only', default=False, action=argparse.BooleanOptionalAction)
    
    parser.add_argument('--do_train_only', default=False, action='store_true')
    parser.add_argument('--do_test_only', default=False, action='store_true')

    ## TRAIN
    parser.add_argument("--max_epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)

    parser.add_argument("--precision", type=str, default=cfg.precision) # 16-mixed, 32
    parser.add_argument("--accumulate_grad_batches", type=int, default=cfg.accumulate_grad_batches)

    ## VALIDATION
    parser.add_argument("--val_interval_epoch", type=int, default=cfg.val_interval_epoch) # 
    # parser.add_argument("--val_interval_step", type=int, default=200)

    args = parser.parse_args()
    return args

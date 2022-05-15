import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='./DataSet/simulation_rabbit_dog_480x480_8/', 
                        help='root directory of dataset') 
    parser.add_argument('--img_wh', nargs="+", type=int, default=[480, 480], help='resolution (img_w, img_h) of the image')
    parser.add_argument('--ctf_wh', nargs="+", type=int, default=[480, 480], help='resolution (ctf_w, ctf_h) of the CTF')

    parser.add_argument('--exp_name', type=str, default= 'test_simulation', help='experiment name')                    
    parser.add_argument('--pai_scale', type=float, default=0.5, help='pai multiply scale')
    parser.add_argument('--simulation', type=bool, default=True, help='simulation: True, real-world data: False')
    parser.add_argument('--joint_training', default=False, action="store_true", help='joint training')
    parser.add_argument('--train_model', type=int, default=0, help='train amp and phase or z') # 0: amp, pha; 1: z

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--ckpt_path', type=str, default = None, help='pretrained checkpoint path to load')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--dataset_name', type=str, default='lensless', choices=['lensless'], help='which dataset to train/val')
    parser.add_argument('--img_num', type=int, default=8, help='number of raw images')
   
    parser.add_argument('--use_disp', default=True, action="store_true", help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0, help='std dev of noise added to regularize sigma')
        
    # loss function
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse'], help='loss to use')
    parser.add_argument('--chunk', type=int, default=250000, help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')    
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'], help='the prefixes to ignore in the checkpoint state dict')

    # params for optimizer and scheduler 
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type', choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--momentum', type=float, default=0.9, help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr', help='scheduler type', choices=['steplr', 'cosine', 'poly'])
                        
    # params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0, help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Gradually warm-up(increasing) learning rate in optimizer')
    
    parser.add_argument('--decay_step', nargs='+', type=int, default=[3000, 6000], help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1, help='learning rate decay amount')
                        
    return parser.parse_args()

import sys
from yaml import dump
from os import path, name
import Src.Utils.Utils as Utils
import numpy as np
import torch
from collections import OrderedDict
from Src.Utils.passenger_utility import validate_choice_utility_policy

class Config(object):
    def __init__(self, args):
        algo_aliases = {"DSPO_plus_SPO": "DRPO"}
        if args.algo_name in algo_aliases:
            args.legacy_algo_name = args.algo_name
            args.algo_name = algo_aliases[args.algo_name]

        # Path setup
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))

        # Reproducibility
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Save results after every certain number of episodes
        self.save_after = args.max_episodes // args.save_count if args.max_episodes >= args.save_count else args.max_episodes

        # Add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        if args.pricing:
            self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name,'pricing', args.algo_name, folder_suffix)
        else:
            self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name,'offering', args.algo_name, folder_suffix)
            
        if name == 'nt':
             suffix = '\\'
        else:
             suffix = '/'
        path_prefix = [self.paths['experiment'], str(args.seed)]
        self.paths['logs'] = path.join(*path_prefix, 'Logs'+suffix)
        self.paths['checkpoint'] = path.join(*path_prefix, 'Checkpoints'+suffix)
        self.paths['results'] = path.join(*path_prefix, 'Results'+suffix)

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                Utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = Utils.Logger(self.paths['logs'], args.log_output)

        #load data
        if args.load_data:
            self.coords,self.dist_matrix,self.n_parcelpoints,self.adjacency,self.service_times = Utils.load_demand_data(self.paths['root'],args.instance,args.data_seed,args.clip_service_time,args.truck_speed,k=args.k,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            self.coords_test,self.dist_matrix_test,self.n_parcelpoints_test,self.adjacency_test,self.service_times_test = Utils.load_demand_data(self.paths['root'],args.instance,args.data_seed_test,args.clip_service_time,args.truck_speed,k=args.k,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            self.choice_util_matrix = Utils.load_choice_utility_matrix(self.paths['root'],args.instance,args.data_seed,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            self.choice_util_matrix_test = Utils.load_choice_utility_matrix(self.paths['root'],args.instance,args.data_seed_test,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            self.walking_distance_matrix = Utils.load_walking_distance_matrix(self.paths['root'],args.instance,args.data_seed,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            self.walking_distance_matrix_test = Utils.load_walking_distance_matrix(self.paths['root'],args.instance,args.data_seed_test,n_passengers=args.n_passengers,yanjiao_prefix=getattr(args, 'yanjiao_prefix', None))
            validate_choice_utility_policy(
                self.choice_util_matrix,
                final_yanjiao_mode=getattr(args, 'final_yanjiao_mode', False),
                allow_derived_choice_utility=getattr(args, 'allow_derived_choice_utility', False),
            )
            validate_choice_utility_policy(
                self.choice_util_matrix_test,
                final_yanjiao_mode=getattr(args, 'final_yanjiao_mode', False),
                allow_derived_choice_utility=getattr(args, 'allow_derived_choice_utility', False),
            )
        else:#only used for debug purpose
            self.coords = Utils.generate_demand_data(100)
            self.dist_matrix = []
            self.n_parcelpoints = 6
            self.adjacency = np.ones(6)
            self.service_times = np.ones(100)
            self.choice_util_matrix = None
            self.walking_distance_matrix = None
            
            self.coords_test = Utils.generate_demand_data(100)
            self.dist_matrix_test = []
            self.n_parcelpoints_test = 6
            self.adjacency_test = np.ones(6)
            self.service_times_test = np.ones(100)
            self.choice_util_matrix_test = None
            self.walking_distance_matrix_test = None

        # Get the domain and algorithm
        self.env = self.get_domain(args.env_name, args=args,path=path.join(self.paths['root'], 'Environments'))
        self.env.seed(seed)
        
        self.test_env = self.get_domain(args.env_name, args=args,path=path.join(self.paths['root'], 'Environments'),test_env=True)
        self.test_env.seed(seed)

        # Set Model
        self.algo = Utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name, load_class=True)

        # GPU
        # args.gpu 可以是 0（使用第一个GPU）或 None/False（不使用GPU）
        # 如果 args.gpu 是整数（包括0），则尝试使用GPU
        if args.gpu is not None and args.gpu >= 0:
            if torch.cuda.is_available():
                gpu_id = int(args.gpu)
                self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cuda:0")
                print(f'Using GPU device: {self.device}')
                print(f'Number of GPUs available: {torch.cuda.device_count()}')
                self.cuda = 1
            else:
                self.device = torch.device("cpu")
                print('CUDA not available, using CPU')
                self.cuda = 0
        else:
            self.device = torch.device("cpu")
            self.cuda = 0

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')


        print("=====Configurations=====\n", args)

    # Load the domain
    def get_domain(self, tag, args, path,test_env=False):
        if tag[:11] == 'Parcelpoint':
            obj = Utils.dynamic_load(path, tag, load_class=True)
            # 获取退出阈值和外部选项效用（如果存在）
            quit_threshold = getattr(args, 'quit_threshold', None)
            outside_option_util = getattr(args, 'outside_option_util', None)
            # 调试输出：验证参数值
            print(f"[DEBUG config.get_domain] quit_threshold = {quit_threshold}, outside_option_util = {outside_option_util}")
            
            if test_env:
                env = obj(model=args.algo_name,max_steps_r=args.max_steps_r,max_steps_p=args.max_steps_p,pricing=args.pricing,n_vehicles=args.n_vehicles,
                      veh_capacity=args.veh_capacity,parcelpoint_capacity=args.parcelpoint_capacity,fraction_capacitated=args.fraction_capacitated,incentive_sens=args.incentive_sens,base_util=args.base_util,
                      home_util=args.home_util,reopt=args.reopt,load_data=args.load_data,coords=self.coords_test,dist_matrix=self.dist_matrix_test,
                      n_parcelpoints=self.n_parcelpoints_test,adjacency=self.adjacency_test,service_times=self.service_times_test,dissatisfaction=self.dissatisfaction,hgs_time=args.hgs_reopt_time,
                      l0_home=args.l0_home,l_mp=args.l_mp,route_label_mode=args.route_label_mode,quit_threshold=quit_threshold,outside_option_util=outside_option_util,
                      choice_util_matrix=getattr(self, 'choice_util_matrix_test', None),walking_distance_matrix=getattr(self, 'walking_distance_matrix_test', None),
                      travel_time_weight=getattr(args, 'travel_time_weight', None),walk_distance_weight=getattr(args, 'walk_distance_weight', None))
            else:
                env = obj(model=args.algo_name,max_steps_r=args.max_steps_r,max_steps_p=args.max_steps_p,pricing=args.pricing,n_vehicles=args.n_vehicles,
                      veh_capacity=args.veh_capacity,parcelpoint_capacity=args.parcelpoint_capacity,fraction_capacitated=args.fraction_capacitated,incentive_sens=args.incentive_sens,base_util=args.base_util,
                      home_util=args.home_util,reopt=args.reopt,load_data=args.load_data,coords=self.coords,dist_matrix=self.dist_matrix,
                      n_parcelpoints=self.n_parcelpoints,adjacency=self.adjacency,service_times=self.service_times,dissatisfaction=self.dissatisfaction,hgs_time=args.hgs_reopt_time,
                      l0_home=args.l0_home,l_mp=args.l_mp,route_label_mode=args.route_label_mode,quit_threshold=quit_threshold,outside_option_util=outside_option_util,
                      choice_util_matrix=getattr(self, 'choice_util_matrix', None),walking_distance_matrix=getattr(self, 'walking_distance_matrix', None),
                      travel_time_weight=getattr(args, 'travel_time_weight', None),walk_distance_weight=getattr(args, 'walk_distance_weight', None))
            return env

if __name__ == '__main__':
    pass

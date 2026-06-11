from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch import float32
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync, name
import importlib
from time import time
import sys
from Environments.OOH.containers import Location,Vehicle,Fleet
from math import trunc, sqrt
import hygese
import json

# 随机种子由 config.py 中的 args.seed 控制，不在这里硬编码
# np.random.seed(0)
# torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, method): # restore
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method
        self.log_path = log_path
        self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())


def total_costs(count_home,service_times,travel_time,discount_costs,charge_revenue,config):
    cost_multiplier = (config.driver_wage+config.fuel_cost) / 3600
    total_costs = (service_times+travel_time)*cost_multiplier + sum(discount_costs) - sum(charge_revenue)
    total_costs += count_home*config.home_failure*config.failure_cost#costs of failed delivery
    
    return total_costs

def plot_training_curves(rewards,config):
    plt.figure()
    plt.ylabel("Monetary unit")
    plt.xlabel("Episode")
    plt.title("Performance (operational costs including pricing revenue/costs)")
    plt.plot(rewards)
    plt.savefig(config.paths['results'] + "training_curve.png")
    plt.close()
    
    np.save(config.paths['results'] + "training_curve", rewards)

def save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
                      parcel_lockers_remaining_capacity,home_delivery_loc,step_time,price_time,config):
    
    np.save(config.paths['results'] + "price_time", price_time)
    np.save(config.paths['results'] + "travel_time", travel_time)
    np.save(config.paths['results'] + "total_cost", total_cost)
    np.save(config.paths['results'] + "actions", actions)
    np.save(config.paths['results'] + "accepted_price", accepted_price)
    np.save(config.paths['results'] + "count_home_delivery", count_home_delivery)
    np.save(config.paths['results'] + "service_time", service_time)
    np.save(config.paths['results'] + "parcel_lockers_remaining_capacity", parcel_lockers_remaining_capacity)
    np.save(config.paths['results'] + "home_delivery_loc", home_delivery_loc)
    np.save(config.paths['results'] + "step_time", step_time)

def plot_test_boxplot(rewards,prices,step_time,config):
    
    plt.figure()
    plt.ylabel("Step time")
    plt.title("Performance")
    plt.boxplot(step_time)
    plt.savefig(config.paths['results'] + "box_eval_step_time.png")
    plt.close()
    
    sum_rewards=[]
    sum_prices=[]
    for i in rewards:
        sum_rewards.append(i[0])
    for i in prices:
        sum_prices.append(i[0])
    
    plt.figure()
    plt.ylabel("Monetary unit")
    plt.title("Performance")
    plt.boxplot(sum_rewards)
    plt.savefig(config.paths['results'] + "box_eval_costs.png")
    plt.close()
    
    plt.figure()
    plt.ylabel("Monetary unit")
    plt.title("Performance")
    plt.boxplot(sum_prices)
    plt.savefig(config.paths['results'] + "box_eval_discounts.png")
    plt.close()
    
def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, file)
        if not exact and name in file:
            return path.join(dir, file)
    # recursive scan
    for file in all_files:
        if file == 'Experiments':
            continue
        _path = path.join(dir, file)
        if path.isdir(_path):
            location = search(_path, name, exact)
            if location:
                return location
    return None

def dynamic_load(dir, name, load_class=False):
    try:
        file_path = search(dir, name)
        if file_path is None:
            raise ValueError(f"Could not find file containing '{name}' in directory '{dir}'")
        
        abs_path = file_path.split('/')[1:]

        if len(abs_path) == 0:
            abs_path = file_path.split('\\')[1:]
        pos = abs_path.index('ooh_code')

        # Remove .py extension from the last element if present
        module_parts = [str(item) for item in abs_path[pos + 1:]]
        if len(module_parts) > 0 and module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]  # Remove .py extension
        
        module_path = '.'.join(module_parts)
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError("Failed to dynamically load the class: " + name + " - " + str(e))

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    dir_path = str.split(dir_path, sep=sepa)[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join(sepa, *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def writeCVRPLIB(fleet,filename,pathh,n_cust,n_veh):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    pathh = pathh + sepa+'Src'+ sepa+'Algorithms'+sepa+'CVRPLIB'+sepa
    folder = str(n_cust+1)+'_'+str(n_veh)+sepa
    if filename==0:
        if not path.exists(pathh+folder):
            create_directory_tree(pathh+folder)
        # else:
        #     while input("Folder exists already, possibly overwriting existing files, do you want to continue? [y/n]") == "n":
        #         exit('Exit program, folder exists')
    route=[]
    for v in range(len(fleet["fleet"])):
        xy=[]
        for i in fleet["fleet"][v]["routePlan"]:
            xy.append(str(i.x)+'\t'+str(i.y)+'\t'+str(i.id_num))
        route.append(xy)
    with open(pathh+folder+'CVRPLIB'+str(filename)+'.txt', 'w') as fp:
        for v in range(len(fleet["fleet"])):
            fp.write("Route_"+str(v)+'\n')
            fp.write('\n'.join(route[v]))
            fp.write('\n')

def readCVRPLIB(pathh,v_cap,n_veh):
    historicRoutes = np.empty(0)
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    pathh = pathh+sepa+'Src'+sepa+'Algorithms'+sepa+'CVRPLIB'+sepa+str(v_cap*n_veh)+'_'+str(n_veh)
    if path.exists(pathh):
        for filename in listdir(pathh):
            f = path.join(pathh, filename)
            # checking if it is a file
            if path.isfile(f):
                file = open(f, "r")
                routeplans=[ [] for i in range(n_veh)]
                idx = -1
                for i in file:
                    if not i.startswith('Route'):
                        loc = i.strip().split('\t')
                        try:
                            loc = Location(float(loc[0]),float(loc[1]),int(loc[2]),0)#we do not care about time here
                        except:
                            break#fewer vehicles than maximum, so we break
                        routeplans[idx].append(loc)
                    else:
                        idx +=1
                vehicles=[]
                for v in range(n_veh):
                    if len(routeplans[v]) > 0:
                        vehicles.append(Vehicle(routeplans[v],v_cap,v))
                historicRoutes = np.append(historicRoutes,Fleet(vehicles))
        return historicRoutes
    else:
        raise ValueError("Failed to load the historic routes: " + str(v_cap*n_veh)+'_'+str(n_veh) )

def sixhump_func(x,y):
    """
    for documentation and visualisation of the sixhump camelback function, see: https://www.sfu.ca/~ssurjano/camel6.html
    """
    return (4-2.1*x**2+(x**4/3))*x**2+x*y+(-4+4*y**2)*x**2+6

def calculate_service_time(coords,clip_service_time):
    """
    We project the coordinates onto the domain [-3,3]x[-2,2] and next calculate service times using the 6-hump camel function
    """
    max_xcoord = max(coords, key=lambda x: x.x).x
    max_ycoord = max(coords, key=lambda y: y.y).y
    min_xcoord = min(coords, key=lambda x: x.x).x
    min_ycoord = min(coords, key=lambda y: y.y).y
    diff_x = max_xcoord-min_xcoord
    diff_y = max_ycoord-min_ycoord
    
    mult_x = 6
    mult_y = 4
    #standardize x-min_x/diff, we use domain: x-[-3,3] y-[-2,2]
    service_times = np.zeros([0])
    for coord in coords:
        x1 = (((coord.x-min_xcoord)/diff_x)*mult_x)-3
        y1 = (((coord.y-min_ycoord)/diff_y)*mult_y)-2
        sixhump = np.around(np.clip(sixhump_func(x1,y1),1,clip_service_time),decimals=2)*60#times 60 to convert from minutes to seconds
        service_times = np.append(service_times,sixhump)
        
    return service_times

def getdistance_euclidean(a,b):
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def _load_demand_data_legacy_unused(pathh,instance,data_seed,clip_service_time,truck_speed,k=20):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    if instance=='Austin' or instance=='Seattle':
        instance_folder = 'Amazon_data'
        instance_size = '_700_'
    elif instance=='Beijing_bus':
        instance_folder = 'Beijing_bus'
        instance_size = '_410_'
        instance_subfolder = 'bus'  # Beijing_bus 数据在 bus 子文件夹中
    else:
        instance_folder = 'HombergerGehring_data'
        instance_size = '_90_'
        instance_subfolder = None
    
    # 构建路径
    if instance == 'Beijing_bus':
        pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa+instance_subfolder+sepa
    else:
        pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa+instance+sepa
    
    if path.exists(pathh):
        # 构建文件路径
        if instance == 'Beijing_bus':
            pathh = pathh+'bus'+instance_size+str(data_seed)
        else:
            pathh = pathh+instance+instance_size+str(data_seed)
        
        f = pathh+"_coords.txt"
        if path.isfile(f):
            file = open(f, "r")
            coords = np.zeros([0])
            for j,i in enumerate(file):
                if not i.startswith('NODE'):
                    loc = i.strip().split('\t')
                    loc = Location(float(loc[1]),float(loc[2]),j-1,0)
                    coords = np.append(coords,loc)
        dist_matrix = np.empty(shape=(0,len(coords)),dtype=int)
        if instance_folder=='Amazon_data':
            f = pathh+"_dist_matrix.txt"
            if path.isfile(f):
                file = open(f, "r")
                for i in file:
                    if not i.startswith('EDGE'):
                        loc = i.strip().split('\t')
                        dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
        else:
            # 对于 HombergerGehring_data 和 Beijing_bus，使用欧氏距离计算距离矩阵
            for i in coords:
                dist = []
                for j in coords:
                    dist.append( int(getdistance_euclidean(i,j)) / truck_speed * 3600)
                dist_matrix = np.vstack([dist_matrix,np.array(dist)])
                    
    else:
         raise ValueError("Failed to load the demand data: " + instance + instance_size + str(data_seed) )
   
    #the number of parcelpoints contained in the dataset
    n_parcelpoints = 0
    if instance=='Austin':
        n_parcelpoints=278
        adjacency_file = pathh+f"_adjacency{k}.npy"
        if path.isfile(adjacency_file):
            adjacency = np.load(adjacency_file)  # k closest parcelpoints to each customer
        else:
            # 如果没有 adjacency 文件，则在线生成并保存（避免手动预处理）
            n_customers = len(dist_matrix) - n_parcelpoints
            adjacency = np.zeros(shape=(n_customers, n_parcelpoints), dtype=int)
            for i in range(0, n_customers):
                closest = np.argsort(dist_matrix[i][-n_parcelpoints:])[:k]
                adjacency[i][closest] = 1
            np.save(pathh+f"_adjacency{k}", adjacency)
    elif instance=='Seattle':
        n_parcelpoints=299
        adjacency_file = pathh+f"_adjacency{k}.npy"
        if path.isfile(adjacency_file):
            adjacency = np.load(adjacency_file)  # k closest parcelpoints to each customer
        else:
            n_customers = len(dist_matrix) - n_parcelpoints
            adjacency = np.zeros(shape=(n_customers, n_parcelpoints), dtype=int)
            for i in range(0, n_customers):
                closest = np.argsort(dist_matrix[i][-n_parcelpoints:])[:k]
                adjacency[i][closest] = 1
            np.save(pathh+f"_adjacency{k}", adjacency)
    elif instance=='Beijing_bus':
        n_parcelpoints=169  # 索引241-409，共169个OOH点
        # 检查是否有 adjacency 文件
        adjacency_file = pathh+f"_adjacency{k}.npy"
        if path.isfile(adjacency_file):
            adjacency = np.load(adjacency_file)
        else:
            # 如果没有 adjacency 文件，为每个客户找到最近的k个OOH点
            # 注意：此修改仅影响 Beijing_bus 数据集，不影响其他数据集
            # 根据 find_closest_parcelpoints 的逻辑：
            # - adjacency矩阵形状：(n_customers, n_parcelpoints)
            # - 客户索引1-240对应adjacency行索引0-239
            # - OOH点索引241-409对应adjacency列索引0-168
            # - dist_matrix中，行索引=客户索引，列索引=位置索引
            # - 代码中 mask[mask.mask].data 获取被掩码的元素，所以 adjacency[i][j]=1 表示可访问
            
            n_customers = 240  # 索引1-240，共240个客户
            
            # 初始化adjacency矩阵：全0表示所有OOH点都不可访问（会被掩码）
            # 然后为每个客户设置最近的k个OOH点为可访问（adjacency=1）
            adjacency = np.zeros(shape=(n_customers, n_parcelpoints), dtype=int)
            
            # 为每个客户找到最近的k个OOH点
            for customer_idx in range(1, n_customers + 1):  # 客户索引1-240
                # 获取该客户到所有OOH点的距离
                # dist_matrix[customer_idx] 是客户customer_idx到所有位置的距离
                # OOH点在dist_matrix中的列索引是241-409
                ooh_distances = dist_matrix[customer_idx][241:410]
                # 找到最近的k个OOH点（相对于OOH点列表的索引0-168）
                closest_indices = np.argsort(ooh_distances)[:k]
                # 根据 find_closest_parcelpoints 的逻辑，设置为1表示可访问
                for j in closest_indices:
                    adjacency[customer_idx - 1][j] = 1  # customer_idx-1是adjacency的行索引
            # 保存生成的 adjacency，便于下次直接加载
            np.save(pathh+f"_adjacency{k}", adjacency)
    else:
        n_parcelpoints=10
        adjacency_file = pathh+f"_adjacency{k}.npy"
        if path.isfile(adjacency_file):
            adjacency = np.load(adjacency_file)
        else:
            # Build k-nearest adjacency for C/R/RC so --k is effective.
            n_customers = len(dist_matrix) - n_parcelpoints
            adjacency = np.zeros(shape=(n_customers, n_parcelpoints), dtype=int)
            k_eff = min(int(k), n_parcelpoints)
            for i in range(0, n_customers):
                closest = np.argsort(dist_matrix[i][-n_parcelpoints:])[:k_eff]
                adjacency[i][closest] = 1
            np.save(pathh+f"_adjacency{k}", adjacency)
        
    
    #service times drawn from 6-hump camelback
    service_times = calculate_service_time(coords,clip_service_time)
    
    return coords,dist_matrix,n_parcelpoints,adjacency,service_times

def load_demand_data(
        pathh, instance, data_seed, clip_service_time, truck_speed,
        k=20, n_passengers=300, yanjiao_prefix=None):
    """Load legacy instances plus the NYC_TLC pilot instance.

    NYC_TLC uses precomputed OSRM duration matrices. Its adjacency matrix keeps
    the legacy runtime convention where row 0 is a depot placeholder and
    customer rows are addressed by Location.id_num.
    """
    if name == 'nt':
        sepa= '\\'
    else:
        sepa='/'

    if instance=='Austin' or instance=='Seattle':
        instance_folder = 'Amazon_data'
        instance_size = '_700_'
        instance_subfolder = None
    elif instance=='Beijing_bus':
        instance_folder = 'Beijing_bus'
        instance_size = '_410_'
        instance_subfolder = 'bus'
    elif instance=='Beijing_Yanjiao':
        instance_folder = 'Beijing_Yanjiao'
        instance_size = f'_{n_passengers}_'
        instance_subfolder = None
    elif instance=='NYC_TLC':
        instance_folder = 'NYC_TLC'
        instance_size = None
        instance_subfolder = 'pilot'
    else:
        instance_folder = 'HombergerGehring_data'
        instance_size = '_90_'
        instance_subfolder = None

    if instance == 'Beijing_bus' or instance == 'NYC_TLC':
        dataset_dir = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa+instance_subfolder+sepa
    elif instance == 'Beijing_Yanjiao':
        dataset_dir = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa
    else:
        dataset_dir = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa+instance+sepa

    if not path.exists(dataset_dir):
         raise ValueError("Failed to load the demand data: " + instance + str(data_seed))

    if instance == 'Beijing_bus':
        base_path = dataset_dir+'bus'+instance_size+str(data_seed)
    elif instance == 'Beijing_Yanjiao':
        if yanjiao_prefix:
            prefix = str(yanjiao_prefix).format(
                n_passengers=n_passengers,
                seed=data_seed,
                data_seed=data_seed,
            )
            base_path = dataset_dir + prefix
        else:
            base_path = dataset_dir+'yanjiao'+instance_size+str(data_seed)
    elif instance == 'NYC_TLC':
        base_path = dataset_dir+'nyc_tlc_pilot_'+str(data_seed)
    else:
        base_path = dataset_dir+instance+instance_size+str(data_seed)

    metadata = None
    metadata_file = base_path+"_metadata.json"
    if path.isfile(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f_meta:
            metadata = json.load(f_meta)

    f = base_path+"_coords.txt"
    if not path.isfile(f):
        raise ValueError("Coordinate file not found: " + f)

    coords = np.zeros([0])
    with open(f, "r") as file:
        for j,i in enumerate(file):
            if not i.startswith('NODE'):
                loc = i.strip().split('\t')
                loc = Location(float(loc[1]),float(loc[2]),j-1,0)
                coords = np.append(coords,loc)

    dist_matrix = np.empty(shape=(0,len(coords)),dtype=int)
    if instance_folder=='Amazon_data':
        f = base_path+"_dist_matrix.txt"
        if path.isfile(f):
            with open(f, "r") as file:
                for i in file:
                    if not i.startswith('EDGE'):
                        loc = i.strip().split('\t')
                        dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
    elif instance == 'NYC_TLC':
        f = base_path+"_duration_matrix.txt"
        if not path.isfile(f):
            raise ValueError("NYC_TLC duration matrix not found: " + f)
        with open(f, "r") as file:
            for i in file:
                    if not i.startswith('EDGE'):
                        loc = i.strip().split('\t')
                        dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
    elif instance == 'Beijing_Yanjiao' and path.isfile(base_path+"_duration_matrix.txt"):
        f = base_path+"_duration_matrix.txt"
        with open(f, "r") as file:
            for i in file:
                if not i.startswith('EDGE'):
                    loc = i.strip().split('\t')
                    dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
        print("[INFO] Loaded Beijing_Yanjiao duration matrix:", f, "shape=", dist_matrix.shape)
    else:
        metric_yanjiao = (
            instance == 'Beijing_Yanjiao'
            and metadata is not None
            and (
                metadata.get("projection") == "metric"
                or metadata.get("coordinate_system") == "local_metric_km"
            )
        )
        for i in coords:
            dist = []
            for j in coords:
                distance = getdistance_euclidean(i,j)
                if metric_yanjiao:
                    dist.append(int(round(distance / truck_speed * 3600)))
                else:
                    dist.append(int(distance) / truck_speed * 3600)
            dist_matrix = np.vstack([dist_matrix,np.array(dist)])

    if len(dist_matrix) == 0:
        raise ValueError("Distance/duration matrix is empty for: " + instance + str(data_seed))

    if instance=='Austin':
        n_parcelpoints=278
    elif instance=='Seattle':
        n_parcelpoints=299
    elif instance=='Beijing_bus':
        n_parcelpoints=169
    elif instance=='Beijing_Yanjiao':
        if metadata is not None:
            n_parcelpoints = int(metadata["n_meeting_points"])
            n_passengers_actual = int(metadata["passengers"])
        else:
            raise ValueError("Beijing_Yanjiao metadata not found: " + metadata_file)
    elif instance=='NYC_TLC':
        metadata_file = base_path+"_metadata.json"
        if not path.isfile(metadata_file):
            raise ValueError("NYC_TLC metadata not found: " + metadata_file)
        with open(metadata_file, "r", encoding="utf-8") as metadata_handle:
            metadata = json.load(metadata_handle)
        n_parcelpoints = int(metadata["n_meeting_points"])
    else:
        n_parcelpoints=10

    adjacency_file = base_path+f"_adjacency{k}.npy"
    if path.isfile(adjacency_file):
        adjacency = np.load(adjacency_file)
    else:
        if instance=='NYC_TLC':
            raise ValueError("NYC_TLC adjacency not found for k=" + str(k) + ": " + adjacency_file)
        n_customers = len(dist_matrix) - n_parcelpoints
        adjacency = np.zeros(shape=(n_customers, n_parcelpoints), dtype=int)
        k_eff = min(int(k), n_parcelpoints)
        if instance=='Beijing_bus':
            for customer_idx in range(1, min(240, n_customers - 1) + 1):
                ooh_distances = dist_matrix[customer_idx][241:410]
                closest_indices = np.argsort(ooh_distances)[:k_eff]
                for j in closest_indices:
                    adjacency[customer_idx - 1][j] = 1
        elif instance=='Beijing_Yanjiao':
            n_pass = n_passengers_actual
            mp_start = n_pass + 1  # meeting points start after depot + homes
            mp_end = mp_start + n_parcelpoints
            for customer_idx in range(1, n_pass + 1):
                ooh_distances = dist_matrix[customer_idx][mp_start:mp_end]
                closest_indices = np.argsort(ooh_distances)[:k_eff]
                for j in closest_indices:
                    adjacency[customer_idx - 1][j] = 1
        else:
            for i in range(0, n_customers):
                closest = np.argsort(dist_matrix[i][-n_parcelpoints:])[:k_eff]
                adjacency[i][closest] = 1
        np.save(base_path+f"_adjacency{k}", adjacency)

    if instance=='NYC_TLC':
        expected_rows = len(dist_matrix) - n_parcelpoints
        expected_cols = n_parcelpoints
        if adjacency.shape != (expected_rows, expected_cols):
            raise ValueError(
                "NYC_TLC adjacency shape mismatch: expected "
                + str((expected_rows, expected_cols))
                + ", found "
                + str(adjacency.shape)
            )

    service_times_file = base_path+"_service_times.txt"
    if path.isfile(service_times_file):
        service_times = np.loadtxt(service_times_file)
    else:
        service_times = calculate_service_time(coords,clip_service_time)

    return coords,dist_matrix,n_parcelpoints,adjacency,service_times


def load_choice_utility_matrix(
        pathh, instance, data_seed, n_passengers=300, yanjiao_prefix=None):
    """Load an optional static customer-choice utility sidecar matrix.

    The sidecar is intentionally optional. Existing instances keep their
    original distance-based MNL behavior when the file is absent.
    """
    if instance != 'Beijing_Yanjiao':
        return None

    if name == 'nt':
        sepa = '\\'
    else:
        sepa = '/'

    dataset_dir = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Beijing_Yanjiao'+sepa
    if yanjiao_prefix:
        prefix = str(yanjiao_prefix).format(
            n_passengers=n_passengers,
            seed=data_seed,
            data_seed=data_seed,
        )
        base_path = dataset_dir + prefix
    else:
        base_path = dataset_dir+'yanjiao'+f'_{n_passengers}_'+str(data_seed)

    npy_file = base_path+"_choice_utility.npy"
    txt_file = base_path+"_choice_utility.txt"
    if path.isfile(npy_file):
        matrix = np.load(npy_file)
        source = npy_file
    elif path.isfile(txt_file):
        matrix = np.loadtxt(txt_file)
        source = txt_file
    else:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Choice utility matrix must be square: " + base_path)
    print("[INFO] Loaded Beijing_Yanjiao choice utility matrix:", source, "shape=", matrix.shape)
    return matrix


def load_walking_distance_matrix(
        pathh, instance, data_seed, n_passengers=300, yanjiao_prefix=None):
    """Load optional home-to-meeting-point distance matrix for choice utility."""
    if instance != 'Beijing_Yanjiao':
        return None

    if name == 'nt':
        sepa = '\\'
    else:
        sepa = '/'

    dataset_dir = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Beijing_Yanjiao'+sepa
    if yanjiao_prefix:
        prefix = str(yanjiao_prefix).format(
            n_passengers=n_passengers,
            seed=data_seed,
            data_seed=data_seed,
        )
        base_path = dataset_dir + prefix
    else:
        base_path = dataset_dir+'yanjiao'+f'_{n_passengers}_'+str(data_seed)

    npy_file = base_path+"_walk_distance.npy"
    txt_file = base_path+"_walk_distance.txt"
    if path.isfile(npy_file):
        matrix = np.load(npy_file)
        source = npy_file
    elif path.isfile(txt_file):
        matrix = np.loadtxt(txt_file)
        source = txt_file
    else:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Walking distance matrix must be square: " + base_path)
    print("[INFO] Loaded Beijing_Yanjiao walking distance matrix:", source, "shape=", matrix.shape)
    return matrix


def generate_demand_data(dim):
    coords = np.zeros([0])
    count = 1
    for i in range(dim):
        for j in range(dim):
            loc = Location(float(i),float(j),count,0)
            coords = np.append(coords,loc)
            count+=1
    
    return coords

def get_dist_mat_HGS(dist_matrix,loc_ids):        
    dist_mat = dist_matrix[loc_ids]
    return dist_mat[:,loc_ids]

def get_fleet(initRouteplan,num_vehicles,vehicleCapacity):
    vehicles = np.empty(shape=(0,num_vehicles))
    for v in range(num_vehicles):
        vehicles = np.append(vehicles,Vehicle(initRouteplan.copy(),vehicleCapacity,v))
    return Fleet(vehicles)

def extract_route_HGS(route,data):
    fleet = get_fleet([],data['num_vehicles'],data['vehicle_capacity'])#reset fleet and write to vehicles again
    veh = 0
    for r in route.routes:
        for i in r:
            loc = Location(data['x_coordinates'][i],data['y_coordinates'][i],data['id'][i],data['time'][i])
            idx = len(fleet["fleet"][veh]["routePlan"])-1
            fleet["fleet"][veh]["routePlan"].insert(idx,loc)
        veh+=1
    return fleet

def find_closest_parcelpoints(pathh,parcelpoints,dist_matrix,instance,data_seed,k=20):
    """
    This function is used to generate the adjacency matrix, we stored them so we do not call this function online
    """
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    shape = (len(dist_matrix)-len(parcelpoints["parcelpoints"]),len(parcelpoints["parcelpoints"]))
    adjacency = np.zeros(shape=shape,dtype=int)
    for i in range(0,len(dist_matrix)-len(parcelpoints["parcelpoints"])):
        closest = np.argsort(dist_matrix[i][-len(parcelpoints["parcelpoints"]):])[:k]#find k closest parcelpoints
        for j in closest:
            adjacency[i][j]=1
    pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Amazon_data'+sepa+instance+sepa
    np.save(pathh+instance+"_700_"+str(data_seed)+f"_adjacency{k}", adjacency)


def get_matrix(coords,dim,hexa=False):
    """
    For hexagon calculation, see: https://stackoverflow.com/a/7714148
    """
    max_xcoord = max(coords, key=lambda x: x.x).x
    max_ycoord = max(coords, key=lambda y: y.y).y
    min_xcoord = min(coords, key=lambda x: x.x).x
    min_ycoord = min(coords, key=lambda y: y.y).y

    customer_cells = np.empty((0,2),dtype=int)
    min_x = min_xcoord
    diff_x = max_xcoord-min_xcoord
    min_y = min_ycoord
    diff_y = max_ycoord-min_ycoord
    
    #hexa params (not used in paper)
    if hexa:
        gridwidth = diff_x/dim
        gridheight = diff_y/dim
        c = gridwidth / 4#TODO: approximation of c, we should calculate this exactly 
        m = c / gridwidth / 2
    
    for i in coords:
        row = trunc(dim* ((i.y - min_y) / diff_y)-1e-5)     
        
        if hexa:
            rowIsOdd = row % 2 == 1   
            if rowIsOdd:#if row is odd number calculte indent of hexa grid
                column = trunc(dim* ((i.x - (gridwidth/2) - min_x) / diff_x)-1e-5)
            relative_y = i.y - (row*gridheight)
            if rowIsOdd:
                relative_x = (i.x - (column*gridwidth)) - (gridwidth/2)
            else:
                relative_x = i.x - (column*gridwidth)
            
            if relative_y < (m * relative_x) + c:#left edge
                row -=1
                if not rowIsOdd:
                    column -=1 
            elif relative_y < (-m * relative_x) - c:#rigt edge
                row -=1
                if rowIsOdd:
                    column +=1 
        else:
            column = trunc(dim* ((i.x - min_x) / diff_x)-1e-5)
            
        
        customer_cells = np.vstack((customer_cells,[column,row]))
    return customer_cells
    

class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using observations
    """
    def __init__(self, max_len, time_intervals, matrix_dim, target_dim, atype, config, stype=float32):

        self.features = torch.zeros((max_len, time_intervals, matrix_dim, matrix_dim), dtype=stype, requires_grad=False,device=config.device)
        self.capacity_features = torch.zeros(max_len, dtype=stype, requires_grad=False, device=config.device)
        self.target = torch.zeros((max_len, target_dim), dtype=atype, requires_grad=False,device=config.device)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype
        self.config = config
        self.matrix_dim = matrix_dim
        self.time_intervals = time_intervals

    @property
    def size(self):
        return self.length

    def reset(self):
        self.length = 0

    def _get(self, idx):
        return self.features[idx], self.capacity_features[idx], self.target[idx]

    def batch_sample(self, batch_size, randomize=True):
        if randomize:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for ids in [indices[i:i + batch_size] for i in range(0, self.length, batch_size)]:
            yield self._get(ids)

    def sample(self, batch_size):
        count = min(batch_size, self.length)
        return self._get(np.random.choice(self.length, count))

    def add(self, features, capacity_features, target):
        mtrx_dim = self.matrix_dim
        time_intervals = self.time_intervals
        if len(features)!=len(target) or len(features) != len(capacity_features):
            raise ValueError("MemoryBuffer: features and target are different length" )
        for i in range(len(features)):
            pos = self.length
            if self.length < self.max_len:
                self.length = self.length + 1
            else:
                pos = np.random.randint(self.max_len)
    
            self.features[pos] = torch.tensor(features[i].reshape(time_intervals,mtrx_dim,mtrx_dim), dtype=self.stype)
            self.capacity_features[pos] = torch.tensor(capacity_features[i], dtype=self.stype)
            self.target[pos] = torch.tensor(target[i][1], dtype=self.atype)
        
    def save(self, filename):
        torch.save(self.features, filename + 'feat.pt')
        torch.save(self.capacity_features, filename + 'cap_feat.pt')
        torch.save(self.target, filename + 'target.pt')
        
        
##for PPO  actor and critic    
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return

#for PPO
class Trajectory:
    """
    Pre-allocated memory interface for storing and using on-policy trajectories
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False)

        self.ctr = 0
        self.max_len = max_len
        self.atype = atype
        self.stype= stype
        self.config = config

    def add(self, s1, a1, dist, r1, s2, done):
        if self.ctr == self.max_len:
            raise OverflowError

        self.s1[self.ctr] = torch.tensor(s1, dtype=self.stype)
        self.a1[self.ctr] = torch.tensor(a1, dtype=self.atype)
        self.dist[self.ctr] = torch.tensor(dist)
        self.r1[self.ctr] = torch.tensor(r1)
        self.s2[self.ctr] = torch.tensor(s2, dtype=self.stype)
        self.done[self.ctr] = torch.tensor(done)

        self.ctr += 1

    def reset(self):
        self.ctr = 0

    @property
    def size(self):
        return self.ctr

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def get_current_transitions(self):
        pos = self.ctr
        return self.s1[:pos], self.a1[:pos], self.dist[:pos], self.r1[:pos], self.s2[:pos], self.done[:pos]

    def get_all(self):
        return self.s1, self.a1, self.dist, self.r1, self.s2, self.done

    def get_latest(self):
        return self._get([-1])

    def batch_sample(self, batch_size, nth_return):
        # Compute the estimated n-step gamma return
        R = nth_return
        for idx in range(self.ctr-1, -1, -1):
            R = self.r1[idx] + self.config.gamma * R
            self.r1[idx] = R

        # Genreate random sub-samples from the trajectory
        perm_indices = np.random.permutation(self.ctr)
        for ids in [perm_indices[i:i + batch_size] for i in range(0, self.ctr, batch_size)]:
            yield self._get(ids)

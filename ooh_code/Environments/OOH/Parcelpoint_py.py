from __future__ import print_function

import numpy as np
import numpy.ma as ma
import sys
from Src.Utils.Utils import get_dist_mat_HGS,get_fleet
from Environments.OOH.containers import Location,ParcelPoint,ParcelPoints,Vehicle,Fleet,Customer
from Environments.OOH.env_utils import utils_env
from Environments.OOH.customerchoice import customerchoicemodel

class Parcelpoint_py(object):
    def __init__(self,
                 model,
                 max_steps_r,
                 max_steps_p,
                 pricing = False,
                 n_vehicles=2,
                 veh_capacity=100,
                 parcelpoint_capacity=25,
                 fraction_capacitated=0.0,
                 incentive_sens=0.99,
                 base_util=0.2,
                 home_util=0.3,
                 reopt=2000,
                 load_data=False,
                 coords=[],
                 dist_matrix=[],
                 n_parcelpoints=6,
                 adjacency=[],
                 service_times=[],
                 dissatisfaction=False,
                 hgs_time=3.0,
                 l0_home=2.5,
                 l_mp=0.75,
                 choice_util_matrix=None,
                 walking_distance_matrix=None,
                 route_label_mode="hgs",
                 quit_threshold=None,  # 保留向后兼容
                 outside_option_util=None,  # 外部选项的效用值u0
                 travel_time_weight=None,  # 在途时间的权重系数（None表示不启用在途时间效用）
                 walk_distance_weight=None):

        #episode length params
        self.max_steps = 0
        self.max_steps_r = max_steps_r
        self.max_steps_p = max_steps_p

        #init fleet and parcelpoints
        self.n_vehicles = n_vehicles
        self.veh_capacity = veh_capacity
        self.pp_capacity = parcelpoint_capacity
        self.fraction_capacitated = fraction_capacitated
        self.data = dict()

        #possible passed on data
        self.coords = coords
        self.dist_matrix = dist_matrix
        self.n_parcelpoints = n_parcelpoints
        self.adjacency = adjacency
        self.service_times = service_times
        self.choice_util_matrix = choice_util_matrix
        self.walking_distance_matrix = walking_distance_matrix
        
        # Service time parameters: l0_home for base home delivery time, l_mp for meeting points
        self.l0_home = l0_home * 60  # Convert minutes to seconds
        self.l_mp = l_mp * 60  # Convert minutes to seconds

        #load data or generate data
        self.load_data = load_data
        self.n_unique_customer_locs = len(self.coords)-self.n_parcelpoints
        if self.load_data:
            print("\n Note: the HGS python implementation (hygese 0.0.0.8) throws an assertion error for coords<0, you will need to outcomment this check in hygese.py \n")
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.fraction_capacitated,self.n_parcelpoints,self.data,self.dist_matrix,hgs_time)
            self.depot = self.coords[0]
            self.parcelPoints = self.utils.get_parcelpoints_from_data(self.coords[-self.n_parcelpoints:],self.n_unique_customer_locs)
            self.get_customer = self.get_new_customer_from_data
            self.num_cust_loc = len(self.dist_matrix)-len(self.parcelPoints["parcelpoints"])-1
            self.dist_scaler = np.amax(self.dist_matrix)
        else:
            if self.fraction_capacitated != 0.0:
                print("Capacitated lockers not supported on generated data")
                sys.exit()
            self.depot = Location(50,50,0,0)
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.fraction_capacitated,self.n_parcelpoints,self.data,self.dist_matrix,hgs_time)
            self.parcelPoints = self.utils.get_parcelpoints()
            self.get_customer = self.generate_new_customer
            self.dist_scaler = 10

        #customers
        self.home_util = home_util
        self.incentive_sens = incentive_sens
        self.dissatisfaction = dissatisfaction

        self.newCustomer = Customer
        self.fleet = get_fleet([self.depot,self.depot],self.n_vehicles,self.veh_capacity)

        #pricing of offering problem variant
        if pricing:
            #self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.dist_matrix,self.n_unique_customer_locs,quit_threshold=quit_threshold,outside_option_util=outside_option_util,travel_time_weight=travel_time_weight,choice_util_matrix=self.choice_util_matrix,walking_distance_matrix=self.walking_distance_matrix,walk_distance_weight=walk_distance_weight)
            self.customerChoice = self.customerchoice.customerchoice_pricing
            self.get_delivery_loc = self.get_delivery_loc_pricing
        else:
            #self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.dist_matrix,self.n_unique_customer_locs,quit_threshold=quit_threshold,outside_option_util=outside_option_util,travel_time_weight=travel_time_weight,choice_util_matrix=self.choice_util_matrix,walking_distance_matrix=self.walking_distance_matrix,walk_distance_weight=walk_distance_weight)
            self.customerChoice = self.customerchoice.customerchoice_offer
            self.get_delivery_loc = self.get_delivery_loc_offer

        self.steps = 0
       # self.max_steps = (self.n_vehicles*self.veh_capacity)
        self.reopt_freq = reopt
        self.route_label_mode = route_label_mode
        
        # 存储当前的在途时间预测（由agent设置）
        self.current_travel_times = None

        self.reset()

    def seed(self, seed):
        """设置随机种子并保存"""
        self.seed_value = seed  # 保存seed值（避免与self.seed冲突）
        np.random.seed(seed)  # 实际设置numpy随机种子
        # 如果使用torch，也需要设置（但这里主要是numpy）

    def reset(self,training=True):
        """
        Sets the environment to default conditions
        """
        # 在reset时重新设置随机种子，确保每次reset都使用相同的随机状态起点
        if hasattr(self, 'seed_value'):
            np.random.seed(self.seed_value)
        
        self.max_steps = np.random.negative_binomial(self.max_steps_r,self.max_steps_p)

        self.fleet = self.utils.reset_fleet(self.fleet,[self.depot,self.depot])
        self.parcelPoints = self.utils.reset_parcelpoints(self.parcelPoints)

        self.steps = 0
        self.service_time = 0
        self.count_home_delivery = 0
        self.total_prices = []
        self.total_discounts = []
        self.quit_count = 0  # 新增：初始化退出计数

        self.data['x_coordinates'] = self.depot.x
        self.data['y_coordinates'] =  self.depot.y
        self.data['id'] = 0
        self.data['time'] = 0
        self.data['vehicle_capacity'] = self.veh_capacity
        self.data['num_vehicles'] = self.n_vehicles

        self.count_dissatisfaction = 0

        # Reset customer choice debug counters to avoid confusing cumulative counts (e.g., "6500次")
        # when comparing algorithms under the same environment parameters.
        if hasattr(self, "customerchoice") and hasattr(self.customerchoice, "reset_debug_counters"):
            try:
                self.customerchoice.reset_debug_counters()
            except Exception:
                pass

        self.curr_state = self.make_state()
        return self.curr_state

    def get_new_customer_from_data(self):
        # 确保索引不超出service_times的边界
        max_idx = min(len(self.service_times), len(self.coords)) - 1
        idx = np.random.randint(1, min(self.num_cust_loc, max_idx + 1))
        home = self.coords[idx]#depot = 0
        home.time=self.steps
        service_time = self.service_times[idx]
        return Customer(home,self.incentive_sens,self.home_util,service_time,idx)

    def generate_new_customer(self):
        # 确保索引不超出service_times的边界
        max_idx = min(len(self.service_times), len(self.coords)) - 1
        idx = np.random.randint(0, max_idx + 1)
        home = self.coords[idx]#depot = 0
        home.time=self.steps
        service_time = self.service_times[idx]
        return Customer(home,self.incentive_sens,self.home_util,service_time,idx)

    def make_state(self):
        self.newCustomer = self.get_customer()
        state = [self.newCustomer,self.fleet,self.parcelPoints,self.steps]
        return state

    def abstract_state_ppo(self,state):
        newcust_x = state[0].home.x
        newcust_y = state[0].home.y

        #for user friendliness, we commented out the state route variables
        # closest_locations = []
        # for v in range(self.n_vehicles):
        #     for loc in sorted(state[1][v]["routePlan"], key=distance_to_home)[:20]:
        #         closest_locations.append(loc)

        return [newcust_x,newcust_y]

    def is_terminal(self):
        if self.steps > self.max_steps:
            return 1
        else:
            return 0

    def get_delivery_loc_pricing(self,action):
        mask = ma.masked_array(self.parcelPoints.parcelpoints, mask=self.adjacency[self.newCustomer.id_num])#only offer 20 closest
        # 如果提供了在途时间预测，传递给客户选择模型
        if self.current_travel_times is not None:
            return self.customerChoice(self.newCustomer,action,mask,travel_times=self.current_travel_times)
        else:
            return self.customerChoice(self.newCustomer,action,mask)

    def get_delivery_loc_offer(self,action):
        #get the chosen delivery location
        # 如果提供了在途时间预测，传递给客户选择模型
        if self.current_travel_times is not None:
            return self.customerChoice(self.newCustomer,action,self.parcelPoints.parcelpoints,travel_times=self.current_travel_times)
        else:
            return self.customerChoice(self.newCustomer,action,self.parcelPoints.parcelpoints)
    
    def set_travel_times(self, travel_times):
        """
        设置当前的在途时间预测（由agent调用）
        
        Args:
            travel_times: 字典，包含：
                - 'home': HOME点的在途时间（秒）
                - 'ooh': 列表，对应每个OOH点的在途时间（秒）
        """
        self.current_travel_times = travel_times
    
    def clear_travel_times(self):
        """清除当前的在途时间预测"""
        self.current_travel_times = None

    def reopt_for_eval(self,data):
        if self.route_label_mode == "hep":
            return self.fleet_route_distance(self.fleet)

        # Normalize scalar/array route ids. In high-quit episodes this can be scalar.
        ids_raw = data.get('id', [])
        if np.isscalar(ids_raw):
            route_ids = np.array([int(ids_raw)])
        else:
            route_ids = np.asarray(ids_raw)

        # Guard tiny instances: HGS is unstable/slow on 0/1-customer routes.
        if route_ids.size <= 1:  # depot-only
            return 0.0
        positive_ids = [int(x) for x in route_ids if int(x) > 0]
        if len(positive_ids) <= 1:
            if len(positive_ids) == 0:
                return 0.0
            cid = positive_ids[0]
            if self.load_data:
                return float(self.dist_matrix[0][cid] + self.dist_matrix[cid][0])
            return 0.0
        
        if self.load_data:
            try:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix, route_ids)
            except Exception as e:
                # 如果距离矩阵计算失败（可能因为所有顾客都退出），返回0
                print(f"  警告: 计算距离矩阵时出错（可能所有顾客都退出）: {e}")
                return 0.0
        _,cost = self.utils.reopt_HGS(data)
        return cost

    def _route_distance(self, loc_a, loc_b):
        if self.load_data:
            a_id = 0 if loc_a is None else int(loc_a.id_num)
            b_id = 0 if loc_b is None else int(loc_b.id_num)
            return float(self.dist_matrix[a_id][b_id])
        if loc_a is None:
            loc_a = self.depot
        if loc_b is None:
            loc_b = self.depot
        return float(self.utils.getdistance_euclidean(loc_a, loc_b))

    def fleet_route_distance(self, fleet):
        distance = 0.0
        for vehicle in fleet["fleet"]:
            route = [loc for loc in vehicle["routePlan"] if int(loc.id_num) > 0]
            prev_loc = None
            for loc in route:
                distance += self._route_distance(prev_loc, loc)
                prev_loc = loc
            if route:
                distance += self._route_distance(prev_loc, None)
        return distance

    #ToDo: cleanup saving statistics, not efficient right now
    def step(self,action):
        self.steps += 1

        #get the customer's choice of delivery location
        loc,accepted_pp,idx,price = self.get_delivery_loc(action)
        
        # 处理退出情况：idx=-2 表示顾客选择退出
        if idx == -2:
            # 顾客退出，不添加到路线中，不更新容量
            # 记录退出统计
            self.quit_count += 1
            
            #info for plots and statistics (退出时使用特殊值)
            # stats格式: (steps, count_home_delivery, service_time, total_prices, parcelpoints, distance, total_discounts, price, quit_count)
            # 注意：distance使用-1表示退出，0表示Home delivery（自己到自己的距离），>0表示OOH点
            stats = self.steps,self.count_home_delivery,self.service_time,self.total_prices,self.parcelPoints.parcelpoints,-1,self.total_discounts,price,self.quit_count
            
            #generate new customer arrival and return state info
            self.curr_state = self.make_state()
            
            return self.curr_state.copy(), self.is_terminal(), stats, self.data
        
        # 正常处理（原有逻辑）
        if price>0:
            self.total_prices.append(price)
        else:
            self.total_discounts.append(price)
        self.data['x_coordinates']= np.append(self.data['x_coordinates'],loc.x)
        self.data['y_coordinates'] = np.append(self.data['y_coordinates'],loc.y)
        self.data['id'] = np.append(self.data['id'],loc.id_num)
        self.data['time'] = np.append(self.data['time'],self.steps)

        #reduce parcelpoint capacity, if chosen
        if accepted_pp:
            self.parcelPoints.parcelpoints[idx-self.n_unique_customer_locs].remainingCapacity -= 1
            # Meeting点服务时间: l_mp = ε (already in seconds)
            self.service_time += self.l_mp
        else:#home delivery
            # 家接送服务时间: l_home(i) = l0_home + l_addr(i)
            # service_times[idx] stores l_addr(i) in seconds, add l0_home (also in seconds)
            self.service_time += (self.l0_home + self.service_times[idx])
            self.count_home_delivery+=1

        if self.dissatisfaction:#perhaps remove, not used so far
            if np.mean(action)>2.75 and np.std(action)<1.0:
                self.count_dissatisfaction+=1

        #construct intermittent route kept in memory during booking horizon
        insertVeh,idx,costs = self.utils.cheapestInsertionRoute(loc,self.fleet)
        self.fleet.fleet[insertVeh].routePlan.insert(idx,loc)

        #re-optimize the intermittent route after X steps, we did not do this for the paper
        if self.steps % self.reopt_freq == 0:#do re-opt using HGS
            if self.load_data:
                self.data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,self.data['id'])
            self.fleet,_ = self.utils.reopt_HGS(self.data)

        #info for plots and statistics
        # stats格式: (steps, count_home_delivery, service_time, total_prices, parcelpoints, distance, total_discounts, price, quit_count)
        stats = self.steps,self.count_home_delivery,self.service_time,self.total_prices,self.parcelPoints.parcelpoints,self.dist_matrix[self.newCustomer.home.id_num][loc.id_num],self.total_discounts,price,self.quit_count

        #generate new customer arrival and return state info
        self.curr_state = self.make_state()
        
        # 清除当前的在途时间预测，为下次预测做准备
        self.clear_travel_times()

        return self.curr_state.copy(), self.is_terminal(), stats, self.data

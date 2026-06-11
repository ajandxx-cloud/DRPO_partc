import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch import float32
from math import sqrt
from Src.Utils.Utils import MemoryBuffer,get_dist_mat_HGS,extract_route_HGS, get_matrix
from Src.Utils.Predictors import CNN_2d, CNN_3d, LinReg, CNN_TravelTime
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw
from math import exp, e
from hygese import AlgorithmParameters, Solver
from operator import itemgetter
from Src.Utils.passenger_utility import (
    utility_home_nonprice,
    utility_meeting_point_nonprice,
)
# This function implements DSPO
class DSPO(Agent):
    def __init__(self, config):
        super(DSPO, self).__init__(config)
               
        self.load_data = config.load_data
        # heuristic parameters
        self.k = config.k
        self.init_theta = config.init_theta_cnn
        self.cool_theta = config.cool_theta_cnn#linear cooling scheme
        
        #problem variant: pricing or offering
        if self.config.pricing:
            self.get_action = self.get_action_pricing
            self.max_p = config.max_price
            self.min_p = config.min_price
        else:
            self.get_action = self.get_action_offer
        
        self.grid_dim = config.grid_dim
        self.initial_phase = True
        
        self.n_layers = config.n_input_layers
        self.memory = MemoryBuffer(max_len=self.config.buffer_size,time_intervals=self.n_layers, matrix_dim=self.grid_dim,
                                     target_dim=1, atype=float32, config=config)  
        
        if config.use3d_conv:
            self.supervised_ml = CNN_3d(self.grid_dim,self.n_layers,config.n_filters,config.dropout)
        elif config.linearModel:
            self.supervised_ml = LinReg(self.grid_dim*self.grid_dim*self.n_layers)
        else:
            self.supervised_ml = CNN_2d(self.grid_dim,self.n_layers,config.n_filters,config.dropout)
        self.features = np.empty((0,self.n_layers*self.grid_dim*self.grid_dim))
        self.cap_features = np.empty((0,1))
        self.interval = int(config.max_steps_r/config.n_input_layers)
        
        self.optimizer = config.optim(self.supervised_ml.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # 在途时间预测器（可选，如果启用）
        self.use_travel_time_prediction = getattr(config, 'use_travel_time_prediction', False)
        if self.use_travel_time_prediction:
            if config.use3d_conv:
                # 3D CNN不支持，使用2D CNN
                self.travel_time_predictor = CNN_TravelTime(self.grid_dim, self.n_layers, config.n_filters, config.dropout)
            elif config.linearModel:
                # 线性模型不支持，使用CNN
                self.travel_time_predictor = CNN_TravelTime(self.grid_dim, self.n_layers, config.n_filters, config.dropout)
            else:
                self.travel_time_predictor = CNN_TravelTime(self.grid_dim, self.n_layers, config.n_filters, config.dropout)
            
            # 如果在途时间预测器需要单独训练，添加优化器
            self.travel_time_optimizer = config.optim(self.travel_time_predictor.parameters(), 
                                                      lr=getattr(config, 'travel_time_learning_rate', config.learning_rate))
            self.travel_time_criterion = nn.HuberLoss(delta=1.0)
            
            # 在途时间训练数据的MemoryBuffer
            self.travel_time_memory = MemoryBuffer(
                max_len=self.config.buffer_size,
                time_intervals=self.n_layers,
                matrix_dim=self.grid_dim,
                target_dim=1,
                atype=float32,
                config=config
            )
            self.travel_time_features = np.empty((0, self.n_layers*self.grid_dim*self.grid_dim))
            self.travel_time_cap_features = np.empty((0, 1))
            self.travel_time_targets = []  # 存储真实的在途时间
        else:
            self.travel_time_predictor = None
            self.travel_time_memory = None
        
        #define learning modules
        self.modules = [('supervised_ml', self.supervised_ml)]
        if self.use_travel_time_prediction and self.travel_time_predictor is not None:
            self.modules.append(('travel_time_predictor', self.travel_time_predictor))
        self.init()#write module to device
        self.device = config.device
        
        if self.load_data:
            self.customer_cell = get_matrix(config.coords,self.grid_dim,config.hexa)
            self.dist_matrix = config.dist_matrix
            self.service_times = config.service_times
            self.adjacency = config.adjacency
            self.choice_util_matrix = getattr(config, 'choice_util_matrix', None)
            self.first_parcelpoint_id = len(self.dist_matrix[0])-config.n_parcelpoints-1
            self.addedcosts = self.addedcosts_distmat
            self.dist_scaler = 1#np.amax(self.dist_matrix)
            self.mnl = self.mnl_distmat
        else:
            self.choice_util_matrix = None
            self.addedcosts = self.addedcosts_euclid
            self.dist_scaler = 1#10
            self.mnl = self.mnl_euclid
        
        #mnl parameters
        self.base_util = config.base_util
        # cost_multiplier: (driver_wage + fuel_cost_per_distance * truck_speed) / 3600 = cost per second
        self.cost_multiplier = (config.driver_wage + config.fuel_cost * config.truck_speed) / 3600
        self.wage = config.driver_wage
        #self.added_costs_home = config.driver_wage*(config.del_time/60)
        self.revenue = config.revenue
        
        # Service time parameters: l0_home for base home delivery time, l_mp for meeting points
        self.l0_home = config.l0_home * 60  # Convert minutes to seconds
        self.l_mp = config.l_mp * 60  # Convert minutes to seconds
        
        #hgs settings
        self.route_label_mode = getattr(config, 'route_label_mode', 'hgs')
        self._terminal_label_cache = None
        ap_final = AlgorithmParameters(timeLimit=config.hgs_final_time)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)#used for final route        
        
        #lambdas
        id_num = lambda x: x.id_num
        self.get_id = np.vectorize(id_num)
                
    def reset(self):
        self._terminal_label_cache = None
        super(DSPO, self).reset()
        
    def get_action_offer(self,state,training):
        if self.initial_phase:
            return self.get_action_offerall(state,training)
        else:
        
            theta = self.init_theta - (state[3] *  self.cool_theta)
            mltplr = self.cost_multiplier
            
            #cheapest insertion costs of every PP in current and historic routes
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            pp_costs = np.full(len(pps),1000000000.0)
            
            #ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat,state[0].home,pps)
            
            for pp in range(len(pps)):
                if state[2]["parcelpoints"][pp].remainingCapacity > 0:#check if parcelpont has remaining capacity               
                    # OOH point cost: includes service time (l_mp) and routing cost
                    pp_costs[pp] = self.l_mp*mltplr + mltplr*((1-theta)*self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location, state[1]) + theta*(costs[pp+2]-costs[0]))
            pp_sorted_args = state[2]["parcelpoints"][np.argpartition(pp_costs, self.k)[:self.k]]
            
            #get k best PPs
            action = self.get_id(pp_sorted_args)
        return action
      
    def get_action_pricing(self,state,training):
        # 获取customerchoice模型（统一使用环境的MNL模型）
        customerchoice_model = None
        if hasattr(self.config, 'env') and self.config.env is not None:
            customerchoice_model = getattr(self.config.env, 'customerchoice', None)
        
        if self.initial_phase:
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            a_hat = np.zeros(len(pps)+1)
            return np.around(a_hat,decimals=2)
        else:
            
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            pp_costs = np.full(len(pps),1000000000.0)
            
            #ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat,state[0].home,pps)
            
            # 预测在途时间（如果启用）
            travel_times = None
            if self.use_travel_time_prediction:
                travel_times = self.get_travel_time_prediction(cur_feat, state[0].home, pps)
                # 将预测的在途时间存储到环境中
                if hasattr(self.config, 'env') and self.config.env is not None:
                    self.config.env.set_travel_times(travel_times)
            
            #1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
            theta = self.init_theta - (state[3] *  self.cool_theta)
            mltplr = self.cost_multiplier
            
            # Home delivery cost: includes service time (l0_home + l_addr(i)) and routing cost
            # state[0].service_time stores l_addr(i), need to add l0_home
            homeCosts = (self.l0_home + state[0].service_time)*mltplr+mltplr*((1-theta)*(self.cheapestInsertionCosts(state[0].home, state[1]) ) + theta*( costs[1]-costs[0] ))
            
            # 使用customerchoice模型的MNL计算HOME delivery效用（如果可用）
            if customerchoice_model is not None:
                home_travel_time = None
                if travel_times is not None and 'home' in travel_times:
                    home_travel_time = travel_times['home']
                # HOME delivery utility without price.
                home_base_util = utility_home_nonprice(
                    self.base_util,
                    state[0].home_util,
                    customerchoice_model.travel_time_weight,
                    home_travel_time,
                )
                sum_mnl = exp(home_base_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
            else:
                # 回退到原来的实现
                home_base_util = utility_home_nonprice(
                    self.base_util,
                    state[0].home_util,
                    None,
                    None,
                )
                sum_mnl = exp(home_base_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
            
            #Slight change compared to paper, to support faster training, without relying on the CVRP solver every time,
            #See bottom of this file for details.
            for idx,pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    # 使用customerchoice模型的MNL方法（如果可用）
                    if customerchoice_model is not None:
                        ooh_travel_time = None
                        if travel_times is not None and 'ooh' in travel_times and idx < len(travel_times['ooh']):
                            ooh_travel_time = travel_times['ooh'][idx]
                        util = customerchoice_model.mnl(state[0], pp, travel_time=ooh_travel_time)
                    else:
                        # 回退到原来的实现
                        util = self.mnl(state[0],pp)
                    # OOH point cost: includes service time (l_mp) and routing cost
                    pp_costs[idx] = self.l_mp*mltplr + mltplr * ((1-theta)* ( self.cheapestInsertionCosts(pp.location, state[1]) )+ theta*(costs[idx+2]-costs[0]) )
                    sum_mnl += exp(util+(state[0].incentiveSensitivity*(pp_costs[idx]-self.revenue)))
            
            # 添加外部选项到MNL分母（如果存在）- 修复：确保与环境的MNL模型一致
            outside_option_util = getattr(self.config, 'outside_option_util', None)
            if outside_option_util is not None:
                sum_mnl += exp(outside_option_util)
           
            #2 obtain lambert w0
            lambertw0 = (lambertw(sum_mnl/e).real+1)/state[0].incentiveSensitivity
            
            # 3 calculate discounts/prices with conservative pricing to avoid over-subsidization
            a_hat = np.zeros(len(pps)+1)
            
            # 调整定价策略：更好地引导客户选择OOH点
            # 降低safety_margin，减少折扣削减，允许更大的价格差异
            safety_margin = 0.1  # 从0.2降低到0.1，减少折扣削减
            
            # Home delivery price: cost - revenue - lambertw0, with safety margin
            home_price_base = homeCosts - self.revenue - lambertw0
            # Apply safety margin: if price is negative (discount), significantly reduce discount
            if home_price_base < 0:
                # More aggressive reduction: reduce discount by safety_margin
                home_price_base = home_price_base * (1 - safety_margin)
            # 提高home delivery最低价格，减少home delivery的吸引力
            home_price_base = max(home_price_base, self.min_p)
            a_hat[0] = home_price_base
            
            for idx,pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    pp_price_base = pp_costs[idx] - self.revenue - lambertw0
                    # Apply safety margin: if price is negative (discount), significantly reduce discount
                    if pp_price_base < 0:
                        # More aggressive reduction: reduce discount by safety_margin
                        pp_price_base = pp_price_base * (1 - safety_margin)
                    # 降低OOH点最低价格，允许更大的折扣，提高OOH点的吸引力
                    pp_price_base = max(pp_price_base, self.min_p)
                    a_hat[idx+1] = pp_price_base
            
            # Clip to valid price range, but ensure discounts are not too large
            # 提高max_discount，允许更大的OOH点折扣
            a_hat = np.clip(a_hat, self.min_p, self.max_p)

            return np.around(a_hat,decimals=2)
    
    def get_action_offerall(self,state,training):   
        #check if pp is feasible
        if self.load_data:
            mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]
        action = np.empty(0,dtype=int)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                action = np.append(action,pp.id_num)
        return action
    
    def addedcosts_euclid(self,route,i,loc):
        costs = self.getdistance_euclidean(route[i-1],loc) + self.getdistance_euclidean(loc,route[i])\
                    - self.getdistance_euclidean(route[i-1],route[i])
        return costs/self.dist_scaler
   
    def addedcosts_distmat(self,route,i,loc):
        costs = self.dist_matrix[route[i-1].id_num][loc.id_num] + self.dist_matrix[loc.id_num][route[i].id_num]\
                         - self.dist_matrix[route[i-1].id_num][route[i].id_num]
        return costs/self.dist_scaler     
                    
    def cheapestInsertionCosts(self,loc,fleet):
        cheapestCosts = float("inf")
        for v in fleet["fleet"]:#note we do not check feasibility of insertion here, let this to HGS
            for i in range(1,len(v["routePlan"])):
               addedCosts = self.addedcosts(v["routePlan"],i,loc)
               if addedCosts < cheapestCosts:
                   cheapestCosts = addedCosts
        
        return cheapestCosts
    
    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
    def get_prediction(self,cur_feat,home,pps):
        time_int = min(int(home.time/self.interval),self.n_layers-1)
        new_feat = torch.cat((2+len(pps))*[cur_feat])
        new_feat[1][time_int][self.customer_cell[home.id_num][0]][self.customer_cell[home.id_num][1]]+=1
        cap_feat = torch.zeros((2+len(pps)))
        cap_feat[0] = 1000000
        cap_feat[1] = 1000000
        for idx,p in enumerate(pps):
            new_feat[idx+2][time_int][self.customer_cell[p.location.id_num][0]][self.customer_cell[p.location.id_num][1]]+=1
            cap_feat[idx+2] = p.remainingCapacity-1
        with torch.no_grad():
            outputs = self.supervised_ml(
                new_feat.to(self.device),
                cap_feat.to(self.device),
            ).view(-1)
        return outputs.detach().cpu().numpy().tolist()
    
    def get_per_customer_travel_times(self, fleet, data, state_at_arrival=None):
        """
        从最终优化路线中计算每个客户的真实在途时间
        
        关键概念：计算如果客户选择HOME或某个OOH点，该点在最终VRP路线中的服务顺序，
        然后根据该服务顺序计算从该点到终点（depot）的累计时间。
        
        具体实现：
        - HOME点：从最终路线中找到HOME点的位置，计算从HOME点到终点（depot）的累计时间
        - OOH点：模拟插入OOH点到客户到达时刻t的fleet状态，找到最佳插入位置，
                 计算从该插入位置到终点（depot）的累计时间
        
        Args:
            fleet: 最终优化后的车队
            data: episode数据，包含客户到达信息
            state_at_arrival: 可选的，客户到达时的状态（用于获取所有OOH点信息）
        
        Returns:
            travel_times: 列表，每个元素是[arrival_time, customer_id, travel_time_dict, ooh_points_info, data_idx]
                travel_time_dict包含：
                    - 'home': 从HOME点到终点（depot）的累计时间（秒）
                    - 'ooh': 列表，从各OOH点到终点（depot）的累计时间（秒）
                ooh_points_info: OOH点信息列表
        """
        if not self.use_travel_time_prediction:
            return None
        
        # 获取车辆速度
        vehicle_speed = getattr(self.config, 'truck_speed', 1.0)
        
        travel_times = []
        customer_arrival_times = {}  # {customer_id: arrival_time}
        customer_data = {}  # {customer_id: (arrival_time, index_in_data)}
        
        # 从data中获取客户到达时间和索引
        for i, (customer_id, arrival_time) in enumerate(zip(data['id'], data['time'])):
            if customer_id > 0:  # 排除depot
                customer_arrival_times[customer_id] = arrival_time
                customer_data[customer_id] = (arrival_time, i)
        
        # 获取所有OOH点的信息
        all_ooh_points = []
        if self.load_data and hasattr(self, 'first_parcelpoint_id'):
            n_total = len(self.dist_matrix[0])
            for ooh_id in range(self.first_parcelpoint_id, n_total):
                all_ooh_points.append(ooh_id)
        else:
            # 对于生成的数据，从fleet中提取所有OOH点
            for v in fleet["fleet"]:
                for location in v["routePlan"]:
                    if location.id_num > 0:
                        if self.load_data:
                            if location.id_num >= self.first_parcelpoint_id:
                                if location.id_num not in all_ooh_points:
                                    all_ooh_points.append(location.id_num)
        
        # 为每个客户计算所有选项的在途时间
        for customer_id, (arrival_time, data_idx) in customer_data.items():
            # 获取HOME点的位置
            home_loc = None
            for v in fleet["fleet"]:
                for loc in v["routePlan"]:
                    if loc.id_num == customer_id:
                        home_loc = loc
                        break
                if home_loc:
                    break
            
            if home_loc is None:
                continue  # 无法找到HOME点，跳过
            
            # 计算如果选择HOME点，从HOME点到终点的累计时间
            # 需要找到HOME点在最终路线中的位置
            home_travel_time = None
            home_position = None
            for v_idx, v in enumerate(fleet["fleet"]):
                for i, location in enumerate(v["routePlan"]):
                    if location.id_num == customer_id:
                        home_position = (v_idx, i)
                        break
                if home_position:
                    break
            
            if home_position:
                # 计算从HOME点到终点的累计时间
                home_travel_time = self._calculate_travel_time_to_end(
                    fleet, home_position[0], home_position[1], vehicle_speed
                )
            
            # 计算如果选择各个OOH点，从各OOH点到终点的累计时间
            # 关键：需要模拟如果客户选择了这个OOH点，它会插入到什么位置
            # 然后计算从该插入位置到终点的累计时间
            ooh_travel_times = []
            ooh_points_info = []
            
            # 获取客户到达时的fleet状态（只包含在时刻t之前已访问的位置）
            # 这样我们可以模拟插入操作
            temp_fleet_at_arrival = self._get_fleet_state_at_time(fleet, arrival_time, vehicle_speed)
            
            for ooh_id in all_ooh_points:
                # 获取OOH点的位置
                ooh_loc = None
                for v in fleet["fleet"]:
                    for loc in v["routePlan"]:
                        if loc.id_num == ooh_id:
                            ooh_loc = loc
                            break
                    if ooh_loc:
                        break
                
                if ooh_loc is None:
                    ooh_travel_times.append(None)
                    ooh_points_info.append({'id': ooh_id, 'location': None})
                    continue
                
                # 在客户到达时刻t的fleet状态中，找到插入OOH点的最佳位置
                best_vehicle_idx, best_insert_idx, _ = self._find_best_insertion_position(
                    ooh_loc, temp_fleet_at_arrival
                )
                
                if best_vehicle_idx is not None and best_insert_idx is not None:
                    # 模拟插入OOH点后的路线
                    # 计算从插入位置到终点的累计时间
                    ooh_travel_time = self._calculate_travel_time_to_end_from_insertion(
                        temp_fleet_at_arrival, best_vehicle_idx, best_insert_idx, vehicle_speed, ooh_loc
                    )
                else:
                    # 如果无法找到插入位置，使用默认值
                    ooh_travel_time = None
                
                ooh_travel_times.append(ooh_travel_time)
                ooh_points_info.append({
                    'id': ooh_id,
                    'location': ooh_loc
                })
            
            travel_time_dict = {
                'home': home_travel_time,
                'ooh': ooh_travel_times
            }
            
            travel_times.append([arrival_time, customer_id, travel_time_dict, ooh_points_info, data_idx])
        
        # 按到达时间排序
        travel_times.sort(key=lambda x: x[0])
        
        return travel_times if len(travel_times) > 0 else None
    
    def _calculate_travel_time_to_end(self, fleet, vehicle_idx, position_idx, vehicle_speed):
        """
        计算从路线中某个位置到终点的累计时间
        
        Args:
            fleet: 车队
            vehicle_idx: 车辆索引
            position_idx: 位置索引（在路线中的位置）
            vehicle_speed: 车辆速度
        
        Returns:
            travel_time: 从该位置到终点的累计时间（秒）
        """
        vehicle = fleet["fleet"][vehicle_idx]
        route = vehicle["routePlan"]
        
        if position_idx >= len(route) - 1:
            # 如果已经是最后一个位置（depot），返回0
            return 0.0
        
        cumulative_time = 0.0
        
        # 从当前位置到终点（depot）的累计时间
        for i in range(position_idx, len(route) - 1):
            current_loc = route[i]
            next_loc = route[i + 1]
            
            if self.load_data:
                distance = self.dist_matrix[current_loc.id_num][next_loc.id_num]
            else:
                distance = self.getdistance_euclidean(current_loc, next_loc)
            
            travel_time = distance / vehicle_speed if vehicle_speed > 0 else distance
            cumulative_time += travel_time
        
        return cumulative_time
    
    def _calculate_travel_time_to_end_from_insertion(self, fleet, vehicle_idx, insert_idx, vehicle_speed, inserted_location):
        """
        计算如果插入某个位置，从插入位置到终点的累计时间
        
        Args:
            fleet: 车队（客户到达时刻t的状态）
            vehicle_idx: 车辆索引
            insert_idx: 插入位置索引
            vehicle_speed: 车辆速度
            inserted_location: 要插入的位置对象
        
        Returns:
            travel_time: 从插入位置到终点的累计时间（秒）
        """
        vehicle = fleet["fleet"][vehicle_idx]
        route = vehicle["routePlan"]
        
        if len(route) == 0:
            return 0.0
        
        cumulative_time = 0.0
        
        # 模拟插入后的路线：... -> prev -> inserted -> next -> ... -> depot
        # 计算从inserted到终点的累计时间
        
        if insert_idx >= len(route):
            # 插入在最后（depot之前）
            if len(route) > 0:
                prev_loc = route[-1]  # 最后一个位置（可能是depot）
                if prev_loc.id_num != 0:  # 不是depot
                    # 从插入位置到depot
                    if self.load_data:
                        distance = self.dist_matrix[inserted_location.id_num][0]
                    else:
                        from Environments.OOH.containers import Location
                        depot = Location(0, 0, 0, 0)
                        distance = self.getdistance_euclidean(inserted_location, depot)
                    cumulative_time = distance / vehicle_speed if vehicle_speed > 0 else distance
            return cumulative_time
        
        # 插入在中间位置
        # 计算：从inserted到next，然后从next到终点
        if insert_idx < len(route):
            next_loc = route[insert_idx]
            
            # 从插入位置到下一个位置
            if self.load_data:
                distance_to_next = self.dist_matrix[inserted_location.id_num][next_loc.id_num]
            else:
                distance_to_next = self.getdistance_euclidean(inserted_location, next_loc)
            
            travel_time_to_next = distance_to_next / vehicle_speed if vehicle_speed > 0 else distance_to_next
            cumulative_time += travel_time_to_next
            
            # 从下一个位置到终点
            next_position_idx = insert_idx
            time_from_next_to_end = self._calculate_travel_time_to_end(fleet, vehicle_idx, next_position_idx, vehicle_speed)
            cumulative_time += time_from_next_to_end
        
        return cumulative_time
    
    def _find_best_insertion_position(self, target_location, fleet):
        """
        找到插入目标位置（HOME或OOH点）的最佳位置
        
        Args:
            target_location: 目标位置对象（Location对象）
            fleet: 当前fleet状态（客户到达时刻t的状态）
        
        Returns:
            (best_vehicle_idx, best_insert_idx, cost): 最佳插入位置
        """
        if target_location is None:
            return None, None, None
        
        best_cost = float("inf")
        best_vehicle_idx = None
        best_insert_idx = None
        
        target_location_id = target_location.id_num
        
        # 遍历所有车辆，找到最佳插入位置
        for v_idx, v in enumerate(fleet["fleet"]):
            route = v["routePlan"]
            if len(route) == 0:
                continue
            
            # 插入位置从1开始（depot之后）
            for i in range(1, len(route)):
                # 计算插入成本
                if self.load_data:
                    added_cost = (self.dist_matrix[route[i-1].id_num][target_location_id] +
                                 self.dist_matrix[target_location_id][route[i].id_num] -
                                 self.dist_matrix[route[i-1].id_num][route[i].id_num])
                else:
                    added_cost = (self.getdistance_euclidean(route[i-1], target_location) +
                                 self.getdistance_euclidean(target_location, route[i]) -
                                 self.getdistance_euclidean(route[i-1], route[i]))
                
                if added_cost < best_cost:
                    best_cost = added_cost
                    best_vehicle_idx = v_idx
                    best_insert_idx = i
        
        # 如果没有找到插入位置（可能路线为空），尝试插入到第一个位置（depot之后）
        if best_vehicle_idx is None:
            for v_idx, v in enumerate(fleet["fleet"]):
                route = v["routePlan"]
                if len(route) > 0:
                    best_vehicle_idx = v_idx
                    best_insert_idx = 1  # 插入在depot之后
                    break
        
        return best_vehicle_idx, best_insert_idx, best_cost
    
    def _get_vehicle_position_at_time(self, fleet, target_time, vehicle_speed):
        """
        找到车辆在指定时刻t时的位置
        
        Args:
            fleet: 最终优化后的车队
            target_time: 目标时刻t（客户到达时间）
            vehicle_speed: 车辆速度
        
        Returns:
            (location_id, location): 车辆在时刻t时的位置（id和Location对象），如果无法确定则返回None
        """
        # 遍历所有车辆，找到在时刻t时正在执行任务的车辆
        for v_idx, v in enumerate(fleet["fleet"]):
            cumulative_time = 0.0
            
            for i, location in enumerate(v["routePlan"]):
                location_id = location.id_num
                
                if location_id == 0:  # depot
                    cumulative_time = 0.0
                    # 如果目标时间在depot停留期间（时间=0），返回depot
                    if target_time <= 0:
                        return (0, location)
                    continue
                
                # 计算到达当前位置的时间
                if i == 0:
                    # 从depot到第一个位置
                    if self.load_data:
                        distance = self.dist_matrix[0][location_id]
                    else:
                        from Environments.OOH.containers import Location
                        depot = Location(0, 0, 0, 0) if not hasattr(self, 'coords') or len(self.coords) == 0 else self.coords[0]
                        distance = self.getdistance_euclidean(depot, location)
                else:
                    # 从前一个位置到当前位置
                    prev_location = v["routePlan"][i-1]
                    if self.load_data:
                        distance = self.dist_matrix[prev_location.id_num][location_id]
                    else:
                        distance = self.getdistance_euclidean(prev_location, location)
                
                travel_time_to_location = distance / vehicle_speed if vehicle_speed > 0 else distance
                arrival_time_at_location = cumulative_time + travel_time_to_location
                
                # 检查目标时间是否在这个位置之前或之后
                if target_time <= arrival_time_at_location:
                    # 目标时间在到达当前位置之前，车辆正在前往当前位置
                    # 返回前一个位置（或depot）
                    if i == 0:
                        # 从depot前往第一个位置，返回depot
                        depot_loc = v["routePlan"][0] if len(v["routePlan"]) > 0 and v["routePlan"][0].id_num == 0 else None
                        if depot_loc:
                            return (0, depot_loc)
                        else:
                            # 如果找不到depot，返回当前位置（简化处理）
                            return (location_id, location)
                    else:
                        # 从前一个位置前往当前位置，返回前一个位置
                        prev_loc = v["routePlan"][i-1]
                        return (prev_loc.id_num, prev_loc)
                
                # 更新累计时间
                cumulative_time = arrival_time_at_location
                
                # 如果这是最后一个位置，且目标时间在到达之后，返回当前位置
                if i == len(v["routePlan"]) - 1:
                    if target_time >= arrival_time_at_location:
                        return (location_id, location)
        
        # 如果无法找到，返回None（所有车辆都在depot或路线为空）
        # 尝试返回depot作为默认位置
        for v in fleet["fleet"]:
            if len(v["routePlan"]) > 0:
                depot = v["routePlan"][0] if v["routePlan"][0].id_num == 0 else None
                if depot:
                    return (0, depot)
        
        return None
    
    def _get_fleet_state_at_time(self, fleet, target_time, vehicle_speed):
        """
        获取车辆在指定时刻t时的fleet状态（只包含在时刻t之前已访问的位置）
        
        Args:
            fleet: 最终优化后的车队
            target_time: 目标时刻t（客户到达时间）
            vehicle_speed: 车辆速度
        
        Returns:
            temp_fleet: 临时fleet对象，包含车辆在时刻t时的状态
        """
        from Environments.OOH.containers import Vehicle, Fleet
        
        temp_vehicles = []
        
        for v_idx, v in enumerate(fleet["fleet"]):
            temp_route = []
            cumulative_time = 0.0
            
            for i, location in enumerate(v["routePlan"]):
                location_id = location.id_num
                
                if location_id == 0:  # depot
                    cumulative_time = 0.0
                    temp_route.append(location)  # 总是包含depot
                    continue
                
                # 计算到达当前位置的时间
                if i == 0:
                    if self.load_data:
                        distance = self.dist_matrix[0][location_id]
                    else:
                        from Environments.OOH.containers import Location
                        depot = Location(0, 0, 0, 0) if not hasattr(self, 'coords') or len(self.coords) == 0 else self.coords[0]
                        distance = self.getdistance_euclidean(depot, location)
                else:
                    prev_location = v["routePlan"][i-1]
                    if self.load_data:
                        distance = self.dist_matrix[prev_location.id_num][location_id]
                    else:
                        distance = self.getdistance_euclidean(prev_location, location)
                
                travel_time_to_location = distance / vehicle_speed if vehicle_speed > 0 else distance
                arrival_time_at_location = cumulative_time + travel_time_to_location
                
                # 如果到达时间在目标时间之前或等于目标时间，包含该位置
                if arrival_time_at_location <= target_time:
                    temp_route.append(location)
                    cumulative_time = arrival_time_at_location
                else:
                    # 如果到达时间在目标时间之后，停止添加位置
                    # 但需要包含当前位置（车辆正在前往该位置）
                    # 简化：不包含该位置，因为车辆还没有到达
                    break
            
            # 创建临时车辆
            if len(temp_route) > 0:
                temp_vehicle = Vehicle(temp_route, v.capacity, v.id_num)
                temp_vehicles.append(temp_vehicle)
        
        # 如果所有车辆都为空，创建一个只包含depot的车辆
        if len(temp_vehicles) == 0:
            for v in fleet["fleet"]:
                if len(v["routePlan"]) > 0:
                    depot = v["routePlan"][0] if v["routePlan"][0].id_num == 0 else None
                    if depot:
                        temp_vehicle = Vehicle([depot], v.capacity, v.id_num)
                        temp_vehicles.append(temp_vehicle)
                        break
        
        temp_fleet = Fleet(temp_vehicles)
        return temp_fleet
    
    def get_travel_time_prediction(self, cur_feat, home, pps):
        """
        预测车辆在途时间（从HOME点或OOH点到终点depot的时间）
        
        关键概念：预测如果客户选择HOME或某个OOH点，该点在最终VRP路线中的服务顺序，
        然后计算从该点到终点（depot）的累计时间。这是基于历史运行情况预测的服务顺序。
        
        Args:
            cur_feat: 当前状态特征 [1, n_layers, grid_dim, grid_dim]
            home: 客户HOME点
            pps: OOH点列表（推荐的OOH点）
        
        Returns:
            travel_times: 字典，包含：
                - 'home': 从HOME点到终点（depot）的累计时间（秒）
                - 'ooh': 列表，从各OOH点到终点（depot）的累计时间（秒），
                         顺序与OOH点列表一致
        """
        if not self.use_travel_time_prediction or self.travel_time_predictor is None:
            # 如果未启用预测，返回None
            return None
        
        try:
            time_int = min(int(home.time/self.interval), self.n_layers-1)
            
            # 为HOME点和每个OOH点构建特征
            # [0] = depot (不使用), [1] = home, [2:] = OOH points
            new_feat = torch.cat((2+len(pps))*[cur_feat])
            
            # HOME点的特征
            new_feat[1][time_int][self.customer_cell[home.id_num][0]][self.customer_cell[home.id_num][1]] += 1
            
            # 容量特征
            cap_feat = torch.zeros((2+len(pps)))
            cap_feat[0] = 1000000  # depot
            cap_feat[1] = 1000000  # home (无容量限制)
            
            # OOH点的特征
            for idx, p in enumerate(pps):
                new_feat[idx+2][time_int][self.customer_cell[p.location.id_num][0]][self.customer_cell[p.location.id_num][1]] += 1
                cap_feat[idx+2] = p.remainingCapacity - 1
            
            # 使用CNN预测在途时间
            travel_times = {'home': None, 'ooh': []}
            
            with torch.no_grad():
                # 预测HOME点的在途时间
                home_feat = new_feat[1].unsqueeze(0).to(self.device)
                home_cap = cap_feat[1].unsqueeze(0).to(self.device)
                home_travel_time = self.travel_time_predictor(home_feat, home_cap).item()
                travel_times['home'] = max(0.0, home_travel_time)  # 确保非负
                
                # 预测每个OOH点的在途时间
                for idx in range(len(pps)):
                    ooh_feat = new_feat[idx+2].unsqueeze(0).to(self.device)
                    ooh_cap = cap_feat[idx+2].unsqueeze(0).to(self.device)
                    ooh_travel_time = self.travel_time_predictor(ooh_feat, ooh_cap).item()
                    travel_times['ooh'].append(max(0.0, ooh_travel_time))  # 确保非负
            
            return travel_times
            
        except Exception as e:
            # 如果预测失败，返回None
            print(f"在途时间预测失败: {e}")
            return None
                                                       
    def mnl_euclid(self,customer,parcelpoint):
        distance = self.getdistance_euclidean(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_walk = -1.0 / max(self.dist_scaler, 1e-8)
        return utility_meeting_point_nonprice(
            self.base_util,
            beta_walk,
            None,
            distance,
            None,
        )
    
    def mnl_distmat(self,customer,parcelpoint):
        if self.choice_util_matrix is not None:
            ci = int(customer.id_num)
            pi = int(parcelpoint.id_num)
            if (
                0 <= ci < self.choice_util_matrix.shape[0]
                and 0 <= pi < self.choice_util_matrix.shape[1]
            ):
                util = float(self.choice_util_matrix[ci][pi])
                if np.isfinite(util):
                    return self.base_util + util
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_walk = -1.0 / max(self.dist_scaler, 1e-8)
        return utility_meeting_point_nonprice(
            self.base_util,
            beta_walk,
            None,
            distance,
            None,
        )
    
    
    def update(self,data,state,done=False):
        #first obtain data      
        if not done:
            self.features = np.vstack(( self.features, self.get_feature_rep(data).flatten()))
            try:
                # 处理 data["id"] 可能是标量或数组的情况
                customer_id = data["id"]
                if np.isscalar(customer_id):
                    customer_id = int(customer_id)
                else:
                    customer_id = int(customer_id[-1])  # 获取最后一个客户ID
                self.cap_features = np.vstack(( self.cap_features, state[2]["parcelpoints"][customer_id].remainingCapacity))
            except:
                self.cap_features = np.vstack(( self.cap_features, 1000000))#home delivery
            
            # 收集在途时间训练数据（如果启用）
            if self.use_travel_time_prediction and self.travel_time_memory is not None:
                # 在episode进行中，我们暂时不收集在途时间数据
                # 因为真实的在途时间需要在episode结束时从最终路线中计算
                pass
            
            return 0.0
        else:
            # Normalize scalar/array route fields. With high quit-rate episodes, data['id']
            # can be scalar (only depot kept), which can break iteration and may trigger
            # expensive/unstable solver calls on empty instances.
            ids_raw = data.get('id', [])
            times_raw = data.get('time', [])
            if np.isscalar(ids_raw):
                route_ids = np.array([int(ids_raw)])
            else:
                route_ids = np.asarray(ids_raw)
            if np.isscalar(times_raw):
                route_times = np.array([times_raw])
            else:
                route_times = np.asarray(times_raw)
            data['id'] = route_ids
            data['time'] = route_times

            # obtain final CVRP schedule after end of booking horizon
            # Skip HGS if no service customer exists (depot-only route).
            positive_customer_count = int(np.sum(route_ids > 0)) if route_ids.size > 0 else 0
            # Very small instances (0/1 service customer) are numerically fragile for HGS
            # and can dominate runtime in high-quit regimes.
            if positive_customer_count <= 1:
                fleet = {"fleet": []}
                cost = 0.0
            else:
                fleet,cost = self._get_terminal_label_route(data, route_ids)
            
            target = self.get_per_customer_costs(fleet)
            target = sorted(target, key=itemgetter(0))#sort in order of arrival (same as features)
            
            # 创建到达时间到成本的映射
            target_dict = {t[0]: t[1] for t in target}  # {arrival_time: cost}
            
            # 匹配特征和目标值：通过到达时间匹配
            # data['id'] 只包含不退出的客户，所以 self.features 的长度应该等于 len(data['id'])
            # 但为了安全，我们只匹配那些在 target 中的客户（即实际被服务的客户）
            filtered_features = []
            filtered_cap_features = []
            filtered_targets = []  # 格式：[[arrival_time, cost], ...]
            
            # 遍历 data 中的客户，只保留那些在 target 中的（即被服务的）
            for i, customer_id in enumerate(data['id']):
                if customer_id > 0:  # 排除depot
                    arrival_time = data['time'][i]
                    if arrival_time in target_dict:
                        # 这个客户被服务了，保留其特征和目标值
                        if i < len(self.features) and i < len(self.cap_features):
                            filtered_features.append(self.features[i])
                            filtered_cap_features.append(self.cap_features[i])
                            # 保持 [time, cost] 格式
                            filtered_targets.append([arrival_time, target_dict[arrival_time]])
            
            # 转换为numpy数组并添加到memory
            if len(filtered_features) > 0:
                filtered_features = np.array(filtered_features)
                filtered_cap_features = np.array(filtered_cap_features)
                penalties = (20 / (filtered_cap_features + 0.1)).flatten()  # 展平为一维数组
                # adjusted_target 格式：[[arrival_time, adjusted_cost], ...]
                # MemoryBuffer.add 期望 target[i][1] 是成本值
                adjusted_target = [[t[0], t[1] + float(p)] for t, p in zip(filtered_targets, penalties)]
                self.memory.add(filtered_features, filtered_cap_features, adjusted_target)
            
            # 收集在途时间训练数据（如果启用）
            if self.use_travel_time_prediction and self.travel_time_memory is not None:
                # 获取车辆速度
                vehicle_speed = getattr(self.config, 'truck_speed', 1.0)
                
                travel_time_targets = self.get_per_customer_travel_times(fleet, data)
                if travel_time_targets is not None and len(travel_time_targets) > 0:
                    # travel_time_targets格式：[arrival_time, customer_id, travel_time_dict, ooh_points_info, data_idx]
                    # 为每个客户的所有选项（HOME和所有OOH点）创建训练样本
                    for arrival_time, customer_id, travel_time_dict, ooh_points_info, data_idx in travel_time_targets:
                        if data_idx >= len(self.features):
                            continue
                        
                        # 获取客户到达时的fleet状态特征
                        # 使用最终优化的fleet，但需要找到车辆在客户到达时刻t的位置
                        # 构建一个临时的fleet状态，模拟客户到达时刻t的fleet状态
                        # 简化：使用最终优化的fleet，但只包含在时刻t之前已访问的位置
                        temp_fleet_state = self._get_fleet_state_at_time(fleet, arrival_time, vehicle_speed)
                        
                        # 使用与预测时相同的特征构建方法
                        customer_feat = self.get_feature_rep_infer(temp_fleet_state)
                        customer_feat_np = customer_feat.squeeze(0).cpu().numpy()  # [n_layers, grid_dim, grid_dim]
                        customer_feat_tensor = torch.tensor(customer_feat_np, dtype=float32, requires_grad=False)
                        
                        # 获取客户HOME点信息
                        customer_home_id = customer_id
                        time_int = min(int(arrival_time/self.interval), self.n_layers-1)
                        
                        # 为HOME点创建训练样本
                        if travel_time_dict.get('home') is not None:
                            # 构建HOME点的特征（与预测时相同）
                            home_feat = customer_feat_tensor.clone()
                            home_feat[time_int][self.customer_cell[customer_home_id][0]][self.customer_cell[customer_home_id][1]] += 1
                            
                            self.travel_time_features = np.vstack((
                                self.travel_time_features,
                                home_feat.numpy().flatten()
                            ))
                            self.travel_time_cap_features = np.vstack((
                                self.travel_time_cap_features,
                                1000000  # HOME点无容量限制
                            ))
                            self.travel_time_targets.append([arrival_time, travel_time_dict['home']])
                        
                        # 为所有OOH点创建训练样本
                        if 'ooh' in travel_time_dict and travel_time_dict['ooh'] is not None:
                            for ooh_idx, (ooh_time, ooh_info) in enumerate(zip(travel_time_dict['ooh'], ooh_points_info)):
                                if ooh_time is not None:
                                    # 构建OOH点的特征（与预测时相同）
                                    ooh_feat = customer_feat_tensor.clone()
                                    ooh_id = ooh_info['id']
                                    ooh_feat[time_int][self.customer_cell[ooh_id][0]][self.customer_cell[ooh_id][1]] += 1
                                    
                                    self.travel_time_features = np.vstack((
                                        self.travel_time_features,
                                        ooh_feat.numpy().flatten()
                                    ))
                                    
                                    # 获取OOH点的容量（从最终fleet中获取，或使用默认值）
                                    ooh_cap = 1000000  # 默认值
                                    # 尝试从fleet中获取OOH点的容量
                                    for v in fleet["fleet"]:
                                        for loc in v["routePlan"]:
                                            if loc.id_num == ooh_id:
                                                # 找到OOH点，但无法直接获取容量，使用默认值
                                                break
                                    
                                    self.travel_time_cap_features = np.vstack((
                                        self.travel_time_cap_features,
                                        ooh_cap
                                    ))
                                    self.travel_time_targets.append([arrival_time, ooh_time])
                    
                    # 将收集的数据添加到MemoryBuffer
                    if len(self.travel_time_targets) > 0:
                        travel_time_target_array = np.array([[t[0], t[1]] for t in self.travel_time_targets])
                        self.travel_time_memory.add(
                            self.travel_time_features,
                            self.travel_time_cap_features.flatten(),
                            travel_time_target_array
                        )
                    
                    # 清空临时数据
                    self.travel_time_features = np.empty((0, self.n_layers*self.grid_dim*self.grid_dim))
                    self.travel_time_cap_features = np.empty((0, 1))
                    self.travel_time_targets = []
            
            self.features = np.empty((0,self.n_layers*self.grid_dim*self.grid_dim))
            self.cap_features = np.empty((0,1))
            #optionally update model
            if self.initial_phase:#train model initial phase            
                if self.memory.length >= self.config.buffer_size:
                    self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
            elif not self.config.only_phase_one:
                #simply update CNN after every new data point collected
                    self.optimize()
        
            return cost

    def _get_terminal_label_route(self, data, route_ids=None):
        if route_ids is None:
            route_ids = data.get('id', [])
            if np.isscalar(route_ids):
                route_ids = np.array([int(route_ids)])
            else:
                route_ids = np.asarray(route_ids)

        cache = self._terminal_label_cache
        if cache is not None and cache.get("data_id") == id(data):
            return cache["fleet"], cache["cost"]

        if self.route_label_mode == "hep":
            env = getattr(self.config, "env", None)
            fleet = getattr(env, "fleet", None)
            if fleet is None:
                fleet = {"fleet": []}
                cost = 0.0
            else:
                cost = self._fleet_route_distance(fleet)
        else:
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix, route_ids)
            fleet,cost = self.reopt_HGS_final(data)#do a final reopt

        self._terminal_label_cache = {
            "data_id": id(data),
            "fleet": fleet,
            "cost": cost,
        }
        return fleet, cost

    def _label_route_distance(self, loc_a, loc_b):
        if self.load_data:
            a_id = 0 if loc_a is None else int(loc_a.id_num)
            b_id = 0 if loc_b is None else int(loc_b.id_num)
            return float(self.dist_matrix[a_id][b_id])
        if loc_a is None:
            loc_a = getattr(getattr(self.config, 'env', None), 'depot', None)
        if loc_b is None:
            loc_b = getattr(getattr(self.config, 'env', None), 'depot', None)
        if loc_a is None or loc_b is None:
            return 0.0
        return float(self.getdistance_euclidean(loc_a, loc_b))

    def _fleet_route_distance(self, fleet):
        distance = 0.0
        for vehicle in fleet["fleet"]:
            route = [loc for loc in vehicle["routePlan"] if int(loc.id_num) > 0]
            prev_loc = None
            for loc in route:
                distance += self._label_route_distance(prev_loc, loc)
                prev_loc = loc
            if route:
                distance += self._label_route_distance(prev_loc, None)
        return distance
    def optimize(self):
        # Take one supervised step
        feat,cap_feat,target = self.memory.sample(batch_size=self.config.batch_size)
        loss = self.self_supervised_update(feat,cap_feat,target)
        print("Huber loss: ", loss)
        
        # 训练在途时间预测模型（如果启用）
        if self.use_travel_time_prediction and self.travel_time_memory is not None and self.travel_time_memory.length > 0:
            travel_time_loss = self.optimize_travel_time()
            if travel_time_loss is not None:
                print("Travel time Huber loss: ", travel_time_loss)
    
    
    def self_supervised_update(self, feat,cap_feat,target):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.supervised_ml(feat,cap_feat)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def optimize_travel_time(self):
        """
        训练在途时间预测模型
        """
        if not self.use_travel_time_prediction or self.travel_time_memory is None or self.travel_time_memory.length == 0:
            return None
        
        feat, cap_feat, target = self.travel_time_memory.sample(batch_size=self.config.batch_size)
        loss = self.travel_time_supervised_update(feat, cap_feat, target)
        return loss
    
    def travel_time_supervised_update(self, feat, cap_feat, target):
        """
        在途时间预测模型的监督学习更新
        """
        if not self.use_travel_time_prediction or self.travel_time_predictor is None:
            return None
        
        # zero the parameter gradients
        self.travel_time_optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.travel_time_predictor(feat, cap_feat)
        loss = self.travel_time_criterion(outputs, target)
        loss.backward()
        self.travel_time_optimizer.step()
        return loss.item()
    
    def initial_phase_training(self, max_epochs=-1):
        initial_losses = []
        travel_time_losses = []
        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for feat,cap_feat,target in self.memory.batch_sample(batch_size=self.config.batch_size, randomize=True):
                loss = self.self_supervised_update(feat,cap_feat,target)
                losses.append(loss)
            initial_losses.append(np.mean(losses))
            
            # 训练在途时间预测模型（如果启用且有数据）
            if self.use_travel_time_prediction and self.travel_time_memory is not None and self.travel_time_memory.length > 0:
                tt_losses = []
                for tt_feat, tt_cap_feat, tt_target in self.travel_time_memory.batch_sample(batch_size=self.config.batch_size, randomize=True):
                    tt_loss = self.travel_time_supervised_update(tt_feat, tt_cap_feat, tt_target)
                    tt_losses.append(tt_loss)
                if len(tt_losses) > 0:
                    travel_time_losses.append(np.mean(tt_losses))
            
            if counter % 1 == 0:
                print("Epoch {} Huber loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if len(travel_time_losses) > 0:
                    print("Epoch {} Travel time Huber loss:: {}".format(counter, np.mean(travel_time_losses[-10:])))
                if self.config.only_phase_one:
                    self.memory.save(self.config.paths['checkpoint']+'initial_' )
                    if self.travel_time_memory is not None:
                        self.travel_time_memory.save(self.config.paths['checkpoint']+'initial_travel_time_')
                    self.save()
                    print("Saved..")
            # Terminate initial phase once it have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break
        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.memory.save(self.config.paths['checkpoint']+'initial_' )
        if self.travel_time_memory is not None:
            self.travel_time_memory.save(self.config.paths['checkpoint']+'initial_travel_time_')
        self.save()
        
    def get_feature_rep(self,data):
        feature = np.zeros((self.n_layers,self.grid_dim,self.grid_dim))
        # 处理 data["id"] 和 data["time"] 可能是标量或数组的情况
        ids = data["id"]
        times = data["time"]
        
        # 如果是标量，转换为数组
        if np.isscalar(ids):
            ids = np.array([ids])
        else:
            ids = np.asarray(ids)
        
        if np.isscalar(times):
            times = np.array([times])
        else:
            times = np.asarray(times)
        
        for i,t in zip(ids, times):
            time_int = min(int(t/self.interval),self.n_layers-1)
            feature[time_int][self.customer_cell[i][0]][self.customer_cell[i][1]]+=1#actual choice of the customer during simulation
        return feature
    
    def get_feature_rep_infer(self,fleet):
        feature = np.zeros((self.n_layers,self.grid_dim,self.grid_dim))
        for v in self._fleet_vehicles(fleet):
            for i in v["routePlan"]:
                time_int = min(int(i.time/self.interval),self.n_layers-1)
                feature[time_int][self.customer_cell[i.id_num][0]][self.customer_cell[i.id_num][1]]+=1
        return torch.tensor(feature,dtype=float32,requires_grad=False).unsqueeze(0)

    def _fleet_vehicles(self, fleet):
        if fleet is None:
            return []
        if isinstance(fleet, dict):
            return fleet.get("fleet", [])
        if hasattr(fleet, "fleet"):
            return fleet.fleet
        return fleet
    
    #NOTE: this implementation slightly differs from the one in the paper, reason for this is that this implementation is way more efficient, so we believe it will be more friendly for public use
    #For our experiments, we did not use this exact method, so there may be some performance loss compared to rsults in the paper using this calculation method.
    #We clal this method "Half-edge partitioning (HEP)", see the paper Dynamic Time Slot Pricing Using Delivery Costs Approximations by Akkerman et al. (2022)
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet,result.cost
    
    def get_per_customer_costs(self,fleet):
        mltplr = self.cost_multiplier
       # addedcosts_home = self.added_costs_home
        costs = []
        for v in fleet["fleet"]:
            if len(v["routePlan"])==1:#when only 1 customer is visited
                costs.append( [v["routePlan"][0].time,mltplr * (self.dist_matrix[0][v["routePlan"][0].id_num])] )
            else:
                for i in range(0,len(v["routePlan"])):
                    #costs is composed of distance*mltplr
                    if i==0:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[0][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                    elif i==len(v["routePlan"])-1:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][0])] )
                    else:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                    
                    # if v["routePlan"][i].id_num < self.first_parcelpoint_id:#customer chose home delivery
                    #     costs[-1][0] += mltplr*self.service_times[v["routePlan"][i].id_num]
        
        return costs

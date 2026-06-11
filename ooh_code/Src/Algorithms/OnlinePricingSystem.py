#!/usr/bin/env python3
"""
Online Ride-sharing Pricing System with CNN-based Cost Prediction

Core Features:
1. CNN predicts insertion costs for boarding points
2. MNL pricing model generates optimal prices
3. Pricing error loss (L2) between predicted and true pricing
4. Strictly online/rolling horizon decision making
5. No future information leakage
"""

import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch import float32
from math import sqrt, exp, e
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy

from Src.Utils.Utils import MemoryBuffer, get_matrix
from Src.Utils.Predictors import CNN_2d, CNN_3d, LinReg
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw


@dataclass
class PricingDecision:
    """单次定价决策的数据结构"""
    customer_id: int
    boarding_points: List  # 备选boarding点
    predicted_costs: np.ndarray  # CNN预测的插入成本
    true_costs: np.ndarray  # 真实插入成本
    predicted_prices: np.ndarray  # 基于预测成本的定价
    true_prices: np.ndarray  # 基于真实成本的定价
    choice_probabilities: np.ndarray  # MNL选择概率
    expected_revenue: float  # 期望收益
    pricing_error: float  # 定价误差 (L2 loss)


@dataclass
class EpisodeResult:
    """单次episode的结果"""
    episode_id: int
    customer_sequence: List[int]  # 乘客到达顺序
    pricing_decisions: List[PricingDecision]  # 每次定价决策
    total_pricing_error: float  # 累计定价误差
    total_expected_revenue: float  # 累计期望收益
    mean_pricing_error: float  # 平均定价误差


class OnlinePricingSystem(Agent):
    """
    在线拼车定价系统
    
    核心流程：
    1. 乘客到达 -> 构造当前状态
    2. CNN预测插入成本 -> 生成预测定价
    3. 计算真实插入成本 -> 生成真实定价
    4. 计算定价误差 -> 更新CNN模型
    5. MNL选择概率 -> 期望收益计算
    """
    
    def __init__(self, config):
        super(OnlinePricingSystem, self).__init__(config)
        
        # ==================== 系统配置 ====================
        self.pricing_enabled = config.pricing
        self.grid_dim = config.grid_dim
        self.n_time_layers = config.n_input_layers
        self.n_boarding_points = config.n_parcelpoints
        
        # ==================== 成本预测模型 ====================
        self._setup_cost_predictor(config)
        self._setup_training_components(config)
        
        # ==================== 定价模型参数 ====================
        self.base_utility = config.base_util
        self.price_sensitivity = config.incentive_sens
        self.max_price = config.max_price
        self.min_price = config.min_price
        self.revenue_per_trip = config.revenue
        
        # ==================== 距离计算 ====================
        self._setup_distance_calculation(config)
        
        # ==================== 数据存储 ====================
        self._setup_data_structures()
        
        # ==================== 性能跟踪 ====================
        self.episode_counter = 0
        self.pricing_error_history = []
        self.revenue_history = []
        
    def _setup_cost_predictor(self, config):
        """设置成本预测模型"""
        if config.use3d_conv:
            self.cost_predictor = CNN_3d(
                self.grid_dim, self.n_time_layers, 
                config.n_filters, config.dropout
            )
        elif config.linearModel:
            self.cost_predictor = LinReg(
                self.grid_dim * self.grid_dim * self.n_time_layers
            )
        else:
            self.cost_predictor = CNN_2d(
                self.grid_dim, self.n_time_layers, 
                config.n_filters, config.dropout
            )
    
    def _setup_training_components(self, config):
        """设置训练组件"""
        self.optimizer = config.optim(
            self.cost_predictor.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        self.device = config.device
        
        # 将模型移到指定设备
        self.cost_predictor.to(self.device)
        
        # 设置modules列表（Agent基类需要）
        self.modules = [('cost_predictor', self.cost_predictor)]
        
        # 记忆缓冲区
        self.memory_buffer = MemoryBuffer(
            max_len=config.buffer_size,
            time_intervals=self.n_time_layers,
            matrix_dim=self.grid_dim,
            target_dim=1,
            atype=float32,
            config=config
        )
    
    def _setup_distance_calculation(self, config):
        """设置距离计算"""
        if config.load_data and hasattr(config, 'coords') and hasattr(config, 'dist_matrix'):
            try:
                self.customer_cell_mapping = get_matrix(
                    config.coords, self.grid_dim, config.hexa
                )
                self.distance_matrix = config.dist_matrix
                self.adjacency_matrix = config.adjacency
                self._calculate_insertion_cost = self._calculate_insertion_cost_matrix
            except Exception as e:
                print(f"距离矩阵设置失败: {e}, 使用欧几里得距离")
                self._calculate_insertion_cost = self._calculate_insertion_cost_euclidean
        else:
            self._calculate_insertion_cost = self._calculate_insertion_cost_euclidean
    
    def _setup_data_structures(self):
        """设置数据结构"""
        self.current_episode_data = {
            'customer_sequence': [],
            'pricing_decisions': [],
            'features': np.empty((0, self.n_time_layers * self.grid_dim * self.grid_dim)),
            'capacity_features': np.empty((0, 1))
        }
        
        self.time_interval = 1  # 时间间隔，可根据需要调整
    
    def get_action(self, state, training=False):
        """
        主要决策函数 - 生成定价方案
        
        Args:
            state: 当前状态 [customer, fleet, boarding_points, step]
            training: 是否处于训练模式
            
        Returns:
            pricing_scheme: 定价方案
        """
        if not self.pricing_enabled:
            return self._generate_default_pricing(state)
        
        # 1. 提取当前状态信息
        current_customer = state[0]
        current_fleet = state[1]
        available_boarding_points = state[2].parcelpoints
        
        # 2. 构造特征表示
        state_features = self._construct_state_features(
            current_fleet, current_customer, available_boarding_points
        )
        
        # 3. CNN预测插入成本
        predicted_costs = self._predict_insertion_costs(
            state_features, current_customer, available_boarding_points
        )
        
        # 4. 计算真实插入成本
        true_costs = self._calculate_true_insertion_costs(
            current_customer, current_fleet, available_boarding_points
        )
        
        # 5. 生成定价方案
        predicted_prices = self._generate_optimal_pricing(
            current_customer, available_boarding_points, predicted_costs
        )
        
        true_prices = self._generate_optimal_pricing(
            current_customer, available_boarding_points, true_costs
        )
        
        # 6. 计算选择概率和期望收益
        choice_probs = self._calculate_choice_probabilities(
            current_customer, available_boarding_points, predicted_prices
        )
        
        expected_revenue = self._calculate_expected_revenue(
            predicted_prices, choice_probs
        )
        
        # 7. 计算定价误差 (L2 loss)
        pricing_error = self._calculate_pricing_error(
            predicted_prices, true_prices
        )
        
        # 8. 存储决策数据
        pricing_decision = PricingDecision(
            customer_id=current_customer.id_num,
            boarding_points=available_boarding_points,
            predicted_costs=predicted_costs,
            true_costs=true_costs,
            predicted_prices=predicted_prices,
            true_prices=true_prices,
            choice_probabilities=choice_probs,
            expected_revenue=expected_revenue,
            pricing_error=pricing_error
        )
        
        self.current_episode_data['pricing_decisions'].append(pricing_decision)
        
        # 9. 存储训练数据
        if training:
            self._store_training_data(
                state_features, predicted_costs, true_costs
            )
        
        # 10. 确保返回的定价数组长度正确
        # 环境期望的是所有boarding points的定价，包括容量为0的
        # 格式：[home_delivery_price, pp1_price, pp2_price, ...]
        final_prices = []
        
        # 添加home delivery价格（第一个元素）
        if len(predicted_prices) > 0:
            final_prices.append(predicted_prices[0])  # home delivery价格
        else:
            final_prices.append(0.0)  # 默认home delivery价格
        
        # 添加所有boarding points的价格
        for i, bp in enumerate(available_boarding_points):
            if bp.remainingCapacity > 0:
                # 找到对应的定价（跳过home delivery）
                bp_index = len([b for b in available_boarding_points[:i] if b.remainingCapacity > 0])
                if bp_index + 1 < len(predicted_prices):  # +1 因为跳过home delivery
                    final_prices.append(predicted_prices[bp_index + 1])
                else:
                    final_prices.append(0.0)  # 默认价格
            else:
                final_prices.append(float('inf'))  # 容量为0的点设为无穷大
        
        return np.array(final_prices)
    
    def _construct_state_features(self, fleet, customer, boarding_points):
        """
        构造状态特征表示
        
        Returns:
            features: 网格特征张量 [n_time_layers, grid_dim, grid_dim]
        """
        features = np.zeros((self.n_time_layers, self.grid_dim, self.grid_dim))
        
        # 检查是否有customer_cell_mapping
        if not hasattr(self, 'customer_cell_mapping'):
            # 如果没有映射，使用简单的网格位置
            self._setup_simple_grid_mapping()
        
        # 添加车辆位置特征
        for vehicle in fleet.fleet:
            for location in vehicle.routePlan:
                try:
                    time_layer = min(int(getattr(location, 'time', 0) / self.time_interval), self.n_time_layers - 1)
                    grid_x, grid_y = self._get_grid_position(location)
                    features[time_layer, grid_x, grid_y] += 1
                except Exception as e:
                    print(f"车辆位置特征添加失败: {e}")
                    continue
        
        # 添加客户位置特征
        try:
            time_layer = min(int(getattr(customer.home, 'time', 0) / self.time_interval), self.n_time_layers - 1)
            grid_x, grid_y = self._get_grid_position(customer.home)
            features[time_layer, grid_x, grid_y] += 1
        except Exception as e:
            print(f"客户位置特征添加失败: {e}")
        
        # 添加boarding点位置特征
        for bp in boarding_points:
            if bp.remainingCapacity > 0:
                try:
                    time_layer = min(int(getattr(bp.location, 'time', 0) / self.time_interval), self.n_time_layers - 1)
                    grid_x, grid_y = self._get_grid_position(bp.location)
                    features[time_layer, grid_x, grid_y] += 1
                except Exception as e:
                    print(f"Boarding点位置特征添加失败: {e}")
                    continue
        
        return torch.tensor(features, dtype=float32, requires_grad=False).unsqueeze(0)
    
    def _setup_simple_grid_mapping(self):
        """设置简单的网格映射（当没有customer_cell_mapping时）"""
        self.customer_cell_mapping = {}
        # 为所有可能的ID创建简单的网格位置
        for i in range(1000):  # 假设最大ID为1000
            x = (i * 7) % self.grid_dim  # 简单的哈希映射
            y = (i * 11) % self.grid_dim
            self.customer_cell_mapping[i] = (x, y)
    
    def _get_grid_position(self, location):
        """获取位置的网格坐标"""
        try:
            if hasattr(self, 'customer_cell_mapping') and hasattr(location, 'id_num') and location.id_num in self.customer_cell_mapping:
                return self.customer_cell_mapping[location.id_num]
            else:
                # 回退到简单的位置计算
                if hasattr(location, 'id_num'):
                    x = (location.id_num * 7) % self.grid_dim
                    y = (location.id_num * 11) % self.grid_dim
                else:
                    # 如果没有id_num，使用默认位置
                    x = 0
                    y = 0
                return (x, y)
        except Exception as e:
            print(f"网格位置计算失败: {e}")
            return (0, 0)  # 默认位置
    
    def _predict_insertion_costs(self, features, customer, boarding_points):
        """
        使用CNN预测插入成本
        
        Returns:
            costs: 预测成本数组 [depot, home, bp1, bp2, ...]
        """
        try:
            # 构造容量特征
            capacity_features = self._construct_capacity_features(customer, boarding_points)
            
            # 检查特征维度并确保正确的形状
            if features.dim() == 3:  # [time, height, width]
                features = features.unsqueeze(0)  # 添加batch维度 -> [1, time, height, width]
            elif features.dim() == 2:  # [height, width]
                features = features.unsqueeze(0).unsqueeze(0)  # 添加batch和时间维度 -> [1, 1, height, width]
            
            # 确保特征形状正确
            if features.dim() != 4:
                print(f"特征维度不正确: {features.dim()}, 期望4维")
                return self._calculate_heuristic_costs(customer, boarding_points)
            
            # 预测成本
            with torch.no_grad():
                predicted_costs = self.cost_predictor(
                    features.to(self.device), 
                    capacity_features.to(self.device)
                )
            
            # 转换为numpy数组
            costs = predicted_costs.cpu().numpy().flatten()
            
            # 确保成本数组长度正确
            expected_length = 2 + len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            if len(costs) != expected_length:
                # 如果长度不匹配，使用启发式成本
                print(f"CNN预测成本长度不匹配: 期望{expected_length}, 实际{len(costs)}")
                return self._calculate_heuristic_costs(customer, boarding_points)
            
            # 确保成本为正数
            costs = np.maximum(costs, 0.1)
            
            return costs
            
        except Exception as e:
            print(f"CNN预测失败: {e}, 使用启发式成本")
            return self._calculate_heuristic_costs(customer, boarding_points)
    
    def _construct_capacity_features(self, customer, boarding_points):
        """构造容量特征"""
        try:
            # [depot_capacity, home_capacity, bp1_capacity, bp2_capacity, ...]
            capacities = [1000, 1000]  # depot和home容量
            
            for bp in boarding_points:
                if bp.remainingCapacity > 0:
                    capacities.append(bp.remainingCapacity)
                else:
                    capacities.append(0)
            
            # 确保容量特征长度与期望的成本数量匹配
            expected_length = 2 + len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            if len(capacities) != expected_length:
                print(f"容量特征长度不匹配: 期望{expected_length}, 实际{len(capacities)}")
                # 调整长度
                if len(capacities) < expected_length:
                    capacities.extend([0] * (expected_length - len(capacities)))
                else:
                    capacities = capacities[:expected_length]
            
            return torch.tensor(capacities, dtype=float32).unsqueeze(0)
        except Exception as e:
            print(f"容量特征构造失败: {e}")
            # 返回默认容量特征
            return torch.tensor([[1000, 1000, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]], dtype=float32)
    
    def _calculate_true_insertion_costs(self, customer, fleet, boarding_points):
        """
        计算真实插入成本
        
        Returns:
            costs: 真实成本数组 [depot, home, bp1, bp2, ...]
        """
        try:
            costs = [0.0]  # depot成本为0
            
            # 计算home delivery成本
            home_cost = self._calculate_insertion_cost(
                customer.home, fleet
            )
            costs.append(home_cost)
            
            # 计算boarding point成本
            for bp in boarding_points:
                if bp.remainingCapacity > 0:
                    bp_cost = self._calculate_insertion_cost(bp.location, fleet)
                    costs.append(bp_cost)
                else:
                    costs.append(float('inf'))
            
            # 确保成本数组长度正确
            expected_length = 2 + len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            if len(costs) != expected_length:
                print(f"真实成本计算长度不匹配: 期望{expected_length}, 实际{len(costs)}")
                # 调整长度
                if len(costs) < expected_length:
                    costs.extend([1000.0] * (expected_length - len(costs)))
                else:
                    costs = costs[:expected_length]
            
            return np.array(costs)
        except Exception as e:
            print(f"真实成本计算失败: {e}")
            # 返回默认成本
            default_length = 2 + len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            return np.array([0.0] + [100.0] * (default_length - 1))
    
    def _calculate_insertion_cost_matrix(self, location, fleet):
        """使用距离矩阵计算插入成本"""
        # 检查距离矩阵是否存在
        if not hasattr(self, 'distance_matrix') or self.distance_matrix is None:
            print("距离矩阵不存在，使用欧几里得距离")
            return self._calculate_insertion_cost_euclidean(location, fleet)
        
        min_cost = float('inf')
        
        try:
            for vehicle in fleet.fleet:
                if len(vehicle.routePlan) <= 1:
                    # 空路线，直接插入
                    if (0 <= location.id_num < len(self.distance_matrix) and 
                        0 <= location.id_num < len(self.distance_matrix[0])):
                        cost = self.distance_matrix[0][location.id_num]
                        min_cost = min(min_cost, cost)
                else:
                    # 寻找最佳插入位置
                    for i in range(1, len(vehicle.routePlan)):
                        prev_loc = vehicle.routePlan[i-1]
                        next_loc = vehicle.routePlan[i]
                        
                        # 检查索引是否有效
                        if (0 <= prev_loc.id_num < len(self.distance_matrix) and 
                            0 <= location.id_num < len(self.distance_matrix) and 
                            0 <= next_loc.id_num < len(self.distance_matrix) and
                            0 <= prev_loc.id_num < len(self.distance_matrix[0]) and 
                            0 <= location.id_num < len(self.distance_matrix[0]) and 
                            0 <= next_loc.id_num < len(self.distance_matrix[0])):
                            
                            # 计算插入成本
                            cost = (self.distance_matrix[prev_loc.id_num][location.id_num] +
                                   self.distance_matrix[location.id_num][next_loc.id_num] -
                                   self.distance_matrix[prev_loc.id_num][next_loc.id_num])
                            
                            min_cost = min(min_cost, cost)
        except Exception as e:
            print(f"距离矩阵计算失败: {e}, 使用默认成本")
            return 1000.0
        
        return min_cost if min_cost != float('inf') else 1000.0
    
    def _calculate_insertion_cost_euclidean(self, location, fleet):
        """使用欧几里得距离计算插入成本"""
        min_cost = float('inf')
        
        try:
            for vehicle in fleet.fleet:
                if len(vehicle.routePlan) <= 1:
                    # 空路线，直接插入
                    # 创建虚拟的depot位置
                    depot_location = type(location)(0, 0, 0, 0)  # 动态创建相同类型的对象
                    cost = self._euclidean_distance(depot_location, location)
                    min_cost = min(min_cost, cost)
                else:
                    # 寻找最佳插入位置
                    for i in range(1, len(vehicle.routePlan)):
                        prev_loc = vehicle.routePlan[i-1]
                        next_loc = vehicle.routePlan[i]
                        
                        # 计算插入成本
                        cost = (self._euclidean_distance(prev_loc, location) +
                               self._euclidean_distance(location, next_loc) -
                               self._euclidean_distance(prev_loc, next_loc))
                        
                        min_cost = min(min_cost, cost)
        except Exception as e:
            print(f"欧几里得距离计算失败: {e}, 使用默认成本")
            return 1000.0
        
        return min_cost if min_cost != float('inf') else 1000.0
    
    def _euclidean_distance(self, loc1, loc2):
        """计算欧几里得距离"""
        try:
            # 检查位置对象是否有x和y属性
            if hasattr(loc1, 'x') and hasattr(loc1, 'y') and hasattr(loc2, 'x') and hasattr(loc2, 'y'):
                return sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
            else:
                # 如果没有x,y属性，尝试使用其他属性或返回默认距离
                print(f"位置对象缺少x,y属性: loc1={type(loc1)}, loc2={type(loc2)}")
                # 尝试使用id_num作为距离的简单近似
                if hasattr(loc1, 'id_num') and hasattr(loc2, 'id_num'):
                    return abs(loc1.id_num - loc2.id_num) * 10.0
                return 100.0  # 默认距离
        except Exception as e:
            print(f"欧几里得距离计算失败: {e}")
            return 100.0  # 默认距离
    
    def _calculate_heuristic_costs(self, customer, boarding_points):
        """计算启发式成本（CNN失败时的回退方案）"""
        costs = [0.0]  # depot
        
        # 计算home delivery成本
        try:
            home_distance = self._euclidean_distance(customer.home, customer.home)
            costs.append(home_distance * 0.1)  # home delivery成本
        except Exception as e:
            print(f"Home delivery成本计算失败: {e}")
            costs.append(50.0)  # 默认成本
        
        # 简单的距离启发式
        for bp in boarding_points:
            if bp.remainingCapacity > 0:
                try:
                    distance = self._euclidean_distance(customer.home, bp.location)
                    costs.append(distance * 0.1)  # 缩放因子
                except Exception as e:
                    print(f"启发式成本计算失败: {e}")
                    costs.append(100.0)  # 默认成本
            else:
                costs.append(float('inf'))
        
        # 确保成本数组长度正确
        expected_length = 2 + len([bp for bp in boarding_points if bp.remainingCapacity > 0])
        if len(costs) != expected_length:
            print(f"启发式成本长度不匹配: 期望{expected_length}, 实际{len(costs)}")
            # 调整长度
            if len(costs) < expected_length:
                costs.extend([100.0] * (expected_length - len(costs)))
            else:
                costs = costs[:expected_length]
        
        return np.array(costs)
    
    def _generate_optimal_pricing(self, customer, boarding_points, costs):
        """
        基于MNL模型生成最优定价
        
        Returns:
            prices: 定价数组 [home, bp1, bp2, ...]
        """
        try:
            # 只考虑有容量的boarding points
            available_bps = [bp for bp in boarding_points if bp.remainingCapacity > 0]
            
            if not available_bps:
                return np.array([])
            
            # 计算效用函数
            utilities = []
            for bp in available_bps:
                if bp == boarding_points[0]:  # home delivery
                    util = self.base_utility + customer.home_util
                else:  # boarding point
                    util = self.base_utility - self._calculate_distance_penalty(
                        customer.home, bp.location
                    )
                utilities.append(util)
            
            # 计算MNL分母
            exp_utilities = [exp(u) for u in utilities]
            sum_exp_utilities = sum(exp_utilities)
            
            # 计算Lambert W函数
            lambert_w = (lambertw(sum_exp_utilities / 2.718281828459045).real + 1) / customer.incentiveSensitivity
            
            # 生成最优定价
            prices = []
            for i, bp in enumerate(available_bps):
                # 定价 = 成本 - 收益 - Lambert W调整
                if i+1 < len(costs):
                    price = costs[i+1] - self.revenue_per_trip - lambert_w
                else:
                    price = 100.0 - self.revenue_per_trip - lambert_w  # 默认成本
                price = max(price, self.min_price)  # 确保不低于最小价格
                prices.append(price)
            
            # 限制定价范围
            prices = np.clip(prices, self.min_price, self.max_price)
            
            return np.array(prices)
            
        except Exception as e:
            print(f"定价计算失败: {e}, 使用默认定价")
            # 返回包含home delivery的默认定价
            available_count = len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            if available_count > 0:
                return np.array([0.0] * available_count)  # 默认为0价格
            else:
                return np.array([])
    
    def _calculate_distance_penalty(self, customer_home, bp_location):
        """计算距离惩罚"""
        try:
            distance = self._euclidean_distance(customer_home, bp_location)
            return exp(-distance / 100.0)  # 距离越远，效用越低
        except Exception as e:
            print(f"距离惩罚计算失败: {e}")
            return 0.0  # 默认惩罚
    
    def _calculate_choice_probabilities(self, customer, boarding_points, prices):
        """
        计算MNL选择概率
        
        Returns:
            probabilities: 选择概率数组
        """
        try:
            utilities = []
            
            # 只考虑有容量的boarding points
            available_bps = [bp for bp in boarding_points if bp.remainingCapacity > 0]
            
            if not available_bps:
                return np.array([])
            
            for i, bp in enumerate(available_bps):
                if i < len(prices):
                    if bp == boarding_points[0]:  # home delivery
                        util = self.base_utility + customer.home_util - prices[i]
                    else:  # boarding point
                        util = self.base_utility - prices[i]
                else:
                    # 如果价格数组长度不够，使用默认价格
                    if bp == boarding_points[0]:  # home delivery
                        util = self.base_utility + customer.home_util
                    else:  # boarding point
                        util = self.base_utility
                utilities.append(util)
            
            # 计算概率
            exp_utilities = [exp(u) for u in utilities]
            sum_exp_utilities = sum(exp_utilities)
            if sum_exp_utilities > 0:
                probabilities = [eu / sum_exp_utilities for eu in exp_utilities]
            else:
                # 如果所有效用都是负无穷，使用均匀分布
                probabilities = [1.0 / len(utilities)] * len(utilities)
            
            return np.array(probabilities)
            
        except Exception as e:
            print(f"选择概率计算失败: {e}")
            # 返回均匀分布
            available_count = len([bp for bp in boarding_points if bp.remainingCapacity > 0])
            if available_count > 0:
                return np.array([1.0 / available_count] * available_count)
            else:
                return np.array([])
    
    def _calculate_expected_revenue(self, prices, probabilities):
        """计算期望收益"""
        try:
            # 确保价格和概率数组长度匹配
            min_length = min(len(prices), len(probabilities))
            if min_length == 0:
                return 0.0
            
            # 只计算有效长度的部分
            valid_prices = prices[:min_length]
            valid_probabilities = probabilities[:min_length]
            
            # 过滤掉无效价格
            valid_mask = (valid_prices != float('inf')) & (valid_prices != float('-inf'))
            if np.sum(valid_mask) == 0:
                return 0.0
            
            valid_prices = valid_prices[valid_mask]
            valid_probabilities = valid_probabilities[valid_mask]
            
            # 重新归一化概率
            if np.sum(valid_probabilities) > 0:
                valid_probabilities = valid_probabilities / np.sum(valid_probabilities)
            
            return np.sum(valid_prices * valid_probabilities)
            
        except Exception as e:
            print(f"期望收益计算失败: {e}")
            return 0.0
    
    def _calculate_pricing_error(self, predicted_prices, true_prices):
        """
        计算定价误差 (L2 loss)
        
        Returns:
            error: L2损失值
        """
        try:
            # 确保价格数组长度匹配
            min_length = min(len(predicted_prices), len(true_prices))
            if min_length == 0:
                return 0.0
            
            # 只计算有效长度的部分
            pred_prices = predicted_prices[:min_length]
            true_prices = true_prices[:min_length]
            
            # 过滤掉无效价格
            valid_mask = (pred_prices != float('inf')) & (pred_prices != float('-inf')) & \
                        (true_prices != float('inf')) & (true_prices != float('-inf'))
            
            if np.sum(valid_mask) == 0:
                return 0.0
            
            valid_pred = pred_prices[valid_mask]
            valid_true = true_prices[valid_mask]
            
            # 计算L2损失
            mse = np.mean((valid_pred - valid_true) ** 2)
            return mse
            
        except Exception as e:
            print(f"定价误差计算失败: {e}")
            return 0.0
    
    def _store_training_data(self, features, predicted_costs, true_costs):
        """存储训练数据"""
        try:
            # 确保特征是正确的形状并展平
            if features.dim() == 4:  # [batch, time, height, width]
                features_flat = features.squeeze(0).flatten().cpu().numpy()
            elif features.dim() == 3:  # [time, height, width]
                features_flat = features.flatten().cpu().numpy()
            elif features.dim() == 2:  # [height, width]
                features_flat = features.flatten().cpu().numpy()
            else:
                features_flat = features.flatten().cpu().numpy()
            
            # 存储到episode数据中
            self.current_episode_data['features'] = np.vstack([
                self.current_episode_data['features'],
                features_flat
            ])
            
            # 计算成本差异作为训练目标
            # 确保成本数组长度匹配
            min_length = min(len(predicted_costs), len(true_costs))
            if min_length > 0:
                pred_costs = predicted_costs[:min_length]
                true_costs = true_costs[:min_length]
                cost_diff = pred_costs - true_costs
            else:
                cost_diff = np.array([0.0])
            
            # 存储容量特征占位符
            self.current_episode_data['capacity_features'] = np.vstack([
                self.current_episode_data['capacity_features'],
                np.array([1.0])  # 占位符
            ])
            
            # 关键：确保特征和目标长度完全匹配
            # 如果长度不匹配，我们需要调整到相同的长度
            if len(features_flat) != len(cost_diff):
                print(f"长度不匹配: 特征{len(features_flat)}, 目标{len(cost_diff)}")
                
                # 选择较小的长度作为最终长度
                final_length = min(len(features_flat), len(cost_diff))
                
                # 截断两个数组到相同长度
                features_flat = features_flat[:final_length]
                cost_diff = cost_diff[:final_length]
                
                print(f"调整后长度: {final_length}")
            
            # 最终验证长度匹配
            if len(features_flat) != len(cost_diff):
                print(f"最终长度仍然不匹配: 特征{len(features_flat)}, 目标{len(cost_diff)}")
                # 如果仍然不匹配，使用最短长度
                min_final_length = min(len(features_flat), len(cost_diff))
                features_flat = features_flat[:min_final_length]
                cost_diff = cost_diff[:min_final_length]
            
            # 添加到记忆缓冲区
            try:
                self.memory_buffer.add(
                    features_flat,
                    np.array([1.0]),  # 容量特征占位符
                    cost_diff
                )
                print(f"成功存储训练数据: 特征{len(features_flat)}, 目标{len(cost_diff)}")
            except Exception as e:
                print(f"记忆缓冲区添加失败: {e}")
                # 如果记忆缓冲区失败，至少保存到episode数据中
                pass
            
        except Exception as e:
            print(f"训练数据存储失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_default_pricing(self, state):
        """生成默认定价（当定价功能关闭时）"""
        try:
            available_bps = [bp for bp in state[2].parcelpoints if bp.remainingCapacity > 0]
            return np.zeros(len(available_bps))
        except Exception as e:
            print(f"默认定价生成失败: {e}")
            return np.array([0.0])  # 返回默认定价
    
    def update(self, data, state, done=False):
        """
        更新函数 - 处理每个决策步骤
        
        Args:
            data: 环境数据
            state: 当前状态
            done: 是否episode结束
        """
        try:
            if not done:
                # 记录乘客到达顺序
                if len(self.current_episode_data['customer_sequence']) == 0:
                    if isinstance(data, dict) and "id" in data:
                        self.current_episode_data['customer_sequence'] = data["id"].copy()
                    else:
                        print(f"数据格式不正确: {type(data)}")
                return 0.0
            else:
                # Episode结束，计算总结果
                episode_result = self._finalize_episode(data)
                
                # 训练模型
                if hasattr(self, 'memory_buffer') and self.memory_buffer.length >= 32:  # 最小批次大小
                    self._train_model()
                
                # 重置episode数据
                self._reset_episode_data()
                
                # 增加episode计数器
                self.episode_counter += 1
                
                return episode_result.total_pricing_error
        except Exception as e:
            print(f"更新函数失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _finalize_episode(self, data):
        """完成episode并计算结果"""
        try:
            pricing_decisions = self.current_episode_data['pricing_decisions']
            
            if not pricing_decisions:
                return EpisodeResult(
                    episode_id=self.episode_counter,
                    customer_sequence=[],
                    pricing_decisions=[],
                    total_pricing_error=0.0,
                    total_expected_revenue=0.0,
                    mean_pricing_error=0.0
                )
            
            # 计算累计指标
            total_pricing_error = sum(decision.pricing_error for decision in pricing_decisions)
            total_expected_revenue = sum(decision.expected_revenue for decision in pricing_decisions)
            mean_pricing_error = total_pricing_error / len(pricing_decisions)
            
            # 创建结果对象
            episode_result = EpisodeResult(
                episode_id=self.episode_counter,
                customer_sequence=self.current_episode_data['customer_sequence'].copy(),
                pricing_decisions=pricing_decisions.copy(),
                total_pricing_error=total_pricing_error,
                total_expected_revenue=total_expected_revenue,
                mean_pricing_error=mean_pricing_error
            )
            
            # 更新性能历史
            self.pricing_error_history.append(mean_pricing_error)
            self.revenue_history.append(total_expected_revenue)
            
            return episode_result
        except Exception as e:
            print(f"Episode结果计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认结果
            return EpisodeResult(
                episode_id=self.episode_counter,
                customer_sequence=[],
                pricing_decisions=[],
                total_pricing_error=0.0,
                total_expected_revenue=0.0,
                mean_pricing_error=0.0
            )
    
    def _train_model(self):
        """训练成本预测模型"""
        try:
            # 检查记忆缓冲区是否有足够的数据
            if self.memory_buffer.length < 32:
                print(f"记忆缓冲区数据不足: {self.memory_buffer.length}/32")
                return
            
            # 采样训练数据
            features, capacity_features, targets = self.memory_buffer.sample(batch_size=32)
            
            # 检查数据有效性
            if features is None or capacity_features is None or targets is None:
                print("训练数据无效")
                return
            
            # 转换为tensor
            features_tensor = torch.tensor(features, dtype=float32).to(self.device)
            capacity_features_tensor = torch.tensor(capacity_features, dtype=float32).to(self.device)
            targets_tensor = torch.tensor(targets, dtype=float32).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.cost_predictor(features_tensor, capacity_features_tensor)
            loss = self.criterion(predictions, targets_tensor)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            print(f"Training loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_episode_data(self):
        """重置episode数据"""
        try:
            self.current_episode_data = {
                'customer_sequence': [],
                'pricing_decisions': [],
                'features': np.empty((0, self.n_time_layers * self.grid_dim * self.grid_dim)),
                'capacity_features': np.empty((0, 1))
            }
        except Exception as e:
            print(f"Episode数据重置失败: {e}")
            # 使用简单的重置
            self.current_episode_data = {
                'customer_sequence': [],
                'pricing_decisions': [],
                'features': np.empty((0, 1)),
                'capacity_features': np.empty((0, 1))
            }
    
    def get_performance_statistics(self):
        """获取性能统计信息"""
        try:
            if not self.pricing_error_history:
                return None
            
            stats = {
                'num_episodes': len(self.pricing_error_history),
                'mean_pricing_error': np.mean(self.pricing_error_history),
                'std_pricing_error': np.std(self.pricing_error_history),
                'mean_revenue': np.mean(self.revenue_history),
                'std_revenue': np.std(self.revenue_history),
                'recent_pricing_error': self.pricing_error_history[-5:] if len(self.pricing_error_history) >= 5 else [],
                'recent_revenue': self.revenue_history[-5:] if len(self.revenue_history) >= 5 else []
            }
            
            return stats
        except Exception as e:
            print(f"性能统计获取失败: {e}")
            return None
    
    def save_results(self, filepath):
        """保存结果到文件"""
        try:
            import json
            from datetime import datetime
            
            stats = self.get_performance_statistics()
            
            if stats:
                with open(filepath, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'performance_statistics': stats,
                        'pricing_error_history': self.pricing_error_history,
                        'revenue_history': self.revenue_history
                    }, f, indent=2)
                
                print(f"结果已保存到: {filepath}")
            else:
                print("没有可保存的结果")
        except Exception as e:
            print(f"结果保存失败: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self):
        """重置系统状态"""
        try:
            # 重置episode数据
            self._reset_episode_data()
            
            # 重置性能历史
            self.pricing_error_history = []
            self.revenue_history = []
            
            # 重置episode计数器
            self.episode_counter = 0
            
            # 重置记忆缓冲区
            if hasattr(self, 'memory_buffer'):
                try:
                    self.memory_buffer.reset()
                except Exception as e:
                    print(f"记忆缓冲区重置失败: {e}")
            
            # 重置模型（如果需要）
            if hasattr(self, 'cost_predictor'):
                try:
                    self.cost_predictor.train()
                except Exception as e:
                    print(f"模型重置失败: {e}")
        except Exception as e:
            print(f"系统重置失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save(self, filepath=None):
        """保存模型"""
        try:
            if filepath is None:
                filepath = f"online_pricing_system_{self.episode_counter}.pth"
            
            torch.save({
                'episode_counter': self.episode_counter,
                'cost_predictor_state_dict': self.cost_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'pricing_error_history': self.pricing_error_history,
                'revenue_history': self.revenue_history
            }, filepath)
            
            print(f"模型已保存到: {filepath}")
        except Exception as e:
            print(f"模型保存失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load(self, filepath):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.cost_predictor.load_state_dict(checkpoint['cost_predictor_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_counter = checkpoint['episode_counter']
            self.pricing_error_history = checkpoint['pricing_error_history']
            self.revenue_history = checkpoint['revenue_history']
            
            print(f"模型已从 {filepath} 加载")
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()

import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch import float32
from math import sqrt
from Src.Utils.Utils import MemoryBuffer, get_dist_mat_HGS, extract_route_HGS, get_matrix
from Src.Utils.Predictors import CNN_2d, CNN_3d, LinReg
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw
from math import exp, e
from hygese import AlgorithmParameters, Solver
from operator import itemgetter

# SPO+损失函数
class SPOPlusLoss(nn.Module):
    def __init__(self, optimization_oracle):
        super(SPOPlusLoss, self).__init__()
        self.optimization_oracle = optimization_oracle

    def forward(self, predicted_costs, true_costs):
        batch_size = predicted_costs.size(0)
        loss = 0.0

        for i in range(batch_size):
            c = true_costs[i]
            c_hat = predicted_costs[i]

            # 计算 c - 2c_hat
            c_minus_2c_hat = c - 2 * c_hat

            # 获取决策
            w_star_modified = self.optimization_oracle(c_minus_2c_hat)
            w_star_c = self.optimization_oracle(c)

            # 计算各项
            term1 = torch.dot(c_minus_2c_hat, w_star_modified)
            term2 = 2 * torch.dot(c_hat, w_star_c)
            term3 = torch.dot(c, w_star_c)

            # 单个样本的损失
            sample_loss = term1 + term2 - term3
            loss += sample_loss

        return loss / batch_size

# class SPOPlusLoss(nn.Module):
#     def __init__(self, optimization_oracle):
#         super(SPOPlusLoss, self).__init__()
#         self.optimization_oracle = optimization_oracle
#
#     def forward(self, predicted_costs, true_costs):
#         batch_size = predicted_costs.size(0)
#
#         # 问题点1: 检查张量类型和梯度要求
#         # 确保所有张量都在同一设备上且类型匹配
#         device = predicted_costs.device
#         predicted_costs = predicted_costs.to(device)
#         true_costs = true_costs.to(device)
#
#         # 初始化损失为零，但必须是可微的
#         loss = torch.tensor(0.0, device=device, requires_grad=True)
#
#         for i in range(batch_size):
#             c = true_costs[i]
#             c_hat = predicted_costs[i]
#
#             # 问题点2: 可能c和c_hat是不兼容的形状
#             # 打印诊断信息
#             print(f"Sample {i}: c shape: {c.shape}, c_hat shape: {c_hat.shape}")
#
#             # 问题点3: 可能optimization_oracle返回None或无效值
#             # 添加错误处理
#             try:
#                 # 计算 c - 2c_hat
#                 c_minus_2c_hat = c - 2 * c_hat
#
#                 # 获取决策
#                 w_star_modified = self.optimization_oracle(c_minus_2c_hat)
#                 w_star_c = self.optimization_oracle(c)
#
#                 # 问题点4: 检查优化预言机返回值
#                 if w_star_modified is None or w_star_c is None:
#                     print(f"Warning: optimization_oracle returned None for sample {i}")
#                     continue
#
#                 # 问题点5: 点积计算可能不正确
#                 # 确保向量维度匹配
#                 if c_minus_2c_hat.shape != w_star_modified.shape or c.shape != w_star_c.shape:
#                     print(f"Shape mismatch: {c_minus_2c_hat.shape} vs {w_star_modified.shape}")
#                     continue
#
#                 # 计算各项
#                 term1 = torch.dot(c_minus_2c_hat, w_star_modified)
#                 term2 = 2 * torch.dot(c_hat, w_star_c)
#                 term3 = torch.dot(c, w_star_c)
#
#                 # 单个样本的损失
#                 sample_loss = term1 + term2 - term3
#
#                 # 问题点6: 损失累加方式可能不正确
#                 # 确保不是简单赋值而是累加
#                 loss = loss + sample_loss
#             except Exception as e:
#                 print(f"Error in SPO+ calculation: {e}")
#                 continue
#
#         # 问题点7: 可能没有除以batch_size
#         if batch_size > 0:
#             loss = loss / batch_size
#
#         # 问题点8: 打印诊断信息
#         print(f"Raw SPO+ loss: {loss.item()}")
#
#         return loss

# This function implements SPO
class SPO(Agent):
    def __init__(self, config):
        super(SPO, self).__init__(config)

        self.load_data = config.load_data
        # heuristic parameters
        self.k = config.k
        self.init_theta = config.init_theta_cnn
        self.cool_theta = config.cool_theta_cnn  # linear cooling scheme

        # 添加SPO+损失函数
        self.spo_criterion = SPOPlusLoss(self.optimization_oracle)
        # 损失函数权重
        self.spo_weight = config.spo_weight if hasattr(config, 'spo_weight') else 0.25

        # problem variant: pricing or offering
        if self.config.pricing:
            self.get_action = self.get_action_pricing
            self.max_p = config.max_price
            self.min_p = config.min_price
        else:
            self.get_action = self.get_action_offer

        self.grid_dim = config.grid_dim
        self.initial_phase = True

        self.n_layers = config.n_input_layers
        self.memory = MemoryBuffer(max_len=self.config.buffer_size, time_intervals=self.n_layers,
                                   matrix_dim=self.grid_dim,
                                   target_dim=1, atype=float32, config=config)

        if config.use3d_conv:
            self.supervised_ml = CNN_3d(self.grid_dim, self.n_layers, config.n_filters, config.dropout)
        elif config.linearModel:
            self.supervised_ml = LinReg(self.grid_dim * self.grid_dim * self.n_layers)
        else:
            self.supervised_ml = CNN_2d(self.grid_dim, self.n_layers, config.n_filters, config.dropout)
        self.features = np.empty((0, self.n_layers * self.grid_dim * self.grid_dim))
        self.cap_features = np.empty((0, 1))
        self.interval = int(config.max_steps_r / config.n_input_layers)

        # self.optimizer = config.optim(self.supervised_ml.parameters(), lr=self.config.learning_rate)
        self.optimizer = torch.optim.AdamW(self.supervised_ml.parameters(), lr=self.config.learning_rate,weight_decay=1e-4)
        self.criterion = nn.HuberLoss(delta=1.0)

        # define learning modules
        self.modules = [('supervised_ml', self.supervised_ml)]
        self.init()  # write module to device
        self.device = config.device

        if self.load_data:
            self.customer_cell = get_matrix(config.coords, self.grid_dim, config.hexa)
            self.dist_matrix = config.dist_matrix
            self.service_times = config.service_times
            self.adjacency = config.adjacency
            self.first_parcelpoint_id = len(self.dist_matrix[0]) - config.n_parcelpoints - 1
            self.addedcosts = self.addedcosts_distmat
            self.dist_scaler = 1  # np.amax(self.dist_matrix)
            self.mnl = self.mnl_distmat
        else:
            self.addedcosts = self.addedcosts_euclid
            self.dist_scaler = 1  # 10
            self.mnl = self.mnl_euclid

        # mnl parameters
        self.base_util = config.base_util
        self.cost_multiplier = (config.driver_wage + config.fuel_cost) / 3600
        self.wage = config.driver_wage
        # self.added_costs_home = config.driver_wage*(config.del_time/60)
        self.revenue = config.revenue

        # hgs settings
        ap_final = AlgorithmParameters(timeLimit=config.hgs_final_time)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)  # used for final route

        # lambdas
        id_num = lambda x: x.id_num
        self.get_id = np.vectorize(id_num)

    def optimization_oracle(self, costs):
        """优化预言机，用于SPO+损失函数"""
        # 如果输入是张量，将其转换为numpy数组
        if isinstance(costs, torch.Tensor):
            costs_np = costs.detach().cpu().numpy()
        else:
            costs_np = costs

        # 创建一个合理范围的决策向量
        decision = np.zeros_like(costs_np)

        # 正规化决策，确保它们在合理范围内
        for i in range(len(costs_np)):
            # 基于成本计算决策，但确保在合理范围内
            # 这里可以根据你的具体问题调整
            cost_value = float(costs_np[i])

            # 限制在合理范围内，防止极端值
            cost_clamped = np.clip(cost_value, -10.0, 10.0)

            # 简单的线性映射，可以根据需要调整
            decision[i] = 0.1 * cost_clamped

        # 将决策转回张量（如果输入是张量）
        if isinstance(costs, torch.Tensor):
            return torch.tensor(decision, device=costs.device, dtype=costs.dtype)
        return decision


    def get_action_offer(self, state, training):
        if self.initial_phase:
            return self.get_action_offerall(state, training)
        else:

            theta = self.init_theta - (state[3] * self.cool_theta)
            mltplr = self.cost_multiplier

            # cheapest insertion costs of every PP in current and historic routes
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"],
                                       mask=self.adjacency[state[0].id_num])  # only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            pp_costs = np.full(len(pps), 1000000000.0)

            # ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat, state[0].home, pps)

            for pp in range(len(pps)):
                if state[2]["parcelpoints"][
                    pp].remainingCapacity > 0:  # check if parcelpont has remaining capacity
                    pp_costs[pp] = mltplr * (
                                (1 - theta) * self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location,
                                                                          state[1]) + theta * (
                                            costs[pp + 2] - costs[0]))
            pp_sorted_args = state[2]["parcelpoints"][np.argpartition(pp_costs, self.k)[:self.k]]

            # get k best PPs
            action = self.get_id(pp_sorted_args)
        return action

    def get_action_pricing(self, state, training):
        if self.initial_phase:
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"],
                                       mask=self.adjacency[state[0].id_num])  # only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            a_hat = np.zeros(len(pps) + 1)
            return np.around(a_hat, decimals=2)
        else:

            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"],
                                       mask=self.adjacency[state[0].id_num])  # only offer 20 closest
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            pp_costs = np.full((len(pps), 1), 1000000000.0)

            # ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat, state[0].home, pps)

            # 1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
            theta = self.init_theta - (state[3] * self.cool_theta)
            mltplr = self.cost_multiplier

            homeCosts = state[0].service_time * mltplr + (
                        (1 - theta) * (self.cheapestInsertionCosts(state[0].home, state[1])) + theta * (
                            costs[1] - costs[0]))
            sum_mnl = exp(
                self.base_util + state[0].home_util + (state[0].incentiveSensitivity * (homeCosts - self.revenue)))

            # Slight change compared to paper, to support faster training, without relying on the CVRP solver every time,
            # See bottom of this file for details.
            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = self.mnl(state[0], pp)
                    pp_costs[idx] = mltplr * (
                                (1 - theta) * (self.cheapestInsertionCosts(pp.location, state[1])) + theta * (
                                    costs[idx + 2] - costs[0]))
                    sum_mnl += exp(util + (state[0].incentiveSensitivity * (pp_costs[idx] - self.revenue)))

            # 2 obtain lambert w0
            lambertw0 = (lambertw(sum_mnl / e).real + 1) / state[0].incentiveSensitivity

            # 3 calculate discounts/prices
            a_hat = np.zeros(len(pps) + 1)
            a_hat[0] = homeCosts - self.revenue - lambertw0
            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    a_hat[idx + 1] = pp_costs[idx] - self.revenue - lambertw0

            a_hat = np.clip(a_hat, self.min_p, self.max_p)
            return np.around(a_hat, decimals=2)

    def get_action_offerall(self, state, training):
        # check if pp is feasible
        if self.load_data:
            mask = ma.masked_array(state[2]["parcelpoints"],
                                   mask=self.adjacency[state[0].id_num])  # only offer 20 closest
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]
        action = np.empty(0, dtype=int)
        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                action = np.append(action, pp.id_num)
        return action

    def addedcosts_euclid(self, route, i, loc):
        costs = self.getdistance_euclidean(route[i - 1], loc) + self.getdistance_euclidean(loc, route[i]) \
                - self.getdistance_euclidean(route[i - 1], route[i])
        return costs / self.dist_scaler

    def addedcosts_distmat(self, route, i, loc):
        costs = self.dist_matrix[route[i - 1].id_num][loc.id_num] + self.dist_matrix[loc.id_num][route[i].id_num] \
                - self.dist_matrix[route[i - 1].id_num][route[i].id_num]
        return costs / self.dist_scaler

    def cheapestInsertionCosts(self, loc, fleet):
        cheapestCosts = float("inf")
        for v in fleet["fleet"]:  # note we do not check feasibility of insertion here, let this to HGS
            for i in range(1, len(v["routePlan"])):
                addedCosts = self.addedcosts(v["routePlan"], i, loc)
                if addedCosts < cheapestCosts:
                    cheapestCosts = addedCosts

        return cheapestCosts

    def getdistance_euclidean(self, a, b):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def get_prediction(self, cur_feat, home, pps):
        time_int = min(int(home.time / self.interval), self.n_layers - 1)
        new_feat = torch.cat((2 + len(pps)) * [cur_feat])
        new_feat[1][time_int][self.customer_cell[home.id_num][0]][self.customer_cell[home.id_num][1]] += 1
        cap_feat = torch.zeros((2 + len(pps)))
        cap_feat[0] = 1000000
        cap_feat[1] = 1000000
        for idx, p in enumerate(pps):
            new_feat[idx + 2][time_int][self.customer_cell[p.location.id_num][0]][
                self.customer_cell[p.location.id_num][1]] += 1
            cap_feat[idx + 2] = p.remainingCapacity - 1
        costs = []
        for i, feat in enumerate(new_feat):
            costs.append(
                self.supervised_ml(feat.unsqueeze(0).to(self.device), cap_feat[i].unsqueeze(0).to(self.device)).item())
        return costs

    def mnl_euclid(self, customer, parcelpoint):
        distance = self.getdistance_euclidean(customer.home, parcelpoint.location)  # distance from parcelpoint to home
        beta_p = -exp(-distance / self.dist_scaler)
        return self.base_util + beta_p

    def mnl_distmat(self, customer, parcelpoint):
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]  # distance from parcelpoint to home
        beta_p = -exp(-distance / self.dist_scaler)
        return self.base_util + beta_p

    def update(self, data, state, done=False):
        # first obtain data
        if not done:
            self.features = np.vstack((self.features, self.get_feature_rep(data).flatten()))
            try:
                self.cap_features = np.vstack(
                    (self.cap_features, state[2]["parcelpoints"][data["id"][-1]].remainingCapacity))
            except:
                self.cap_features = np.vstack((self.cap_features, 1000000))  # home delivery
            return 0.0
        else:
            # obtain final CVRP schedule after end of booking horizon
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix, data['id'])
            fleet, cost = self.reopt_HGS_final(data)  # do a final reopt

            target = self.get_per_customer_costs(fleet)
            target = sorted(target, key=itemgetter(0))  # sort in order of arrival (same as features)
            penalties = 20 / (self.cap_features + 0.1)
            adjusted_target = [t + p for t, p in zip(target, penalties)]
            self.memory.add(self.features, self.cap_features, adjusted_target)

            self.features = np.empty((0, self.n_layers * self.grid_dim * self.grid_dim))
            self.cap_features = np.empty((0, 1))
            # optionally update model
            if self.initial_phase:  # train model initial phase
                if self.memory.length >= self.config.buffer_size:
                    self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
            elif not self.config.only_phase_one:
                # simply update CNN after every new data point collected
                self.optimize()

            return cost

    def optimize(self):
        # 采样一批数据
        feat, cap_feat, target = self.memory.sample(batch_size=self.config.batch_size)

        # 更新模型
        combined_loss = self.self_supervised_update(feat, cap_feat, target)

        # 计算单独的损失值用于显示
        with torch.no_grad():
            outputs = self.supervised_ml(feat, cap_feat)
            huber_loss = self.criterion(outputs, target).item()
            spo_loss = self.spo_criterion(outputs, target).item()

        print(f"Combined loss: {combined_loss:.4f}, Huber loss: {huber_loss:.4f}, SPO+ loss: {spo_loss:.4f}")
        return combined_loss


    def self_supervised_update(self, feat, cap_feat, target):
        # ...
        outputs = self.supervised_ml(feat, cap_feat)
        huber_loss = self.criterion(outputs, target)
        spo_plus_loss = self.spo_criterion(outputs, target)

        # 动态SPO+缩放
        current_epoch = getattr(self, 'current_epoch', 0)
        self.current_epoch = current_epoch + 1

        # 随训练进展增加SPO+影响
        if self.current_epoch < 100:
            # 初期阶段 - 低缩放因子
            spo_scale = 0.06
            spo_weight = 0.2
        elif self.current_epoch < 300:
            # 中期阶段 - 逐渐增加
            progress = (self.current_epoch - 100) / 200
            spo_scale = 0.06 + 0.06 * progress  # 从0.06增加到0.12
            spo_weight = 0.2 + 0.2 * progress  # 从0.2增加到0.4
        else:
            # 后期阶段 - 较高权重
            spo_scale = 0.12
            spo_weight = 0.4

        # 记录使用的参数
        if self.current_epoch % 10 == 0:
            print(f"Epoch {self.current_epoch}: weight={spo_weight:.3f}, scale={spo_scale:.3f}")

        combined_loss = (1 - spo_weight) * huber_loss + spo_weight * spo_scale * spo_plus_loss
        combined_loss.backward()
        self.optimizer.step()
        # return combined_loss.item(), huber_loss.item(), spo_plus_loss.item()
        return combined_loss.item()

    # def self_supervised_update2(self, feat, cap_feat, target):
    #     # ...
    #     outputs = self.supervised_ml(feat, cap_feat)
    #     huber_loss = self.criterion(outputs, target)
    #
    #     # 记录原始损失值（用于日志）
    #     raw_spo_plus_loss = self.spo_criterion(outputs, target)
    #
    #     # 应用额外的缩放来控制SPO+损失数值
    #     # 动态调整缩放因子以保持损失在合理范围
    #     if raw_spo_plus_loss > 1000:
    #         extra_scale = 0.001  # 对高SPO+损失应用更强缩放
    #     else:
    #         extra_scale = 0.01  # 对正常SPO+损失应用适度缩放
    #
    #     # 动态权重和缩放（保留现有逻辑）
    #     current_epoch = getattr(self, 'current_epoch', 0)
    #     self.current_epoch = current_epoch + 1
    #
    #     if self.current_epoch < 100:
    #         spo_scale = 0.06 * extra_scale
    #         spo_weight = 0.2
    #     elif self.current_epoch < 300:
    #         progress = (self.current_epoch - 100) / 200
    #         spo_scale = (0.06 + 0.06 * progress) * extra_scale
    #         spo_weight = 0.2 + 0.2 * progress
    #     else:
    #         spo_scale = 0.12 * extra_scale
    #         spo_weight = 0.4
    #
    #     # 记录使用的参数
    #     if self.current_epoch % 10 == 0:
    #         print(
    #             f"Epoch {self.current_epoch}: weight={spo_weight:.3f}, scale={spo_scale:.6f}, raw_spo={raw_spo_plus_loss:.1f}")
    #
    #     # 计算组合损失
    #     scaled_spo_loss = spo_scale * raw_spo_plus_loss
    #     combined_loss = (1 - spo_weight) * huber_loss + spo_weight * scaled_spo_loss
    #
    #     combined_loss.backward()
    #
    #     # 添加梯度裁剪以增加稳定性
    #     torch.nn.utils.clip_grad_norm_(self.supervised_ml.parameters(), max_norm=4.0)
    #
    #     self.optimizer.step()
    #
    #     return combined_loss.item()


    def initial_phase_training(self, max_epochs=-1):
        initial_losses = []
        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for feat, cap_feat, target in self.memory.batch_sample(batch_size=self.config.batch_size, randomize=True):
                loss = self.self_supervised_update(feat, cap_feat, target)
                losses.append(loss)
            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} Huber loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.memory.save(self.config.paths['checkpoint'] + 'initial_')
                    self.save()
                    print("Saved..")
            # Terminate initial phase once it have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break
        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.memory.save(self.config.paths['checkpoint'] + 'initial_')
        self.save()

    def get_feature_rep(self, data):
        feature = np.zeros((self.n_layers, self.grid_dim, self.grid_dim))
        for i, t in zip(data["id"], data["time"]):
            time_int = min(int(t / self.interval), self.n_layers - 1)
            feature[time_int][self.customer_cell[i][0]][
                self.customer_cell[i][1]] += 1  # actual choice of the customer during simulation
        return feature

    def get_feature_rep_infer(self, fleet):
        feature = np.zeros((self.n_layers, self.grid_dim, self.grid_dim))
        for v in fleet:
            for i in v["routePlan"]:
                time_int = min(int(i.time / self.interval), self.n_layers - 1)
                feature[time_int][self.customer_cell[i.id_num][0]][self.customer_cell[i.id_num][1]] += 1
        return torch.tensor(feature, dtype=float32, requires_grad=False).unsqueeze(0)

    # NOTE: this implementation slightly differs from the one in the paper, reason for this is that this implementation is way more efficient, so we believe it will be more friendly for public use
    # For our experiments, we did not use this exact method, so there may be some performance loss compared to rsults in the paper using this calculation method.
    # We clal this method "Half-edge partitioning (HEP)", see the paper Dynamic Time Slot Pricing Using Delivery Costs Approximations by Akkerman et al. (2022)
    def reopt_HGS_final(self, data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0  # depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)
        # update current routes
        fleet = extract_route_HGS(result, data)
        return fleet, result.cost

    def get_per_customer_costs(self, fleet):
        mltplr = self.cost_multiplier
        # addedcosts_home = self.added_costs_home
        costs = []
        for v in fleet["fleet"]:
            if len(v["routePlan"]) == 1:  # when only 1 customer is visited
                costs.append([v["routePlan"][0].time, mltplr * (self.dist_matrix[0][v["routePlan"][0].id_num])])
            else:
                for i in range(0, len(v["routePlan"])):
                    # costs is composed of distance*mltplr
                    if i == 0:
                        costs.append([v["routePlan"][i].time, mltplr * (
                                    0.5 * self.dist_matrix[0][v["routePlan"][i].id_num] + 0.5 *
                                    self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i + 1].id_num])])
                    elif i == len(v["routePlan"]) - 1:
                        costs.append([v["routePlan"][i].time, mltplr * (
                                    0.5 * self.dist_matrix[v["routePlan"][i - 1].id_num][
                                v["routePlan"][i].id_num] + 0.5 * self.dist_matrix[v["routePlan"][i].id_num][0])])
                    else:
                        costs.append([v["routePlan"][i].time, mltplr * (
                                    0.5 * self.dist_matrix[v["routePlan"][i - 1].id_num][
                                v["routePlan"][i].id_num] + 0.5 * self.dist_matrix[v["routePlan"][i].id_num][
                                        v["routePlan"][i + 1].id_num])])

                    # if v["routePlan"][i].id_num < self.first_parcelpoint_id:#customer chose home delivery
                    #     costs[-1][0] += mltplr*self.service_times[v["routePlan"][i].id_num]

        return costs


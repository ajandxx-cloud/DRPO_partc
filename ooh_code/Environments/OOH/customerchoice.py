from math import exp
from numpy.random import gumbel
import numpy as np
import sys
from Src.Utils.passenger_utility import (
    utility_home_nonprice,
    utility_meeting_point_nonprice,
)

class customerchoicemodel(object):
    def __init__(self,
                 base_util,
                 dist_scaler,
                 euclidean,
                 dist_mat,
                 n_cust,
                 quit_threshold=None,  # 保留向后兼容
                 outside_option_util=None,  # 外部选项的效用值u0（None表示不启用外部选项）
                 travel_time_weight=None,  # 在途时间的权重系数（None表示不启用在途时间效用）
                 choice_util_matrix=None,
                 walking_distance_matrix=None,
                 walk_distance_weight=None):
        self.euclidean_distance = euclidean
        self.dist_scaler = dist_scaler
        self.base_util = base_util
        self.dist_mat = dist_mat
        self.n_cust = n_cust
        self.quit_threshold = quit_threshold  # 保留向后兼容
        self.outside_option_util = outside_option_util  # u0: 外部选项的吸引力
        self.travel_time_weight = travel_time_weight  # 在途时间的权重系数（负值，因为时间越长效用越低）
        self.choice_util_matrix = choice_util_matrix
        self.walking_distance_matrix = walking_distance_matrix
        self.walk_distance_weight = walk_distance_weight
        
        # 调试输出：验证参数传递
        self._verbose_debug = bool(getattr(sys, "_debug_utils", False))
        if self._verbose_debug:
            print(f"[DEBUG customerchoicemodel.__init__] outside_option_util = {self.outside_option_util}, quit_threshold = {self.quit_threshold}")
        if len(self.dist_mat)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid

    def reset_debug_counters(self):
        """
        Reset debug counters so that printed "[DEBUG ... 选择统计]" does not accumulate across
        training + evaluation runs (which can be confusing when comparing algorithms).
        """
        # pricing debug
        if hasattr(self, "_debug_choice_stats"):
            self._debug_choice_stats = {'outside': 0, 'home': 0, 'ooh': 0}
        if hasattr(self, "_debug_noise_count_pricing"):
            self._debug_noise_count_pricing = 0
        if hasattr(self, "_debug_count_pricing"):
            self._debug_count_pricing = 0
        # offer debug (if used)
        if hasattr(self, "_debug_utils_collector_offer"):
            self._debug_utils_collector_offer = []
        if hasattr(self, "_debug_utils_collector"):
            self._debug_utils_collector = []
        
    def mnl_euclid(self,customer,parcelpoint,travel_time=None):
        """
        multi-nomial logit model calculating euclidean distance
        travel_time: 车辆在途时间（秒），如果提供则添加到效用中
        """
        distance = self._walking_distance(customer, parcelpoint)
        beta_walk = self.walk_distance_weight
        if beta_walk is None:
            beta_walk = -1.0 / max(self.dist_scaler, 1e-8)
        util = utility_meeting_point_nonprice(
            self.base_util,
            beta_walk,
            self.travel_time_weight,
            distance,
            None,
        )
        # 如果启用了在途时间效用，则添加在途时间项
        if self.travel_time_weight is not None and travel_time is not None:
            # 在途时间越长，效用越低（travel_time_weight应该是负值）
            util = utility_meeting_point_nonprice(
                self.base_util,
                beta_walk,
                self.travel_time_weight,
                distance,
                travel_time,
            )
        return util

    def mnl_distmat(self,customer,parcelpoint,travel_time=None):
        """
        multi-nomial logit model using distance matrix
        travel_time: 车辆在途时间（秒），如果提供则添加到效用中
        """
        util = self._choice_sidecar_util(customer, parcelpoint)
        if util is None:
            distance = self._walking_distance(customer, parcelpoint)
            beta_walk = self.walk_distance_weight
            if beta_walk is None:
                beta_walk = -1.0 / max(self.dist_scaler, 1e-8)
            util = utility_meeting_point_nonprice(
                self.base_util,
                beta_walk,
                self.travel_time_weight,
                distance,
                None,
            )
        # 如果启用了在途时间效用，则添加在途时间项
        if self.travel_time_weight is not None and travel_time is not None:
            # 在途时间越长，效用越低（travel_time_weight应该是负值）
            if util is None:
                distance = self._walking_distance(customer, parcelpoint)
                beta_walk = self.walk_distance_weight
                if beta_walk is None:
                    beta_walk = -1.0 / max(self.dist_scaler, 1e-8)
                util = utility_meeting_point_nonprice(
                    self.base_util,
                    beta_walk,
                    self.travel_time_weight,
                    distance,
                    travel_time,
                )
            else:
                util += self.travel_time_weight * travel_time
        return util

    def _choice_sidecar_util(self, customer, parcelpoint):
        matrix = self.choice_util_matrix
        if matrix is None:
            return None
        ci = int(customer.id_num)
        pi = int(parcelpoint.id_num)
        if ci < 0 or pi < 0 or ci >= matrix.shape[0] or pi >= matrix.shape[1]:
            return None
        value = float(matrix[ci][pi])
        if not np.isfinite(value):
            return None
        return self.base_util + value

    def _walking_distance(self, customer, parcelpoint):
        matrix = self.walking_distance_matrix
        if matrix is not None:
            ci = int(customer.id_num)
            pi = int(parcelpoint.id_num)
            if 0 <= ci < matrix.shape[0] and 0 <= pi < matrix.shape[1]:
                value = float(matrix[ci][pi])
                if np.isfinite(value):
                    return value
        if len(self.dist_mat) > 0:
            return float(self.dist_mat[customer.id_num][parcelpoint.id_num])
        return float(self.euclidean_distance(customer.home, parcelpoint.location))
    
    def customerchoice_offer(self,customer,action,parcelpoints,travel_times=None):
        """
        Customer choice model for offering decision with outside option (k=0) in MNL framework
        外部选项作为MNL模型中的一个选项，与其他DRT选项一起参与选择
        
        travel_times: 字典，包含每个选项的车辆在途时间（秒）
            - 'home': HOME点的在途时间
            - 'ooh': 列表，对应每个OOH点的在途时间，顺序与pps一致
            如果为None，则不使用在途时间效用
        """
        pps = parcelpoints[action-self.n_cust]
        
        # 如果启用了外部选项机制，将其作为选项k=0
        if self.outside_option_util is not None:
            # 形状: (len(action)+2, 1)
            # [0] = 外部选项(k=0), [1] = home delivery, [2:] = OOH points
            shape = (len(action)+2, 1)
            utils = np.empty(shape)
            
            # k=0: 外部选项
            utils[0] = self.outside_option_util
            
            # k=1: Home delivery
            utils[1] = utility_home_nonprice(
                self.base_util,
                customer.home_util,
                None,
                None,
            )
            # 如果提供了在途时间，添加到HOME delivery的效用中
            if travel_times is not None and 'home' in travel_times:
                if self.travel_time_weight is not None:
                    utils[1] += self.travel_time_weight * travel_times['home']
            
            # k=2,3,...: OOH points
            for idx, pp in enumerate(pps):
                travel_time = None
                if travel_times is not None and 'ooh' in travel_times and idx < len(travel_times['ooh']):
                    travel_time = travel_times['ooh'][idx]
                utils[idx+2] = self.mnl(customer, pp, travel_time=travel_time)
            
            # 添加Gumbel噪声
            # 使用较小的噪声标准差（0.3），与customerchoice_pricing保持一致
            gumbel_scale = 0.3  # 从1.0降低到0.3，减少噪声影响
            utils = np.add(utils, gumbel(0, gumbel_scale, np.shape(utils)))
            
            # 选择
            idx = np.argmax(utils)
            
            if idx == 0:
                return None, False, -2, 0  # 退出
            elif idx == 1:
                return customer.home, False, -1, 0  # home delivery
            else:
                pp_idx = idx - 2
                return pps[pp_idx].location, True, pps[pp_idx].id_num, 0  # OOH point
        
        # 原有的quit_threshold机制（向后兼容）
        shape = (len(action)+1, 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        # 如果提供了在途时间，添加到HOME delivery的效用中
        if travel_times is not None and 'home' in travel_times:
            if self.travel_time_weight is not None:
                utils[0] += self.travel_time_weight * travel_times['home']
        for idx,pp in enumerate(pps):
            travel_time = None
            if travel_times is not None and 'ooh' in travel_times and idx < len(travel_times['ooh']):
                travel_time = travel_times['ooh'][idx]
            utils[idx+1] = self.mnl(customer,pp,travel_time=travel_time)
        
        # 检查退出条件：如果启用退出功能，且所有选项的效用都低于阈值，则退出
        if self.quit_threshold is not None:
            max_util = np.max(utils)
            if max_util < self.quit_threshold:
                return None, False, -2, 0
        
        # 使用较小的噪声标准差（0.3），与customerchoice_pricing保持一致
        gumbel_scale = 0.3  # 从1.0降低到0.3，减少噪声影响
        utils = np.add(utils,gumbel(0, gumbel_scale, np.shape(utils)))
        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, 0#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,0#accept offer
    
    def _print_utility_statistics_offer(self):
        """打印offer方法的效用值统计信息"""
        if not hasattr(self, '_debug_utils_collector_offer') or len(self._debug_utils_collector_offer) == 0:
            return
        
        import sys
        if not (hasattr(sys, '_debug_utils') and sys._debug_utils):
            return
        
        max_utils = [d['max_util'] for d in self._debug_utils_collector_offer]
        min_utils = [d['min_util'] for d in self._debug_utils_collector_offer]
        home_utils = [d['home_util'] for d in self._debug_utils_collector_offer]
        quit_count = sum(1 for d in self._debug_utils_collector_offer if d['will_quit'])
        
        print(f"\n  [Utility Statistics Offer] (前{len(self._debug_utils_collector_offer)}个顾客):")
        print(f"    最大效用: 平均={np.mean(max_utils):.3f}, 最小={np.min(max_utils):.3f}, "
              f"最大={np.max(max_utils):.3f}, 中位数={np.median(max_utils):.3f}")
        print(f"    最小效用: 平均={np.mean(min_utils):.3f}, 最小={np.min(min_utils):.3f}, "
              f"最大={np.max(min_utils):.3f}, 中位数={np.median(min_utils):.3f}")
        print(f"    Home delivery效用: 平均={np.mean(home_utils):.3f}, 最小={np.min(home_utils):.3f}, "
              f"最大={np.max(home_utils):.3f}, 中位数={np.median(home_utils):.3f}")
        print(f"    退出顾客数: {quit_count}/{len(self._debug_utils_collector_offer)} ({100*quit_count/len(self._debug_utils_collector_offer):.1f}%)")
        print(f"    建议阈值范围: [{np.min(max_utils):.2f}, {np.max(max_utils):.2f}] "
              f"(当前阈值={self.quit_threshold:.2f})")

    def customerchoice_pricing(self,customer,action,parcelpoints,travel_times=None):
        """
        Customer choice model for the pricing decision with outside option (k=0) in MNL framework
        外部选项作为MNL模型中的一个选项，与其他DRT选项一起参与选择
        
        travel_times: 字典，包含每个选项的车辆在途时间（秒）
            - 'home': HOME点的在途时间
            - 'ooh': 列表，对应每个OOH点的在途时间，顺序与pps一致
            如果为None，则不使用在途时间效用
        """
        # 防御式：确保 action 是一维向量，避免上层算法返回 (1,N)/(N,1) 导致广播错误
        action = np.asarray(action).reshape(-1)
        pps = parcelpoints[parcelpoints.mask].data
        
        # 如果启用了外部选项机制，将其作为选项k=0
        if self.outside_option_util is not None:
            # 调试输出：验证是否进入外部选项分支
            if self._verbose_debug and not hasattr(self, '_debug_outside_option_check_pricing'):
                self._debug_outside_option_check_pricing = True
                print(f"[DEBUG customerchoice_pricing] 使用外部选项机制，outside_option_util = {self.outside_option_util}")
            
            # 形状: (len(pps)+2, 1)
            # [0] = 外部选项(k=0), [1] = home delivery, [2:] = OOH points
            shape = (len(pps)+2, 1)
            utils = np.empty(shape)
            
            # k=0: 外部选项（退出DRT系统，选择其他出行方式）
            utils[0] = self.outside_option_util  # u0: 外部选项的固定效用
            
            # k=1: Home delivery（基础效用）
            utils[1] = utility_home_nonprice(
                self.base_util,
                customer.home_util,
                None,
                None,
            )
            # 如果提供了在途时间，添加到HOME delivery的效用中
            if travel_times is not None and 'home' in travel_times:
                if self.travel_time_weight is not None:
                    utils[1] += self.travel_time_weight * travel_times['home']
            
            # k=2,3,...: OOH points（基础效用）
            for idx, pp in enumerate(pps):
                travel_time = None
                if travel_times is not None and 'ooh' in travel_times and idx < len(travel_times['ooh']):
                    travel_time = travel_times['ooh'][idx]
                utils[idx+2] = self.mnl(customer, pp, travel_time=travel_time)
            
            # 添加价格影响（只对DRT选项，不包括外部选项）
            # action[0] = home price, action[1:] = OOH prices
            utils[1] += customer.incentiveSensitivity * action[0]  # home delivery
            for idx in range(len(pps)):
                if idx+1 < len(action):
                    utils[idx+2] += customer.incentiveSensitivity * action[idx+1]  # OOH points

            # Ensure 1D numeric arrays (NumPy>=2.0 no longer allows float(array([x]))).
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            utils = np.asarray(utils, dtype=np.float64).reshape(-1)
            
            # 调试输出：查看效用值和选择概率（只打印前几次，帮助诊断问题）
            if not hasattr(self, '_debug_count_pricing'):
                self._debug_count_pricing = 0
            
            if self._verbose_debug and self._debug_count_pricing < 10:
                print(f"\n[DEBUG customerchoice_pricing] 客户选择时的效用值（加噪声前）:")
                # 将numpy数组转换为标量
                outside_util = float(utils[0])
                home_util_val = float(utils[1])
                print(f"  外部选项: {outside_util:.3f}")
                print(f"  Home delivery: {home_util_val:.3f} (价格: {action[0]:.2f}, base_util+home_util: {self.base_util + customer.home_util:.3f}, 价格影响: {customer.incentiveSensitivity * action[0]:.3f})")
                for idx in range(len(pps)):
                    if idx+2 < len(utils):
                        price = action[idx+1] if idx+1 < len(action) else 0
                        ooh_util_val = float(utils[idx+2])
                        base_util_ooh = ooh_util_val - customer.incentiveSensitivity * price
                        print(f"  OOH点{idx}: {ooh_util_val:.3f} (价格: {price:.2f}, 基础效用: {base_util_ooh:.3f}, 价格影响: {customer.incentiveSensitivity * price:.3f})")
                
                # 计算理论退出概率（MNL模型，无噪声）
                exp_utils = np.exp(utils)
                exit_prob = float(exp_utils[0]) / float(exp_utils.sum())
                drt_utils = [float(utils[1])] + [float(utils[i]) for i in range(2, len(utils))]
                min_drt_util = min(drt_utils)
                max_drt_util = max(drt_utils)
                print(f"  理论退出概率（无噪声）: {exit_prob*100:.2f}%")
                print(f"  DRT选项效用范围: [{min_drt_util:.3f}, {max_drt_util:.3f}]")
                print(f"  效用差距: min(DRT)-外部={min_drt_util-outside_util:.3f}, max(DRT)-外部={max_drt_util-outside_util:.3f}")
                self._debug_count_pricing += 1
            
            # 添加Gumbel噪声（所有选项，包括外部选项）
            # 使用较小的噪声标准差（0.3），减少随机性，使选择更接近理论概率
            # 这样可以实现"外部竞争由低到高，退出率由低到高"的效果
            gumbel_scale = 0.3  # 从1.0降低到0.3，减少噪声影响
            gumbel_noise = gumbel(0, gumbel_scale, np.shape(utils))
            utils_with_noise = np.add(utils, gumbel_noise)
            
            # 调试输出：显示加上噪声后的实际效用值（只打印前几次）
            if not hasattr(self, '_debug_noise_count_pricing'):
                self._debug_noise_count_pricing = 0
                self._debug_choice_stats = {'outside': 0, 'home': 0, 'ooh': 0}
            
            if self._verbose_debug and self._debug_noise_count_pricing < 10:
                print(f"\n[DEBUG customerchoice_pricing] 加上噪声后的效用值:")
                print(f"  外部选项: {float(utils_with_noise[0]):.3f} (原始: {float(utils[0]):.3f}, 噪声: {float(gumbel_noise[0]):.3f})")
                print(f"  Home delivery: {float(utils_with_noise[1]):.3f} (原始: {float(utils[1]):.3f}, 噪声: {float(gumbel_noise[1]):.3f})")
                for i in range(min(3, len(pps))):
                    if i+2 < len(utils_with_noise):
                        print(f"  OOH点{i}: {float(utils_with_noise[i+2]):.3f} (原始: {float(utils[i+2]):.3f}, 噪声: {float(gumbel_noise[i+2]):.3f})")
                self._debug_noise_count_pricing += 1
            
            # 根据MNL模型选择（argmax of utils with Gumbel noise）
            # 注意：utils_with_noise 是形状为 (len(pps)+2, 1) 的2D数组
            # 我们需要将其展平或使用 axis=None 来获取全局最大值索引
            utils_flat = utils_with_noise.flatten()  # 展平为1D数组
            idx = np.argmax(utils_flat)
            
            # 统计选择结果
            if self._verbose_debug:
                if idx == 0:
                    self._debug_choice_stats['outside'] += 1
                elif idx == 1:
                    self._debug_choice_stats['home'] += 1
                else:
                    self._debug_choice_stats['ooh'] += 1
            
            # 每100次选择后打印统计信息
            total_choices = sum(self._debug_choice_stats.values()) if self._verbose_debug else 0
            if self._verbose_debug and total_choices > 0 and total_choices % 100 == 0:
                print(f"\n[DEBUG customerchoice_pricing] 选择统计（最近{total_choices}次）:")
                print(f"  外部选项: {self._debug_choice_stats['outside']} ({self._debug_choice_stats['outside']/total_choices*100:.1f}%)")
                print(f"  Home delivery: {self._debug_choice_stats['home']} ({self._debug_choice_stats['home']/total_choices*100:.1f}%)")
                print(f"  OOH点: {self._debug_choice_stats['ooh']} ({self._debug_choice_stats['ooh']/total_choices*100:.1f}%)")
            
            if idx == 0:
                # 选择外部选项（退出DRT系统）
                return None, False, -2, 0
            elif idx == 1:
                # 选择home delivery
                return customer.home, False, -1, action[0]
            else:
                # 选择OOH point
                pp_idx = idx - 2
                return pps[pp_idx].location, True, pps[pp_idx].id_num, action[pp_idx] if pp_idx < len(action)-1 else 0
        
        # 原有的quit_threshold机制（向后兼容）
        # 调试输出：如果进入这个分支，说明outside_option_util是None
        if self._verbose_debug and not hasattr(self, '_debug_quit_threshold_check'):
            self._debug_quit_threshold_check = True
            print(f"[DEBUG customerchoice_pricing] 使用quit_threshold机制，outside_option_util = {self.outside_option_util}, quit_threshold = {self.quit_threshold}")
        
        shape = (len(pps)+1, 1)
        utils= np.empty(shape)
        utils[0] = utility_home_nonprice(
            self.base_util,
            customer.home_util,
            None,
            None,
        )
        # 如果提供了在途时间，添加到HOME delivery的效用中
        if travel_times is not None and 'home' in travel_times:
            if self.travel_time_weight is not None:
                utils[0] += self.travel_time_weight * travel_times['home']
        for idx,pp in enumerate(pps):
            travel_time = None
            if travel_times is not None and 'ooh' in travel_times and idx < len(travel_times['ooh']):
                travel_time = travel_times['ooh'][idx]
            utils[idx+1] = self.mnl(customer,pp,travel_time=travel_time)
        
        utils = np.add(utils,customer.incentiveSensitivity*action.reshape((len(action),1)))#incentive
        
        # 检查退出条件：如果启用退出功能，且所有选项的效用都低于阈值，则退出
        if self.quit_threshold is not None:
            max_util = np.max(utils)
            if max_util < self.quit_threshold:
                return None, False, -2, 0
        
        # 使用较小的噪声标准差（0.3），与customerchoice_pricing保持一致
        gumbel_scale = 0.3  # 从1.0降低到0.3，减少噪声影响
        utils = np.add(utils,gumbel(0, gumbel_scale, np.shape(utils)))
        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, action[0]#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,action[idx-1]#accept offer
    
    def _print_utility_statistics(self):
        """打印效用值统计信息"""
        if not hasattr(self, '_debug_utils_collector') or len(self._debug_utils_collector) == 0:
            return
        
        import sys
        if not (hasattr(sys, '_debug_utils') and sys._debug_utils):
            return
        
        max_utils = [d['max_util'] for d in self._debug_utils_collector]
        min_utils = [d['min_util'] for d in self._debug_utils_collector]
        home_utils = [d['home_util'] for d in self._debug_utils_collector]
        quit_count = sum(1 for d in self._debug_utils_collector if d['will_quit'])
        
        print(f"\n  [Utility Statistics] (前{len(self._debug_utils_collector)}个顾客):")
        print(f"    最大效用: 平均={np.mean(max_utils):.3f}, 最小={np.min(max_utils):.3f}, "
              f"最大={np.max(max_utils):.3f}, 中位数={np.median(max_utils):.3f}")
        print(f"    最小效用: 平均={np.mean(min_utils):.3f}, 最小={np.min(min_utils):.3f}, "
              f"最大={np.max(min_utils):.3f}, 中位数={np.median(min_utils):.3f}")
        print(f"    Home delivery效用: 平均={np.mean(home_utils):.3f}, 最小={np.min(home_utils):.3f}, "
              f"最大={np.max(home_utils):.3f}, 中位数={np.median(home_utils):.3f}")
        print(f"    退出顾客数: {quit_count}/{len(self._debug_utils_collector)} ({100*quit_count/len(self._debug_utils_collector):.1f}%)")
        print(f"    建议阈值范围: [{np.min(max_utils):.2f}, {np.max(max_utils):.2f}] "
              f"(当前阈值={self.quit_threshold:.2f})")

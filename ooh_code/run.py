import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env#env used for training
        self.test_env = self.config.test_env#seperate env used for testing only
        self.state_dim = np.shape(self.env.reset())[0]
        
        #to ensure we do not exceed the fleet capacity
        self.max_steps = int(config.n_vehicles*config.veh_capacity)-1
        
        print("State space: {}".format(self.state_dim))

        self.model = config.algo(config=config)

    # Main training/simulation loop
    def train(self):
        # Learn the model on the environment
        rewards = []
        episode_quit_counts = []  # 新增：记录每个episode的退出数

        checkpoint = self.config.save_after
        start_ep = 0

        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step = 0
            done = False

            while not done:
                action = self.model.get_action(state, training=True)
                new_state, done, stats,route_data = self.env.step(action=action)
                state = new_state
                step += 1
                _ = self.model.update(route_data,state,False)
                if step >= self.max_steps or done:
                    travel_time = self.model.update(route_data,state,True)#do full update when episode is done
                    rewards.append(Utils.total_costs(stats[1],stats[2],travel_time,stats[3],stats[6],self.config))
                    # 记录episode结束时的退出数
                    quit_count_episode = stats[8] if len(stats) > 8 else self.env.quit_count
                    episode_quit_counts.append(quit_count_episode)
                    break
            
            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' episodes :' +str(time()-t0))
                # 打印退出统计（最近checkpoint个episode）
                recent_quit_counts = episode_quit_counts[-checkpoint:] if len(episode_quit_counts) >= checkpoint else episode_quit_counts
                if len(recent_quit_counts) > 0:
                    total_recent_quits = sum(recent_quit_counts)
                    print('Recent {} episodes - Quit count: {}, Avg per episode: {:.2f}'.format(
                        len(recent_quit_counts), total_recent_quits, np.mean(recent_quit_counts)))
                Utils.plot_training_curves(rewards,self.config)
                #Utils.save_plots_stats(run_stats,travel_time,run_time,actions=actions,config=self.config,episode=episode)
               
                t0 = time()
    

    def eval(self, episodes=1):
        # Evaluate the model, see run_ppo - eval() for some interesting statistics to save,
        # we removed these statistics tracking from run.py to make the code a bit more readable,
        # but you can add those statistics tracking easily to this file again
        total_cost = []
        actions = []
        accepted_price = []
        price_time=[]
        home_delivery_loc = []
        step_time = []
        quit_counts = []  # 新增：记录每个episode的退出数
        for episode in range(episodes):
            state = self.test_env.reset()
            step = 0
            done = False
            while not done:
                t1 = time()
                action = self.model.get_action(state, training=False)
                new_state, done, stats,route_data = self.test_env.step(action=action)
                price_time.append([stats[7],step])
                actions.append([*action,step,episode])
                home_delivery_loc.append([stats[5],step,episode])
                state = new_state
                step += 1
                step_time.append(time()-t1)
                if step >= self.max_steps:
                    break
            # 在episode结束时记录退出数
            quit_count_episode = stats[8] if len(stats) > 8 else self.test_env.quit_count
            quit_counts.append(quit_count_episode)
         
        # 打印退出统计
        # 正确计算：
        # - distance == -1: 退出
        # - distance == 0: Home delivery（自己到自己的距离）
        # - distance > 0: OOH点
        total_customers_accepted = sum(1 for i in home_delivery_loc if i[0] >= 0)  # >=0 表示接受服务（包括Home delivery和OOH点）
        total_quits = sum(1 for i in home_delivery_loc if i[0] == -1)  # -1 表示退出
        # 总客户数 = 所有到达的客户（包括退出的）
        total_customers = len(home_delivery_loc)
        
        # 验证数据一致性
        if total_customers != total_customers_accepted + total_quits:
            print(f"警告：数据不一致！总客户数({total_customers}) != 接受服务({total_customers_accepted}) + 退出({total_quits})")
            print(f"差异：{total_customers - total_customers_accepted - total_quits}个客户")
        if total_customers > 0:
            quit_rate = total_quits / total_customers
            print('\n=== Customer Exit Statistics (eval) ===')
            print('Total customers: ', total_customers)
            print('Accepted customers: ', total_customers_accepted)
            print('Quit customers: ', total_quits)
            print('Quit rate: {:.2f}%'.format(quit_rate * 100))
            print('Average quits per episode: {:.2f}'.format(total_quits / episodes))
            print('========================================\n')
        
        #directly save statistics
        #Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
        #                      parcel_lockers_remaining_capacity,home_delivery_loc,step_time,price_time,self.config)
        
        return total_cost, accepted_price,step_time

    def eval2(self, episodes=1):
            # Evaluate the model
            # Todo: cleanup, looks a bit messy
            travel_time = []
            total_cost = []
            actions = []
            accepted_price = []
            accepted_discount = []
            count_home_delivery = []
            service_time = []
            parcel_lockers_remaining_capacity = []
            home_delivery_loc = []
            step_time = []
            quit_counts = []  # 新增：记录每个episode的退出数
            for episode in range(episodes):
                # print('episode' ,episode)
                state = self.test_env.reset()
                step = 0
                done = False
                while not done:
                    t1 = time()
                    action = self.model.get_action(state, training=True)

                    new_state, done, stats, route_data = self.test_env.step(action=action)
                    actions.append([*action, step, episode])
                    # accepted_price.append([stats[3],step,episode])
                    home_delivery_loc.append([stats[5], step, episode])
                    state = new_state
                    step += 1
                    step_time.append(time() - t1)
                    if step >= self.max_steps:
                        break
                # 在episode结束时记录退出数（从stats中获取，stats[8]是quit_count）
                quit_count_episode = stats[8] if len(stats) > 8 else self.test_env.quit_count
                quit_counts.append(quit_count_episode)
                travel_time.append([self.env.reopt_for_eval(route_data), episode])  # short HGS (re-opt) call
                # total_cost.append([Utils.total_costs(stats[1],stats[2],travel_time,stats[3],self.config)[0][0][0],episode])
                service_time.append([stats[2], episode])
                count_home_delivery.append([stats[1], episode])
                accepted_price.append(stats[3])
                accepted_discount.append(stats[6])
                for i in stats[4]:
                    parcel_lockers_remaining_capacity.append([i.remainingCapacity, i.location.x, i.location.y, episode])

            cnt = 0
            trvl = 0
            trvl_list = []
            srvc = 0
            fail = 0
            distance = 0
            count_pp = 0
            for i in home_delivery_loc:
                if i[0] > 0:
                    count_pp += 1
                distance += i[0]

            # ?? +
            cost_multiplier = (self.config.driver_wage + self.config.fuel_cost * self.config.truck_speed) / 3600
            for i in range(0, len(count_home_delivery)):
                cnt += count_home_delivery[i][0]
                trvl += (travel_time[i][0] * cost_multiplier)
                trvl_list.append(travel_time[i][0] * cost_multiplier)
                srvc += (service_time[i][0] * cost_multiplier)
                fail += count_home_delivery[i][
                            0] * self.config.home_failure * self.config.failure_cost  # costs of failed delivery

            # 安全地连接列表，处理空列表情况
            if len(accepted_discount) > 0:
                # 过滤掉空列表，只连接非空列表
                non_empty_discounts = [d for d in accepted_discount if len(d) > 0]
                if len(non_empty_discounts) > 0:
                    d_list = np.concatenate(non_empty_discounts)
                else:
                    d_list = np.array([])
            else:
                d_list = np.array([])
            
            if len(accepted_price) > 0:
                # 过滤掉空列表，只连接非空列表
                non_empty_prices = [p for p in accepted_price if len(p) > 0]
                if len(non_empty_prices) > 0:
                    r_list = np.concatenate(non_empty_prices)
                else:
                    r_list = np.array([])
            else:
                r_list = np.array([])
            
            # 计算退出统计
            # 正确计算：
            # - distance == -1: 退出
            # - distance == 0: Home delivery（自己到自己的距离）
            # - distance > 0: OOH点
            total_customers_accepted = sum(1 for i in home_delivery_loc if i[0] >= 0)  # >=0 表示接受服务（包括Home delivery和OOH点）
            total_quits = sum(1 for i in home_delivery_loc if i[0] == -1)  # -1 表示退出
            # 总客户数 = 所有到达的客户（包括退出的）
            total_customers = len(home_delivery_loc)
            
            # 验证数据一致性
            if total_customers != total_customers_accepted + total_quits:
                print(f"警告：数据不一致！总客户数({total_customers}) != 接受服务({total_customers_accepted}) + 退出({total_quits})")
                print(f"差异：{total_customers - total_customers_accepted - total_quits}个客户")
            
            print('percentage home delivery: ', cnt / len(home_delivery_loc) if len(home_delivery_loc) > 0 else 0)
            print('travel costs: ', trvl / episodes)
            print('service costs: ', srvc / episodes)
            print('failure costs: ', fail / episodes)
            # 安全地计算平均值和标准差，处理空数组
            print('Avg. Charge: ', np.mean(r_list) if len(r_list) > 0 else 0.0, 'std.: ', np.std(r_list) if len(r_list) > 0 else 0.0)
            print('Avg. Discount: ', -np.mean(d_list) if len(d_list) > 0 else 0.0, 'std.: ', np.std(d_list) if len(d_list) > 0 else 0.0)
            print('Charge revenue: ', np.sum(r_list) / episodes if len(r_list) > 0 else 0.0)
            print('Discount costs: ', -np.sum(d_list) / episodes if len(d_list) > 0 else 0.0)
            # 计算总收益：Charge revenue - Discount costs
            total_revenue = (np.sum(r_list) + np.sum(d_list)) / episodes if (len(r_list) > 0 or len(d_list) > 0) else 0.0  # d_list是负数，所以相加
            # 计算基础票价收益：每个被服务的客户都会产生基础收益
            base_revenue = total_customers_accepted * self.config.revenue / episodes
            print('Base revenue: ', base_revenue)
            # 计算运营成本（不包括收入）
            operating_costs = (trvl + srvc + fail) / episodes
            # 计算净利润：总收益 + 基础票价收益 - 运营成本
            net_profit = total_revenue + base_revenue - operating_costs
            print('Net profit: ', net_profit)
            # 安全地计算total costs，处理空数组
            sum_d = np.sum(d_list) if len(d_list) > 0 else 0.0
            sum_r = np.sum(r_list) if len(r_list) > 0 else 0.0
            print('total costs: ', (trvl + srvc + fail - sum_d - sum_r) / episodes)
            print('average travelled by customers: ', distance / count_pp if count_pp > 0 else 0)
            
            # 打印退出统计信息
            print('\n=== Customer Exit Statistics ===')
            print('Total customers: ', total_customers)
            print('Accepted customers: ', total_customers_accepted)
            print('Quit customers: ', total_quits)
            if total_customers > 0:
                quit_rate = total_quits / total_customers
                print('Quit rate: {:.2f}%'.format(quit_rate * 100))
            else:
                print('Quit rate: 0.00% (no customers)')
            print('Average quits per episode: {:.2f}'.format(total_quits / episodes))
            if len(quit_counts) > 0:
                print('Quit counts per episode: min={}, max={}, mean={:.2f}, std={:.2f}'.format(
                    min(quit_counts), max(quit_counts), np.mean(quit_counts), np.std(quit_counts)))
            print('===============================\n')

            return total_cost, accepted_price, step_time

def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    
    #evaluate model
    rewards,prices,step_time = solver.eval2(args.eval_episodes)
    #Utils.plot_test_boxplot(rewards,prices,step_time,config)
    
    print('total timing: ', time()-t)

if __name__== "__main__":
        main(train=True)

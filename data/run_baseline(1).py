import numpy as np
import Src.Utils.Utils as Utils
from Src.Algorithms.Baseline import Baseline
from Src.parser import Parser
from Src.config import Config
from time import time
from  Environments.OOH import Parcelpoint_py

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env:Parcelpoint_py = self.config.env  # env used for training
        self.test_env:Parcelpoint_py = self.config.test_env  # seperate env used for testing only
        self.state_dim = np.shape(self.env.reset())[0]

        # to ensure we do not exceed the fleet capacity
        #为车队总的容量
        self.max_steps = int(config.n_vehicles * config.veh_capacity) - 1

        print("State space: {}".format(self.state_dim))


        self.model:Baseline = config.algo(config=config)

    # Main training/simulation loop
    def train(self):
        # Learn the model on the environment
        rewards = []

        checkpoint = self.config.save_after
        start_ep = 0

        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            '''
            [self.newCustomer,self.fleet,self.parcelPoints,self.steps]
            '''
            state = self.env.reset()
            self.model.reset()

            step = 0
            done = False

            while not done:
                action = self.model.get_action(state, training=True)

                new_state, done, stats, route_data = self.env.step(action=action)
                state = new_state
                step += 1
                _ = self.model.update(route_data, state, False)
                if step >= self.max_steps or done:
                    travel_time = self.model.update(route_data, state, True)  # do full update when episode is done
                    rewards.append(Utils.total_costs(stats[1], stats[2], travel_time, stats[3], stats[6], self.config))
                    break
            print('总共 ',step)
            if episode % checkpoint == 0 or episode == self.config.max_episodes - 1:
                print('time required for ' + str(checkpoint) + ' episodes :' + str(time() - t0))
                Utils.plot_training_curves(rewards, self.config)
                # Utils.save_plots_stats(run_stats,travel_time,run_time,actions=actions,config=self.config,episode=episode)

                t0 = time()

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
        for episode in range(episodes):
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

        # 工资 +
        cost_multiplier = (self.config.driver_wage + self.config.fuel_cost * self.config.truck_speed) / 3600
        for i in range(0, len(count_home_delivery)):
            cnt += count_home_delivery[i][0]
            trvl += (travel_time[i][0] * cost_multiplier)
            trvl_list.append(travel_time[i][0] * cost_multiplier)
            srvc += (service_time[i][0] * cost_multiplier)
            fail += count_home_delivery[i][
                        0] * self.config.home_failure * self.config.failure_cost  # costs of failed delivery

        d_list = np.concatenate(accepted_discount)
        r_list = np.concatenate(accepted_price)
        print('percentage home delivery: ', cnt / len(home_delivery_loc))
        print('travel costs: ', trvl / episodes)
        print('service costs: ', srvc / episodes)
        print('failure costs: ', fail / episodes)
        print('Avg. Charge: ', np.mean(r_list), 'std.: ', np.std(r_list))
        print('Avg. Discount: ', -np.mean(d_list), 'std.: ', np.std(d_list))
        print('Charge revenue: ', np.sum(r_list) / episodes)
        print('Discount costs: ', -np.sum(d_list) / episodes)
        print('total costs: ', (trvl + srvc + fail - np.sum(d_list) - np.sum(r_list)) / episodes)
        print('average travelled by customers: ', distance / count_pp)

        return total_cost, accepted_price, step_time

    def eval(self, episodes=1):

        # Evaluate the model, see run_ppo - eval() for some interesting statistics to save,
        # we removed these statistics tracking from run.py to make the code a bit more readable,
        # but you can add those statistics tracking easily to this file again
        total_cost = []
        actions = []
        accepted_price = []
        price_time = []
        count_home_delivery = []
        home_delivery_loc = []
        step_time = []
        travel_time = []
        for episode in range(episodes):
            state = self.test_env.reset()
            step = 0
            done = False
            while not done:
                t1 = time()
                action = self.model.get_action(state, training=False)
                # print('提供的选择',action)
                new_state, done, stats, route_data = self.test_env.step(action=action)
                # print('选择家的人',stats[1])
                price_time.append([stats[7], step])
                actions.append([*action, step, episode])
                # print('stats inx 5(dist_matrix)',stats[5])
                home_delivery_loc.append([stats[5], step, episode])
                state = new_state
                step += 1
                step_time.append(time() - t1)
                if step >= self.max_steps:
                    break
            travel_time.append([self.env.reopt_for_eval(route_data), episode])  # short HGS (re-opt) call
            count_home_delivery.append([stats[1], episode])

            print('total step', step)
        trvl = 0
        cnt = 0
        cost_multiplier = (self.config.driver_wage + self.config.fuel_cost * self.config.truck_speed) / 3600
        for i in range(0, len(count_home_delivery)):
            cnt += count_home_delivery[i][0]
            trvl += (travel_time[i][0] * cost_multiplier)

        # directly save statistics
        # Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
        #              parcel_lockers_remaining_capacity,home_delivery_loc,step_time,price_time,self.config)

        print('travel costs: ', trvl / episodes)
        print('choice home delivery: ', cnt / episodes)
        return total_cost, accepted_price, step_time


def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()
    args.algo_name = 'Baseline'
    #args.max_episodes=10
    args.max_steps_r=90
    config = Config(args)
    #[-5,5]
    sequence = np.arange(0, 5.1, 0.1)
    for  x in sequence:
        args.home_util = x
        print('home util', x)
        config = Config(args)
        solver = Solver(config=config)

    # if train:
    #     solver.train()

    #evaluate model
        rewards, prices, step_time = solver.eval2(10)
        #Utils.plot_test_boxplot(rewards, prices, step_time, config)

    print('total timing: ', time() - t)


if __name__ == "__main__":
    main(train=False)


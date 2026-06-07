import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

class Solver:
    def __init__(self, config):
        self.config = config
        self.env = self.config.env
        self.test_env = self.config.test_env
        self.state_dim = np.shape(self.env.reset())[0]
        self.max_steps = int(config.n_vehicles*config.veh_capacity)-1
        print("State space: {}".format(self.state_dim))
        self.model = config.algo(config=config)

    def train(self):
        returns = []
        total_loss_actor = 0
        total_loss_critic = 0
        checkpoint = self.config.save_after
        rm, start_ep = 0, 0
        t0 = time()
        self.episode = 0
        for episode in range(start_ep, self.config.max_episodes):
            episode_loss_actor = []
            episode_loss_critic = []
            state = self.env.reset()
            self.model.reset()
            step, total_r = 0, 0
            done = False
            while not done:
                action, a_hat = self.model.get_action(self.env.abstract_state_ppo(state), state, training=True)
                new_state, done, stats, route_data = self.env.step(action=action)
                reward = 0.01
                loss_actor, loss_critic = self.model.update(
                    self.env.abstract_state_ppo(state), action, a_hat, reward,
                    self.env.abstract_state_ppo(new_state), done
                )
                episode_loss_actor.append(loss_actor)
                episode_loss_critic.append(loss_critic)
                state = new_state
                total_r += reward
                step += 1
                if step >= self.max_steps or done:
                    travel_time = self.model.update_route(route_data, state, True)
                    reward = -Utils.total_costs(stats[1], stats[2], travel_time, stats[3], stats[6], self.config)
                    loss_actor, loss_critic = self.model.update(
                        self.env.abstract_state_ppo(state), action, a_hat, reward,
                        self.env.abstract_state_ppo(new_state), done
                    )
                    episode_loss_actor.append(loss_actor)
                    episode_loss_critic.append(loss_critic)
                    total_r += reward
                    break

            total_loss_actor = total_loss_actor*0.99 + 0.01*np.average(episode_loss_actor)
            total_loss_critic = total_loss_critic*0.99 + 0.01*np.average(episode_loss_critic)
            rm = 0.9*rm + 0.1*total_r

            if episode % checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' :' + str(time()-t0))
                print('Episode '+str(episode)+' / current actor loss: ' + str(total_loss_actor))
                print('Episode '+str(episode)+' / current critic loss: ' + str(total_loss_critic))
                returns.append(total_r)
                Utils.plot_training_curves(returns, config=self.config)
                t0 = time()

    def eval(self, episodes=1):
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
                 action, a_hat = self.model.get_action(self.test_env.abstract_state_ppo(state), state, training=True)
                 new_state, done, stats, route_data = self.test_env.step(action=action)
                 actions.append([*action, step, episode])
                 home_delivery_loc.append([stats[5], step, episode])
                 state = new_state
                 step += 1
                 step_time.append(time()-t1)
                 if step >= self.max_steps:
                     break
             travel_time.append([self.env.reopt_for_eval(route_data), episode])
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
         distance = 0
         count_pp = 0
         for i in home_delivery_loc:
             if i[0] > 0:
                 count_pp += 1
             distance += i[0]

         cost_multiplier = (self.config.driver_wage+self.config.fuel_cost*self.config.truck_speed) / 3600
         for i in range(0, len(count_home_delivery)):
             cnt += count_home_delivery[i][0]
             trvl += (travel_time[i][0]*cost_multiplier)
             trvl_list.append(travel_time[i][0]*cost_multiplier)
             srvc += (service_time[i][0]*cost_multiplier)

         d_list = np.concatenate(accepted_discount)
         r_list = np.concatenate(accepted_price)
         print('percentage home delivery: ', cnt/len(home_delivery_loc))
         print('travel costs: ', trvl/episodes)
         print('service costs: ', srvc/episodes)
         print('Avg. Charge: ', np.mean(r_list), 'std.: ', np.std(r_list))
         print('Avg. Discount: ', -np.mean(d_list), 'std.: ', np.std(d_list))
         print('Charge revenue: ', np.sum(r_list)/episodes)
         print('Discount costs: ', -np.sum(d_list)/episodes)
         print('total costs: ', (trvl+srvc-np.sum(d_list)-np.sum(r_list))/episodes)
         print('average travelled by customers: ', distance/count_pp)
         return total_cost, accepted_price, step_time

def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()
    config = Config(args)
    solver = Solver(config=config)
    if train:
        solver.train()
    print('total timing: ', time()-t)

if __name__== "__main__":
        main(train=True)
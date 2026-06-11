import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Seed for reproducibility
        parser.add_argument("--seed", default=0, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=50, help="Number of checkpoints for saving results and model", type=int)
        parser.add_argument("--log_output", default="term_file", help="Log all the print outputs", choices=["term_file", "term", "file"])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model checkpoints")

        # For documentation purposes
        now = datetime.now()
        timestamp = str(now.month) + "|" + str(now.day) + "|" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default="default", help="folder name suffix")
        parser.add_argument("--experiment", default="run", help="Name of the experiment")

        parser.add_argument("--algo_name",default="DSPO_MenuSelection_SPO",help="Policy/algorithm used, capital sensitive",
            choices=["DSPO", "DSPO_MenuSelection_MSE",  "Baseline", "DSPO_MenuSelection_SPO", "DRPO", "DSPO_plus_SPO", "MenuSelection_SPO_A","MenuSelection_SPO_B","MenuSelection_SPO_C"],
        )
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)

        # Environment parameters
        self.environment_parameters(parser)

        # General settings for algorithms
        self.DSPO_parameters(parser)
        self.Heuristic_parameters(parser)
        self.Baseline_parameters(parser)
        self.PPO_parameters(parser)

        self.parser = parser

    def environment_parameters(self, parser):
        # Debug default for outside option utility; can be overridden by CLI
        DEBUG_OUTSIDE_OPTION_UTIL = -1.0  # None disables outside option

        parser.add_argument("--env_name", default="Parcelpoint_py", help="Environment to run the code")
        parser.add_argument("--max_episodes", default=int(80), help="maximum number of training episodes", type=int)
        parser.add_argument("--eval_episodes", default=20, help="number of evaluation episodes after training", type=int)

        # Episode length parameters (Gamma / negative binomial in env.reset)
        parser.add_argument("--max_steps_r", default=90, help="maximum customers per episode r of gamma dist.", type=int)
        parser.add_argument("--max_steps_p", default=0.5, help="maximum customers per episode p of gamma dist. [0,1]", type=float)

        # Data loading
        parser.add_argument(
            "--load_data",
            default=True,
            help="whether to load location data from file or to generate data (only used for debug)",
            type=self.str2bool,
        )
        parser.add_argument(
            "--instance",
            default="RC",
            help="which instance to load",
            choices=["Austin", "Seattle", "C", "R", "RC", "Beijing_bus", "Beijing_Yanjiao", "NYC_TLC"],
        )

        parser.add_argument("--data_seed", default=0, help="which dataset seed to load for training", choices=[0, 1, 2, 3], type=int)
        parser.add_argument("--data_seed_test", default=1, help="which dataset seed to load for testing", choices=[0, 1, 2, 3], type=int)

        parser.add_argument("--pricing", default=True, help="if we use pricing or offering decision space", type=self.str2bool)
        parser.add_argument("--max_price", default=2.0, help="max delivery charge >0", type=float)
        parser.add_argument("--min_price", default=-10.0, help="max discount <0", type=float)

        # Fixed number of parcelpoints/meeting points offered (pricing mode uses adjacency built with k)
        parser.add_argument("--k", default=10, help="Number of parcelpoints to offer to customer", type=int)

        # Instance-specific passenger count (used by Beijing_Yanjiao)
        parser.add_argument("--n_passengers", default=300, help="Number of passengers/home locations (for Beijing_Yanjiao)", type=int)
        parser.add_argument(
            "--yanjiao_prefix",
            default=None,
            help=(
                "Optional Beijing_Yanjiao data-file prefix template. "
                "Use {n_passengers} and {seed}, e.g. yanjiao_het_home_{n_passengers}_{seed}. "
                "Defaults to yanjiao_{n_passengers}_{seed}."
            ),
        )

        parser.add_argument("--n_vehicles", default=15, help="number of vehicles", type=int)
        parser.add_argument("--veh_capacity", default=12, help="capacity per vehicle per day", type=int)
        parser.add_argument("--fraction_capacitated", default=0.38, help="pfraction of lockers capacitated", type=float)
        parser.add_argument("--parcelpoint_capacity", default=50, help="capacity of capacitated lockers", type=int)

        parser.add_argument("--incentive_sens", default=-0.25, help="sensitivty of customer to incentives", type=float)
        parser.add_argument("--base_util", default=-1.0, help="base utility across all alternatives", type=float)
        parser.add_argument("--home_util", default=1.4, help="utility given to home delivery", type=float)
        parser.add_argument("--dissatisfaction", default=False, help="customer dissatisfaction penalty when all delivery options have too high prices", type=self.str2bool)

        parser.add_argument("--revenue", default=50, help="revenue per customer, only used for pricing decision", type=float)
        parser.add_argument("--fuel_cost", default=0.6, help="costs of fuel per distance unit", type=float)
        parser.add_argument("--truck_speed", default=30, help="distance travelled per hour", type=float)
        parser.add_argument("--clip_service_time", default=10, help="maximum service time in minutes", type=float)
        parser.add_argument("--driver_wage", default=30, help="salary of driver per hour", type=float)

        # Service time parameters for home delivery and meeting points
        parser.add_argument("--l0_home", default=2.5, help="base service time for home delivery in minutes", type=float)
        parser.add_argument("--l_mp", default=0.75, help="service time for meeting points (OOH) in minutes", type=float)

        parser.add_argument("--home_failure", default=0.1, help="the probability of delivery failure for home delivery", type=float)
        parser.add_argument("--failure_cost", default=20.0, help="the monetary costs of a delivery failure", type=float)

        parser.add_argument("--reopt", default=10000000, help="re-opt frequency of cheapest insertion route using HGS", type=int)
        parser.add_argument("--hgs_reopt_time", default=1.1, help="re-opt HGS time limit", type=float)
        parser.add_argument("--hgs_final_time", default=1.5, help="HGS time limit for obtaining final routing schedule", type=float)
        parser.add_argument(
            "--route_label_mode",
            default="hgs",
            choices=["hgs", "hep"],
            help="Route label source: hgs uses Hygese final routes; hep uses cheapest-insertion routes with half-edge labels for fast screening.",
        )

        # Quit threshold (backward compatible)
        parser.add_argument("--quit_threshold", default=None, help="Customer quit threshold: if all options' utility < threshold, customer quits. None means disabled.", type=float)

        # Optional mode flags
        parser.add_argument("--find_threshold", action="store_true", help="Quickly search quit threshold mode")
        parser.add_argument("--threshold_percentile", type=int, default=100, help="Quit threshold percentile (100=all quit, 95=95%% quit)")

        # Outside option utility
        parser.add_argument(
            "--outside_option_util",
            type=float,
            default=DEBUG_OUTSIDE_OPTION_UTIL,
            help="Outside option utility u0 (None disables). Larger means more attractive outside option.",
        )
        parser.add_argument("--travel_time_weight", default=None, help="Negative utility weight for predicted in-vehicle travel time.", type=float)
        parser.add_argument("--walk_distance_weight", default=None, help="Negative utility weight for home-to-meeting-point distance.", type=float)
        parser.add_argument("--final_yanjiao_mode", default=False, help="Enable strict final Yanjiao utility/data guards.", type=self.str2bool)
        parser.add_argument("--allow_derived_choice_utility", default=False, help="Allow audited derived choice_util_matrix in final Yanjiao mode.", type=self.str2bool)

    def DSPO_parameters(self, parser):
        parser.add_argument("--grid_dim", default=11, help="division of operational area in X*X clusters", type=int)
        parser.add_argument("--hexa", default=False, help="division of operational area in hexagional grid instead of squares (beta)", type=self.str2bool)
        parser.add_argument("--n_input_layers", default=2, help="divide feature map in X time intervals", type=int)
        parser.add_argument("--only_phase_one", default=False, help="when True, we stop learning after an initial data collection phase", type=self.str2bool)
        parser.add_argument("--initial_phase_epochs", default=50, help="maximum number of training epochs", type=int)
        parser.add_argument("--buffer_size", default=int(500), help="Size of memory buffer", type=int)
        parser.add_argument("--batch_size", default=256, help="Batch size", type=int)
        parser.add_argument("--learning_rate", default=1e-3, help="learning rate", type=float)

        parser.add_argument("--init_theta_cnn", default=0.75, help="initial weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta_cnn", default=(1 / 850), help="weight reduction for cheapest insertion", type=float)

        parser.add_argument("--linearModel", default=False, type=self.str2bool, help="To use a linear regression model instead of a CNN/MLP")
        parser.add_argument("--optim", default="adam", help="Optimizer type", choices=["adam", "sgd", "rmsprop"])
        parser.add_argument("--use3d_conv", default=False, type=self.str2bool, help="Use 3D convolution instead of 2D")
        parser.add_argument("--use_travel_time_prediction", default=False, type=self.str2bool, help="Use predicted in-vehicle travel time in choice utility")
        parser.add_argument("--travel_time_learning_rate", default=None, help="Learning rate for travel-time predictor; defaults to learning_rate", type=float)
        parser.add_argument("--n_filters", default=16, help="number of filters in first convolutional layer (2nd is 2*X)", type=int)
        parser.add_argument("--dropout", default=0.05, help="dropout rate of the FC layers", type=float)
        parser.add_argument("--menu_size", default=3, help="Menu size L: number of OOH points shown to customer (used by MenuSelection variants)", type=int)
        parser.add_argument("--max_discount_per_customer", default=4.0, help="Maximum discount per customer for SPO variants", type=float)
        parser.add_argument("--menu_spo_warmup_episodes", default=30, help="Warmup episodes for MenuSelection SPO", type=int)
        parser.add_argument("--phase2_huber_weight", default=0.35, help="Phase 2 Huber weight for SPO variants", type=float)
        parser.add_argument("--spo_warmup_episodes", default=5, help="Warmup episodes before enabling SPO+ loss", type=int)
        parser.add_argument("--spo_rampup_episodes", default=10, help="Rampup episodes for SPO+ loss weight", type=int)
        parser.add_argument("--spo_loss_weight", default=0.7, help="Maximum SPO+ loss weight in mixed objective", type=float)
        parser.add_argument("--spo_buffer_size", default=None, help="Replay size for terminal labeled SPO+ menu samples; defaults to buffer_size", type=int)
        parser.add_argument("--spo_batch_size", default=32, help="Number of terminal labeled menus per SPO+ update", type=int)
        parser.add_argument("--spo_label_sample_size", default=0, help="Max pricing decisions per episode to label for SPO+; 0 keeps all decisions", type=int)

    def Heuristic_parameters(self, parser):
        parser.add_argument("--init_theta", default=1.0, help="weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta", default=1 / 850, help="weight reduction for cheapest insertion", type=float)

    def Baseline_parameters(self, parser):
        parser.add_argument("--save_routes", default=False, help="Used to generate and save routes for use inside Heuristic", type=self.str2bool)
        parser.add_argument("--price_pp", default=-0.0, help="fixed fee price to offer for all parcelpoints", type=float)
        parser.add_argument("--price_home", default=0.0, help="fixed fee price to offer for home delivery", type=float)

    def PPO_parameters(self, parser):
        parser.add_argument("--actor_lr", default=1e-4, help="(1e-2) Learning rate of actor", type=float)
        parser.add_argument("--critic_lr", default=1e-2, help="(1e-2) Learning rate of critic", type=float)
        parser.add_argument("--state_lr", default=1e-1, help="Learning rate of state features", type=float)
        parser.add_argument("--batch_size_ppo", default=100, help="Batch size", type=int)
        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis", type=self.str2bool)
        parser.add_argument("--fourier_order", default=3, help="Order of fourier basis, (if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--hiddenLayerSize", default=16, help="size of hiddenlayer of critic", type=int)
        parser.add_argument("--hiddenActorLayerSize", default=8, help="size of hiddenlayer", type=int)
        parser.add_argument("--gamma", default=0.999, help="Discounting factor", type=float)
        parser.add_argument("--gauss_variance", default=2, help="Variance for gaussian policy", type=float)
        parser.add_argument("--clipping_factor", default=0.2, help="PPO clipping factor", type=float)
        parser.add_argument("--td_lambda", default=0.95, help="lambda factor for calculating advantages", type=float)
        parser.add_argument("--policy_update_epochs", default=25, help="number of epochs with which we perform policy updates in PPO", type=int)
        parser.add_argument("--critic_update_epochs", default=25, help="number of epochs with which we perform critic updates in PPO", type=int)

    def str2bool(self, text):
        if text == "True":
            arg = True
        elif text == "False":
            arg = False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        return arg

    def get_parser(self):
        return self.parser


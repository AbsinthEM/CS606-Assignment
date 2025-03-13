import os
import copy
import random
import gymnasium as gym
import numpy as np
import numpy.random as rnd
from operators import *
from psp import PSP, Parser
from src.alns import ALNS
from src.settings import DATA_PATH

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class pspAlnsEnv(gym.Env):
    def __init__(self, config, **kwargs):
        # Parameters
        self.config = config["environment"]
        self.instances_folder = self.config["instances_folder"]

        if isinstance(self.config["instances"], list):
            self.instances = [
                i
                for i in range(self.config["instances"][0], self.config["instances"][1])
            ]
        else:
            self.instances = [self.config["instances"]]

        self.psp = None
        self.rnd_state = None
        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        # // Add code here to include other states that require reset
        # --------------------- Provided to students
        self.improvement = None
        self.current_improved = None

        # // Add aditional attributes if required to store additional state observations
        # ---------------------------------------

        self.cost_difference_from_best = None
        self.current_updated = None

        self.task_assignment_rate = None
        self.avg_worker_utilization = None
        # Stagnation counter
        self.stagcount = None

        # # Simulated annealing acceptance criteria
        self.max_temperature = 10.0
        self.temperature = 10.0
        self.cooling_rate = 0.99

        # Gym Environment Parameters
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = self.config[
            "iterations"
        ]  # max number of generations in an episode

        # Defining Action and Observation Space
        # ----------------------------------------------------------------------
        # // Modify action and observation space as you see fit
    
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 10])

        # Expand observation space to include additional features
        self.observation_space = gym.spaces.Box(
            shape=(9,), low=0, high=100, dtype=np.float64
        )
        # ----------------------------------------------------------------------

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1

        # state = np.array(
        #     [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
        #      self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
        #     dtype=np.float64).squeeze()

        # ---------------------
        # // Add additional state observations here
        state = np.array(
            [
                self.improvement,  # Improvement made
                self.cost_difference_from_best,  # Cost difference from the best solution
                is_current_best,  # Is the current solution the best solution?
                self.stagcount / 30,  # Stagnation counter (normalizaed) 
                self.iteration / self.max_iterations,  # Search progress
                self.current_updated,  # Has the current solution been updated?
                self.current_improved,  # Has the current solution improved?
                self.task_assignment_rate,  # Task assignment rate
                self.avg_worker_utilization  # Average worker utilization
            ],  # State observations
            dtype=np.float64,
        ).squeeze()
        # ---------------------

        return state

    def reset(self, seed=None, options=None, run = 0):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """
        
        if run:
            self.instance_path = os.path.join(
                DATA_PATH,
                self.instances_folder,
                self.instances[0]
            )
            SEED = seed
            self.rnd_state = rnd.RandomState(SEED)
            
        
        else:
            # if training: #TODO - random, else, use set seed
            SEED = random.randint(0, 100000)
            self.rnd_state = rnd.RandomState(SEED)

            # Select Problem Instances Randomly
            self.instance = random.choice(self.instances)
            print(f"DATA_PATH: {DATA_PATH}")
            self.instance_path = os.path.join(
                DATA_PATH,
                self.instances_folder,
                self.config["instances_folder"]
                + "_instance_"
                + str(self.instance)
                + ".json",
            )
       

        parsed = Parser(self.instance_path)
        psp = PSP(parsed.name, parsed.workers, parsed.tasks, parsed.Alpha)
        psp.random_initialize(SEED)

        self.psp = psp
        self.initial_solution = psp
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # Adding of Destroy and Repair Operators
        # // You should import and add your operators.py functions here
        self.dr_alns = ALNS(self.rnd_state)

        # Add all destruction operators (imported from operators.py)
        self.dr_alns.add_destroy_operator(destroy_random)
        self.dr_alns.add_destroy_operator(destroy_time_blocks)
        self.dr_alns.add_destroy_operator(destroy_cost)

        # Add all repair operators (imported from operators.py)
        self.dr_alns.add_repair_operator(repair_greedy)
        self.dr_alns.add_repair_operator(repair_time_blocks)
        self.dr_alns.add_repair_operator(repair_regret)


        # reset tracking values
        # // Add code here to reset the additional
        # observation states that you defined
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0

        # Calculate initial task assginment rate
        total_tasks = len(self.initial_solution.tasks)
        assigned_tasks = total_tasks - len(self.initial_solution.unassigned)
        self.task_assignment_rate = (assigned_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        # Calculate initial worker utilization
        total_workers = len(self.initial_solution.workers)
        active_workers = sum(1 for w in self.initial_solution.workers if len(w.tasks_assigned) > 0)
        self.avg_worker_utilization = (active_workers / total_workers) * 100 if total_workers > 0 else 0

        self.iteration, self.reward = 0, 0
        self.done = False

        return self.make_observation(), {}
        

    def step(self, action):
        self.iteration += 1
        self.stagcount += 1
        self.current_updated = 0
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0
        self.current_improved = 0
        # // Add code here to "step" the additional
        # observation states that you defined

        # Lower the temperature parameter (simulated annealing)
        self.temperature = max(0.5, self.temperature * self.cooling_rate)

        current = self.current_solution
        best = self.best_solution

        d_idx, r_idx = action[0], action[1]
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx]

        factors = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
            4: 0.5,
            5: 0.6,
            6: 0.7,
            7: 0.8,
            8: 0.9,
            9: 1.0,
        }
        destory_factor = factors[action[2]]
        # self.temperature = (1/(action[3]+1)) * self.max_temperature

        destroyed = d_operator(current, self.rnd_state, destory_factor)

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state)

        # Calculate current metrics before considering the candidate solution
        original_task_assignment_rate = self.task_assignment_rate
        original_avg_worker_utilization = self.avg_worker_utilization

        new_best, new_current = self.consider_candidate(best, current, candidate)

        self.reward_and_update(new_best, best, new_current, current)

        # Update task assignment rate
        total_tasks = len(self.current_solution.tasks)
        assigned_tasks = total_tasks - len(self.current_solution.unassigned)
        new_task_assignment_rate = (assigned_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        self.task_assignment_rate = new_task_assignment_rate

        # Update worker utilization
        total_workers = len(self.current_solution.workers)
        active_workers = sum(1 for w in self.current_solution.workers if len(w.tasks_assigned) > 0)
        new_avg_worker_utilization = (active_workers / total_workers) * 100 if total_workers > 0 else 0
        self.avg_worker_utilization = new_avg_worker_utilization

        # Add additional reward based on task assignment improvement
        if new_task_assignment_rate > original_task_assignment_rate:
            self.reward += 0.5 * (new_task_assignment_rate - original_task_assignment_rate) / 10

        self.cost_difference_from_best = min(100, (
            self.best_solution.objective() / max(1, self.current_solution.objective())
        ) * 100)

        state = self.make_observation()

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True
            # Additional reward at the end of the episode based on the improvement of the final solution compared to the initial solution
            initial_obj = self.initial_solution.objective()
            final_obj = self.current_solution.objective()

            if initial_obj != 0:
              improvement_ratio = (initial_obj - final_obj) / abs(initial_obj)
              # Add end-of-episode reward
              self.reward += max(0, improvement_ratio * 5)

        return state, self.reward, self.done, False, {}

    def reward_and_update(self, new_best, best, new_current, current):
        # ------------------------------------------------------------
        # // Modify Reward Function Here as you see fit
        if new_best != best and new_best is not None:
            # Found a new global optimal solution
            improvement_percent = 0
            if best.objective() != 0:
              improvement_percent = (best.objective() - new_best.objective()) / abs(best.objective()) * 100

            self.best_solution = new_best
            self.current_solution = new_best
            self.current_updated = 1

            # Dynamic reward based on improvement percentage
            base_reward = 5
            improvement_reward = min(5, abs(improvement_percent) / 2)
            self.reward += base_reward + improvement_reward

            self.stagcount = 0
            self.current_improved = 1

            # Additional reward based on stagnation count
            # give more reward if the new optimal solution is found after a long stagnation
            if self.stagcount > 10:
              # Reward for breaking stagnation after a long time
              self.reward += 2

        elif new_current != current and new_current.objective() < current.objective():
            # Found a better local solution but not the global optimal
            improvement_percent = 0
            if current.objective() != 0:
                improvement_percent = (current.objective() - new_current.objective()) / abs(current.objective()) * 100

            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1

            # Dynamic reward based on improvement percentage
            local_reward = min(2, abs(improvement_percent) / 5)
            self.reward += local_reward

            # Give more encouragement during stagnation
            if self.stagcount > 20:
              # Small reward for breaking stagnation
              self.reward += 0.5

        elif new_current != current and new_current.objective() >= current.objective():
            # Accepted a solution worse than the current one (using simulated annealing)
            self.current_solution = new_current
            self.current_updated = 1

            # Adjust penalty for accepting a worse solution depending on the search progress
            penalty = -0.1 * (1 - min(1, self.iteration / (self.max_iterations * 0.6)))
            self.reward += penalty

        if new_current.objective() < current.objective():
            self.improvement = 1
        # ------------------------------------------------------------

    # def consider_candidate(self, best, curr, cand):
        # -----------------------------------------------------
        # // Modify acceptance criteria as you see fit
        # Hill Climbing
        # if cand.objective() < best.objective():
        #     return cand, cand
        # else:
        #     return None, curr

        # // You could try other strategies like:
        # 1. Simulated Annealing ?
        # 2. Record to Record Travel ?
        # ------------------------------------------------------

    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
        if cand.objective() < best.objective():
            # If the candidate solution is better than the best solution
            # update both best and current solutions
            return cand, cand
        
        elif cand.objective() < curr.objective():
           # If the candidate solution is better than the current solution, only update the current solution
           return None, cand

        else:
          # If the candidate solution is worse than the current solution
          # accept it accroding to the simulated annealing standard
          diff = cand.objective() - curr.objective()

          # Calculate the acceptance probability
          # higher acceptance probability: smaller the difference and higher the temperature
          # Avoid division by zero
          prob_denom = max(1e-10, self.temperature)
          
          # Prevent numerical overflow
          # if the exponent is too large (in absolute value), directly return 0
          exp_term = -diff / prob_denom
          if exp_term < -700:  # np.exp's numerical limit
            accept_probability = 0.0
          else:
            accept_probability = np.exp(exp_term)

          # Decrease the probability of accepting worse solutions as the search progresses
          progress_factor = 1 - (self.iteration / self.max_iterations) * 0.5
          adjusted_probability = min(1.0, accept_probability * progress_factor)

          if self.rnd_state.random() < adjusted_probability:
              # Accept worse solutions based on the probability
              return None, cand
          else:
              # Reject worse solutions
              return None, curr
    # --------------------------------------------------------------------------------------------------------------------

    def run(self, model, seed = None, episodes = 1):
        """
        Use a trained model to select actions.
        """
        try:
            for episode in range(episodes):
                self.done = False
                state, _ = self.reset(seed = seed, run = 1)

                while not self.done:
                    state = np.array(state)
                    action, _ = model.predict(state, deterministic=True)
                    state, reward, self.done, _, info = self.step(action)

                    # Optional: Debugging/logging
                    # print(f"State: {state}, Reward: {reward}, Done: {self.done}")

        except KeyboardInterrupt:
            print("Execution interrupted. Stopping.")

    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(2):
            self.done = False
            state = self.reset()
            print("start episode: ", episode)
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print(
                    "step {}, action: {}, Current: {}, Best: {}, Reward: {:2.3f}".format(
                        self.iteration,
                        action,
                        self.current_solution.objective(),
                        self.best_solution.objective(),
                        reward,
                    )
                )
              
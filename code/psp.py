import copy
import json
import random

from src.alns import State


### Parser to parse instance json file ###
# You should not change this class!
class Parser(object):
    def __init__(self, json_file):
        """initialize the parser, saves the data from the file into the following instance variables:
        -
        Args:
            json_file::str
                the path to the xml file
        """
        self.json_file = json_file
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.name = self.data["name"]
        self.Alpha = self.data["ALPHA"]
        self.T = self.data["T"]
        self.BMAX = self.data["BMax"]
        self.WMAX = self.data["WMax"]
        self.RMIN = self.data["RMin"]

        self.workers = [
            Worker(worker_data, self.T, self.BMAX, self.WMAX, self.RMIN)
            for worker_data in self.data["Workers"]
        ]
        self.tasks = [Task(task_data) for task_data in self.data["Tasks"]]


class Worker(object):
    def __init__(self, data, T, bmax, wmax, rmin):
        """Initialize the worker
        Attributes:
            id::int
                id of the worker
            skills::[skill]
                a list of skills of the worker
            available::{k: v}
                key is the day, value is the list of two elements,
                the first element in the value is the first available hour for that day,
                the second element in the value is the last available hour for that day, inclusively
            bmax::int
                maximum length constraint
            wmax::int
                maximum working hours
            rmin::int
                minimum rest time
            rate::int
                hourly rate
            tasks_assigned::[task]
                a list of task objects
            blocks::{k: v}
                key is the day where a block is assigned to this worker
                value is the list of two elements
                the first element is the hour of the start of the block
                the second element is the hour of the start of the block
                if a worker is not assigned any tasks for the day, the key is removed from the blocks dictionary:
                        Eg. del self.blocks[D]

            total_hours::int
                total working hours for the worker

        """
        self.id = data["w_id"]
        self.skills = data["skills"]
        self.T = T
        self.available = {int(k): v for k, v in data["available"].items()}
        # the constant number for f2 in the objective function
        self.bmin = 4
        self.bmax = bmax
        self.wmax = wmax
        self.rmin = rmin

        self.rate = data["rate"]
        self.tasks_assigned = []
        self.blocks = {}
        self.total_hours = 0

    def can_assign(self, task):
        # // Implement Code Here
        ## check skill set
        if task.skill not in self.skills:
            return False

        ## check available time slots
        if task.day not in self.available:
            return False

        day_available = self.available[task.day]
        if task.hour < day_available[0] or task.hour > day_available[1]:
            return False

        ## cannot do two tasks at the same time
        for assigned_task in self.tasks_assigned:
            if assigned_task.day == task.day and assigned_task.hour == task.hour:
                return False

        ## If no other tasks assigned in the same day
        if task.day not in self.blocks:
        #   ## check if task.hour within possible hours for current day
            for day, block in self.blocks.items():
                time_diff = abs(task.day - day) * 24
                if task.day > day:
                    time_diff += task.hour - block[1]
                else:
                    time_diff += block[0] - task.hour

                if time_diff < self.bmin:
                    return False

        #   ## check if after total_hours < wmax after adding block
            if self.total_hours + 1 > self.wmax:
                return False

            return True

        ## If there are other tasks assigned in the same day
        current_block = self.blocks[task.day]

        if current_block[0] <= task.hour <= current_block[1]:
            return True

        ## if the task fits within the existing range
        new_block_start = min(current_block[0], task.hour)
        new_block_end = max(current_block[1], task.hour)
        new_block_length = new_block_end - new_block_start


        ## otherwise check if new range after task is assigned is rmin feasible
        if new_block_length > self.bmax:
            return False

        ## check if new range after task is assigned is within bmax and wmax
        additional_hours = new_block_length - (current_block[1] - current_block[0] + 1)
        if additional_hours + self.total_hours > self.wmax:
            return False

        return True


    def assign_task(self, task):
        # // Implement Code Here
        self.tasks_assigned.append(task)

        if task.day not in self.blocks:
            self.blocks[task.day] = [task.hour, task.hour]
            self.total_hours += 1
        else:
            current_block = self.blocks[task.day]
            new_block_start = min(current_block[0], task.hour)
            new_block_end = max(current_block[1], task.hour)

            original_block_length = current_block[1] - current_block[0]
            new_block_length = new_block_end - new_block_start
            additional_hours = new_block_length - original_block_length

            self.blocks[task.day] = [new_block_start, new_block_end]
            self.total_hours += additional_hours

    def remove_task(self, task_id):
        # // Implement Code Here
        task_to_remove = None
        for task in self.tasks_assigned:
            if task.id == task_id:
                task_to_remove = task
                break
        if task_to_remove is None:
            return False

        self.tasks_assigned.remove(task_to_remove)

        day_task = [task for task in self.tasks_assigned if task.day == task_to_remove.day]
        if not day_task:
            original_block_length = self.blocks[task_to_remove.day][1] - self.blocks[task_to_remove.day][0] + 1
            self.total_hours -= original_block_length
            del self.blocks[task_to_remove.day]

            return True

        min_hour = min(task.hour for task in day_task)
        max_hour = max(task.hour for task in day_task)

        original_block = self.blocks[task_to_remove.day]
        original_block_length = original_block[1] - original_block[0] + 1
        new_block_length = max_hour - min_hour + 1
        hours_diff = abs(new_block_length - original_block_length)

        self.blocks[task_to_remove.day] = [min_hour, max_hour]
        self.total_hours -= hours_diff

        return True

    def get_objective(self):
        t = sum(x[1] - x[0] + 1 for x in self.blocks.values())
        return t * self.rate

    def __repr__(self):
        if len(self.blocks) == 0:
            return ""
        return "\n".join(
            [
                f"Worker {self.id}: Day {d} Hours {self.blocks[d]} Tasks {sorted([t.id for t in self.tasks_assigned if t.day == d])}"
                for d in sorted(self.blocks.keys())
            ]
        )


class Task(object):
    def __init__(self, data):
        self.id = data["t_id"]
        self.skill = data["skill"]
        self.day = data["day"]
        self.hour = data["hour"]


### PSP state class ###
# PSP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class PSP(State):
    def __init__(self, name, workers, tasks, alpha):
        """Initialize the PSP state
        Args:
            name::str
                name of the instance
            workers::[Worker]
                workers of the instance
            tasks::[Task]
                tasks of the instance
        """
        self.name = name
        self.workers = workers
        self.tasks = tasks
        self.Alpha = alpha
        # the tasks assigned to each worker, eg. [worker1.tasks_assigned, worker2.tasks_assigned, ..., workerN.tasks_assigned]
        self.solution = []
        self.unassigned = list(tasks)

    # Greedy algorithm implementation based on task clustering by time block
    def random_initialize(self, seed=None):
        """
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        """
        if seed is None:
            seed = 606

        random.seed(seed)
        # -----------------------------------------------------------
        # // Implement Code Here
        # // This should contain your construction heuristic for initial solution
        # // Use Worker class methods to check if assignment is valid
        # -----------------------------------------------------------

        # Sort tasks by day and hour
        sorted_tasks = sorted(self.tasks, key=lambda t: (t.day, t.hour))

        # Group tasks by day and skill
        tasks_by_day_skill = {}
        for task in sorted_tasks:
            key = (task.day, task.skill)

            if key not in tasks_by_day_skill:
                tasks_by_day_skill[key] = []

            tasks_by_day_skill[key].append(task)

        # Sort the groups by size (descending)
        # prefer day and skill combinations with more tasks
        sorted_groups = sorted(tasks_by_day_skill.items(), key=lambda x: len(x[1]), reverse=True)

        assigned_task_ids = set()

        for (day, skill), tasks in sorted_groups:
            tasks = [t for t in tasks if t.id not in assigned_task_ids]
            if not tasks:
                continue

            # Sort tasks by hour
            tasks.sort(key=lambda t: t.hour)

            # Find workers with the required skill and availability for the day
            suitable_workers = []
            for worker in self.workers:
                if skill in worker.skills and day in worker.available:
                    suitable_workers.append(worker)

            if not suitable_workers:
                continue

            # Sort workers by hourly rate (ascending)
            suitable_workers.sort(key=lambda w: w.rate)

            # Try to create optimal time blocks for each worker
            for worker in suitable_workers:
                # Find tasks that can be assigned to this worker
                assignable_tasks = [t for t in tasks if t.id not in assigned_task_ids and worker.can_assign(t)]

                if not assignable_tasks:
                    continue

                # Sort tasks by hour
                assignable_tasks.sort(key=lambda t: t.hour)

                # Group tasks into time blocks
                task_blocks = []
                current_task_block = [assignable_tasks[0]]

                for i in range(1, len(assignable_tasks)):
                    prev_task = assignable_tasks[i - 1]
                    curr_task = assignable_tasks[i]

                    # If tasks are continuous or close together, add to the current block
                    if curr_task.hour - prev_task.hour <= 3:
                        current_task_block.append(curr_task)
                    else:
                        # a new task block
                        task_blocks.append(current_task_block)
                        current_task_block = [curr_task]

                # Add the last task block
                if current_task_block:
                    task_blocks.append(current_task_block)

                for task_block in task_blocks:
                    can_assign_block = True
                    worker_copy = copy.deepcopy(worker)

                    for task in task_block:
                        if not worker_copy.can_assign(task):
                            can_assign_block = False
                            break
                        worker_copy.assign_task(task)

                    # Assign the entire task block
                    if can_assign_block:
                        for task in task_block:
                            worker.assign_task(task)
                            assigned_task_ids.add(task.id)
                            self.unassigned.remove(task)
                    # Try to assign tasks individually
                    else:
                        for task in task_block:
                            if task.id not in assigned_task_ids and worker.can_assign(task):
                                worker.assign_task(task)
                                assigned_task_ids.add(task.id)
                                self.unassigned.remove(task)

        # Try to assign remaining tasks individually
        remaining_tasks = list(self.unassigned)

        for task in sorted(remaining_tasks, key=lambda t: (t.day, t.hour)):
            # Find the best worker for the task
            best_worker = None
            min_cost = float("inf")

            for worker in self.workers:
                if worker.can_assign(task):
                    # Calculate additional cost
                    if task.day not in worker.blocks:
                        # a new block
                        cost = max(worker.rate, 50)
                    else:
                        # Extend existing block
                        original_block = worker.blocks[task.day]
                        original_length = original_block[1] - original_block[0] + 1

                        new_start = min(task.hour, original_block[0])
                        new_end = max(task.hour, original_block[1])
                        new_length = new_end - new_start + 1

                        cost = worker.rate * (new_length - original_length)

                    if cost < min_cost:
                        min_cost = cost
                        best_worker = worker

            if best_worker:
                best_worker.assign_task(task)
                self.unassigned.remove(task)

        self.solution = [worker.tasks_assigned for worker in self.workers]

        return self.objective()

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """Calculate the objective value of the state
        Return the total cost of each worker + unassigned cost
        """
        f1 = len(self.unassigned)
        f2 = sum(max(worker.get_objective(), 50) for worker in self.workers if worker.get_objective() > 0)
        return self.Alpha * f1 + f2

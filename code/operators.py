import random
import numpy as np
from psp import PSP


### Destroy operators ###
def destroy_random(current: PSP, random_state, destroy_factor=0.15):
    """Random Removal
    Randomly removes a percentage of assigned tasks from the solution.
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
        destroy_factor::float
            a factor controlling the percentage of tasks to remove (0.1-1.0)
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()

    # Fast collection of assigned tasks
    assigned_tasks = []
    for worker_idx, worker in enumerate(destroyed.workers):
        for task in worker.tasks_assigned:
            assigned_tasks.append((worker_idx, task))

    # Skip if no tasks are assigned
    if not assigned_tasks:
        return destroyed

    # Use destroy_factor to determine the removal percentage
    # Scale destroy_factor to get appropriate removal percentage (0.05-0.30)
    removal_percentage = 0.05 + destroy_factor * 0.25
    num_to_remove = max(1, int(removal_percentage * len(assigned_tasks)))
    num_to_remove = min(num_to_remove, len(assigned_tasks))

    # Use numpy for efficient selection
    if len(assigned_tasks) > num_to_remove:
        indices_to_remove = random_state.choice(len(assigned_tasks), num_to_remove, replace=False)
    else:
        indices_to_remove = np.arange(len(assigned_tasks))

    # Remove selected tasks
    for idx in indices_to_remove:
        worker_idx, task = assigned_tasks[idx]
        worker = destroyed.workers[worker_idx]
        if worker.remove_task(task.id):
            destroyed.unassigned.append(task)

    return destroyed


def destroy_time_blocks(current: PSP, random_state, destroy_factor=0.15):
    """Time Block Destruction
    Selects random time blocks and removes all tasks in those blocks.
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
        destroy_factor::float
            a factor controlling the number of blocks to destroy (0.1-1.0)
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()

    # Fast collection of blocks
    all_time_blocks = []
    for worker_idx, worker in enumerate(destroyed.workers):
        for day, block in worker.blocks.items():
            all_time_blocks.append((worker_idx, day, block))

    # Skip if no block exists
    if not all_time_blocks:
        return destroyed

    # Use destroy_factor to determine the number of blocks to remove
    # More blocks are selected with higher destroy_factor
    max_blocks = max(1, int(len(all_time_blocks) * destroy_factor))
    num_blocks = min(max_blocks, len(all_time_blocks))

    # Use numpy for efficient selection
    blocks_to_destroy = random_state.choice(len(all_time_blocks), num_blocks, replace=False)

    # For each selected block, remove all tasks in that block
    for idx in blocks_to_destroy:
        worker_idx, day, block = all_time_blocks[idx]
        worker = destroyed.workers[worker_idx]

        # Find out tasks in the block - directly check using original list
        for task in worker.tasks_assigned[:]:  # Use copy of list to allow modification
            if task.day == day and block[0] <= task.hour <= block[1]:
                if worker.remove_task(task.id):
                    destroyed.unassigned.append(task)

    return destroyed


def destroy_cost(current: PSP, random_state, destroy_factor=0.15):
    """Cost-Based Worker Removal
    Removes tasks from workers with high cost-to-task ratio.
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
        destroy_factor::float
            a factor controlling destruction intensity (0.1-1.0)
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()

    # Calculate cost efficiency for each worker
    worker_efficiency = []
    for idx, worker in enumerate(destroyed.workers):
        # Skip workers with no assigned tasks
        if not worker.tasks_assigned:
            continue

        # Calculate worker's total cost
        cost = max(worker.get_objective(), 50)

        # Calculate cost per task
        cost_per_task = cost / len(worker.tasks_assigned)
        worker_efficiency.append((idx, cost_per_task))

    # Skip if no workers have tasks
    if not worker_efficiency:
        return destroyed

    # Sort workers by cost efficiency (descending)
    worker_efficiency.sort(key=lambda x: x[1], reverse=True)

    # Use destroy_factor to determine the percentage of workers to select
    # and percentage of tasks to remove per worker
    worker_percentage = 0.1 + destroy_factor * 0.3  # 0.1-0.4
    task_removal_percentage = 0.4 + destroy_factor * 0.4  # 0.4-0.8
    
    num_workers = max(1, min(int(len(worker_efficiency) * worker_percentage), 5))
    selected_workers = [worker_efficiency[i][0] for i in range(num_workers)]

    # Remove a percentage of tasks from each selected worker
    for worker_idx in selected_workers:
        worker = destroyed.workers[worker_idx]
        tasks = worker.tasks_assigned[:]

        # Skip if worker has no tasks
        if not tasks:
            continue

        # Determine number of tasks to remove using destroy_factor
        num_to_remove = max(1, int(task_removal_percentage * len(tasks)))

        # Use numpy for efficient selection
        if len(tasks) > num_to_remove:
            indices_to_remove = random_state.choice(len(tasks), num_to_remove, replace=False)
            indices_to_remove = sorted(indices_to_remove, reverse=True)  # Sort for backward removal
        else:
            indices_to_remove = range(len(tasks) - 1, -1, -1)  # Reverse range

        # Remove selected tasks
        for idx in indices_to_remove:
            task = tasks[idx]
            if worker.remove_task(task.id):
                destroyed.unassigned.append(task)

    return destroyed


### Repair operators ###
def calculate_worker_cost(worker):
    """Helper function to quickly calculate a worker's current cost"""
    cost = 0
    for day, block in worker.blocks.items():
        block_length = block[1] - block[0] + 1
        block_cost = max(block_length * worker.rate, 50)
        cost += block_cost
    return cost


def calculate_cost_increase(worker, task):
    """Helper function to quickly estimate cost increase for adding a task"""
    # New block cost
    if task.day not in worker.blocks:
        return max(worker.rate, 50)

    # Existing block cost
    current_block = worker.blocks[task.day]
    current_length = current_block[1] - current_block[0] + 1
    current_cost = max(current_length * worker.rate, 50)

    # New block cost after addition
    new_start = min(current_block[0], task.hour)
    new_end = max(current_block[1], task.hour)
    new_length = new_end - new_start + 1
    new_cost = max(new_length * worker.rate, 50)

    return new_cost - current_cost


def repair_greedy(destroyed: PSP, random_state):
    """Greedy Insertion
    Tries to assign unassigned tasks based on minimal cost increase.
    Args:
        destroyed::PSP
            a PSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::PSP
            the PSP object after repairing
    """
    repaired = destroyed.copy()

    # Make a copy of unassigned tasks and shuffle it
    unassigned_tasks = repaired.unassigned[:]
    # Use numpy for efficient shuffling
    indices = np.arange(len(unassigned_tasks))
    random_state.shuffle(indices)
    unassigned_tasks = [unassigned_tasks[i] for i in indices]

    repaired.unassigned = []

    for task in unassigned_tasks:
        # Find all eligible workers
        best_worker = None
        best_cost_increase = float('inf')

        for worker in repaired.workers:
            if worker.can_assign(task):
                # Fast cost increase calculation without deepcopy
                cost_increase = calculate_cost_increase(worker, task)

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_worker = worker

        # If there are eligible workers, assign to the one with minimum cost increase
        if best_worker:
            best_worker.assign_task(task)
        else:
            repaired.unassigned.append(task)

    return repaired


def repair_time_blocks(destroyed: PSP, random_state):
    """Time Block Optimization Repair
    Prioritizes assigning tasks to existing time blocks to minimize new block creation.
    Args:
        destroyed::PSP
            a PSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::PSP
            the PSP object after repairing
    """
    repaired = destroyed.copy()

    # Group unassigned tasks by day using numpy for efficiency
    tasks_by_day = {}
    for task in repaired.unassigned:
        if task.day not in tasks_by_day:
            tasks_by_day[task.day] = []
        tasks_by_day[task.day].append(task)

    repaired.unassigned = []

    for day, day_tasks in tasks_by_day.items():
        # Shuffle tasks within each day
        indices = np.arange(len(day_tasks))
        random_state.shuffle(indices)
        day_tasks = [day_tasks[i] for i in indices]

        for task in day_tasks:
            # Initialize for best worker search
            best_score = float('inf')
            best_worker = None

            # Limit worker consideration to improve performance
            for worker in repaired.workers:
                if not worker.can_assign(task):
                    continue

                # Fast scoring without deepcopy
                if day in worker.blocks:
                    # Worker already has blocks on this day
                    block = worker.blocks[day]

                    # Task fit within existing block (best case)
                    if block[0] <= task.hour <= block[1]:
                        score = 1 * worker.rate
                    # Task extends existing block (second best)
                    elif task.hour == block[0] - 1 or task.hour == block[1] + 1:
                        score = 2 * worker.rate
                    # Extend block larger on the same day (third best)
                    else:
                        score = 3 * worker.rate
                else:
                    # New block on a new day (worst)
                    score = 4 * worker.rate

                if score < best_score:
                    best_score = score
                    best_worker = worker

            # Assign task to best worker if found
            if best_worker:
                best_worker.assign_task(task)
            else:
                repaired.unassigned.append(task)

    return repaired


def repair_regret(destroyed: PSP, random_state):
    """Regret-Based Insertion
    Assigns tasks based on regret value - the difference between best and second-best options.
    Args:
        destroyed::PSP
            a PSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::PSP
            the PSP object after repairing
    """
    repaired = destroyed.copy()

    unassigned_tasks = repaired.unassigned[:]
    repaired.unassigned = []

    # Limit the number of tasks to process per iteration for performance
    max_tasks_to_process = min(len(unassigned_tasks), 50)

    # Continue until all tasks are processed or reached limit
    while unassigned_tasks and max_tasks_to_process > 0:
        regret_values = []
        tasks_to_process = unassigned_tasks[:max_tasks_to_process]

        # Calculate regret value for each unassigned task
        for i, task in enumerate(tasks_to_process):
            # Find eligible workers with a limit for performance
            cost_increases = []
            count = 0

            for worker in repaired.workers:
                if worker.can_assign(task) and count < 10:  # Limit to 10 workers
                    # Fast cost calculation
                    cost_increase = calculate_cost_increase(worker, task)
                    cost_increases.append(cost_increase)
                    count += 1

                    # Stop early if we have enough workers
                    if count >= 10:
                        break

            # Sort cost increases using numpy
            if cost_increases:
                cost_increases = np.sort(cost_increases)

                # Calculate regret value (difference between best and second best)
                regret = 0
                if len(cost_increases) >= 2:
                    # K-regret where K=2
                    regret = cost_increases[1] - cost_increases[0]

                # Store task index and regret value
                regret_values.append((i, regret, cost_increases[0] if len(cost_increases) > 0 else float('inf')))

        # If no tasks can be assigned, break
        if not regret_values:
            repaired.unassigned.extend(unassigned_tasks)
            break

        # Select task with highest regret
        regret_values.sort(key=lambda x: (x[1], -x[2]), reverse=True)  # Sort by regret (high) then cost (low)
        task_idx = regret_values[0][0]
        task = tasks_to_process[task_idx]

        # Find the best worker for this task - reuse earlier computation
        best_worker = None
        best_cost_increase = float('inf')

        for worker in repaired.workers:
            if worker.can_assign(task):
                # Fast cost calculation
                cost_increase = calculate_cost_increase(worker, task)

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_worker = worker

        # Assign task to best worker if found
        if best_worker:
            best_worker.assign_task(task)
        else:
            repaired.unassigned.append(task)

        # Remove task from unassigned list
        unassigned_tasks.remove(task)

        # Decrease max tasks to process counter
        max_tasks_to_process -= 1

    # Add any remaining unprocessed tasks back to unassigned
    if unassigned_tasks:
        repaired.unassigned.extend(unassigned_tasks)

    return repaired
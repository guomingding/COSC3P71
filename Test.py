import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from Chromosome import Chromosone
from Gene import Gene


class ExperimentRunner:
    def __init__(self, test_case, population_size=250):
        self.test_case = test_case
        self.population_size = population_size
        self.load_data()

    def load_data(self):
        # Load courses
        self.courses = []
        with open(f"{self.test_case}/courses.txt", "r") as file:
            next(file)
            for line in file:
                name, professor, students, duration = line.strip().split(",")
                self.courses.append({
                    "name": name,
                    "prof": professor,
                    "students": int(students),
                    "dur": int(duration)
                })

        # Load rooms
        self.rooms = []
        with open(f"{self.test_case}/rooms.txt", "r") as file:
            next(file)
            for line in file:
                name, capacity = line.strip().split(",")
                self.rooms.append({
                    "name": name,
                    "capacity": int(capacity)
                })

        # Load timeslots
        self.timeslots = []
        with open(f"{self.test_case}/timeslots.txt", "r") as file:
            next(file)
            for line in file:
                day, hour = line.strip().split(",")
                self.timeslots.append({
                    "day": day,
                    "hour": int(hour)
                })

        # Set static variables for Chromosome class
        Chromosone.courses = self.courses
        Chromosone.rooms = self.rooms
        Chromosone.timeslots = self.timeslots

    def run_single_experiment(self, crossover_rate, mutation_rate, elitism_rate, max_generations=1000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize population
        population = []
        for _ in range(self.population_size):
            chrom = self.generate_chromosone()
            population.append(chrom)

        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Track metrics
        best_fitness_history = []
        avg_fitness_history = []
        generation = 0
        best_solution = None

        while generation < max_generations:
            population = self.evolve_population(population, elitism_rate, mutation_rate, crossover_rate)

            # Calculate metrics
            best_fitness = population[0].fitness
            avg_fitness = sum(c.fitness for c in population) / len(population)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if best_fitness == 1.0:
                best_solution = deepcopy(population[0])
                break

            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

            generation += 1

        if best_solution is None:
            best_solution = deepcopy(population[0])

        return {
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'best_solution': best_solution,
            'generations': generation
        }

    def generate_chromosone(self):
        classes = []
        unique_days = list(set(slot['day'] for slot in self.timeslots))

        for j in range(len(self.courses)):
            day_index = random.choice(unique_days)
            available_slots = [i for i, slot in enumerate(self.timeslots) if slot["day"] == day_index]
            time_index = random.choice(available_slots)

            suitable_rooms = [room for room in self.rooms if room["capacity"] >= self.courses[j]["students"]]
            if not suitable_rooms:
                raise ValueError(f"No suitable room found for course {self.courses[j]['name']}")

            room_index = self.rooms.index(random.choice(suitable_rooms))
            classes.append(Gene(j, time_index, day_index, room_index))

        return Chromosone(classes)

    def tournament_selection(self, population):
        tournament = random.sample(population, 2)
        return sorted(tournament, key=lambda x: x.fitness, reverse=True)[0]

    def uniform_crossover(self, parent1, parent2):
        child1 = Chromosone()
        child2 = Chromosone()

        for i in range(len(parent1.class_list)):
            if random.random() < 0.5:
                child1.add_gene(deepcopy(parent1.class_list[i]))
                child2.add_gene(deepcopy(parent2.class_list[i]))
            else:
                child1.add_gene(deepcopy(parent2.class_list[i]))
                child2.add_gene(deepcopy(parent1.class_list[i]))

        child1.update_fitness()
        child2.update_fitness()
        return child1, child2

    def evolve_population(self, population, elitism_rate, mutation_rate, crossover_rate):
        new_population = []
        population_size = len(population)
        elitism_count = max(1, int(elitism_rate * population_size))

        # Add elite chromosomes
        new_population.extend(deepcopy(population[:elitism_count]))

        # Generate rest of the population
        while len(new_population) < population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            if random.random() < crossover_rate:
                child1, child2 = self.uniform_crossover(parent1, parent2)

                if random.random() < mutation_rate:
                    child1.mutate_class_l(1.0)
                if random.random() < mutation_rate:
                    child2.mutate_class_l(1.0)

                new_population.extend([child1, child2])
            else:
                new_population.extend([deepcopy(parent1), deepcopy(parent2)])

        # Trim if needed
        new_population = new_population[:population_size]
        return sorted(new_population, key=lambda x: x.fitness, reverse=True)


def run_experiments():
    # Configuration parameters
    test_cases = ['t1', 't2']
    experiment_configs = [
        {'crossover_rate': 1.0, 'mutation_rate': 0.0, 'elitism_rate': 0.01},
        {'crossover_rate': 1.0, 'mutation_rate': 0.1, 'elitism_rate': 0.01},
        {'crossover_rate': 0.9, 'mutation_rate': 0.0, 'elitism_rate': 0.01},
        {'crossover_rate': 0.9, 'mutation_rate': 0.1, 'elitism_rate': 0.01},
        {'crossover_rate': 0.95, 'mutation_rate': 0.2, 'elitism_rate': 0.01},  # Custom settings
    ]
    num_runs = 5

    results = {}

    for test_case in test_cases:
        runner = ExperimentRunner(test_case)
        results[test_case] = {}

        for config in experiment_configs:
            config_name = f"CR{config['crossover_rate']}_MR{config['mutation_rate']}_ER{config['elitism_rate']}"
            results[test_case][config_name] = []

            print(f"\nRunning experiments for {test_case} with configuration: {config_name}")

            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs}")
                seed = random.randint(1, 10000)
                result = runner.run_single_experiment(
                    crossover_rate=config['crossover_rate'],
                    mutation_rate=config['mutation_rate'],
                    elitism_rate=config['elitism_rate'],
                    seed=seed
                )
                result['seed'] = seed
                results[test_case][config_name].append(result)

    return results


def analyze_results(results):
    analysis = {}

    for test_case in results:
        analysis[test_case] = {}

        for config in results[test_case]:
            runs = results[test_case][config]

            # Calculate statistics
            best_fitness_values = [run['best_solution'].fitness for run in runs]
            generations = [run['generations'] for run in runs]

            analysis[test_case][config] = {
                'min_fitness': min(best_fitness_values),
                'max_fitness': max(best_fitness_values),
                'mean_fitness': np.mean(best_fitness_values),
                'median_fitness': np.median(best_fitness_values),
                'std_fitness': np.std(best_fitness_values),
                'avg_generations': np.mean(generations),
                'std_generations': np.std(generations)
            }

    return analysis


def plot_results(results):
    for test_case in results:
        for config in results[test_case]:
            runs = results[test_case][config]

            # Plot average fitness over generations
            plt.figure(figsize=(12, 6))

            # Calculate average fitness histories
            max_gen = max(len(run['best_fitness_history']) for run in runs)
            avg_best_fitness = np.zeros(max_gen)
            avg_pop_fitness = np.zeros(max_gen)
            num_runs = len(runs)

            for run in runs:
                for gen, (best_fit, avg_fit) in enumerate(zip(run['best_fitness_history'],
                                                              run['avg_fitness_history'])):
                    avg_best_fitness[gen] += best_fit
                    avg_pop_fitness[gen] += avg_fit

            avg_best_fitness /= num_runs
            avg_pop_fitness /= num_runs

            plt.plot(avg_best_fitness, label='Average Best Fitness')
            plt.plot(avg_pop_fitness, label='Average Population Fitness')

            plt.title(f'{test_case} - {config}')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'./expriment_res/results_{test_case}_{config}.png')
            plt.close()


if __name__ == "__main__":
    results = run_experiments()
    analysis = analyze_results(results)
    plot_results(results)

    # Save results to file
    with open('./expriment_res/experiment_results.txt', 'w') as f:
        for test_case in analysis:
            f.write(f"\nResults for {test_case}:\n")
            for config in analysis[test_case]:
                f.write(f"\nConfiguration: {config}\n")
                for metric, value in analysis[test_case][config].items():
                    f.write(f"{metric}: {value:.4f}\n")

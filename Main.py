import random

from Chromosome import Chromosone


def evolve_population(population, elitismRate, mutationRate, crossoverRate, gen):
    new_population = []
    elites = population[:int(elitismRate * len(population))]

    new_parents = []
    for i in range(int(len(population) / 2)):
        tournament = random.sample(population, 2)
        res = sorted(tournament, key=lambda chrom: chrom.fitness, reverse=True)
        new_parents.append(res[0])
        new_parents.append(res[1])

    random.shuffle(new_parents)

    for i in range(0, len(new_parents), 2):
        if i + 1 < len(new_parents) and random.random() < crossoverRate:
            child1 = Chromosone()
            child2 = Chromosone()
            mask = ""
            for i in range(len(new_parents[i].class_list)):
                mask += str(random.choice([0, 1]))

            for i in range(len(mask)):
                child1.add_gene(new_parents[i].class_list[i] if mask[i] == '1' else new_parents[i + 1].class_list[i])
                child2.add_gene(new_parents[i + 1].class_list[i] if mask[i] == '1' else new_parents[i].class_list[i])
                child1.update_fitness()
                child2.update_fitness()
        else:
            child1, child2 = new_parents[i], new_parents[i + 1]

        child1.mutate_class_l(mutationRate)
        child2.mutate_class_l(mutationRate)

        new_population.append(child1)
        new_population.append(child2)

    new_population.extend(elites)
    new_population = sorted(new_population, key=lambda chrom: chrom.fitness, reverse=True)
    return new_population[:len(population)]

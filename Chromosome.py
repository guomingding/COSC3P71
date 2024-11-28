import random


class Chromosone:
    courses = None
    rooms = None
    timeslots = None

    def __init__(self, class_list=None):
        if class_list is None:
            self.class_list = []
        else:
            self.class_list = class_list
        self.fitness = self.calc_fitness()

    def add_gene(self, gene):
        self.class_list.append(gene)

    def update_fitness(self):
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        room_conflicts_m = [[0] * len(self.class_list) for _ in range(len(self.class_list))]
        prof_conflicts_m = [[0] * len(self.class_list) for _ in range(len(self.class_list))]

        courses = Chromosone.courses
        rooms = Chromosone.rooms

        for i, class1 in enumerate(self.class_list):
            if courses[class1.ds['course']]['students'] > rooms[class1.ds['room']]["capacity"]:
                room_conflicts_m[i][i] += 2

            for j in range(i + 1, len(self.class_list)):
                class2 = self.class_list[j]

                if class1.ds['room'] == class2.ds['room'] and class1.ds['day'] == class2.ds['day'] and class1.ds['time'] == class2.ds['time']:
                    room_conflicts_m[i][j] += 3
                    room_conflicts_m[j][i] += 3

                if courses[class1.ds['course']]['prof'] == courses[class2.ds['course']]['prof'] and class1.ds['day'] == class2.ds['day'] and class1.ds['time'] == class2.ds['time']:
                    prof_conflicts_m[i][j] += 1
                    prof_conflicts_m[j][i] += 1

        total_conflicts = sum(sum(row) for row in room_conflicts_m) + sum(sum(row) for row in prof_conflicts_m)
        return 1 / (1 + total_conflicts) if total_conflicts > 0 else 1

    def mutate_class_l(self, mutationRate):
        for i in range(len(self.class_list)):
            if random.random() < mutationRate:
                self.class_list[i].mutate(Chromosone.courses, Chromosone.rooms, Chromosone.timeslots)
        self.update_fitness()

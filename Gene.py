import random


class Gene:
    def __init__(self, course, time, day, room):
        self.ds = {'course': course, 'day': day, 'time': int(time), 'room': room}

    def mutate(self, courses, rooms, times):
        self.ds['day'] = random.choice(list(set(slot['day'] for slot in times)))
        self.ds['time'] = random.choice([i for i, slot in enumerate(times) if slot["day"] == self.ds['day']])

        suitable_rooms = [room for room in rooms if room["capacity"] >= courses[self.ds['course']]["students"]]
        if not suitable_rooms:
            raise ValueError(f"No suitable room found for course {courses[self.ds['course']]['name']} with {courses[self.ds['course']]['students']} students.")
        self.ds['room'] = rooms.index(random.choice(suitable_rooms))

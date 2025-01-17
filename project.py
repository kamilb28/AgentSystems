import numpy as np
import itertools
import pygame
import random
import copy

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

CARS = {
    "types": ["car", "bus", "taxi", "lorry"],
    "probability": [0.8, 0.1, 0.05, 0.05]
}


class CarType:
    def __init__(self, type=None):
        if type is None:
            self.type = random.choices(
                CARS["types"], weights=CARS["probability"], k=1)[0]
        else:
            self.type = type

    def get_length(self):
        if self.type == "car" or self.type == "taxi":
            return 5  # 5 meters
        if self.type == "bus":
            return 12  # 12 meters
        if self.type == "lorry":
            return 16   # 15 meters
        return 5

    def get_color(self):
        if self.type == "car":
            return pygame.Color("blue")
        if self.type == "taxi":
            return pygame.Color("yellow")
        if self.type == "bus":
            return pygame.Color("green")
        if self.type == "lorry":
            return pygame.Color("red")
        return pygame.Color("orange")

    def get_acceleration(self):
        if self.type == "car" or self.type == "taxi":
            return 2  # 2 m/s^2
        if self.type == "bus":
            return 1.5  # 1.5 m/s^2
        if self.type == "lorry":
            return 1.2   # 1.2 m/s^2
        return 1

    def get_max_speed(self):
        if self.type == "car" or self.type == "taxi":
            return 25  # ~ 90kmh
        if self.type == "bus":
            return 22  # ~ 80kmh
        if self.type == "lorry":
            return 19   # ~ 70kmh
        return 19


class Car:
    def __init__(self, min_speed_on_lane, max_speed_on_lane, position):
        self.car_type = CarType()
        self.min_speed = 0
        self.max_speed = self.car_type.get_max_speed()
        self.speed = random.randint(min_speed_on_lane, min(
            max_speed_on_lane, self.max_speed))
        self.position = position
        self.car_size = self.car_type.get_length()
        self.lateral_position = 0  # Position between lanes (0 = centered)
        self.target_lane = None  # Target lane number for changing lanes
        self.source_lane = None
        self.is_changing_lane = False
        self.lateral_speed = 40  # Much faster lane changes

    def increase_speed(self, delta_time):
        if self.speed < self.max_speed:
            self.speed += self.car_type.get_acceleration() * delta_time

    def set_speed(self, new_speed):
        if new_speed > self.speed or new_speed > self.max_speed:
            return
        self.speed = max(0, new_speed)

    def randomize_speed(self, probability):
        if self.speed > 4 and random.random() < probability:
            self.speed -= random.choice(range(2, 4))
            self.speed = max(1, self.speed)

    def move(self, delta_time):
        self.position += self.speed * delta_time

        # Handle lateral movement for lane changes
        if self.is_changing_lane:
            target_lateral = 30 if self.target_lane > self.source_lane else -30
            move_direction = 1 if target_lateral > self.lateral_position else -1
            self.lateral_position += self.lateral_speed * delta_time * move_direction

            # Check if we've reached or passed the target position
            if (move_direction > 0 and self.lateral_position >= target_lateral) or \
               (move_direction < 0 and self.lateral_position <= target_lateral):
                self.lateral_position = 0
                self.is_changing_lane = False
                return True
        return False


class CROSSROAD:
    class LANE:
        def __init__(self, lenght, start_point, new_car_probabulity, min_speed=None, max_speed=None):
            self.lenght = lenght
            self.start_point = start_point
            self.new_car_probabulity = new_car_probabulity
            self.min_speed = min_speed
            self.max_speed = max_speed

    lane1 = LANE(lenght=400, start_point=0, new_car_probabulity=.01,
                 min_speed=19, max_speed=25)  # m/s into km/h ~ 70km/h - 90km/h
    lane2 = LANE(lenght=400, start_point=0,
                 new_car_probabulity=.03, min_speed=19, max_speed=25)
    lane3 = LANE(lenght=400, start_point=0, new_car_probabulity=.02,
                 min_speed=11, max_speed=17)  # ~ 40 km/h - 60km/h
    lane4 = LANE(lenght=250, start_point=0, new_car_probabulity=.1,
                 min_speed=11, max_speed=14)  # ~ 40km/h - 50km/h
    left_turn_line = LANE(lenght=200, start_point=200, new_car_probabulity=0)

    class LIGHTS:
        class LIGHT_CYCLE:
            def __init__(self, time, straight, turn_left):
                self.time = time
                self.straight = straight
                self.turn_left = turn_left
        cycle = [
            LIGHT_CYCLE(time=45, straight=True, turn_left=False),
            LIGHT_CYCLE(time=15, straight=True, turn_left=True),
            LIGHT_CYCLE(time=45, straight=False, turn_left=False),
        ]
        last_time_change = 0

        def curr_cycle(self):
            return self.cycle[0]

        def next(self):
            last_cycle = self.cycle.pop(0)
            self.cycle.append(last_cycle)


class Model:
    def __init__(self):
        self.time = 0
        self.road = [CROSSROAD.left_turn_line, CROSSROAD.lane1,
                     CROSSROAD.lane2, CROSSROAD.lane3, CROSSROAD.lane4]
        self.cars = [[], [], [], [], []]  # cars on each line
        self.cars_to_change_line = [[] for _ in range(
            len(self.cars))]  # cars that wants to change lane
        self.lights = CROSSROAD.LIGHTS()

        self.car_img = pygame.image.load('car-icon.png')
        self.bus_img = pygame.image.load('bus-icon.png')
        self.taxi_img = pygame.image.load('taxi-icon.png')
        self.lorry_img = pygame.image.load('lorry-icon.png')

    def randomlyAddNewCar(self):
        for i, line in enumerate(self.road):
            # at least 5m delay beetween spowning cars
            if random.random() < line.new_car_probabulity and (not self.cars[i] or self.cars[i][-1].position > 20):
                self.cars[i].append(
                    Car(line.min_speed, line.max_speed, position=0))

    # make them WANT to change lane (lane can be taken)
    def makeCarChangeLane(self, line_from, car_index, car_position):
        line_offset = random.choice([-1, 1])
        line_to = line_from + line_offset
        # check if this not a wrong line to change (out of index) //TODO implement change to first lane
        if not (1 <= line_to < len(self.cars)) or \
                not (self.road[line_to].start_point < car_position < self.road[line_to].start_point + self.road[line_to].lenght):     # cant change to shorter lanes
            line_to = line_from - line_offset
        car = self.cars[line_from][car_index]
        if not car.is_changing_lane:  # Only add if car isn't already changing lanes
            self.cars_to_change_line[line_from].insert(0, (car_index, line_to))

    def update(self, delta_time):
        self.time += 1
        self.cars_to_change_line = [[] for _ in range(len(self.cars))]

        # Change Lights color
        if (self.time - self.lights.last_time_change) % (self.lights.curr_cycle().time // delta_time) == 0:
            self.lights.last_time_change = self.time
            self.lights.next()

        # Acceleration
        for car_line in self.cars:
            for car in car_line:
                car.increase_speed(delta_time)

        # First car slow down on red light
        for line_number, lane_with_cars in enumerate(self.cars):
            if not lane_with_cars:
                continue
            car = lane_with_cars[0]  # first car
            # stop 5 meters in front of line end
            distance_to_lights = self.road[line_number].lenght - \
                car.position - 5
            if distance_to_lights < car.speed:
                if line_number == 0 and not self.lights.curr_cycle().turn_left:
                    car.set_speed(distance_to_lights)
                elif line_number == 4:  # last line ends before lights
                    car.set_speed(distance_to_lights)
                    # if speed is under 50kmh there is 5% chance it will WANT to change lane
                    if car.speed < 14 and random.random() < 0.05:
                        self.makeCarChangeLane(
                            line_from=line_number, car_index=0, car_position=car.position)
                elif not self.lights.curr_cycle().straight:
                    car.set_speed(distance_to_lights)

        # Slowing Down
        cars_to_change_line = [[] for _ in range(len(self.cars))]
        for lane_number, lane_with_cars in enumerate(self.cars):
            # enumerate cars but without the first one on the lane
            for i in range(1, len(lane_with_cars)):
                car = lane_with_cars[i]
                next_car = lane_with_cars[i-1]
                distance = (next_car.position - next_car.car_size) - \
                    car.position - .5  # 50cm safe distance beetween cars
                if distance < car.speed:
                    car.set_speed(distance)
                    # if car slowed down and speed is under 50kmh there is 1% chance it will WANT to change lane
                    if car.speed < 14 and random.random() < 0.01:
                        self.makeCarChangeLane(
                            line_from=lane_number, car_index=i, car_position=car.position)

        # Process lane change
        for lane_number, lane_changes in enumerate(self.cars_to_change_line):
            for car_index, target_lane in lane_changes:
                car = self.cars[lane_number][car_index]

                # Check if target lane is free
                target_lane_cars = self.cars[target_lane]
                safe_distance = 15  # Increased safe distance to ensure comfort

                can_change_lane = True
                # Check target lane
                for other_car in target_lane_cars:
                    # Calculate the space needed
                    distance_front = other_car.position - car.position
                    distance_back = car.position - \
                        (other_car.position + other_car.car_size)

                    # Check if car is too close either from front or back
                    if abs(distance_front) < safe_distance or abs(distance_back) < safe_distance:
                        can_change_lane = False
                        break

                    # Additional check for cars that are changing lanes
                    if other_car.is_changing_lane:
                        can_change_lane = False
                        break

                if can_change_lane and not car.is_changing_lane:
                    # Start lane change
                    car.is_changing_lane = True
                    car.source_lane = lane_number
                    car.target_lane = target_lane
                    car.lateral_position = 0

        # Randomization
        # for car_line in self.cars:
            # for car in car_line:
                # car.randomize_speed(probability=0.05 * delta_time) # I want have 5% probability in one secound

        # Car motion and lane change completion
        # List of (car, from_lane, to_lane) for completed lane changes
        cars_to_move = []

        for line_index, car_line in enumerate(self.cars):
            # Iterate backwards to safely remove
            for car_index in range(len(car_line) - 1, -1, -1):
                car = car_line[car_index]
                lane_change_complete = car.move(delta_time)

                if lane_change_complete:
                    # Queue the car for moving to its target lane
                    cars_to_move.append(
                        (car, car.source_lane, car.target_lane))
                    car_line.pop(car_index)
                elif car.position >= self.road[line_index].lenght:
                    car_line.pop(car_index)

        # Process completed lane changes
        for car, from_lane, to_lane in cars_to_move:
            self.cars[to_lane].append(car)
            self.cars[to_lane].sort(key=lambda c: c.position, reverse=True)

        self.randomlyAddNewCar()

    def draw(self, screen, w_width, w_height):
        black = (0, 0, 0)
        white = (255, 255, 255)
        grey = (128, 128, 128)
        block_width = w_width / \
            max(self.road, key=lambda lane: lane.lenght).lenght
        block_height = 30

        # draw road
        pos_x = 0
        pos_y = (w_height-(5*block_height))//2

        for line_number, line in enumerate(self.road):
            # LANES
            rect = pygame.Rect(
                pos_x + (block_width * line.start_point),
                pos_y + (block_height * line_number),
                block_width * line.lenght,
                block_height
            )
            pygame.draw.rect(screen, grey, rect, 0)

        for line_num, car_line in enumerate(self.cars):
            for car in car_line:
                # CARS
                x = (block_width) * (car.position - car.car_size)
                y = pos_y + (block_height * line_num)
                if car.is_changing_lane:
                    y += car.lateral_position
                rect = pygame.Rect(
                    x, y, int(block_width * car.car_size), block_height)
                pygame.draw.rect(screen, car.car_type.get_color(), rect, 0)
                if car.car_type.type == "bus":
                    screen.blit(pygame.transform.scale(
                        self.bus_img, (int(block_width * car.car_size), block_height)), rect)
                elif car.car_type.type == "taxi":
                    screen.blit(pygame.transform.scale(
                        self.taxi_img, (int(block_width * car.car_size), block_height)), rect)
                elif car.car_type.type == "lorry":
                    screen.blit(pygame.transform.scale(
                        self.lorry_img, (int(block_width * car.car_size), block_height)), rect)
                else:
                    screen.blit(pygame.transform.scale(
                        self.car_img, (int(block_width * car.car_size), block_height)), rect)

        # last line dosent have lights
        for line_number, line in enumerate(self.road[:-1]):
            # LIGHTS
            x = w_width - 20
            y = pos_y + (block_height * line_number)
            rect = pygame.Rect(x, y, 20, block_height)
            if (self.lights.curr_cycle().turn_left and line_number == 0) or (self.lights.curr_cycle().straight and line_number != 0):
                pygame.draw.rect(screen, pygame.Color("green"), rect, 0)
            else:
                pygame.draw.rect(screen, pygame.Color("red"), rect, 0)

        pygame.display.flip()


def run():
    import time
    pygame.init()
    w_width = 1800
    w_height = 300
    screen = pygame.display.set_mode([w_width, w_height])
    model = Model()
    screen.fill((255, 255, 255))

    clock = pygame.time.Clock()  # Create a clock to control frame rate

    running = True
    stop = False
    model.draw(screen, w_width, w_height)
    while running:
        # Limit to 30 FPS and get delta_time in seconds
        delta_time = clock.tick(60) / 125.0
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    stop = not stop

                if event.key == K_LEFT:
                    model.lights.next()

        if not stop:
            model.update(delta_time)  # Update the model
            model.draw(screen, w_width, w_height)  # Redraw the screen
            pygame.display.flip()  # Update the display

    pygame.quit()


run()

import random
import numpy as np
from easydict import EasyDict as edict


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"

class DropMemoryBank:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS',
                 category_uniform=True):
        # type contains: none, uniform, uncertainty, confidence, HUS, uncertainty_confidence, uncertainty_uniform, UHUS
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = max(self.capacity / self.num_class, 1)

        self.data = [[] for _ in range(self.num_class)]
        self.confidence_threshold = confidence_threshold
        self.uncertain_threshold = uncertainty_threshold
        self.type = type
        self.category_uniform = category_uniform

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)
        return classes

    def get_non_empty_classes(self):
        per_class_dist = self.per_class_dist()
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied > 0:
                classes.append(i)
        return classes

    def add_instance(self, instance):
        # assert (len(instance) == 3)
        self.add_age()
        x, prediction, uncertainty, confidence = instance['data'], instance['prediction'], instance['uncertainty'], \
            instance['confidence']
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=1)

        if self.get_occupancy() < self.capacity:
            self.data[prediction].append(new_item)
        else:
            if self.remove_instance(instance):
                self.data[prediction].append(new_item)

    def remove_instance(self, instance):
        if self.type == 'none':
            pass

        from collections import namedtuple
        if instance['confidence'] >= self.confidence_threshold and instance['uncertainty'] <= self.uncertain_threshold:
            if self.category_uniform:
                class_index = random.choice(self.get_majority_classes())
            else:
                class_index = random.choice(self.get_non_empty_classes())
            # randomly drop one instance from the class
            self.data[class_index].pop(random.randint(0, len(self.data[class_index]) - 1))
            return True
        else:
            return False

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []
        tmp_uncertainty = []
        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)
                tmp_uncertainty.append(item.uncertainty)

        return tmp_data, tmp_uncertainty

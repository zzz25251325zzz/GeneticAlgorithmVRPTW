import math
import random

class City(object):
   def __init__(self, city_id, x, y, demand, lower, upper, service):
      self.city_id = city_id
      self.x = x
      self.y = y
      self.demand = demand
      self.upper = upper
      self.lower = lower
      self.service = service
      
   def __eq__(self, other):
      return self.city_id == other.city_id
   
   def __ne__(self, other):
      return not self.__eq__(other)

def dist(s, e):
    return math.sqrt((s.x - e.x) ** 2 + (s.y - e.y) ** 2)

class TourManager(object):
   def __init__(self, capacity):
      self.destination_cities = []
      self.car_limit = capacity
      self.cache = None

   def calculate_distance(self):
      self.cache = []
      for i in range(len(self.destination_cities)):
         tmp = []
         for j in range(len(self.destination_cities)):
            d = dist(self.destination_cities[i], self.destination_cities[j])
            tmp.append(d)
         self.cache.append(tmp)
   
      
def init(fileName):
   with open(fileName) as file:
      content = file.readlines() 
      content = filter(None, [x.strip() for x in content]) 
      cities = int(content[0])
      car_limit = int(content[1])
      tour_manager = TourManager(car_limit)

      for i in range(0, cities + 1):
         tmp = content[2 + i].split()
         x = float(tmp[1])
         y = float(tmp[2])
         demand = float(tmp[3])
         lower = float(tmp[4])
         upper = float(tmp[5])
         service = float(tmp[6])
         tour_manager.destination_cities.append(City(i, x, y, demand, lower, upper, service))
      tour_manager.calculate_distance()
      return tour_manager

from init import *
import ga_method as ga
import matplotlib.pyplot as plt
import numpy as np
import sys
from parmap import parmap

def process_result(result, draw, iterations, debug_time=False):        
    min_sol = None
    best_stats = None
    scores = []
    num_vehicles = []
    min_score = 100000
    min_vehicles = 1000
    for i in range(iterations):
        sol = result[i][0]
        stats = result[i][1]
        
        s = sol.score()
        v = len(sol.routes)
        scores.append(s)
        num_vehicles.append(v)
        if v < min_vehicles:
            min_vehicles = v
            min_score = s
            min_sol = sol
            best_stats = stats
        elif v == min_vehicles:
            if s < min_score:
                min_score = s
                min_sol = sol
                best_stats = stats
                
    print "GA best solution:"
    min_sol.print_sol()
    if (debug_time):
        time = min_sol.calculate_avail_time(min_sol.routes)
        min_sol.print_avail_time(min_sol.routes, time)
    print "GA average solution: %.2f/%.2f" % (sum(scores) / len(scores), sum(num_vehicles) / len(num_vehicles))
    
    if (draw):
        best, aver = zip(*best_stats)
        x = range(len(best_stats))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        bscore, blen = zip(*best)
        ascore, alen = zip(*aver)
        ax2.plot(x, blen, 'b^:')
        ax2.plot(x, alen, 'r^:')
        ax.plot(x, bscore, 'b-')
        ax.plot(x, ascore, 'r-')
        plt.show()
    

def main():
    num_args = len(sys.argv)
    if num_args < 2:
        print "Usage: python main.py [instance] [plot] [iterations]"
        sys.exit()

    f = sys.argv[1]
    draw = True
    iterations = 10
    if num_args > 3:    
        draw = (sys.argv[2] == "True")
        iterations = int(sys.argv[3])
        
    tour_manager = init(f)

    def run(num):
        print "Running iteration %d" % (num + 1)
        return ga.ga_sol(tour_manager, 50, 50, 300)

    result = map(run, range(iterations))
    process_result(result, draw, iterations)

if __name__=='__main__':
    main()

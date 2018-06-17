# -*- coding: utf-8 -*-
import random
import math
import copy
import functools
from timeit import default_timer as timer
import numpy as np

#paper: http://old.mii.lt/files/mii_dis_2014_vaira.pdf


def dist(s, e):
    return math.sqrt((s.x - e.x) ** 2 + (s.y - e.y) ** 2)

class Solution(object):
    def __init__(self, cities, capacity, cache):
        self.routes = []
        self.unsolved = []
        self.nodes = cities[1:]
        self.start = cities[0]
        self.capacity = capacity
        self.cache = cache

    def dist(self, s, e):
        return self.cache[s.city_id][e.city_id]
    
    def calculate_waiting_time(self, arrival, node):
        if arrival <= node.lower:
            return node.lower - arrival
        return 0
    
    #kiểm tra constraint: nếu add node vào route thì có gì bị vi phạm không.
    def satisfied(self, arc, node, route, avail):
        current = sum([self.get_city(x).demand for x in route])
        cap_sat = current + node.demand <= self.capacity
        if not cap_sat:
            return cap_sat
        
        if arc[3] == 0:
            leaving_time = 0
        else:
            leaving_time = avail[arc[3] - 1][1]
        new_time_arrival = self.dist(arc[0], node) + leaving_time
        new_time_sat = new_time_arrival <= node.upper
        if not new_time_sat:
            return new_time_sat
        new_waiting_time = self.calculate_waiting_time(new_time_arrival, node)
        added_time = self.dist(arc[0], node) + new_waiting_time + node.service + self.dist(node, arc[1])
        added_time = self.dist(arc[0], node) + new_waiting_time + node.service + self.dist(node, arc[1]) - self.dist(arc[0], arc[1])
        for a in avail[arc[3]:]:
            if added_time > a[3]:
                return False
            if added_time > a[2]:
                added_time = added_time - a[2]
            else:
                break
            
        return True

    def get_city(self, idx):
        if idx != 0:
            return self.nodes[idx - 1]
        return self.start
        
    def make_arcs(self, ri, i = 0, index=False):
        if len(ri) == 0: return []
        if index:
            r = [self.get_city(x) for x in ri]
        else:
            r = ri
        return list(zip([self.start] + r, r + [self.start], [i]*(len(r) + 1), range(len(r) + 1)))

    #nếu thêm city ins vào giữa arc (a, b) thì tổng độ dài quãng đường mới là bao nhiêu
    def cost(self, arc, ins):
        return self.dist(arc[0], ins) + self.dist(ins, arc[1]) - self.dist(arc[0], arc[1])

    #tổng độ dài của route của tất cả các xe là bao nhiêu 
    def score(self):
        score = 0
        for r in self.routes:
            k = list(zip([self.start] + r, r + [self.start]))
            score = score + sum(map(lambda x: self.dist(x[0], x[1]), k))
        return score

    def calculate_time(self, node, arc, leave):
        travel = self.dist(arc[0], node)
        arrival_time = leave + travel
        waiting_time = self.calculate_waiting_time(arrival_time, node)
        leaving_time = arrival_time + waiting_time + node.service
        avail_time = node.upper - arrival_time
        return [arrival_time, leaving_time, waiting_time, avail_time]
    
    def update_avail_time(self, routes, node, arc, old_avail):
        if arc[2] == len(routes):
            time_new_node = self.calculate_time(node, arc, 0)
            back_arc = [arc[1], arc[0], arc[2], arc[3]]
            time_back = self.calculate_time(self.start, back_arc, time_new_node[1])
            old_avail.append([time_new_node, time_back])
        else:
            if arc[3] == 0:
                leaving_time = 0
            else:
                leaving_time = old_avail[arc[2]][arc[3] - 1][1]
            new_node_time = self.calculate_time(node, arc, leaving_time)
            added_time = new_node_time[1] + self.dist(node, arc[1]) - old_avail[arc[2]][arc[3]][0]
            for i in range(len(old_avail[arc[2]])):
                if i < arc[3]:
                    continue
                old_avail[arc[2]][i][0] += added_time
                if added_time > old_avail[arc[2]][i][2]:
                    next_added_time = added_time - old_avail[arc[2]][i][2]
                    old_avail[arc[2]][i][2] = 0
                    old_avail[arc[2]][i][1] += next_added_time
                    old_avail[arc[2]][i][3] -= added_time
                    added_time = next_added_time
                else:
                    old_avail[arc[2]][i][2] -= added_time
                    old_avail[arc[2]][i][3] -= added_time
                    break
            old_avail[arc[2]].insert(arc[3], new_node_time)
        return old_avail

    def calculate_avail_time(self, rs):
        avail = []
        for i in range(len(rs)):
            all_arcs = self.make_arcs(rs[i], i)
            last_leave = 0
            avail_i = []
            for a in all_arcs:
                t = self.calculate_time(a[1], a, last_leave)
                last_leave = t[1]
                avail_i.append(t)
            avail.append(avail_i)
        return avail
                
    def print_avail_time(self, rs, avail):
        if len(rs) != len(avail):
            print "Different length: %d %d" % (len(rs), len(avail))
        for i in range(len(rs)):
            if len(rs[i]) + 1 != len(avail[i]):
                print "Different route length: %d %d %d" % (i, len(rs[i]), len(avail[i]))
            for j in range(len(rs[i])):
                print "City %d [%.2f %.2f]: A:%.2f W:%.2f L:%.2f %.2f" % (rs[i][j].city_id, rs[i][j].lower, rs[i][j].upper, avail[i][j][0], avail[i][j][2], avail[i][j][1], avail[i][j][3])
            print "Reach depot at: %.2f, avail: %.2f, wait: %.2f, leave: %.2f" % (avail[i][-1][0], avail[i][-1][3], avail[i][-1][2], avail[i][-1][1])
            
    #insert các nodes trong nodes_orig vào rs_orig
    #policy: chọn một node random trong nodes_orig, sau đó chọn arcs không vi phạm constraints nào khi add node vào giữa, và có cost(arc, node) nhỏ nhất. 
    def insert(self, rs_orig, nodes_orig):
        unsolved = []
        nodes = [x.city_id for x in nodes_orig]
        rs = [[nodei.city_id for nodei in temp] for temp in rs_orig]
        avail_time = self.calculate_avail_time(rs_orig)
        sum_time = 0
        node_count = 0
        index_perm = np.random.permutation(len(nodes))
        for ind in index_perm:
            #i = random.randint(0, len(nodes) - 1)
            n = nodes_orig[ind]
            new_arc = []
            #chọn arcs không vi phạm constraint nếu thêm node vào giữa
            for i in range(len(rs)):
                feasible = [x for x in self.make_arcs(rs[i], i, True) if self.satisfied(x, n, rs[i], avail_time[i])]
                new_arc.extend(feasible)

            if len(new_arc) == 0 and n.demand <= self.capacity:
                avail_time = self.update_avail_time(rs, n, [self.start, n, len(rs), 0], avail_time)
                rs.append([n.city_id])
            else:
                if len(new_arc) != 0:
                    #chọn arc minimize cost(arc, node)
                    arc = min(new_arc, key=lambda x: self.cost(x, n))
                    new_idx = arc[3]
                    avail_time = self.update_avail_time(rs, n, arc, avail_time)
                    rs[arc[2]].insert(new_idx, n.city_id)
                else:
                    unsolved.append(n)
        rs_full = [[self.get_city(x) for x in temp] for temp in rs]
        return rs_full, unsolved
        
    def create(self):
        self.routes, self.unsolved = self.insert(self.routes, self.nodes)

    def print_sol(self):
        for i in range(len(self.routes)):
            cost = [x.demand for x in self.routes[i]]  
        print "Total cost: %.3f" % self.score()
        for i in range(len(self.routes)):
            print "Vehicle %d (%d/%d): %s" % (i + 1, sum(cost), self.capacity, str([x.city_id for x in self.routes[i]]))
        if len(self.unsolved) != 0:
            print "Warning: this solution has unresolved cities: %s" % (str([x.city_id for x in self.unsolved]))

def len_lists(ls):
    return sum([len(l) for l in ls])

def index2d(ls, idx):
    if len(ls) == 0 or len_lists(ls) <= idx:
        raise Exception("Index out of range")
    sz = [len_lists(ls[:i+1]) for i in range(len(ls))]
    for i in range(len(sz)):
        if idx < sz[i]:
            prev = 0 if i == 0 else sz[i - 1]
            return (i, idx - prev)

#so sánh hai solution: return < 0 nếu os tốt hơn s. 
def compare_sol(s, os):
    if len(s.unsolved) != len(os.unsolved):
        return len(os.unsolved) - len(s.unsolved)
    if len(s.routes) != len(os.routes):
        return len(os.routes) - len(s.routes)
    return os.score() - s.score()

def print_partial_sol(unsolved, routes):
    print unsolved
    for r in routes:
        print "%s" % ([x.city_id for x in r])

class Ga(object):
    def __init__(self, tour, init_pop=10, iterations=50, time=60):
        self.init_pop = init_pop
        self.iterations = iterations # nếu main population qua iterations generation vẫn không tiến triển gì thì trả kết quả về. 
        self.time = time
        self.population = []
        self.tour = tour
        self.ps1 = init_pop # số solution được xem xét mỗi generation 
        self.pl1 = self.ps1 / 10 # số cặp bố mẹ được chọn để crossover trong main population 
        self.mp = 0.1 # probability of mutation 
        self.ps2 = self.ps1 / 5 # số solutions được generate từ mutation 
        self.pl2 = self.pl1 / 5 # số cặp bố mẹ được chọn để crossover trong mutated population 
        self.ipop2 = iterations / 10 # nếu mutated population qua ipop2 iterations vẫn không improve thì trả về 
        self.stats = []
        self.timing_info = [0] * 10
        self.counting = [0] * 10
        random.seed()

    def dist(self, s, e):
        return self.tour.cache[s.city_id][e.city_id]
    
    #tạo ra init_pop solutions đầu tiên 
    def init(self):
        for i in range(self.init_pop):
            sol = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
            sol.create()
            self.population.append(sol)

    #chọn một cặp solutions từ ranked - rank càng cao càng có cơ hội được chọn
    def choose(self, ranked):
        c = range(len(ranked))
        choices = []
        m = range(1, len(c) + 1)
        n = [1 for x in range(len(m))]
        for j in range(1, len(m)):
            n[j] = n[j - 1] + m[j]

        def get_j():
            num = random.random() * n[-1]
            j = 0
            while num > n[j]:
                j = j + 1
            return j
        
        fj = get_j()
        choices.append(ranked[fj])
        sj = get_j()
        while sj == fj:
            sj = get_j()
        choices.append(ranked[sj])
        return choices

    def run_auxiliary(self, aux):
        runiter = 0
        last_best = None
        while runiter < self.ipop2:
            aux = sorted(aux, key=functools.cmp_to_key(compare_sol))
            aux = aux[len(aux) - self.ps2:]
            #nếu lần trước không tiến triển gì so với lần trước nữa thì tăng runiter để terminate nếu runiter >= ipop2.
            if self.not_improved(last_best, aux[-1]):
                runiter = runiter + 1
            else:
                runiter = 0
            last_best = aux[-1]
            #chọn pl2 cặp để crossover 
            for i in range(self.pl2):
                mom, dad = self.choose(aux)
                fst = self.crossover(mom, dad)
                snd = self.crossover(dad, mom)
                aux.extend([fst, snd])
                #mutate with probability 0.1 
                if random.random() < self.mp:
                    routes, nodes = self.mutation(fst)
                    s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
                    s.routes, s.unsolved = s.insert(routes, nodes)
                    aux.append(s)
        return aux

    #từ một partial solution (với một số nodes extract từ routes) tạo ra một mutated population có số solutions là ps2. Sau khi run_auxiliary, trả lại cá thể tốt nhất. 
    def best_auxiliary(self, routes, nodes):
        aux_pop = []
        for i in range(self.ps2):
            s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
            s.routes, s.unsolved = s.insert(routes, nodes)
            aux_pop.append(s)
        aux_pop = self.run_auxiliary(aux_pop)
        aux_pop = sorted(aux_pop, key=functools.cmp_to_key(compare_sol))
        return aux_pop[-1]

    def aver_best(self, pop):
        best = (pop[-1].score(), len(pop[-1].routes))
        aver = (sum([p.score() for p in pop]) / len(pop), sum([len(p.routes) for p in pop]) / (len(pop) * 1.0))
        return (best, aver)
    
    def not_improved(self, last, cur):
        if not last:
            return False
        return compare_sol(last, cur) >= 0

    #similar to run_auxiliary 
    def run(self):        
        runiter = 0
        self.stats = []
        start = timer()
        elapsed = 0
        last_best = None

        total_iter = 0

        while runiter < self.iterations and elapsed < self.time:
            total_iter += 1
            pop = sorted(self.population, key=functools.cmp_to_key(compare_sol))
            self.population = pop[len(pop) - self.ps1:]
            self.stats.append(self.aver_best(self.population))
            if self.not_improved(last_best, self.population[-1]):
                runiter = runiter + 1
            else:
                runiter = 0
            last_best = self.population[-1]
            for i in range(self.pl1):
                mom, dad = self.choose(self.population)
                fst = self.crossover(mom, dad)
                snd = self.crossover(dad, mom)
                self.population.extend([fst, snd])
                #thay vì đơn giản là mutate như run_auxiliary thì chỉ chọn cá thể tốt nhất từ mutated population để thêm vào. 
                if random.random() < self.mp:
                    routes, nodes = self.mutation(fst)
                    aux = self.best_auxiliary(routes, nodes)
                    self.population.append(aux)
            elapsed = timer() - start
        pop = sorted(self.population, key=functools.cmp_to_key(compare_sol))
        elapsed = timer() - start
        print "Takes: %.3f seconds (%d) score %.3f/%d" % (elapsed, total_iter, pop[-1].score(), len(pop[-1].routes))
        return pop[-1]
        
    def test(self):
        self.init()
        sol = self.run()
        #print self.timing_info
        #print self.counting
        return sol, self.stats

    #mutation operator: chọn một trong 4 với probability (gần) bằng nhau.
    def mutation(self, orig):
        t = random.random()
        if t <= 0.25:
            return self.simple_mutation(orig)
        if t <= 0.5:
            return self.cluster_mutation(orig)
        if t <= 0.75:
            return self.routes_mutation(orig)
        return self.time_mutation(orig)

    #crossover operator: chọn một trong 3 với probability bằng nhau
    def crossover(self, mom, dad):
        t = random.random()
        if t < 0.33:
            child = self.common_arc_crossover(mom, dad)
        elif t < 0.66:
            child = self.common_node_crossover(mom, dad)
        else:
            child = self.longest_common_sequence_crossover(mom, dad)
        return child
    
    #hai arc có bắt đầu và kết thúc ở cùng một thành phố không 
    def same_arc(self, a, other):
        return a[0] == other[0] and a[1] == other[1]

    def get_elem(self, routes, idx):
        tidx = index2d(routes, idx)
        n = routes[tidx[0]].pop(tidx[1])
        return n

    #chọn num nodes bất kì và xóa khỏi route hiện có, trả lại partial solution gồm route còn lại và các node đã trích ra. 
    def simple_mutation(self, orig):
        ist = timer()
        routes = copy.deepcopy(orig.routes)
        nodes = copy.deepcopy(orig.unsolved)
        num = int(random.random() * 0.5 *
                  (len(self.tour.destination_cities) - len(nodes)))
        for i in range(num):
            idx = random.randint(0, len_lists(routes) - 1)
            nodes.append(self.get_elem(routes, idx))
        routes = filter(lambda x: len(x) > 0, routes)
        if random.random() < 0.1:
            s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
            s.routes = routes
            s.unsolved = nodes
            new_routes, new_unsolved = self.detour_mutation(s)
            return new_routes, new_unsolved
        iend = timer() - ist
        self.timing_info[1] += iend
        self.counting[1] += 1

        return routes, nodes

    #chọn một node bất kì cộng với num nodes gần nó nhất, trả lại partial solution gồm route còn lại và các node đã trích ra. 
    def cluster_mutation(self, orig):
        ist = timer()
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        num = int(random.random() * 0.5 *
                  (len(self.tour.destination_cities) - len(unsolved)))
        idx = random.randint(0, len_lists(routes) - 1)
        n = self.get_elem(routes, idx)
        unsolved.append(n)
        flat = [item for r in routes for item in r]
        flat_idx = list(zip(flat, range(len(flat))))
        flat_idx = sorted(flat_idx, key=lambda x: self.dist(x[0], n))
        to_del = []
        for i in range(num):
            node = flat_idx.pop(0)
            to_del.append(node[1])
        for i in sorted(to_del, reverse=True):
            node = self.get_elem(routes, i)
            unsolved.append(node)
        routes = filter(lambda x: len(x) > 0, routes)
        iend = timer() - ist
        self.timing_info[2] += iend
        self.counting[2] += 1

        return routes, unsolved

    #chọn num routes bất kì và xóa đi, trả lại partial solution gồm các route còn lại và các node trước ở trong route bị xóa. 
    def routes_mutation(self, orig):
        ist = timer()
        num = int(random.random() * 0.5 * len(orig.routes))
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        for i in range(num):
            idx = random.randint(0, len(routes) - 1)
            r = routes.pop(idx)
            unsolved.extend(r)
        routes = filter(lambda x: len(x) > 0, routes)
        iend = timer() - ist
        self.timing_info[3] += iend
        self.counting[3] += 1

        return routes, unsolved

    def detour_mutation(self, orig):
        ist = timer()
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        num = int(random.random() * 0.5 *
                  (len(self.tour.destination_cities) - len(unsolved)))
        detour = []
        for i in range(len(routes)):
            arcs = orig.make_arcs(routes[i], i)
            tmp = []
            for j in range(len(arcs) - 1):
                lr = self.dist(arcs[j][0], arcs[j][1]) + self.dist(arcs[j][1], arcs[j + 1][1]) - self.dist(arcs[j][0], arcs[j + 1][1])
                tmp.append(lr)
            detour.extend(tmp)

        flat_idx = list(zip(detour, range(len(detour))))

        detour_sorted = sorted(flat_idx, key=lambda x: x[0], reverse=True)

        to_del = []
        for i in range(num):
            node = detour_sorted.pop(0)
            to_del.append(node[1])
        for i in sorted(to_del, reverse=True):
            node = self.get_elem(routes, i)
            unsolved.append(node)
        routes = filter(lambda x: len(x) > 0, routes)
        iend = timer() - ist
        self.timing_info[4] += iend
        self.counting[4] += 1

        return routes, unsolved

    def time_mutation(self, orig):
        ist = timer()
        def get_arrival(time, routes, idx):
            tidx = index2d(routes, idx)
            arrival_time = time[tidx[0]][tidx[1]][0]
            return arrival_time
        
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        num = int(random.random() * 0.5 *
                  (len(self.tour.destination_cities) - len(unsolved)))

        time = orig.calculate_avail_time(routes)
        idx = random.randint(0, len_lists(routes) - 1)
        arrival_time = get_arrival(time, routes, idx)
        tidx = index2d(routes, idx)
        time[tidx[0]].pop(tidx[1])
        n = self.get_elem(routes, idx)
        unsolved.append(n)

        flat = [item for r in routes for item in r]
        flat_idx = list(zip(flat, range(len(flat))))
        flat_idx = sorted(flat_idx, key=lambda x: abs(arrival_time - get_arrival(time, routes, x[1])))
        to_del = []
        for i in range(num):
            node = flat_idx.pop(0)
            to_del.append(node[1])
        for i in sorted(to_del, reverse=True):
            node = self.get_elem(routes, i)
            unsolved.append(node)
        routes = filter(lambda x: len(x) > 0, routes)
        iend = timer() - ist
        self.timing_info[5] += iend
        self.counting[5] += 1

        return routes, unsolved
        
    #chọn các arc giống nhau từ bố và mẹ, thêm vào solution con. Những arc không giống nhau thì xóa đi và cho các thành phố từ các arc bị xóa đó vào unsolved. Sau đó tạo full solution từ các route còn lại và các thành phố đó.
    def common_arc_crossover(self, mom, dad):
        ist = timer()
        s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
        unsolved = [x.city_id for x in mom.unsolved]
        dad_arcs = []
        mom_arcs = []
        both_arcs = [[] for i in range(len(mom.routes))]
        for r in dad.routes:
            dad_arcs.extend(dad.make_arcs(r))
        for i in range(len(mom.routes)):
            arcs = mom.make_arcs(mom.routes[i], i)
            mom_arcs.extend(arcs)

        for a in mom_arcs:
            for b in dad_arcs:
                if self.same_arc(a, b):
                #giữ lại các arc giống nhau 
                    both_arcs[a[2]].append(a)
                    break
            else:
                #bỏ các thành phố không nằm trong common arc vào unsolved
                if a[0].city_id != 0:
                    unsolved.append(a[0].city_id)
        
        both_arcs = filter(lambda x: len(x) > 0, both_arcs)
        routes = []
        for r in both_arcs:
            part = []
            for arc in r:
                added_nodes = [x for x in arc[:2] if x not in part and x.city_id != 0]
                part.extend(added_nodes)
                for p in part:
                    if p.city_id in unsolved:
                        unsolved.remove(p.city_id)
            if len(part) > 1:
                routes.append(part)
            elif len(part) > 0:
                unsolved.append(part[0].city_id)

        unsolved_full = [self.tour.destination_cities[x] for x in unsolved]
        new_routes, new_unsolved = self.minimize_num_routes(routes, unsolved_full, len(mom.routes))
        s.routes, s.unsolved = s.insert(new_routes, new_unsolved)
        iend = timer() - ist
        self.timing_info[6] += iend
        self.counting[6] += 1

        return s

    def from_set(self, s, l):
        ls = []
        lidx = [x.city_id for x in l]
        for i in range(len(lidx)):
            if lidx[i] in s:
                ls.append(l[i])
        return ls
    
    #chọn các node giống nhau thuộc cùng một route từ bố và mẹ, xóa các node còn lại. Sau đó tạo full solution từ các route còn lại và các thành phố đó.
    def common_node_crossover(self, mom, dad):
        ist = timer()
        s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
        unsolved = set([x.city_id for x in dad.unsolved])
        routes = []
        dad_set = [set([x.city_id for x in r]) for r in dad.routes]
        mom_set = [set([x.city_id for x in r]) for r in mom.routes]
        for r in dad_set:
            mi = map(lambda x: set.intersection(x, r), mom_set)
            idx, cs = max(enumerate(mi), key=lambda(_, x): len(x))
            if len(cs) > 1:
                routes.append(self.from_set(cs, mom.routes[idx]))
            else:
                idx = -1
            for i in range(len(mi)):
                if i != idx:
                    unsolved.update(mi[i])
        
        routes = filter(lambda x: len(x) > 0, routes)
        unsolved_full = [self.tour.destination_cities[x] for x in unsolved]
        new_routes, new_unsolved = self.minimize_num_routes(routes, unsolved_full, len(mom.routes))

        s.routes, s.unsolved = s.insert(new_routes, new_unsolved)
        iend = timer() - ist
        self.timing_info[7] += iend
        self.counting[7] += 1

        return s

    def minimize_num_routes(self, routes, unsolved, parent_size):
        if parent_size == 0:
            return (routes, unsolved)
        
        if random.random() >= 0.5:
            remove = 1
        else:
            remove = 0
        desired_size = parent_size - remove
        while(len(routes) > desired_size):
            idx = random.randint(0, len(routes) - 1)
            r = routes.pop(idx)
            unsolved.extend(r)
            routes = filter(lambda x: len(x) > 0, routes)
        return (routes, unsolved)

    def largest_common_node(self, orig, rs):
        def count_common(r1, r2):
            return len([r for r in r1 if r in r2])
        return max(enumerate(rs), key = lambda(_, a): count_common(a, orig))[0]

    def longest_common_sequence(self, r1, r2):
        table = [[0 for x in xrange(len(r2) + 1)] for x in xrange(len(r1) + 1)]
        r = []
        for i in range(len(r1) + 1):
            for j in range(len(r2) + 1):
                if i == 0 or j == 0:
                    table[i][j] = 0
                else:
                    if r1[i - 1] == r2[j - 1]:
                        table[i][j] = table[i - 1][j - 1] + 1
                    else:
                        table[i][j] = max(table[i - 1][j], table[i][j - 1])
        i = len(r1)
        j = len(r2)
        while i > 0 and j > 0:
            if r1[i - 1] == r2[j - 1]:
                r.append(r1[i - 1])
                i -= 1
                j -= 1
            else:
                if table[i - 1][j] > table[i][j - 1]:
                    i -= 1
                elif table[i - 1][j] < table[i][j - 1]:
                    j -= 1
                else:
                    if random.random() > 0.5:
                        i -= 1
                    else:
                        j -= 1
        r.reverse()
        return r
        
    def longest_common_sequence_crossover(self, mom, dad):
        ist = timer()
        s = Solution(self.tour.destination_cities, self.tour.car_limit, self.tour.cache)
        temp = [x.city_id for x in mom.unsolved]
        routes = []
        for r in mom.routes:
            ind = self.largest_common_node(r, dad.routes)
            seq = self.longest_common_sequence(r, dad.routes[ind])
            rnew = []
            for n in r:
                if n in seq:
                    rnew.append(n)
                else:
                    temp.append(n.city_id)
            if len(rnew) > 1:
                routes.append(rnew)
            else:
                temp.extend([x.city_id for x in rnew])
        unsolved_full = [self.tour.destination_cities[x] for x in temp]
        new_routes, new_unsolved = self.minimize_num_routes(routes, unsolved_full, len(mom.routes))

        s.routes, s.unsolved = s.insert(new_routes, new_unsolved)
        iend = timer() - ist
        self.timing_info[8] += iend
        self.counting[8] += 1

        return s


    
def ga_sol(tour, num_start, num_iter, time=60):
    ga = Ga(tour, num_start, num_iter, time)
    return ga.test()

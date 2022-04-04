# This is a sample Python script.
import random
import itertools

import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class Osobnik():
    def __init__(self, n=5, res=7):
        self.res = res
        self.n = n
        self.x1 = 0.
        self.x2 = 0.
        self.J = 0.0
        #  chyba jednak losowanie róbmy na u_bin >> bo to jednak troche inaczej niz mi się wydawało działa to zamienianie
        # zerknij na str. 46 ksiazki
        self.u_bin = np.random.randint(2, size=(n,res))
        self.u = np.zeros(n)
        self.bin2dec()

    def bin2dec(self):
        g1 = 0.
        g2 = 10.
        for i in range(self.n):
            u_bin_val = 0.0
            for j in range (self.res):
                u_bin_val += (2**(self.res -1 -j)) * self.u_bin[i][j]
            self.u[i] = np.round(g1 + (u_bin_val* g2/((2**self.res)-1)), decimals=1)
        pass

    def count_J(self):
        for i in range(1, self.n):
            x1_prev = self.x1
            x2_prev = self.x2
            self.x1 = x2_prev
            self.x2 = 2 * x2_prev - x1_prev + (1/(self.n**2))*self.u[i]
        self.J = self.x1 - ((1 / (2 * self.n)) * (np.sum(self.u) - self.u[self.n - 1]))
        pass

    def mutacja(self):
        rand=random.random()
        if rand<=0.1:
            # print("Wykonuje mutacje")
            k=random.randint(0,self.n-1)
            k_id=random.randint(0,self.res-1)
            if self.u_bin[k][k_id] == 0:
                self.u_bin[k][k_id] = 1
            else:
                self.u_bin[k][k_id]=0
            self.bin2dec()


#         na ubin i u


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# def get_random_pairs(numbers):
#     # Generate all possible non-repeating pairs
#     pairs = list(itertools.combinations(numbers, 2))
#
#     # Randomly shuffle these pairs
#     random.shuffle(pairs)
#     return pairs

def krzyzowanie(population,pop_num):
    # losowanie par
    ids=list(range(0,pop_num))
    random.shuffle(ids)
    pairs=[]
    while(ids):
        px=ids.pop()
        py=ids.pop()
        pair=(px,py)
        pairs.append(pair)
    p_k=np.random.rand(int(pop_num/2))
    # id gdzie prawdopodobienstwa krzyz jest git
    crossover_ids=np.where(p_k>0.6)
    for el in crossover_ids[0]:
        selected_pair=pairs[int(el)]
        parent_1=population[selected_pair[0]]
        parent_2=population[selected_pair[1]]
        parent_1.u_bin=parent_1.u_bin.flatten()
        parent_2.u_bin=parent_2.u_bin.flatten()
        rand=random.randint(0,parent_1.u_bin.size-1)
        temp1=parent_1.u_bin[:rand]
        # temp2=parent_2.u_bin[rand:]
        parent_1.u_bin[:rand]=parent_2.u_bin[:rand]
        parent_2.u_bin[:rand]=temp1
        parent_1.u_bin=parent_1.u_bin.reshape((parent_1.n,parent_1.res))
        parent_2.u_bin=parent_2.u_bin.reshape((parent_2.n,parent_2.res))
        population[selected_pair[0]]=parent_1
        population[selected_pair[0]].bin2dec()
        population[selected_pair[1]]=parent_2
        population[selected_pair[1]].bin2dec()

    # return pairs

# metoda proporcjonalna
def selekcja(population,pop_num):
    population.sort(key=lambda x: x.J)
    min_C = population[0].J
    J_sum=0
    f_przystosowania=np.ndarray(pop_num,dtype=Osobnik)
    for i,osobnik in enumerate(population):
        f_przystosowania[i]=osobnik.J-min_C
        J_sum+=f_przystosowania[i]
    f_przystosowania_relative=f_przystosowania/J_sum
    # sum_rel=sum(f_przystosowania_relative)
    dystryb=np.cumsum(f_przystosowania_relative)
    list_of_ids=[]
    for i in range(population.__len__()):
        rand=random.random()
        j=0
        while(rand > dystryb[j]):
            j+=1
        list_of_ids.append(j)
    new_population=[]
    for el in list_of_ids:
        new_population.append(population[el])

    return new_population

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pop_num = 100
    population = []
    # population = np.ndarray(pop_num, dtype=Osobnik)
    for i in range(pop_num):
        # population[i] = Osobnik()
        os=Osobnik()
        os.count_J()
        population.append(os)
    # ola = Osobnik()
    for el in population:
        print(el.J,end=" ")
    print()
    print("Nowa populacja")
    new_population=selekcja(population,pop_num)
    for el in new_population:
        print(el.J,end=" ")
    print()
    for el in new_population:
        print(el.u,end=" ")
    print()
    print("Mutacja")
    for el in new_population:
        el.mutacja()
        print(el.u, end=" ")
    krzyzowanie(new_population,pop_num)
    print()
    print("Krzyżowanie")
    for el in new_population:
        el.mutacja()
        print(el.u, end=" ")
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

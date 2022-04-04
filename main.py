import random
import numpy as np
import copy
import matplotlib.pyplot as plt


class Osobnik():
    def __init__(self, n=5, res=20):
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
        g2 = 1.
        for i in range(self.n):
            u_bin_val = 0.0
            for j in range (self.res):
                u_bin_val += (2**(self.res -1 -j)) * self.u_bin[i][j]
            self.u[i] = np.floor((g1 + (u_bin_val* g2/((2**self.res)-1)))*1000000)/1000000
        self.count_J()

    def count_J(self):
        self.x1 = 0.0
        self.x2 = 0.0
        for i in range(1, self.n):
            x1_prev = self.x1
            x2_prev = self.x2
            self.x1 = x2_prev
            self.x2 = 2 * x2_prev - x1_prev + (1/(self.n**2))*self.u[i-1]
        self.J = self.x2 - ((1 / (2 * self.n)) * np.sum(np.power(self.u,2)))


    def mutacja(self):
        rand=random.random()
        if rand<=0.01:
            # print("Wykonuje mutacje")
            k=random.randint(0,self.n-1)
            k_id=random.randint(0,self.res-1)
            if self.u_bin[k][k_id] == 0:
                self.u_bin[k][k_id] = 1
            else:
                self.u_bin[k][k_id]=0
            self.bin2dec()


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
    crossover_ids=np.where(p_k<0.6)
    for el in crossover_ids[0]:
        selected_pair=pairs[int(el)]
        parent_1=copy.copy(population[selected_pair[0]])
        parent_2=copy.copy(population[selected_pair[1]])
        parent_1.u_bin=parent_1.u_bin.flatten()
        parent_2.u_bin=parent_2.u_bin.flatten()
        rand=random.randint(0,parent_1.u_bin.size-1)
        temp1=parent_1.u_bin[:rand]
        # temp2=parent_2.u_bin[rand:]
        parent_1.u_bin[:rand]=parent_2.u_bin[:rand]
        parent_2.u_bin[:rand]=temp1
        parent_1.u_bin=parent_1.u_bin.reshape((parent_1.n,parent_1.res))
        parent_2.u_bin=parent_2.u_bin.reshape((parent_2.n,parent_2.res))
        population[selected_pair[0]]=copy.copy(parent_1)
        population[selected_pair[0]].bin2dec()
        population[selected_pair[1]]=copy.copy(parent_2)
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

def j_mean_in_population(x_population):
    j_m = 0
    for el in x_population:
        j_m +=el.J
    return (j_m/pop_num)

def u_mean_in_population(population,pop_num):
    return_list = []
    np_pop = np.zeros(pop_num)
    for j in range(population[0].n):
        for i, el in enumerate(population):
            np_pop[i]=el.u[j]
        return_list.append(np.mean(np_pop))
    return return_list

def standard_deviation(population, pop_num, type="J"):
    return_list=[]
    np_pop=np.zeros(pop_num)
    if type == "J":
        for i,el in enumerate(population):
            np_pop[i]=el.J
        return_list.append(np.std(np_pop))
    elif type =="u":
        for j in range(population[0].n):
            for i,el in enumerate(population):
                np_pop[i]=el.u[j]
            return_list.append(np.std(np_pop))
    return return_list
    # np_pop=np.asarrat(population)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l_epok = 1000
    pop_num = 70
    population = []
    j_avg = []
    j_std = []
    j_max = []
    x1_max= []
    x2_max =[]


    # population = np.ndarray(pop_num, dtype=Osobnik)
    for i in range(pop_num):
        # population[i] = Osobnik()
        os=Osobnik()
        # os.count_J()
        population.append(os)
    # ola = Osobnik()
    u_std=[]
    u_avg = []
    for i in range(population[0].n):
        u_std.append([])
        u_avg.append([])
    for el in population:
        print(el.J,end=" ")
    print()
    for i in range (l_epok):
        print()
        print("Nowa populacja "+str(i))
        new_population=selekcja(population,pop_num)

        krzyzowanie(new_population,pop_num)
        # print()
        # print("Krzyżowanie")
        # for el in new_population:
        #     print(el.u, end=" ")
        # print()
        # print("Mutacja")
        for el in new_population:
            el.mutacja()
            # print(el.u, end=" ")

        for el in new_population:
            print(el.J,end=" ")
        print()
        for el in new_population:
            print(el.u,end=" ")
        population = copy.copy(new_population)
        population.sort(key=lambda x: x.J)
        # rysowanko wykresów
        j_avg.append(j_mean_in_population(population))
        j_std.append(standard_deviation(population,pop_num,"J")[0])
        j_max.append(population[pop_num-1].J)
        x1_max.append(population[pop_num-1].x1)
        x2_max.append(population[pop_num - 1].x2)
        u_std_list=standard_deviation(population,pop_num,"u")
        u_avg_list=u_mean_in_population(population,pop_num)
        for k in range(population[0].n):
            u_std[k].append(u_std_list[k])
            u_avg[k].append(u_avg_list[k])
    print()
    print(f'Maxymalne J={j_max[l_epok - 1]}')
    print(f'Średnie J={j_avg[l_epok - 1]}')

    x_num = np.linspace(1,l_epok, l_epok)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot()
    plt.grid()
    ax.set_title("Średnia wartości wskaźnika jakości sterowania J w kolejnych pokoleniach \n"
                 "liczba parametrów N = "+ str(population[0].n))
    ax.set_xlabel('Pokolenie')
    ax.set_ylabel("J [u]")
    ax.plot(x_num, j_avg, label = "AVG J")
    ax.fill_between(x_num, j_avg - 2 * np.asarray(j_std), j_avg + 2 * np.asarray(j_std), alpha=0.2, label = "STD J")
    ax.plot(x_num, j_max ,label = "MAX J")
    ax.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot()
    for i in range(population[0].n):
        ax.plot(x_num, u_avg[i],'-x',label=f'U [{i}]')
        ax.fill_between(x_num, np.asarray(u_avg[i]) - 2 * np.asarray(u_std[i]), np.asarray(u_avg[i]) + 2 * np.asarray(u_std[i]), alpha=0.2)
    ax.set_title("U [" + str(i) + "]")
    ax.set_title("Parametry sterowania u[k] w kolejnych pokoleniach \n"
                 "liczba parametrów N = " + str(population[0].n))
    ax.set_xlabel('Pokolenie')
    ax.set_ylabel("u[k]")
    ax.legend(loc='upper right', ncol=5)
    plt.show()
    #
    x_num = np.linspace(1, l_epok, l_epok)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_title("X1_X2")
    ax.plot(x1_max, x2_max, x_num,label=f'X1_X2')
    ax.set_title(" \n"
                 "Optymalne wartości parametrów dla modelu stanowego \nliczba parametrów N = " + str(population[0].n))
    ax.set_xlabel("x1 max")
    ax.set_ylabel("x2 max")
    ax.set_zlabel('Pokolenie')

    ax.legend(loc='upper right')
    plt.show()


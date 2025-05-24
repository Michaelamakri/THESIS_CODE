import numpy as np
import random
import matplotlib.pyplot as plt
import csv
No=1e-10
current_population = [] 

def dbm_to_mw(dbm):
    y=10**(dbm/10)
    return y
def fitness(individual):
    global current_population
    Pc_dbm, Pd_dbm, htx2rx, dtx2rx, dd2d, a, dc2d, hd2d, hc2d = individual

    Pd = dbm_to_mw(Pd_dbm)
    Pc = dbm_to_mw(Pc_dbm)

    signal = Pd * abs(htx2rx) ** 2 * dtx2rx ** (-a)

    interference = 0
    for other in current_population:
        if other == individual:
            continue
        Pd_j = dbm_to_mw(other[1])
        hd2d_j = other[7]
        dd2d_j = other[4]
        interference += Pd_j * abs(hd2d_j) ** 2 * dd2d_j ** (-a)

    interference += Pc * abs(hc2d) ** 2 * dc2d ** (-a)

    sinr = signal / (interference + No)
    sinr_db = 10 * np.log10(sinr)

    return sinr_db
def create_initial_population(size):
    population=[]

    for i in range(size):
        Pc = random.uniform(10,24)
        Pd = random.uniform(10,24)

        htx2rx = np.random.rayleigh(scale=0.5)
        hd2d = np.random.rayleigh(scale=0.5)
        hc2d = np.random.rayleigh(scale=0.5)
        dtx2rx=random.uniform(10,100)
        dd2d=random.uniform(10,100)
        a=2.75
        dc2d=random.uniform(10,100)
        individual=[Pc,Pd,htx2rx,dtx2rx,dd2d,a,dc2d,hd2d,hc2d]
        population.append(individual)
    return population

def selection(population,fitnesses,tournament_size=3):
    selected=[]
    tournament_size = min(len(population),tournament_size)
    for i in range(len(population)):
        tournament=random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected
# Crossover function
def crossover(parent1, parent2):
    genes = random.random()
    child1=[]
    child2=[]
    y1=zip(parent1,parent2)
    for i,j in y1:
        child1.append(genes*i+(1-genes)*j)
        child2.append(genes*j+(1-genes)*i)
    return child1, child2

def mutation(individual,mutation_rate, lower_bound, upper_bound):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            factor= random.uniform(0.9, 1.1)
            individual[i]=individual[i]*factor
            if (i==2) or (i==7) or (i==8):
                individual[i] = max(min(individual[i],5.0),0.1)
            elif (i==3) or (i==4) or (i==6):
                individual[i]=max(min(individual[i],100),5)
            elif (i==0) or (i==1):
                individual[i]=max(min(individual[i],24),10)
    return individual

def main():
    global current_population;
    generations=100
    mutation_rate=0.1
    lower_bound=0.1
    upper_bound=5.0
    best_fitnesses=[]
    
    population_ex=create_initial_population(100)
    open("best_fits.txt",'w').close()

    for i in range(generations):
        current_population=population_ex
        fitnesses_ex = [fitness(individual) for individual in population_ex]
        best_fitness = max(fitnesses_ex)
        best_fitnesses.append(best_fitness)
        filename = "generation"+str(i)+".txt"
        with open(filename,'w') as file:
            file.write("generation"+str(i+1)+":\n")
            
            for j,individual in enumerate(population_ex):
                f=fitness(individual)
                file.write("Individual"+str(j+1)+":\n")
                file.write("Pc(dbm):"+str(individual[0])+"\n")
                file.write("Pd(dbm):"+str(individual[1])+"\n")
                file.write("htx2rx:"+str(individual[2])+"\n")
                file.write("dtx2rx:"+str(individual[3])+"\n")
                file.write("dd2d:"+str(individual[4])+"\n")
                file.write("dc2d:"+str(individual[6])+"\n")
                file.write("hd2d:"+str(individual[7])+"\n")
                file.write("hc2d:"+str(individual[8])+"\n")
                file.write("FITNESS:"+str(f)+"\n")
                file.write("list:"+str(individual)+"\n")
        filename1="best_fits.txt"
        with open(filename1,'a') as file:
            file.write("best fitness for generation "+str(i)+" : "+str(best_fitness)+"\n")
        with open(f"fits_{i}.csv", mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=';')
            writer.writerow(["Individual", "Fitness"])
            for j, fitness_value in enumerate(fitnesses_ex):
                writer.writerow([j + 1,fitness_value])
                

        with open("bestfits.csv", mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for value in best_fitnesses:
                writer.writerow([value])  
        elite_index = np.argmax(fitnesses_ex)
        elite_individual = population_ex[elite_index]
        selected=selection(population_ex,fitnesses_ex)
        next_gen=[]

        for i in range(0, len(selected), 2):
            if i+1<len(selected):
                parent_ex1 = selected[i]
                parent_ex2 = selected[i+1]
                child_ex1,child_ex2=crossover(parent_ex1,parent_ex2)
                next_gen.extend([child_ex1,child_ex2])
            else:
                next_gen.append(parent_ex1)

        next_gen = [mutation(child, mutation_rate, lower_bound, upper_bound) for child in next_gen]
        next_gen[0]=elite_individual 
        population_ex=next_gen
if __name__ == "__main__":
    main()

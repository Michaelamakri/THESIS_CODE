import numpy as np
import random
import matplotlib.pyplot as plt
import csv 

No=1e-10
SINR_TARGET=30
initial_distances = [] 
current_population = [] 


def dbm_to_mw(dbm):
    y=(10**(dbm/10))
    return y
def mw_to_dbm(mw):
    y=10*np.log10(mw)
    return y
#Η συνάρτηση fitness περιγράφει την συνάρτηση που αφορά το SINR και τα ονοματα των
#παραμέτρων προκύπτουν από την μαθηματική σχέση
def fitness(individual):
    global current_population
    global dtx2rx,dd2d,dc2d# για να μπορουμε μετα την 2η γενια να έχουμε ίδιες αποστάσεις στο μοντέλο
    Pc_dbm, Pd_dbm, htx2rx, dtx2rx, dd2d, a, dc2d, hd2d, hc2d = individual#κάθε individual έχει αυτές τις τιμές
    Pd = dbm_to_mw(Pd_dbm)
    Pc = dbm_to_mw(Pc_dbm)
    signal = Pd*abs(htx2rx)**2*dtx2rx**(-a)
    
    interference = 0
    for other in current_population:#υπολογισμός παρεμβολών από τους χρήστες
        if other == individual:
            continue
        Pcother=dbm_to_mw(other[0])
        Pdother=dbm_to_mw(other[1])
        hd2dother=other[7]
        dd2dother=other[4]
        interference =interference+Pdother* (abs(hd2dother)**2)* (dd2dother**(-a))

    interference =interference+Pc*(abs(hc2d)**2)*(dc2d**(-a))
    sinr=(signal)/(interference+No)

    sinr_db = 10*np.log10(sinr)
    return sinr_db

def create_initial_population(size):
    population=[]
    for i in range(size):
        Pc_dBm =-40
        Pd_dBm =23
        htx2rx = max(np.random.rayleigh(scale=2), 0.5)
        hd2d=max(np.random.rayleigh(scale=2), 0.5)
        hc2d=max(np.random.rayleigh(scale=2), 0.5)
        dtx2rx=np.random.uniform(10,100)
        dd2d=np.random.uniform(10,100)
        a=2.75
        dc2d=np.random.uniform(10,100)
        individual=[Pc_dBm,Pd_dBm,htx2rx,dtx2rx,dd2d,a,dc2d,hd2d,hc2d]
        population.append(individual)       
    return population

def selection(population,fitnesses,tournament_size=4):
    selected=[]
    x=len(population)
    tournament_size = min(x,tournament_size)#επιλογή tournament selection 
    for i in range(x):
        tournament=random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    genes = random.random()
    child1=[]
    child2=[]
    for i in range(len(parent1)):
        if i in [3, 4,5, 6]: #για την διατηρηση σταθερών αποστάσεων τις αποκλείουμε από τοcrossover
            child1.append(parent1[i])
            child2.append(parent1[i])
        else:
            child1.append(genes*parent1[i]+(1-genes)*parent2[i])#γονιδια
            child2.append(genes*parent2[i]+(1-genes)*parent1[i])
    return child1, child2
    
def mutation(individual,mutation_rate,bestfit):
    
    for i in range(len(individual)):
        if i in [3, 4,6]:
            continue
        if  random.random()< mutation_rate:
            factor= random.uniform(0.9, 1.1)
            if (i==0) :
                individual[i]=individual[i]*factor
            elif (i==1):
                individual[i]=individual[i]*factor#έλεγχος ισχύος
                if (bestfit<SINR_TARGET):
                    k=1
                    individual[1]=min(individual[1]+k,40)
                if (bestfit>SINR_TARGET):
                    k=1
                    individual[1]=individual[1]-k
    return individual


def main():
    global current_population;
    generations=100
    mutation_rate=0.1
    best_fitnesses=[]
    
    population_ex=create_initial_population(50)
      
    open("best_fits.txt",'w').close()

    for i in range(generations):
        current_population=population_ex
        fitnesses_ex = [fitness(individual) for individual in population_ex]
        best_fitness = max(fitnesses_ex)
        print("Generation:",i,"bestfitness:",best_fitness)
        best_fitnesses.append(best_fitness)
        #αποθήκευση σε txt αρχεία για να μπορούν να διαβαστουν τα αποτελεσματα για κάθε γενιά
        filename = "generation"+str(i)+".txt"
        with open(filename,'w') as file:
            file.write("generation"+str(i+1)+":\n")
            
            for j,individual in enumerate(population_ex):
                f=fitness(individual)
                if i in [1,99]:
                    print(individual[0],individual[1],individual[6])
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
        #αποθήκευση σε csv για επεξεργασία από matlab
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
        

        next_gen = [mutation(child, mutation_rate,best_fitness) for child in next_gen]
        next_gen[0]=elite_individual 
        population_ex=next_gen
    #Plot
    

    best_individual = population_ex[np.argmax([fitness(ind) for ind in population_ex])]  
    distances = np.linspace(10, 100, 200)  
    sinrs = []

    for d in distances:
        modified = best_individual.copy()
        modified[3] = d 
        sinr_val = fitness(modified)
        sinrs.append(sinr_val)

    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, sinrs, label="SINR vs dtx2rx", color='blue')
    plt.xlabel("Απόσταση Πομπού-Δέκτη (dtx2rx) [m]")
    plt.ylabel("SINR(dB)")
    plt.title("Εξέλιξη του SINR ως προς την Απόσταση Πομπού-Δέκτη")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
                    
if __name__ == "__main__":
    main()


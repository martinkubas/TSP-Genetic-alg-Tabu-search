
import numpy as np
import random
import matplotlib.pyplot as plt

class Traveler:
    def __init__(self, tour):
        self.tour = tour
        self.tourDst = totalTourDistance(self.tour, cityCoords)

def printGraph(gen, traveler=None):

    x_coords = [city[0] for city in cityCoords]
    y_coords = [city[1] for city in cityCoords]
    plt.scatter(x_coords, y_coords)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(i), fontsize=9, ha='right')

    if traveler:
        tour = traveler.tour
        for i in range(len(tour)):
            plt.plot([x_coords[tour[i]], x_coords[tour[(i + 1) % len(tour)]]],
                     [y_coords[tour[i]], y_coords[tour[(i + 1) % len(tour)]]], color='blue', linewidth=2)
            plt.title(f"generation: {gen}")

    plt.show()

def printPath(traveler, cityCoords):
    for i in range (0, len(traveler.tour)):
        if(i == len(traveler.tour) - 1):
            print(distTwoCities(cityCoords[traveler.tour[i]], cityCoords[traveler.tour[0]]), traveler.tour[i], cityCoords[i],"->" ,traveler.tour[0], cityCoords[0])
        else:
            print(distTwoCities(cityCoords[traveler.tour[i]], cityCoords[traveler.tour[i+1]]), traveler.tour[i],cityCoords[i],"->" ,traveler.tour[i+1], cityCoords[i+1])

    print(traveler.tourDst)


def distTwoCities(city1, city2):
    return int(np.sqrt(np.sum((np.array(city1) - np.array(city2))**2)))

def totalTourDistance(tour, cityCoords):
    totalLen = 0
    for i in range (0, len(tour)):
        if(i == len(tour) - 1):
            totalLen += distTwoCities(cityCoords[tour[i]], cityCoords[tour[0]])
        else:   #z posledneho mesta na prve
            totalLen += distTwoCities(cityCoords[tour[i]], cityCoords[tour[i+1]])
    return totalLen


def initPopulation(popSize, cities):
    population = []
    for i in range(popSize):
        traveler = Traveler(list(np.random.permutation(cities)))
        population.append(traveler)
    return population

def fitnessProbabilities(population):
    populationDistances = [traveler.tourDst for traveler in population]

    maxDist = max(populationDistances)
    popFitness = [maxDist - dist for dist in populationDistances]   #odpocitame dlzku cesty od najdlhsej cesty, aby najkratsia cesta mala najvacsie cislo
    popFitnessSum = sum(popFitness)
    if popFitnessSum == 0:
        return [1.0 / len(popFitness)] * len(popFitness)
    probabilities = [fitness / popFitnessSum for fitness in popFitness]     #vypocet pravdepodobnosti, ze dana cesta sa dostane do dalsej generacie

    return probabilities

def fitnessRoulete(population, genFitnessProbs):
    genFitnessProbs = np.array(genFitnessProbs)
    fitnessProbsCumSum = genFitnessProbs.cumsum()

    rand = random.random()                               #random cislo medzi 0 a 1
    boolProbability = fitnessProbsCumSum > rand               #hladame najblizsie vacsie cislo v kumulativnom sumari ako to random cislo
    boolProbability = np.array(boolProbability)

    return population[np.where(boolProbability)[0][0]] #vrati cestujuceho, ktoreho sme nahodne vybrali


def tournamentSelection(population, tournamentSize = 5):         #nahodne vybere 3 travelerov, a vrati toho, ktory ma najmensiu dlzku cesty
    tournament = random.sample(population, tournamentSize)
    best = tournament[0]
    for traveler in tournament:
        if(traveler.tourDst < best.tourDst):
            best = traveler

    return best

def mutate (traveler, mutRate):     #prejde vsetkymi mestami, a ked random cislo je mensie ako cislo mutacie, tak sa vymeni s druhym nahodnym mestom
    for swapped1 in range (0, len(traveler.tour)):
        if(random.random() < mutRate):
            swapped2 = int(random.random() * len(traveler.tour))
            traveler.tour[swapped1], traveler.tour[swapped2] = traveler.tour[swapped2], traveler.tour[swapped1]

    traveler.tourDst = totalTourDistance(traveler.tour, cityCoords)


def crossover(parent1, parent2):
    size = len(parent1.tour)

    start = random.randint(0, size - 1)
    end = random.randint(0, size - 1)
    if start > end:
        start, end = end, start     #random segment z prveho rodica

    child_tour = [None] * size      #vytvorenie pola o dlzke size naplnene None
    for i in range(start, end):
        child_tour[i] = parent1.tour[i]

    for i in range(size):
        if child_tour[i] is None:           #ak je zatial toto pole prazdne tak sa pozrie na druheho rodica
            p2city = parent2.tour[i]        #ak je na tom indexe v druhom rodicovi to mesto, ktore uz je naplnene z prveho rodica
                                            #tak sa pozrieme na akom indexe ma rodic 1 to vybrate mesto a podla toho indexu vyberieme mesto z druheho rodica
            while p2city in child_tour:
                index = parent1.tour.index(p2city)
                p2city = parent2.tour[index]

            child_tour[i] = p2city

    return Traveler(child_tour)


def geneticAlg(cities, popSize, generations, mutRate):
    population = initPopulation(popSize, cities)
    print("Zadajte typ vyberu rodicov: ")
    print("1. ruleta")
    print("2. turnaj")
    print("3. ruleta/turnaj")
    userinput = input()

    for generation in range(generations):
        genFitnessProbs = fitnessProbabilities(population)

        best_individual = min(population, key=lambda traveler: traveler.tourDst)
        # Print the best individual's tour graph
        if(generation % 50 == 0):
            print(best_individual.tourDst)
            printGraph(generation, best_individual)


        parents = []
        for i in range(popSize):
            if(userinput == "1"):
                parents.append(fitnessRoulete(population, genFitnessProbs))
            elif(userinput == "2"):
                parents.append(tournamentSelection(population))
            else:
                if(i % 2 == 0):
                    parents.append(fitnessRoulete(population, genFitnessProbs))
                else:
                    parents.append(tournamentSelection(population))



        newGen = []
        for i in range(0, popSize, 2):
            parent1 = parents[i]
            if(i + 1 == popSize):
                parent2 = parents[0]
            else:
                parent2 = parents[i + 1]
            child = crossover(parent1, parent2)
            mutate(child, mutRate)
            newGen.append(child)

        population = newGen






cities = 40
popSize = 150
generations = 670
mutRate = 0.01



cityCoords = []
for city in range(0, cities):
    cityCoords.append((random.randrange(0, 200), random.randrange(0, 200)))

printGraph(0)

geneticAlg(cities, popSize, generations, mutRate)



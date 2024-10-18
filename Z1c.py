
import numpy as np
import random
import matplotlib.pyplot as plt

cities = 20
cityCoords = [(random.randrange(0, 200), random.randrange(0, 200)) for _ in range(cities)]  #suradnice miest
"""cityCoords = [(130, 80), (190, 70), (50, 130), (120, 110), (140, 190),
              (90, 60), (170, 30), (160, 120), (180, 40), (70, 50),
              (200, 100), (40, 90), (100, 20), (30, 170), (10, 150),
              (80, 140), (150, 180), (60, 200), (110, 160), (20, 10)]"""

"""cityCoords = [(130, 80), (190, 70), (50, 130), (120, 110), (140, 190),
    (90, 60), (170, 30), (160, 120), (180, 40), (70, 50),
    (200, 100), (40, 90), (100, 20), (30, 170), (10, 150),
    (80, 140), (150, 180), (60, 200), (110, 160), (20, 10),
    (15, 95), (175, 75), (95, 195), (85, 175), (45, 105),
    (155, 95), (165, 145), (135, 35), (185, 55), (25, 85)]"""

"""cityCoords = [(130, 80), (190, 70), (50, 130), (120, 110), (140, 190),
              (90, 60), (170, 30), (160, 120), (180, 40), (70, 50),
              (200, 100), (40, 90), (100, 20), (30, 170), (10, 150),
              (80, 140), (150, 180), (60, 200), (110, 160), (20, 10),
              (15, 95), (175, 75), (95, 195), (85, 175), (45, 105),
              (155, 95), (165, 145), (135, 35), (185, 55), (25, 85),
              (75, 25), (195, 115), (115, 135), (65, 165), (105, 85),
              (5, 115), (125, 155), (145, 45), (135, 165), (55, 145)]"""

class Traveler:
    def __init__(self, tour):
        self.tour = tour
        self.tourDst = totalTourDistance(self.tour)

def main():
    print("Choose algorithm: ")
    print("1. Genetic Algorithm")
    print("2. Tabu Search")
    userChoice = input()

    if userChoice == "1":
        bestTour = geneticAlg()
        printGraph("Best genetic alg result", bestTour)
        printPath(bestTour)
    else:
        bestTour = tabuSearch()
        printGraph("Best tabu search result", bestTour)
        printPath(bestTour)

def printPathDecline(bestPathLengths, title="Path Length Decline", avgPathLengths = None):
    generations = list(range(len(bestPathLengths)))
    plt.plot(generations, bestPathLengths, label="Best Path Length", color="blue")
    if avgPathLengths:
        plt.plot(generations, avgPathLengths, label="Average Path Length", color="red")
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Path Length')
    plt.legend()
    plt.grid(True)
    plt.show()

def printGraph(gTitle, traveler=None):

    xCoords = [city[0] for city in cityCoords]
    yCoords = [city[1] for city in cityCoords]
    plt.scatter(xCoords, yCoords)
    for i, (x, y) in enumerate(zip(xCoords, yCoords)):
        plt.text(x, y, str(i), fontsize=9, ha='right')

    if traveler:
        tour = traveler.tour
        for i in range(len(tour)):
            plt.plot([xCoords[tour[i]], xCoords[tour[(i + 1) % len(tour)]]],
                     [yCoords[tour[i]], yCoords[tour[(i + 1) % len(tour)]]], color='blue', linewidth=2)
            plt.title(gTitle)

    plt.show()

def printPath(traveler):
    for i in range (0, len(traveler.tour)):
        if(i == len(traveler.tour) - 1):
            print(f"path length: {distTwoCities(cityCoords[traveler.tour[i]], cityCoords[traveler.tour[0]])}",
                  f"city1: {traveler.tour[i]}", f"{cityCoords[i]}",
                  "->" ,
                  f"city2: {traveler.tour[0]}", f"{cityCoords[0]}")
        else:
            print(f"path length: {distTwoCities(cityCoords[traveler.tour[i]], cityCoords[traveler.tour[i + 1]])}",
                  f"city1: {traveler.tour[i]}", f"{cityCoords[i]}",
                  "->",
                  f"city2: {traveler.tour[i + 1]}", f"{cityCoords[i + 1]}")

    print(f"Total path length: {traveler.tourDst}")


def distTwoCities(city1, city2):
    return int(np.sqrt(np.sum((np.array(city1) - np.array(city2))**2)))

def totalTourDistance(tour):
    totalLen = 0
    for i in range (0, len(tour)):
        if(i == len(tour) - 1):
            totalLen += distTwoCities(cityCoords[tour[i]], cityCoords[tour[0]])
        else:   #z posledneho mesta na prve
            totalLen += distTwoCities(cityCoords[tour[i]], cityCoords[tour[i+1]])
    return totalLen


def initPopulation(popSize): #vrati list cestovatelov s nahodnou permutaciou miest
    population = []
    for i in range(popSize):
        traveler = Traveler(list(np.random.permutation(cities)))
        population.append(traveler)
    return population



def fitnessRanks(population):   #vrati poradie v akom je kazdy cestovatel podla dlzky cesty
    populationDistances = [traveler.tourDst for traveler in population]


    maxDist = max(populationDistances)
    popFitness = [maxDist - dist for dist in populationDistances]   #odpocitame dlzku cesty od najdlhsej cesty, aby najkratsia cesta mala najvacsie cislo

    sortedFitness = sorted(popFitness)
    ranks = [sortedFitness.index(fitness) + 1 for fitness in popFitness]    #ulozi index sortedFitness do indexu cestovatela

    return ranks

def fitnessRoulete(population, genFitnessRanks):    #vrati nahodne vybrateho cestujuceho na zaklade fitnessRanks funkcie
    fitnessProbsCumSum = np.array(genFitnessRanks).cumsum()     #kumulativna suma [1, 3, 6, 10...]

    total = fitnessProbsCumSum[-1]  #[-1] accesne posledny element, co je celkova suma

    rand = random.randint(1, total)
    for i, cumSum in enumerate(fitnessProbsCumSum):
        if rand <= cumSum:
            return population[i]


def tournamentSelection(population, tournamentSize = 2):         #nahodne vybere tournamentSize travelerov, a vrati toho, ktory ma najmensiu dlzku cesty
    tournament = random.sample(population, tournamentSize)
    best = tournament[0]
    for traveler in tournament:
        if(traveler.tourDst < best.tourDst):
            best = traveler

    return best

def mutate (traveler):     #nahodne vyberie zaciatocne a koncove mesto, a obrati poradie cesty medzi nimi
        tour = traveler.tour
        start, end = random.randint(0, len(traveler.tour) - 1), random.randint(0, len(traveler.tour) - 1)
        if start > end:
            start, end = end, start

        while start < end:
            traveler.tour[start], traveler.tour[end] = traveler.tour[end], traveler.tour[start]
            start += 1
            end -= 1

        traveler.tourDst = totalTourDistance(traveler.tour)


def crossover(parent1, parent2):        #vyberie segment cesty z prveho rodica a doplni zvysok z druheho
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


def geneticAlg():
    popSize = 150
    generations = 550
    mutRate = 0.25
    crossoverRate = 0.9
    eliteSize = 3

    bestPaths = []
    avgPaths = []

    population = initPopulation(popSize)    #inicializuje populaciu s random cestou
    bestIndividual = min(population, key=lambda individual: individual.tourDst)
    print("Enter type of selection: ")
    print("1. rank roulette")
    print("2. tournament")
    userinput = input()

    for generation in range(generations):

        population.sort(key=lambda traveler: traveler.tourDst)  #zoradenie populácie, pre neskorsi elitizmus

        bestIndividual = population[0]                                                  #uchovanie najlepsieho
        bestPaths.append(bestIndividual.tourDst)                                        #a priemerneho jedinca z generacie
                                                                                        #pre neskorsie
        avgPathLength = sum(traveler.tourDst for traveler in population) / popSize      #generovanie
        avgPaths.append(avgPathLength)                                                  #grafu



        if generation % 50 == 0:               #kazdu 50tu generaciu spravi graf najlepsieho jedinca
            print(bestIndividual.tourDst)
            printGraph(f"generation: {generation}", bestIndividual)


        parents = []
        for i in range(popSize):
            if userinput == "1":
                genFitnessRanks = fitnessRanks(population)
                parents.append(fitnessRoulete(population, genFitnessRanks))
            elif userinput == "2":
                parents.append(tournamentSelection(population))
            else:
                return 1


        newGen = []
        for i in range (0, eliteSize):      #najlepsi jedinci idu automaticky do novej generacie neznemeni
            newGen.append(population[i])
        for i in range(0, popSize):
            parent1 = parents[i]

            if i + 1 == popSize:
                parent2 = parents[0]
            else:
                parent2 = parents[i + 1]

            if random.random() < crossoverRate:
                child = crossover(parent1, parent2) #krizenie
            else:
                child = parent1

            if random.random() < mutRate:
                mutate(child)   #nasledna mutacia

            newGen.append(child)    #zaradenie dietata do novej generacie



        population = newGen

    printPathDecline(bestPaths,"Result of genetic algorithm", avgPaths)
    return bestIndividual

def generateNeighbours(currentTraveler):
    neighbors = []
    for i in range(len(currentTraveler.tour)):
        for j in range(i + 1, len(currentTraveler.tour)):
            neighborTour = currentTraveler.tour.copy()
            neighborTour[i], neighborTour[j] = neighborTour[j], neighborTour[i]  # swap miest
            neighborTraveler = Traveler(neighborTour)
            neighbors.append((neighborTraveler, (i, j)))

    return neighbors

def tabuSearch():
    maxIterations = 1000
    initialTabuSize = 20


    initialTour = Traveler(list(np.random.permutation(cities)))         #nahodna permutacia miest pre inicializaciu
    bestTraveler = initialTour
    currentTraveler = bestTraveler
    tabuList = []
    bestPaths = []

    for iteration in range(maxIterations):

        tabuSize = initialTabuSize + (iteration // 100) * 20        #zvacsenie tabuSize kazdych 100 iteracii

        currentNeighbors = generateNeighbours(currentTraveler)

        filteredNeighbors = [
            neighbor for neighbor in currentNeighbors
            if neighbor[1] not in tabuList or neighbor[0].tourDst < bestTraveler.tourDst    #ak neni v tabu liste, alebo je lepsi jak momentalna najlepsia trasa
        ]

        if not filteredNeighbors:
            filteredNeighbors = currentNeighbors


        bestNeighbor = min(filteredNeighbors, key=lambda neighbor: neighbor[0].tourDst)

        currentTraveler, swap = bestNeighbor[0], bestNeighbor[1]

        tabuList.append(swap)   #zaradime momentalny swap(i,j) miest do tabu listu, nech nemozu byt v takom poradi najblizsie

        if len(tabuList) > tabuSize:
            tabuList.pop(0)

        if currentTraveler.tourDst < bestTraveler.tourDst:
            bestTraveler = currentTraveler


        if iteration % 50 == 0:
            print(bestTraveler.tourDst)
            printGraph(f"iteration: {iteration}", bestTraveler)
            bestPaths.append(bestTraveler.tourDst)

        if iteration % 50 == 0:             #kazdu x iteraciu sa cestujuci akokeby znovuinicializuje
            currentTraveler = Traveler(list(np.random.permutation(cities)))
            tabuList.clear()

        bestPaths.append(bestTraveler.tourDst)



    printPathDecline(bestPaths, "Decline of best Tabu search path")
    return bestTraveler






main()

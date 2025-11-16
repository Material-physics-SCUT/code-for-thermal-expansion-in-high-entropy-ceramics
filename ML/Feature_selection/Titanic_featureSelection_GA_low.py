#coding=utf-8

import random

# Based on the pre-generation and their performances, 
# generate a new generation by Given finess list, return the selected index


def roulette_wheel(fitness_ls):
    # If the fitness_ls = [1, 3, 2], the roulette_wheel is like [0, 1, 4, 6]
    roulette_wheel = [0]
    for i, fitness in enumerate(fitness_ls):
        roulette_wheel.append(fitness + roulette_wheel[i])
    # print(roulette_wheel)
    while True: 
        # Rotate the wheel
        pointer1=random.random()*roulette_wheel[-1]
        # print(pointer1)
        pointer2=random.random()*roulette_wheel[-1]
        # print(pointer2)
        # pointer = random.randint(0, int(roulette_wheel[-1]))
        select1=[]
        select2=[]
        for i, num in enumerate(roulette_wheel):
            if roulette_wheel[i] <= pointer1 and pointer1 <= roulette_wheel[i + 1]:
                select1.append(i)
                # print(select1)
            if roulette_wheel[i] <= pointer2 and pointer2 <= roulette_wheel[i + 1]:
                select2.append(i)
                # print(select2)
                
        if fitness_ls[select1[0]]>fitness_ls[select2[0]]:
            # print('1:',select1[0],fitness_ls[select1[0]],select2[0],fitness_ls[select2[0]])
            return select2[0]
        if fitness_ls[select1[0]]<fitness_ls[select2[0]]:
            # print('2:',select1[0],fitness_ls[select1[0]],select2[0],fitness_ls[select2[0]])
            return select1[0]
        
def feature_selection(population, num_features, feature_combines, fitness_ls):
    # Initialize the population if is the first generation
    if feature_combines == [] and fitness_ls == []:
        # for i in range(population): 
        while len(feature_combines) < population: # 
            # Lowest possible for feature_combine, i.e., 0
            low = 0
            # Highest possible for feature_combine, i.e., 3 if there are 
            # two feature, the corresponding binary form is 11
            high = 2 ** num_features - 1 
            # print(high)
            feature_combine = bin(random.randint(low, high))[2:] #i.e., '0b11' to '11'
            # print(feature_combine)
            # print(feature_combine.count(str(1)))
            # print(num_features - len(feature_combine))      
            if feature_combine.count(str(1)) > 10: # 
                continue
            else:
                feature_combine = '0' * (num_features - len(feature_combine)) + feature_combine # i.e., 00011
                # print( feature_combine )
                map_combine = [i == '1' for i in feature_combine] # i.e., [False, False, False, True, True]
                # print(map_combine )
                feature_combines.append(map_combine)
        # print(feature_combines)
        return(feature_combines)
    
    # Normalize the fitness_ls:
    # 标准化fitness
    fitness_ls = [((i / sum(fitness_ls)) * 100) ** 2 for i in fitness_ls]
    # print(fitness_ls)
    
    # Roulette wheel selection based on fitness_ls
    # Here we select N/2 population, into N/4 pairs, each pair generates 4 offsprings
    # So the next population remains the same N
    selections = []
    for i in range(int(population * 0.8)): # 
        try:
            index = roulette_wheel(fitness_ls)
            selections.append(feature_combines[index])
        except Exception as e:
            print(e)
            print('The chosen index by roulette_wheel is: ', index)
    # print(selections)
    
    # Crossover to generate offsprings
    # The pairing and crossover is a random-wise
    offsprings = [] # 
    # print(len(selections))
    while len(selections) > 1: # 
        random.shuffle(selections)
        father = selections.pop() # 
        # print(father)
        mother = selections.pop()
        # print(mother)
        
        for g in range(8): # 
            offspring = [0] * len(father)
            # print(offspring)
            for i, bit in enumerate(offspring):
                offspring[i] = [father, mother][random.randint(0, 1)][i]
            # print(offspring)
            offsprings.append(offspring)
    # print(len(offsprings))
                   
    # Mutation one bit, 50% probality
    # avoid 0000, which means there is no feature selected
    offsprings2=[] #       
    for offspring in offsprings:
        index = random.randint(0, len(offspring) - 1)
        offspring[index] = random.randint(0, 1) # i.e., [False, True, False,False, False, 0, True, True]
        # print(offspring)
        while 1 not in offspring:
            index = random.randint(0, len(offspring) - 1)
            offspring[index] = random.randint(0, 1)
        # print(offspring)
        # print(offspring.count(1))
        if offspring.count(1)<=5: # 
            offsprings2.append(offspring)   
        if(len(offsprings2)==100): # 
            break
    # print(offsprings2)
        
    # Return the result
    # [0110] = [False, True, True, False]
    for i, offspring in enumerate(offsprings2):
        offsprings2[i] = [i == 1 for i in offspring]
    return(offsprings2)

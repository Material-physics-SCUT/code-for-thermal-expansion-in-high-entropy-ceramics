

import pandas as pd
import numpy as np
import random
import Titanic_featureSelection_GA_low as tfg
from ABR import skl
import os


data = pd.read_excel('criteria_train.xlsx')  

X = data.iloc[:,2:].apply(lambda x: round(x, 4)) 

x_train11 = X.values
y_train = data.iloc[:,1].values

columns=data.columns[2 :]
columns_new=[]
columns_new.append('代数')
columns_new.append('精确度')
columns_new.append('判据个数')

columns_new2=['代数','本代个体数','最高精度','平均精度','最低精度','最低精度判据个数']
for i in range(len(columns)):
    columns_new.append(columns[i])
    columns_new2.append(columns[i])

columns_new.append('最佳参数')
columns_new2.append('最佳参数')

#Initialize params for GA
generation = 0
population = 100 #
num_features = data.shape[1]-2
feature_combines = []
fitness_ls = []

MAXGEN=50
trace = np.zeros((MAXGEN, 2))

offspring_outs=[]
offspring_outs_Gen=[]
#Train the model by GA(feature selection) and SVM
while generation<MAXGEN:
    print('\n')
    print('Generation: ', generation,end='')
    

    feature_combines = tfg.feature_selection(population, num_features, feature_combines, fitness_ls)
    fitness_ls = []
    best_params_ls = []
    for feature_combine in feature_combines:
        offspring_out=[]
        x_train=pd.DataFrame(x_train11)
        x_train = x_train[x_train.columns[feature_combine]]
        

        try:
            sls13=skl()
            score=sls13.training(x_train.copy(),y_train.copy())[0]
            best_params=sls13.training(x_train.copy(),y_train.copy())[1]
            
            offspring_out.append(generation)
            offspring_out.append(score)
            offspring_out.append(feature_combine.count(1))
            for i in feature_combine:
                if i:
                    offspring_out.append('True')
                else:
                    offspring_out.append('')
            
            offspring_out.append(best_params)
            
            offspring_outs.append(offspring_out)
            
            fitness_ls.append(score)
            best_params_ls.append(best_params)
        except Exception as e:
            print(e)
            score = 0
            fitness_ls.append(score)
        
    trace[generation, 0] = max(fitness_ls)
    trace[generation, 1] =(sum(fitness_ls)/len(fitness_ls))
    print(',本代共'+str(len(fitness_ls))+'个')
    print('Max score: ', max(fitness_ls))
    print('Average score: ', sum(fitness_ls) / len(fitness_ls))
    print('Min score: ', min(fitness_ls))
    
    
    sheet2_out=[]
    sheet2_out.append(generation)
    sheet2_out.append(len(fitness_ls))
    sheet2_out.append(max(fitness_ls))
    sheet2_out.append(sum(fitness_ls) / len(fitness_ls))
    sheet2_out.append(min(fitness_ls))
    sheet2_out.append(feature_combines[fitness_ls.index(min(fitness_ls))].count(1))
    for i in feature_combines[fitness_ls.index(min(fitness_ls))]:
        if i:
            sheet2_out.append('True')
        else:
            sheet2_out.append('')
            
    sheet2_out.append(best_params_ls[fitness_ls.index(min(fitness_ls))])
    
    offspring_outs_Gen.append(sheet2_out)
    # print(feature_combines[fitness_ls.index(max(fitness_ls))])

    # if max(fitness_ls) > 0.8 and sum(fitness_ls) / len(fitness_ls) > 0.77:
    #     print(feature_combines[fitness_ls.index(max(fitness_ls))])
    #     break
    generation += 1

filefullpath='melt_ABR.xlsx' 
if os.path.exists(filefullpath):
    os.remove(filefullpath)

with pd.ExcelWriter(filefullpath) as writer:
    apd11=pd.DataFrame(offspring_outs,columns=columns_new)   
    apd12=pd.DataFrame(offspring_outs_Gen,columns=columns_new2)
    
    apd11.to_excel(writer,sheet_name='详细信息') 
    apd12.to_excel(writer,sheet_name='每一代精度')  

os.startfile(filefullpath)


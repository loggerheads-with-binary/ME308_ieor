import config as cfg 
from pulp import *
import numpy as np 
import pandas as pd 

import logging 
logger = logging.getLogger('single-model')
logger.setLevel(logging.INFO)

def setup(sheet):
    pass

    model = LpProblem("Single_Solution_Model" , LpMinimize)
    logger.info("Setup model skeleton")

    vars = list(map(str , range(1,6)))
    Perm = np.transpose(np.array(sheet.get("B2:B6") , dtype = 'float64') )[0]
    Temp = LpVariable.matrix("T" , vars , cat = "Integer" , lowBound = 0)
    Overtime = LpVariable.matrix("O" , vars , cat = "Integer" , lowBound = 0)

    logger.info("Setup decision variables")

    cost_matrix = np.transpose(np.array(sheet.get("H2:J6") , dtype = 'float64' ) ) 
    time_matrix = np.array(sheet.get("E2:G6") , dtype = 'float64' )  

    tau = float(sheet.get('time')[0][0])

    obj = lpSum(    Perm*(tau*np.transpose(cost_matrix[0])) 
                +   Temp*(tau*np.transpose(cost_matrix[1]))  
                +   Overtime*(np.transpose(cost_matrix[2])) 
    )



    model += obj 
    logger.info("Setup objective")

    demand = np.array([sheet.get(f'demand_{i}') for i in range(1 , 4)] , dtype = 'float64')

    ##Setting up constraints 

    ##Overtime constraints 
    for i in range(0,5):

        constraint = lpSum(tau*(Perm[i] + Temp[i])) >= lpSum(Overtime[i]) , f'Overtime-{i+1}'
        model += constraint

    logger.info("Setup overtime constraints")

    ##Productivity Constraints
    for i in range(0 , 5):

        constraint  = lpSum(tau*(Perm[i] + Temp[i]) + Overtime[i]) >= lpSum(demand*np.transpose(time_matrix[i])) , f'Productivity-{i+1}'
        model += constraint 
    
    logger.info('Setup Productivity Constraints')

    ##Continuity Constraints 
    for i in range( 0 , 4):

        Q_i1 = lpSum(tau*(Temp[i+1] + Perm[i+1]) + Overtime[i+1])
        D_i1 = sum([demand[j]*time_matrix[i+1][j] for j in range(3)]) 

        Q_i =  lpSum(tau*(Temp[i] + Perm[i]) + Overtime[i])
        D_i = sum([demand[j]*time_matrix[i][j] for j in range(3)]) 

        constraint = (Q_i1*1/D_i1)  >= (Q_i*1/D_i)
        
        model += constraint  , f'Continuity-{i+1},{i+2}'

    logger.info("Setup Continuity Constraints")

    return model 

pprint = lambda model : list( map( lambda x : print(f"{x.name}:{x.value()}") , model.variables() ) ) 

def update_sheet(sheet , solutions):
    pass

    from gspread.cell import Cell

    vals = {x : x.value() for x in solutions}

    cells = [Cell(row = i+1 , col = 2 , value = vals[key]) for ]
    cells.extend([row = i+1 , col = 3 , value = vals['']])

def solve(model):

    model.solve(PULP_CBC_CMD())
    logger.info("Model has been solved")

    status =  LpStatus[model.status]
    logger.critical(f"Model Status: {status}")

    return model , model.variables()

if __name__ == '__main__':

    import coloredlogs, pretty_traceback
    pretty_traceback.install()
    coloredlogs.install(fmt = '[%(name)s] %(asctime)s %(levelname)s : %(message)s' , level = logging.INFO)
    
    client = cfg.sheets_init()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1vZziAEudgcfPcukHLTKArho_m1_DwXZ2rK2zdlFv-5Y/edit")

    Single_Model = sheet.get_worksheet(1)

    model = setup(Single_Model)
    model , solutions  = solve(model)   
    pprint(model)
    print(model)

    solutions = model.variables()

execute = lambda : update_sheet( Single_Model , solutions) 

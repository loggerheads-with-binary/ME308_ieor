import config as cfg 
from pulp import *
import numpy as np 
import pandas as pd 

import logging 
logger = logging.getLogger('base-model')
logger.setLevel(logging.DEBUG)

def solve(model):

    model.solve(PULP_CBC_CMD())
    logger.info("Model has been solved")

    status =  LpStatus[model.status]
    logger.critical(f"Model Status: {status}")

    return model 

def printf(model):

    list( map( lambda x : print(f"{x.name}:{x.value()}") , model.variables() ) )  

def sheet_update(sheet , solutions):

    import gspread 
    from gspread.cell import Cell

    global cells 
    
    logger.info("Setting up cell values")
    logger.debug(f'Values: {solutions}')
    cells = [Cell(row = i , col = 2 , value = value) for i , value in enumerate(solutions , 2)]

    sheet.update_cells(cells)
    logger.warning("Cells have been updated on the google sheet")
    
make_float = lambda array : np.array(array).astype('float64')

def setup(sheet):

    global model , DV, obj, constraint, time_matrix, cost, demand, tau
    
    logger.debug("Setting up model")
    model = LpProblem("Linear Programming Problem" , LpMinimize)

    ##Create Demand Matrices 
    demand = np.transpose(np.array(sheet.get("H3:H5") , dtype = 'float64'))[0]
    time_matrix = np.transpose(np.array(sheet.get("C2:E6") , dtype = 'float64'))
    cost = np.array(sheet.get("F2:F6") , dtype = 'float64')

    vars = [str(i) for i in range(1 , 6)]
    DV = LpVariable.matrix("N" , vars , cat = "Integer" , lowBound = 0)
    obj = lpSum(DV*cost )
    model += obj 
    logger.debug("Set up objective")

    tau = float(sheet.get("H6")[0][0])

    for i in range(0,5):

        constraint = lpSum(DV[i]*tau) >= lpSum(sum([demand[j]*time_matrix[j][i] for j in range(3)])) , f"Productivity-{i+1}"
        model += constraint 

    logger.debug("Setup all productivity constraints")

    for i in range(0 , 4):
        
        constraint =    lpSum(DV[i+1]*1/sum([demand[j]*time_matrix[j][1+i] for j in range(3)])) >= \
                        lpSum(DV[i]*1/sum([demand[j]*time_matrix[j][i] for j in range(3)])), f"Continuity-{i+1},{i+2}"

        model += constraint

    logger.debug("Setup all continuity constraints")


    return model 

if __name__ == '__main__':

    import coloredlogs, pretty_traceback
    pretty_traceback.install()
    coloredlogs.install(fmt = '[%(name)s] %(asctime)s %(levelname)s : %(message)s' , level = logging.DEBUG)
    
    client = cfg.sheets_init()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1vZziAEudgcfPcukHLTKArho_m1_DwXZ2rK2zdlFv-5Y/edit")

    Base_Model = sheet.get_worksheet(0)   

    model = setup(Base_Model)
    model = solve(model)

    print(model)
    printf(model)

execute = lambda : sheet_update(Base_Model , list(map(lambda x : x.value() , model.variables()))) 
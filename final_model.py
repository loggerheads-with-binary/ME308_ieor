import config as cfg 

from pulp import *
import numpy as np 

from argparse import Namespace  
import logging
from datetime import datetime
import os 

logger = logging.getLogger('final')
CLIENT = cfg.sheets_init()

import pretty_traceback , coloredlogs

coloredlogs.install(fmt = '[%(name)s] %(asctime)s %(levelname)s : %(message)s' , level = logging.INFO)
pretty_traceback.install()

#get_vars = lambda model : {v.name : v.value() for v in model.variables()}
get_vars = solutions = lambda model:  {v.name : v.value() for v in model.variables()}


#Outputs different types of workers and their total numbers
def get_count(model , month = None ):

	vars = solutions(model)

	if month == None:
		
		OP = list()
		OT = list()
		Temp = list()

		for i in range(12):
			for j in range(5):

				OP.append(vars[f'OP_{i+1}_{1+j}'])
				OT.append(vars[f'OT_{i+1}_{j+1}'])
				Temp.append(vars[f'Temp_{i+1}_{j+1}'])

		OP = np.reshape(np.array(OP) , (12,5))
		OT = np.reshape(np.array(OT) , (12 , 5))
		Temp = np.reshape(np.array(Temp) , (12, 5))

		return {'OP' : OP , 'OT' : OT , 'Temp' : Temp}

	else : 

		OP = list()
		OT = list()
		Temp = list() 

		for i in range(5):

			OP.append(vars[f'OP_{month}_{i+1}'])
			OT.append(vars[f'OT_{month}_{i+1}'])
			Temp.append(vars[f'Temp_{month}_{i+1}'])

		return {'OP': np.array(OP), 'OT' : np.array(OT) , 'Temp' : np.array(Temp)}

#Forecasts a model to return optimized number of permanent workers
def Forecast(params):

	model = LpProblem("Final-Model-Forecast" , LpMinimize)

	Perm = LpVariable.matrix('N' , range(1,6), cat = 'Integer' , lowBound = 0)
	#OP = LpVariable.matrix('OP' , range(1,6) , cat = 'Integer' , lowBound = 0)
	OP = np.zeros(5)

	logger.info("Forecast:DV setup")
	
	##Objective Setup 
	obj = lpSum((params.tau*np.matmul( np.transpose(params.cost)[0], np.transpose(Perm))) + 
				np.matmul(np.transpose(params.cost)[2] , np.transpose(OP)))

	model += obj

	logger.info("Forecast:Objective setup")

	##Constraints => Overtime 
	for i in range(5):

		c = lpSum(params.tau*Perm[i]) >= lpSum(OP[i]) 
		model += c , f'Overtime-{i+1}'

	##Constraints => Productivity 
	for i in range(5):

		c = lpSum(params.tau*Perm[i] + OP[i] ) >= \
				lpSum(np.matmul(params.time[i] , np.transpose(params.mean_demand))) 
		model += c , f'Productivity-{i+1}' 

	##Constraint => Continuity 
	for i in range(1,5):

		c = lpSum(params.tau*Perm[i] + OP[i])*1/lpSum(np.matmul(params.time[i] , np.transpose(params.mean_demand))) >= \
			lpSum(params.tau*Perm[i-1] + OP[i-1])*1/lpSum(np.matmul(params.time[i-1] , np.transpose(params.mean_demand)))

		model += c , f'Continuity-{i},{i+1}'

	logger.info("Forecast: Constraints Setup")

	return model 

#Produces optimized workforce for actual demand simulation 
def Actual(Perm, params):

	model = LpProblem("Final-Model-Forecast" , LpMinimize)

	dtype = list()
	for i in range(1, 13):
		for j in range(1, 6):
			dtype.append( f'{i}_{j}' )

	#Perm = LpVariable.matrix('N' , range(1,13) , cat = 'Integer' , lowBound = 0)
	OP = LpVariable.matrix('OP' , dtype , cat = 'Integer' , lowBound = 0)
	#OT = LpVariable.matrix('OT' , dtype , cat = 'Integer' , lowBound = 0)
	OT = LpVariable.matrix('OT' , dtype , cat = 'Integer' , lowBound = 0 )
	Temp = LpVariable.matrix('Temp' , dtype , cat = 'Integer' , lowBound = 0)
	
	Temp = np.reshape(Temp , (12,5))
	OP = np.reshape(OP , (12, 5))
	OT = np.reshape(OT , (12,5))



	logger.info("Realization:DV setup")
	
	##Objective Setup 
	obj = lpSum(12*params.tau*np.matmul( np.transpose(params.cost)[0], np.transpose(Perm))) \
		+ lpSum(params.tau*np.matmul(Temp , np.transpose(np.transpose(params.cost)[1]))) \
		+ lpSum(np.matmul(OP , np.transpose(np.transpose(params.cost)[2]))) \
		+ lpSum(np.matmul(OT , np.transpose(np.transpose(params.cost)[3])))

	model += obj

	logger.info("Realization:Objective setup")

	##Constraints => Overtime 
	for j in range(12):
		for i in range(5):

			c = lpSum(params.tau*Perm[i]) >= lpSum(OP[j][i]) 
			model += c , f'Overtime[{j+1}]-Perm-{i+1}'

			c = lpSum(params.tau*Temp[j][i]) >= lpSum(OT[j][i]) 
			model += c , f'Overtime[{j+1}]-Temp-{i+1}'

	##Constraints => Productivity 
	for j in range(12):
		for i in range(5):

			c = lpSum(params.tau*Perm[i] + OT[j][i] + params.tau*Temp[j][i] + OP[j][i] ) >= \
					lpSum(np.matmul(params.time[i] , np.transpose(params.demand[j] ))) 
			model += c , f'Productivity[{j+1}]-{i+1}' 

	##Constraint => Continuity 
	for j in range(12):
		for i in range(1,5):

			c = lpSum(params.tau*Perm[i] + OP[j][i])*1/lpSum(np.matmul(params.time[i] , np.transpose(params.demand[j]))) >= \
				lpSum(params.tau*Perm[i-1] + OP[j][i-1])*1/lpSum(np.matmul(params.time[i-1] , np.transpose(params.demand[j])))

			model += c , f'Continuity[{j+1}]-{i},{i+1}'

	logger.info("Realization: Constraints Setup")

	return model 

##Returns Isometric Flexiblity of a given model and demand 
IF = lambda Perm , params: min([(params.tau*Perm[i]/np.matmul(params.time[i] , np.transpose(params.mean_demand) )) for i in range(5)] )

#Solves a LPP model using CBC Solver
def solve(model):

	model.solve(PULP_CBC_CMD(gapRel = (10**-5) ))
	logger.info("Model has been solved")

	status =  LpStatus[model.status]
	logger.critical(f"Model Status: {status}")

	return model

#Returns necessary parameters like time,cost,total working hours,etc
def get_params(ps , ds):

	pass     

	logger.info("Obtaining variables from sheet")

	time = np.array( ps.get('A12:C16') , dtype = 'float64') 
	demand = np.array( ds.get('B2:D13') , dtype = int)
	cost = np.array(ps.get('A3:D7') , dtype = 'float64' )
	mean_demand = np.array(ds.get('B16:D16') , dtype = int)
	tau = int(ps.get('G5')[0][0])
	total_prod_time = np.array(ps.get('A17:C17') , dtype = float)

	logger.info("Variables obtained")

	params = Namespace(**{	'time' : time , 'demand' : demand , 'cost' : cost , 
							'mean_demand' : mean_demand , 'tau' : tau , 'prod_time' : total_prod_time})
	return params 

#Returns permanent workers as an array
def perm(model):

	tt = get_vars(model)
	perm_ = [tt[f'N_{i}'] for i in range(1,6)]

	return perm_

#Writes the values onto the results sheet in the main sheet
def write_values(model , params,  perm_, fs):

	from gspread.cell import Cell

	cells = list()
	p_cells = [Cell(row = 2+i , col = 2 , value = val) for i , val in enumerate(perm_)]
	vals = get_vars(model)

	for j in range(12):
		for i in range(5):
			
			cells.append(Cell(col = 5 , row = 2 + (j*5 ) + i , value = vals[f'Temp_{j+1}_{i+1}']))
			cells.append(Cell(col = 6 , row = 2 + (j*5 ) + i , value = vals[f'OP_{1+j}_{i+1}']))
			cells.append(Cell(col = 7 , row = 2 + (j*5 ) + i , value = vals[f'OT_{1+j}_{i+1}']))

	t_cost = Cell(col = 2 , row = 8 , value = model.objective.value())

	fs.update_cells(p_cells)
	fs.update_cells(cells)
	fs.update_cells([t_cost])

#Returns the sheet, parameters subsheet and  demand subsheet    
def get_sheets():


	sheet = CLIENT.open_by_url('https://docs.google.com/spreadsheets/d/1vZziAEudgcfPcukHLTKArho_m1_DwXZ2rK2zdlFv-5Y/edit#gid=1001286541')


	param_sheet = sheet.get_worksheet(2)
	demand_sheet = sheet.get_worksheet(3)
	
	
	return sheet, param_sheet, demand_sheet

#Returns a normal distribution
def norm_distro(  meanA , meanB, meanC , stdA = 0.1  , stdB = 0.1, stdC = 0.1):

	a = np.random.normal(meanA , meanA*stdA , 12 )
	b = np.random.normal(meanB, meanB*stdB , 12 )
	c = np.random.normal(meanC, meanC*stdC , 12 )

	return np.array(np.absolute(np.transpose(np.array([a , b , c] ) )  ) , dtype = int ) + 1 

#Beta parameters from mean and COV
def beta_params(mean , COV): 
	std = mean*COV 
	alpha = ((1-mean)*((mean/std)**2)  - mean ) 
	beta = alpha/mean*(1-mean)

	logger.warning(f"Mean={mean} , Std ={std} :: Alpha= {alpha} , Beta= {beta}")

	return alpha, beta

##Returns a distribution with a scaling factor
def beta_distro(sf : int , meanA , meanB, meanC , stdA = 0.1  , stdB = 0.1, stdC = 0.1):

	a = np.random.beta( *beta_params(meanA, stdA), 12)
	b = np.random.beta( *beta_params(meanB, stdB), 12)
	c = np.random.beta( *beta_params(meanC, stdC), 12) 

	return np.array(np.absolute(np.transpose(sf*np.array([a , b , c] ) )  ) , dtype = int ) + 1 

##Returns the Theoretical Minimum Cost of a certain demand 
def tmc(params):

	time = np.transpose(params.time)
	cost = np.transpose(params.cost)[0]

	mod1 = np.dot(time[0] , cost)
	mod2 = np.dot(time[1] , cost)
	mod3 = np.dot(time[2] , cost)

	tmc = 0

	for i in range(12):
		tmc += np.dot(np.array([mod1, mod2 , mod3]) , params.demand[i])

	return tmc 

#Writes the demand array onto the sheet
def demand_write(dem, ds):

	from gspread.cell import Cell 

	cells= list()
	for j in range(12):
		for i in range(3):
			cells.append(Cell(row = 2 + j , col = 2 + i , value = dem[j][i] ))

	ds.update_cells(cells)

##Returns TMC of the mean demand 
def tmc_mean(params):


	time = np.transpose(params.time)
	cost = np.array(np.transpose(params.cost)[0])
	
	mod1 = np.matmul(time[0] , np.transpose(cost))
	mod2 = np.matmul(time[1] , np.transpose(cost))
	mod3 = np.matmul(time[2] , np.transpose(cost))

	return mod1*params.mean_demand[0][0] + mod2*params.mean_demand[0][1] + mod3*params.mean_demand[0][2] 
	
#Writes any array onto any set of rows and columns of any sheet	
def write_anywhere(arr , sheet , row_init , col_init , type_ = int):

	from gspread.cell import Cell

	rows, cols = arr.shape
	cells = list()

	for i in range(rows):
		for j in range(cols):

			cells.append(Cell(row = row_init + i , col = col_init + j , value = type_(arr[i][j])))

	sheet.update_cells(cells)
	return True 

#Returns Cost to Utility Ratio(CUR)
def cur(model , params):

	cost = model.objective.value()
	
	time = 0 

	for i in range(12):
		time += np.sum(np.matmul(params.time , np.transpose(params.demand[i])))

	return cost/time

#Returns CUR for the mean 
def cur_mean(perm_ , params):

	cost = np.dot(np.transpose(params.cost)[0] , np.array(perm_))*params.tau
	time = np.sum(np.matmul(params.time , np.transpose(params.mean_demand)))

	return cost/time 

#Returns Capacity Utilization(CU)
def cu(model , params , perm_,  month = False ):

	if month == False:
		##All months

		tmp = get_count(model , None)
		OP = tmp['OP']
		OT = tmp['OT']
		Temp = tmp['Temp']

		time_supplied = 12*200*np.sum(perm_) + 200*np.sum(Temp) + np.sum(OT) + np.sum(OP) 
		time_required = 0  

		for i in range(12):
			time_required += np.sum(np.matmul( params.prod_time , np.transpose(params.demand[i])) ) 

		return time_required/time_supplied

	else: 

		tmp = get_count(model , month)
		OP = tmp['OP']
		OT = tmp['OT']
		Temp = tmp['Temp']

		time_supplied = 12*200*np.sum(perm_) + 200*np.sum(Temp) + np.sum(OT) + np.sum(OP) 
		time_required = np.sum(np.matmul( params.prod_time , np.transpose(params.demand[month]))) 

		return time_required/time_supplied


def cu_mean(perm_ , params):

	time_required = np.sum(np.matmul(params.prod_time , np.transpose(params.mean_demand)))
	time_supplied = 200*np.sum(perm_)

	return time_required/time_supplied

#Returns Cost per hour 
cph_mean = lambda perm_ , params : cur_mean(perm_ , params)*cu_mean(perm_ , params)
cph = lambda model , params, perm_ , month = False : cur(model , params)*cu(model , params, perm_ , month)

##Runs the entire model for a given mean and COV
def run_all(params , t = 'norm', std : float = 0.1 , sheet = None  ):

	if t == 'norm':
		params.demand = norm_distro(params.mean_demand[0][0] , params.mean_demand[0][1] , params.mean_demand[0][2] , std, std, std)

	elif t == 'beta':
		params.demand = beta_distro(10**5 ,params.mean_demand[0][0]/(10**5) , 
									params.mean_demand[0][1]/(10**5) , params.mean_demand[0][2]/(10**5)  , 
									std , std , std)

	_model = Forecast(params)
	perm_ = perm(solve(_model))
	bcost = _model.objective.value() 

	model = Actual(perm_ , params)
	model = solve(model)

	write_values(model , params, perm_ , sheet.get_worksheet(8))

	cu_ratio =  cur(model , params)
	cu_mean_ratio = cur_mean(perm_ , params)

	cph_ratio = cph(model , params, perm_)
	cph_mean_ratio = cph_mean(perm_ , params)
 
	tmc_value = tmc(params)
	tmc_mean_value = tmc_mean(params)

	cu_value = cu(model , params, perm_ , False)

	COP = (bcost/tmc_mean_value)/(model.objective.value()/tmc_value)
	CPH = cu_ratio*cu_value 
	Efficiency = cph_mean_ratio/cph_ratio 

	return {'STD' : std , 'CUR' : cu_ratio , 'CPH' : CPH , 'efficiency' : Efficiency , 
						'COP' : COP , 'TMC' : tmc_value , 'TMCMEAN' : tmc_mean_value , 
						'CU' : cu_value }

METRICS = ('STD' , 'CUR' , 'CPH' , 'efficiency' , 'COP' , 'TMC' , 'TMCMEAN' , 'CU')
STD_VALS = (0.1 , 0.2 , 0.3 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9,  1 , 1.25 , 1.5 )

##Creates a perfect metric for a given mean and different values of COV
def metric_make():

	s, ps , ds = get_sheets()
	params = get_params(ps , ds)
	#params.mean_demand = np.transpose(params.mean_demand)

	_norm = list()
	_beta = list() 
	N = len(STD_VALS)

	for _idx , std in enumerate(STD_VALS):

		_norm.append(run_all(params, 'norm' , std , s))
		_beta.append(run_all(params , 'beta' , std , s))
		
		logger.info(f'Proc {_idx+1}/{N} completed')

	import pandas as pd 

	df_norm = pd.DataFrame(_norm)
	df_beta = pd.DataFrame(_beta)

	df_norm.to_csv('norm.csv')
	df_beta.to_csv('beta.csv')

	return True 

##Analyzes Capacity Utilization(CU) pattern over a year
def cu_analyze(params , model , perm_ ):

	from gspread.cell import Cell 

	md0 , md1 , md2 = params.mean_demand[0]

	r = np.zeros((12 , 2))

	for month in range(12):

		d0, d1, d2 = params.demand[month]

		rat = (d0/md0) + (d1/md1) + (d2/md2)
		rat /= 3 
		cu_val = cu(model , params , perm_ , month)

		r[month][0] = rat 
		r[month][1] = cu_val

	

	return r 


	sheet , param_sheet , demand_sheet = get_sheets() 

	logger.info("Access to sheets established")

	params = get_params(param_sheet , demand_sheet)
	model = Forecast(params)
	model = solve(model)
	model = Actual(model)
	model = solve(model)
	write_values(model , os.path.join( cfg.PROG_PATH , 'final_' + str(datetime.strftime(datetime.now() , '%d-%m @ %H-%M-%S')) + '.csv' ) )
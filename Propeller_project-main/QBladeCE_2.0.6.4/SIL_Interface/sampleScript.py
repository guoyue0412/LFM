from ctypes import *
from QBladeLibrary import QBladeLibrary

#loading the QBlade library from the folder below the location of sampleScript.py, if calling this script not from the script folder directly you need to use an absolute path instead!
 
# QBLIB = QBladeLibrary(b"D:/QBlade_code/QBladeCE_2.0.6.4/QBladeCE_2.0.6.dll")

QBLIB = QBladeLibrary("D:\\QBlade_code\\QBladeCE_2.0.6.4\\QBladeCE_2.0.6.dll")    

#creation of a QBlade instance from the library
QBLIB.createInstance(1,32)

#loading a project or sim-file, in this case the DTU_10MW_Demo project or simulation definition file
#QBLIB.loadSimDefinition(b"./DTU_10MW_Demo.sim") #uncomment this line to load a simulation definition file
QBLIB.loadProject(b"D:\QBlade_code\QBladeCE_2.0.6.4\SIL_Interface\NREL_5MW_Sample.qpr") 

#initializing the sim and ramp-up phase, call before starting the simulation loop
QBLIB.initializeSimulation()

#we will run the simulation for 500 steps before storing the results
number_of_timesteps = 500

#start of the simulation loop
for i in range(number_of_timesteps):

    #advance the simulation
    QBLIB.advanceTurbineSimulation() 	
    
    #assign the c-type double array 'loads' with length [6], initialized with zeros
    loads = (c_double * 6)(0,0,0,0,0,0) 
    #retrieve the tower loads and store the in the array 'loads' by calling the function getTowerBottomLoads_at_num()
    QBLIB.getTowerBottomLoads_at_num(loads,0)
    
    #uncomment the next line to try changing the position of the turbine dynamically
    #QBLIB.setTurbinePosition_at_num(-0.2*i,0,0,0,i*0.1,i*0.1,0) 
    
    #example how to extract a variable by name from the simulation, call as often as needed with different variable names, extracting rpm and time in the lines below
    rpm = QBLIB.getCustomData_at_num(b"Rotational Speed [rpm]",0,0) 
    time = QBLIB.getCustomData_at_num(b"Time [s]",0,0) #example how to extract the variable 'Time' by name from the simulation
    AoA = QBLIB.getCustomData_at_num(b"Angle of Attack at 0.25c (at section) Blade 1 [deg]",0.85,0) #example how to extract the variable 'Angle of Attack' by name at 85% blade length from the simulation 
    
    #example how to extract a 3 length double array with the x,y,z windspeed components at a global position of x=-50,Y=0,Z=100m from the simulation
    windspeed = (c_double * 3)(0,0,0) 
    QBLIB.getWindspeed(-50,0,100,windspeed)
    
    #assign the c-type double array 'ctr_vars' with length [5], initialized with zeros
    ctr_vars = (c_double * 5)(0); 
    #advance the turbine controller and store the controller signals in the array 'ctr_vars'
    QBLIB.advanceController_at_num(ctr_vars,0)
    
    #pass the controller signals in 'ctr_vars' to the turbine by calling setControlVars_at_num(ctr_vars,0) 
    QBLIB.setControlVars_at_num(ctr_vars,0) 
    
    #print out a few of the recorded data, in this case torque, tower bottom force along z (weight force) and rpm
    print("Time:","{:3.2f}".format(time),"   Windspeed:","{:2.2f}".format(windspeed[0]),"  Torque:","{:1.4e}".format(ctr_vars[0]),"    RPM:","{:2.2f}".format(rpm),"   Pitch:","{:2.2f}".format(ctr_vars[2]),"   AoA at 85%:","{:2.2f}".format(AoA))
#the simulation loop ends here after all 'number_of_timesteps have been evaluated
	
#storing the finished simulation in a project as DTU_10MW_Demo_finished.qpr, you can open this file to view the results of the simulation inside QBlade's GUI
QBLIB.storeProject(b"./NREL_5MW_Sample_completed.qpr")

#closing the QBlade instance to free memory
QBLIB.closeInstance()

#unloading the QBlade library
del QBLIB.lib 
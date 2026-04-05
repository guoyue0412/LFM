%%
% This is a minimum working example on how to interface with the QBlade
% library. A project called 'NREL_5MW_Sample.qpr' has been provided to work with. 
% This script establishes a connection to the QBlade library, calls the library
% QBLIB and then initializes QBlade and the project. It runs the
% simulation for 500 simulation steps, and outputs some values. The time length
% of the simulation equals the number of simulation steps times the time
% step with which the simulation has been set up. At the end, the QBlade
% instance is closed again.

%%
clear all
close all 
clc

% create an object of the class 'QBladeLibrary' that contains all interface
% functions
QBLIB = QBladeLibrary('../QBladeCE_2.0.6.dll');

QBLIB.createInstance(1,32);

% since matlab is unable to display the console output from the library, we
% store the output in a log file
QBLIB.setLogFile('./LogFile.txt')

QBLIB.loadProject('NREL_5MW_Sample.qpr')

QBLIB.initializeSimulation()

number_of_timesteps = 500; 

f = waitbar(0,'Initializing Simulation') ;

for i = 1:1:number_of_timesteps
    
	%advance the simulation
	QBLIB.advanceTurbineSimulation()
	
	%assign the c-type double array 'loads' with length [6], initialized with zeros
	loads = libpointer('doublePtr',zeros(6,1));
	%retrieve the tower loads and store the in the array 'loads' by calling the function getTowerBottomLoads_at_num()
	QBLIB.getTowerBottomLoads_at_num(loads,0);
	%dereferencing the 'loads' pointer and accessing its first value
	loads.Value(1);
	
	%uncomment the next line to try changing the position of the turbine dynamically
	%QBLIB.setTurbinePosition_at_num(-0.2*i,0,0,0,i*0.1,i*0.1,0)
	
	%example how to extract a variable by name from the simulation, call as often as needed with different variable names, extracting rpm and time in the lines below
	rpm = QBLIB.getCustomData_at_num('Rotational Speed [rpm]',0,0);
	t = QBLIB.getCustomData_at_num('Time [s]',0,0);  %example how to extract the variable 'Time' by name from the simulation
	AoA = QBLIB.getCustomData_at_num('Angle of Attack at 0.25c (at section) Blade 1 [deg]',0.85,0); %example how to extract the variable 'Angle of Attack' by name at 85% blade length from the simulation 
	
	%example how to extract a 3 length double array with the x,y,z windspeed components at a global position of x=-50,Y=0,Z=100m from the simulation
	windspeed = libpointer('doublePtr',zeros(3,1)); 
	QBLIB.getWindspeed(-50,0,100,windspeed);
	
	%assign the c-type double array 'ctr_vars' with length [5], initialized with zeros
	ctr_vars = libpointer('doublePtr',zeros(5,1));
	%advance the turbine controller and store the controller signals in the array 'ctr_vars'
	QBLIB.advanceController_at_num(ctr_vars,0)
	
	%passthe controller signals in 'ctr_vars' to the turbine by calling setControlVars_at_num(ctr_vars,0) 
	QBLIB.setControlVars_at_num(ctr_vars,0)
    
	fprintf('Time: %3.2f	Windspeed: %2.2f    Torque: %1.4e	RPM: %2.2f	Pitch: %2.2f    AoA at 85%%: %2.2f\n',t,windspeed.Value(1),ctr_vars.Value(1),rpm,ctr_vars.Value(3),AoA);

	waitbar(i/number_of_timesteps,f,'QBlade Simulation Running')

end

close(f)

QBLIB.storeProject('./NREL_5MW_Sample_completed.qpr')

QBLIB.closeInstance()

QBLIB.unload()


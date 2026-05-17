classdef QBladeLibrary
    properties
        lib % DLL handle
    end
    
    methods
        % Constructor
        function obj = QBladeLibrary(dllPath)
            % Load DLL
            obj.lib = loadlibrary(dllPath,'QBladeLibInclude.h','alias','QBLIB');
            calllib('QBLIB','setLibraryPath',dllPath)
        end
        
        % Destructor
        function unload(obj)
            % Unload Library
            if libisloaded('QBLIB')
                unloadlibrary 'QBLIB'
            end;
        end
        
        % Function to call library function
        function createInstance(obj,clDevice,groupSize)
            calllib('QBLIB', 'createInstance', clDevice, groupSize);
        end
        
        function loadProject(obj,str)
            calllib('QBLIB', 'loadProject', str);
        end
        
        function loadSimDefinition(obj,str)
            calllib('QBLIB', 'loadSimDefinition', str);
        end
        
        function initializeSimulation(obj)
            calllib('QBLIB', 'initializeSimulation');
        end
        
        function runFullSimulation(obj)
            calllib('QBLIB', 'runFullSimulation');
        end
        
        function advanceController_at_num(obj,vars,num)
            calllib('QBLIB', 'advanceController_at_num', vars, num);
        end
        
        function advanceTurbineSimulation(obj)
            calllib('QBLIB', 'advanceTurbineSimulation');
        end
        
        function storeProject(obj,str)
            calllib('QBLIB', 'storeProject',str);
        end
        
        function closeInstance(obj)
            calllib('QBLIB', 'closeInstance');
        end
        
        function setLogFile(obj,str)
            calllib('QBLIB', 'setLogFile',str);
        end
        
        function loadTurbulentWindBinary(obj,str)
            calllib('QBLIB', 'loadTurbulentWindBinary', str);
        end
        
        function addTurbulentWind(obj,windspeed, refheight, hubheight, dimensions, gridPoints, length, dT, turbulenceClass, turbulenceType, seed, vertInf, horInf, removeFiles)
            calllib('QBLIB', 'addTurbulentWind', windspeed, refheight, hubheight, dimensions, gridPoints, length, dT, turbulenceClass, turbulenceType, seed, vertInf, horInf, removeFiles);
        end
        
        function setPowerLawWind(obj,windspeed,horAngle,vertAngle,shearExponent,referenceHeight)
            calllib('QBLIB', 'setPowerLawWind',windspeed,horAngle,vertAngle,shearExponent,referenceHeight);
        end
        
        function setDebugInfo(obj,isDebug)
            calllib('QBLIB', 'setDebugInfo', isDebug);
        end
        
        function setTimestepSize(obj,timestep)
            calllib('QBLIB', 'setTimestepSize', timestep);
        end
        
        function setRPMPrescribeType_at_num(obj,type,num)
            calllib('QBLIB', 'setRPMPrescribeType_at_num',type,num);
        end
        
        function setRampupTime(obj,time)
            calllib('QBLIB', 'setRampupTime',time);
        end
        
        function setInitialConditions_at_num(obj,yaw,pitch,azimuth,rpm,num)
            calllib('QBLIB', 'setInitialConditions_at_num',yaw,pitch,azimuth,rpm,num);
        end
        
        function setTurbinePosition_at_num(obj,x,y,z,xrot,yrot,zrot,num)
            calllib('QBLIB', 'setTurbinePosition_at_num',x,y,z,xrot,yrot,zrot,num);
        end
        
        function setControlVars_at_num(obj,vars,num)
            calllib('QBLIB', 'setControlVars_at_num',vars,num);
        end
        
        function setExternalAction(obj,action,id,val,pos,dir,isLocal,num)
            calllib('QBLIB', 'setExternalAction',action,id,val,pos,dir,isLocal,num);
        end
        
        function getWindspeed(obj,x,y,z,velocity)
            calllib('QBLIB', 'getWindspeed',x,y,z,velocity);
        end
        
        function getWindspeedArray(obj,posx,posy,posz,velx,vely,velz,arraySize)
            calllib('QBLIB', 'getWindspeedArray',posx,posy,posz,velx,vely,velz,arraySize);
        end
        
        function getTowerBottomLoads_at_num(obj,loads,num)
            calllib('QBLIB', 'getTowerBottomLoads_at_num',loads,num);
        end
        
        function getTurbineOperation_at_num(obj,vars,num)
            calllib('QBLIB', 'getTurbineOperation_at_num',vars,num);
        end
        
        function output = getCustomData_at_num(obj,var,i,j)
            output = calllib('QBLIB','getCustomData_at_num',var,i,j);
        end

        function output = getCustomSimulationData(obj,var)
            output = calllib('QBLIB','getCustomSimulationData',var);
        end
        
    end
end


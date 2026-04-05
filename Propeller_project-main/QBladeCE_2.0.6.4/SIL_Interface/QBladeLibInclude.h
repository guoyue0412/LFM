

void setLibraryPath(char *str);

void createInstance(int clDevice, int groupSize);
void loadProject(char *str);
void loadSimDefinition(char *str);
void initializeSimulation();
void runFullSimulation();

void advanceController_at_num(double *vars, int num);
void advanceTurbineSimulation();

void storeProject(char *str);
void closeInstance();
void setLogFile(char *str);

void loadTurbulentWindBinary(char *str);
void addTurbulentWind(double windspeed, double refheight, double hubheight, double dimensions, int gridPoints, double length, double dT, char *turbulenceClass, char *turbulenceType, int seed, double vertInf, double horInf, bool removeFiles);

void setPowerLawWind(double windspeed, double horAngle, double vertAngle, double shearExponent, double referenceHeight);
void setDebugInfo(bool isDebug);
void setTimestepSize(double timestep);
void setRPMPrescribeType_at_num(int type, int num);
void setRampupTime(double time);
void setInitialConditions_at_num(double yaw, double pitch, double azimuth, double rpm, int num);
void setTurbinePosition_at_num(double x, double y, double z, double rotx, double roty, double rotz, int num);
void setControlVars_at_num(double *vars, int num);
void setExternalAction(char *action, char *id, double val, double pos, char *dir, bool isLocal, int num);

void getWindspeed(double posx, double posy, double posz, double *velocity);
void getWindspeedArray(double *posx, double *posy, double *posz, double *velx, double *vely, double *velz, int arraySize);
void getTowerBottomLoads_at_num(double *loads, int num);
void getTurbineOperation_at_num(double *vars, int num);
double getCustomData_at_num(char *str, double pos, int num);
double getCustomSimulationData(char *str);
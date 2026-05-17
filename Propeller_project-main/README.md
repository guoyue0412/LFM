# Propeller Simulation via QBlade DLL Interface

This repository contains a Python-based simulation pipeline for evaluating propeller aerodynamic performance using [QBlade 2.0.6.4](https://qblade.org/). It supports batch simulations across different **RPM**, **wind speed**, and **inflow angle** conditions, and allows for **parametric geometry modification** based on control points or direct section data.

Plot functions are needed to visualize the results.

The simulation is built on top of the official QBlade DLL and performs full-loop automation including:

- `.sim` file generation based on Excel definitions
- `.bld` geometry manipulation using B-spline interpolation
- batch execution of QBlade simulations via DLL interface
- automatic data extraction, structuring and saving

---

## 📦 Key Features

- ✅ **Single & Batch Simulation** supported via `SIMULATION.run_one_simulation()` and `run_all_simulation()`
- 📊 **Output Data Includes:** thrust, power, torque, efficiency (`eta`), `Ct`, `Cp`, etc.
- 🌀 **Blade Geometry Control:** through `change_propeller_geometry()` using section arrays or 8D control points(4D for chord,the left for twist)
- 🔄 **Auto Save:** simulation results and geometries are stored in `.pkl` files for downstream optimization or training
- 🧩 **new functions:** for single simulation,the progress bar is added to show the simulation progress.
  
### 📊 Output Data Description (`self.one_simulation_data`)

Each simulation returns a `pandas.DataFrame` containing detailed time-series and averaged aerodynamic data:

| Column Name   | Unit     | Description |
|---------------|----------|-------------|
| `Time`        | s        | Simulation time steps（real_time-the last 120 steps） |
| `Thrust`      | N        | Instantaneous axial force (the last 120 steps)|
| `Power`       | kW       | Instantaneous aerodynamic power (the last 120 steps)|
| `Torque`      | Nm       | Instantaneous torque (the last 120 steps)|
| `Thrust_y`    | N        | Instantaneous force in hub Y direction (the last 120 steps)|
| `Thrust_z`    | N        | Instantaneous force in hub Z direction (the last 120 steps)|
| `RPM`         | rpm      | Rotational speed of the propeller |
| `WIND_SPEED`  | m/s      | Inflow wind speed |
| `ANGLE`       | degrees  | Inflow angle relative to X-axis |
| `THRUST`      | N        | Time-averaged axial force (`-1 * mean(Thrust)`) |
| `POWER`       | W        | Time-averaged power (`-1 * mean(Power) × 1000`) |
| `TORQUE`      | Nm       | Time-averaged torque (`-1 * mean(Torque)`) |
| `THRUST_Y`    | N        | Time-averaged force in hub Y direction (`-1 * mean(Thrust_y)`) |
| `THRUST_Z`    | N        | Time-averaged force in hub Z direction (`-1 * mean(Thrust_z)`) |
| `Ct`          | -        | Thrust coefficient: $$C_t = \frac{T}{\rho n^2 D^4}$$ |
| `Cp`          | -        | Power coefficient: $$C_p = \frac{P}{\rho n^3 D^5}$$ |
| `eta`         | -        | Propeller efficiency: $$\eta = \frac{V}{nD} \cdot \frac{C_t}{C_p}$$ (set to 0 if wind speed = 0) |

> **Notes**:

> - $$\( n = \frac{\text{RPM}}{60} \): rotational frequency (Hz) $$  
> - $$\( D = 2R \): propeller diameter (m), where radius \( R = 0.127 \, \text{m} \) $$ 
> - All mean values are computed over the final 120 time steps of the simulation

### 📦 Aggregated Simulation Results (`self.all_simulation_data`)

The `self.all_simulation_data` dictionary stores all simulation outputs for a given propeller geometry. Its structure is as follows:

```python
{
  "RPM1000_Wind5_Angle0": pd.DataFrame(...),  # simulation result for specific condition
  "RPM1000_Wind5_Angle15": pd.DataFrame(...),
  ...
  "geometry": np.ndarray  # geometry array for this batch (shape: [n_sections, 3])
}
```

#### 🧩 Key Structure

- Each key (except "geometry") represents one simulation condition and follows the format:

```python
RPM{rpm_value}_Wind{wind_speed}_Angle{angle}   (For example: RPM1200_Wind8_Angle10)
```

- Each value is a pandas.DataFrame containing time-resolved and averaged outputs for that condition. See the section one_simulation_data for details.
- The "geometry" key stores a NumPy array of shape (22,3)where:
   > Column 1: radial position 𝑟
   > Column 2: chord length c(r)
   > Column 3: twist angle 𝜃(𝑟)

#### 💡 Typical Use Case

- The entire self.all_simulation_data object is serialized and stored as:

   > geometry_simulation_dict.pkl: for all geometries and all simulation results
   > propeller_simulation_data_for_one.pkl: for current geometry’s simulation batch

---

## 🧰 Dependencies

The code is written in Python 3.8 and uses the following libraries(at leastpython3.1):

```bash
numpy
scipy
pandas
ctypes
re
pickle
typing
```

---

## 🚀 How to Use the Code?

1. Clone the repository:

   ```bash
   git clone https://github.com/yourname/propeller_simulation.git
   ```

2. Navigate to the simulation folder:

   ```bash
   cd propeller_simulation/Simulation_QBlade
   ```

3. Refer to `example.ipynb` to understand:
   - how to initialize the `SIMULATION` class  
   - how to run simulations  
   - how to change the geometry and update `.bld` files  
   - how to save and load result dictionaries

4. After changing propeller geometry, the system will automatically save simulation results into:

   ```Bash
   geometry_simulation_dict.pkl
   propeller_simulation_data_for_one.pkl
   ```

   These can be reloaded without re-running simulations.

---

## 🗂️ Repository Structure

```Bash
propeller_simulation/
│
├── Simulation_QBlade/             # Core Python simulation code
│   ├── class_sim/simulation.py    # SIMULATION class definition
│   ├── example.ipynb              # Usage examples
│
├── Qblade_data/                   # Input files for QBlade DLL interface
│   ├── QBlade_bld/                # .bld geometry files
│   ├── QBlade_sim/                # .sim simulation parameter templates
│   ├── QBR_file/                  # QBlade project files (.qpr/.qbr)
│   └── QBladeCE_2.0.6.dll         # DLL interface (not included in repo)
```

---

## ⚠️ Notes

- The QBlade DLL version used is **2.0.6.4**
- The `.dll` file is **not included** due to licensing; please download it manually from the [QBlade official site](https://qblade.org/)
- The code is under continuous development — bug reports and suggestions are welcome!

---

## ✍️ Author

Guo Yue – [github.com/guoyue0412](https://github.com/guoyue0412)

If you'd like to contribute, please fork the repository and submit a pull request!

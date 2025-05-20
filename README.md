# ☀️ solarwind-analysis-tools

Tools for processing, analyzing, and visualizing solar wind and geomagnetic data from the ACE satellite and OMNI datasets.  
Developed as part of a doctoral research project in Space Geophysics at INPE (Brazil).

## 📦 Description

**solarwind-analysis-tools** is a Python-based tool for merging, analyzing, and plotting solar wind and geomagnetic parameters from ACE and OMNI data.  
It enables quick diagnostics and high-resolution visualizations of space weather conditions, suitable for both research and educational purposes.

## 🚀 Features

- Automatic merge of ACE and OMNI datasets by timestamp  
- Clean, high-resolution plots for:
  - Magnetic field components (Bx, By, Bz)
  - Proton density
  - Solar wind speed
  - Dynamic pressure
  - Proton temperature (with scientific notation)
  - Alpha ratio
  - Sym-H index
  - AE index
- Customizable date ranges for event selection
- Figures suitable for scientific presentations and publications

## 📁 Project Structure

```plaintext
solarwind-analysis-tools/
├── data/                # Raw HDF and TXT data files
├── figures/             # Output directory for generated plots
├── notebooks/           # Optional Jupyter notebooks
├── solarwind_plot.py    # Main plotting script
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
```

## 🧑‍💻 How to Use

### 1. Prepare Your Data

Place the required data files inside the `data/` folder:

- **ACE data**: `ACE_BROWSE_YYYY-001_to_YYYY-365.HDF`
- **OMNI data**: `omni_minYYYY.dat`

> Note: Make sure the files follow the correct naming convention and format.

---

### 2. Set the Time Range

Open `solarwind_plot.py` and define the desired time interval:

```python
start = '2015-06-21 00:00:00'
end   = '2015-06-23 23:59:59'
```

---

### 3. Run the Script

From your terminal, run:

```bash
python solarwind_plot.py
```

The output figure will be saved in the `figures/` folder as a PNG file.

---

## 📊 Example Output

*You can add an example plot image here once generated, e.g.:*

![Example Plot](figures/example_plot.png)

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `pandas`
- `numpy`
- `matplotlib`
- `h5py`

**Optional:**
- `jupyter` — for notebook-based exploration

---

## 🧪 Applications

- Event-based case studies (e.g., ICMEs, HSSs, geomagnetic storms)
- Preprocessing for statistical space weather analysis
- High-quality plots for scientific communication
- Educational demonstrations of solar wind dynamics

---

## 📄 License

Distributed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this code with proper attribution.

---

## 🙋‍♀️ Author

**Karen Ferreira**  
Doctoral Researcher in Space Geophysics  
[INPE - Instituto Nacional de Pesquisas Espaciais, Brazil](https://www.inpe.br)  
GitHub: [@karen-ferreira](https://github.com/karen-ferreira)

---

> Feel free to contribute, suggest improvements, or report issues!

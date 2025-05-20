# ☀️ solarwind-analysis-tools

Tools for processing, analyzing, and visualizing solar wind and magnetospheric data from ACE satellite and OMNI data.  
Developed as part of a doctoral research project in Space Geophysics at INPE (Brazil).

## 📦 Description

This Python-based tool merges and plots solar wind and geomagnetic parameters from various datasets (e.g., ACE, OMNI), allowing quick diagnostics and high-resolution visual analysis of space weather conditions.

## 🚀 Features

- Automatic merge of ACE and OMNI datasets by timestamp  
- High-resolution plots of:
  - Magnetic field (Bx, By, Bz)
  - Proton density
  - Solar wind speed
  - Dynamic pressure
  - Proton temperature
  - Alpha ratio
  - Sym-H index
  - AE index
- Customizable date ranges
- Scientific, clean, and readable figures for publications or reports

## 📁 File Structure

```plaintext
solarwind-analysis-tools/
├── data/                # Raw HDF and TXT data files
├── figures/             # Automatically generated figures
├── notebooks/           # Jupyter notebooks (optional)
├── solarwind_plot.py    # Main analysis script
├── requirements.txt     # Python dependencies
└── README.md            # You are here

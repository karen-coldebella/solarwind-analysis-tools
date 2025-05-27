# Solar Wind Analysis Tools

Tools for analyzing solar wind conditions using data from the ACE (Advanced Composition Explorer) mission and OMNIweb data.  
This repository provides Python scripts to load, process, and visualize key solar wind parameters such as magnetic field intensity, solar wind speed, proton density, and more.

---

## Features

- Reads CDF data files from ACE (MAG and SWEPAM instruments) and OMNI.
- Processes solar wind data for specific time intervals.
- Generates and saves plots for scientific analysis and reporting.

---

## Download the data
https://cdaweb.gsfc.nasa.gov/


## Requirements

- Python 3.9 or later
- Required Python packages:
  - `pandas`
  - `matplotlib`
  - `cdflib`

### Install with pip:

```bash
pip install pandas matplotlib cdflib
```

### Or, if using a Conda environment:

```bash
conda install pandas matplotlib -c conda-forge
pip install cdflib
```

---

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/solarwind-analysis-tools.git
cd solarwind-analysis-tools
```

Activate your environment (if applicable):

```bash
conda activate solarwind-analysis-tools
```

Ensure the required libraries are installed (see section above).

---

## Usage

To run the main analysis script:

```bash
python ace_solarwind_analysis.py
```

By default, it processes the available CDF files in the `solarwind-data` directory and generates plots saved in the `Figures` folder.

---

## Project Structure

```
solarwind-analysis-tools/
├── ace_solarwind_analysis.py       # Main analysis script
├── README.md                       # Project description
├── LICENSE                         # License information
├── Figures/                        # Output plots
│   └── May30.png
├── solarwind-data/                 # Input data (CDF files)
│   ├── ac_h0s_mfi_*.cdf
│   ├── ac_h0s_swe_*.cdf
│   └── omni_hro2s_*.cdf
└── .gitignore
```

---

## Author

Developed by **Karen Ferreira**  
📧 karen.coldebella@gmail.com

---

## License

This project is licensed under the [MIT License](LICENSE).

"""
Reading and processing OMNI Web CDF data, manually downloaded from
https://cdaweb.gsfc.nasa.gov/. Data covers the period from 2013-05-30 to 2013-06-02.
Includes solar wind parameters, magnetic field components, geomagnetic indices, and alpha ratio (NaNp_Ratio).

PySPEDAS was not used because the alpha-ratio variable is not available when downloading data through it.

Author: Karen Ferreira  
August 2025
"""


import os
import cdflib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# 1 INTERVALO DE TEMPO
start_time = datetime(2013, 5, 30, 0, 0)
end_time = datetime(2013, 6, 3, 0, 0)


# 2 CAMINHO E ARQUIVO OMNI
caminho = r'C:\Users\karen\Programas\solarwind-analysis-tools\Data\OMNI'
#O arquivo contém dados de 28 de maio a 05 de setembro de 2013
#cobrindo o período em que houveram CIR enquanto THE e VAP estavam com apogeu no lado noturno da magnetosfera
arquivo_omni = 'omni_hro2s_1min_20130528000000_20130905000000_cdaweb.cdf'
caminho_arquivo_omni = os.path.join(caminho, arquivo_omni)


# 3 ABRIR ARQUIVO OMNI
cdf_omni = cdflib.CDF(caminho_arquivo_omni)
# Exibir variáveis
print(cdf_omni.cdf_info().rVariables)
print(cdf_omni.cdf_info().zVariables)


# 4 EXTRACAO VARIÁVEIS OMNI
epoch_raw_omni = cdf_omni.varget('Epoch')
epoch_omni = cdflib.cdfepoch.to_datetime(epoch_raw_omni)
epoch_np_omni = np.array(epoch_omni, dtype='datetime64[ns]')

SYMH_index = cdf_omni.varget('SYM_H')
AE_index = cdf_omni.varget('AE_INDEX')
flow_speed = cdf_omni.varget('flow_speed')
proton_density = cdf_omni.varget('proton_density')
temperature = cdf_omni.varget('T')

# Campo magnético componentes
Bz = cdf_omni.varget('BZ_GSE')
Bx = cdf_omni.varget('BX_GSE')
By = cdf_omni.varget('BY_GSE')

# B total (magnitude do campo B) 
Btotal = cdf_omni.varget('F')

# Beta do plasma
beta_plasma = cdf_omni.varget('Beta')

# Alpha ratio: no seu arquivo OMNI a variável que pode se aproximar é 'NaNp_Ratio'
# que é a razão de densidade de Na (alfa) para Np (prótons)
alpha_ratio = cdf_omni.varget('NaNp_Ratio')


# 5 TRATAMENTO DOS DADOS
fill_values = [-1.00e+31, 99999.898438, 999.98999, 9999999.0, 9.999, 9999.990234]  # adiciona o novo valor que aparece

def clean_data(array):
    for val in fill_values:
        array = np.where(np.isclose(array, val, atol=1e-5), np.nan, array)
    return array

# Limpar dados espúrios OMNI
SYMH_index = clean_data(SYMH_index)
AE_index = clean_data(AE_index)
flow_speed = clean_data(flow_speed)
proton_density = clean_data(proton_density)
temperature = clean_data(temperature)

Bx = clean_data(Bx)
By = clean_data(By)
Bz = clean_data(Bz)
Btotal = clean_data(Btotal)

beta_plasma = clean_data(beta_plasma)
alpha_ratio = clean_data(alpha_ratio)

# 6 TRANSFORMAR DADOS TRATADOS EM DATAFRAME OMNI
df_omni = pd.DataFrame({
    'SYMH_index': SYMH_index,
    'AE_index': AE_index,
    'flow_speed': flow_speed,
    'proton_density': proton_density,
    'temperature': temperature,
    'Bx': Bx,
    'By': By,
    'Bz': Bz,
    'Btotal': Btotal,
    'beta_plasma': beta_plasma,
    'alpha_ratio': alpha_ratio
}, index=pd.to_datetime(epoch_np_omni))


# 7 CRIAR OS GRÁFICOS

# Filtrar df_omni pelo intervalo de tempo desejado
df_plot = df_omni[(df_omni.index >= start_time) & (df_omni.index <= end_time)]

epoch = df_plot.index

fig, axs = plt.subplots(10, 1, figsize=(14, 20), sharex=True)
fig.subplots_adjust(hspace=0.25)

# SYMH_index
axs[0].plot(epoch, df_plot['SYMH_index'], color='black')
axs[0].set_ylabel('SYMH (nT)', fontsize=14)
axs[0].grid()
axs[0].tick_params(axis='both', labelsize=12)
axs[0].text(0.02, 0.85, '(a)', transform=axs[0].transAxes, fontsize=14, fontweight='bold')

# AE_index
axs[1].plot(epoch, df_plot['AE_index'], color='black')
axs[1].set_ylabel('AE (nT)', fontsize=14)
axs[1].grid()
axs[1].tick_params(axis='both', labelsize=12)
axs[1].text(0.02, 0.85, '(b)', transform=axs[1].transAxes, fontsize=14, fontweight='bold')

# flow_speed
axs[2].plot(epoch, df_plot['flow_speed'], color='black')
axs[2].set_ylabel('Flow Speed (km/s)', fontsize=14)
axs[2].grid()
axs[2].tick_params(axis='both', labelsize=12)
axs[2].text(0.02, 0.85, '(c)', transform=axs[2].transAxes, fontsize=14, fontweight='bold')

# proton_density
axs[3].plot(epoch, df_plot['proton_density'], color='black')
axs[3].set_ylabel('Proton Density (cm⁻³)', fontsize=14)
axs[3].grid()
axs[3].tick_params(axis='both', labelsize=12)
axs[3].text(0.02, 0.85, '(d)', transform=axs[3].transAxes, fontsize=14, fontweight='bold')

# temperature
axs[4].plot(epoch, df_plot['temperature'], color='black')
axs[4].set_ylabel('Temperature (K)', fontsize=14)
axs[4].grid()
axs[4].tick_params(axis='both', labelsize=12)
axs[4].text(0.02, 0.85, '(e)', transform=axs[4].transAxes, fontsize=14, fontweight='bold')

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((3, 5))
axs[4].yaxis.set_major_formatter(formatter)

# Btotal
axs[5].plot(epoch, df_plot['Btotal'], color='black')
axs[5].set_ylabel('|B| (nT)', fontsize=14)
axs[5].grid()
axs[5].tick_params(axis='both', labelsize=12)
axs[5].text(0.02, 0.85, '(f)', transform=axs[5].transAxes, fontsize=14, fontweight='bold')

# Bz
axs[6].plot(epoch, df_plot['Bz'], color='black')
axs[6].set_ylabel('Bz (nT) GSE', fontsize=14)
axs[6].grid()
axs[6].tick_params(axis='both', labelsize=12)
axs[6].text(0.02, 0.85, '(g)', transform=axs[6].transAxes, fontsize=14, fontweight='bold')

# Bx and By
axs[7].plot(epoch, df_plot['Bx'], color='black', label='Bx')
axs[7].plot(epoch, df_plot['By'], color='red', label='By')
axs[7].set_ylabel('B (nT) GSE', fontsize=14)
axs[7].grid()
axs[7].tick_params(axis='both', labelsize=12)
axs[7].legend(loc='upper right', fontsize='small')
axs[7].text(0.02, 0.85, '(h)', transform=axs[7].transAxes, fontsize=14, fontweight='bold')

# beta_plasma
axs[8].plot(epoch, df_plot['beta_plasma'], color='black')
axs[8].set_ylabel('Plasma Beta', fontsize=14)
axs[8].grid()
axs[8].tick_params(axis='both', labelsize=12)
axs[8].text(0.02, 0.85, '(i)', transform=axs[8].transAxes, fontsize=14, fontweight='bold')

# alpha_ratio
axs[9].plot(epoch, df_plot['alpha_ratio'], color='black')
axs[9].set_ylabel('Alpha Ratio', fontsize=14)
axs[9].grid()
axs[9].tick_params(axis='both', labelsize=12)
axs[9].set_xlabel('Time (UT)', fontsize=14)

#Configura as ticks no eixo x
locator = mdates.HourLocator(interval=6)
def custom_formatter(x, pos):
    dt = mdates.num2date(x)
    if dt.hour == 0 and dt.minute == 0:
        return dt.strftime('%m/%d %H:%M')
    else:
        return dt.strftime('%H:00')

axs[9].xaxis.set_major_locator(locator)
axs[9].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
axs[9].text(0.02, 0.85, '(j)', transform=axs[9].transAxes, fontsize=14, fontweight='bold')

plt.setp(axs[9].get_xticklabels(), rotation=45)
fig.align_ylabels(axs)
plt.tight_layout()

output_folder = r'C:\Users\karen\Programas\solarwind-analysis-tools\Figures'
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, 'OMNI_event_1.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
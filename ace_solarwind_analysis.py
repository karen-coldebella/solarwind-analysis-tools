import os
import pandas as pd
import cdflib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.ticker import ScalarFormatter


# 1 INTERVALO DE TEMPO
# === Definir intervalo de tempo desejado ===
start_time = datetime(2013, 5, 31, 12, 0)
end_time = datetime(2013, 6, 3, 0, 0)


# 2 ABRIR OS DADOS E EXTRAIR VARIÁVEIS
# === Caminho para os dados ===
caminho = r'C:\Users\karen\Programas\solarwind-analysis-tools\solarwind-data'

arquivo_swe = 'ac_h0s_swe_20130530000009_20130605235904_cdaweb.cdf'
arquivo_mfi = 'ac_h0s_mfi_20130530000004_20130605235947_cdaweb.cdf'
arquivo_omni = 'omni_hro2s_1min_20130530000000_20130606000000_cdaweb.cdf'

caminho_arquivo_swe = os.path.join(caminho, arquivo_swe)
caminho_arquivo_mfi = os.path.join(caminho, arquivo_mfi)
caminho_arquivo_omni = os.path.join(caminho, arquivo_omni)

# === Abrir arquivos ===
cdf_swe = cdflib.CDF(caminho_arquivo_swe)
cdf_mfi = cdflib.CDF(caminho_arquivo_mfi)
cdf_omni = cdflib.CDF(caminho_arquivo_omni)

# === Ver informações dos CDFs ===
info_swe = cdf_swe.cdf_info()
print('SWE rVariables:', info_swe.rVariables)
print('SWE zVariables:', info_swe.zVariables)

info_mfi = cdf_mfi.cdf_info()
print('MFI rVariables:', info_mfi.rVariables)
print('MFI zVariables:', info_mfi.zVariables)

info_omni = cdf_omni.cdf_info()
print('OMNI rVariables:', info_omni.rVariables)
print('OMNI zVariables:', info_omni.zVariables)


# === Extrair variáveis SWE ===
epoch_raw_swe = cdf_swe.varget('Epoch')
epoch_swe = cdflib.cdfepoch.to_datetime(epoch_raw_swe)
epoch_np_swe = np.array(epoch_swe, dtype='datetime64[ns]')

Np = cdf_swe.varget('Np')           # Densidade do vento solar (cm⁻³)
Vp = cdf_swe.varget('Vp')           # Velocidade do vento solar (km/s)
Tpr = cdf_swe.varget('Tpr')         # Temperatura dos prótons (K)
alpha_ratio = cdf_swe.varget('alpha_ratio')  # Razão alfa


# === Extrair variáveis MFI ===
epoch_raw_mfi = cdf_mfi.varget('Epoch')
epoch_mfi = cdflib.cdfepoch.to_datetime(epoch_raw_mfi)
epoch_np_mfi = np.array(epoch_mfi, dtype='datetime64[ns]')

BGSEc = cdf_mfi.varget('BGSEc')        # Matriz (N, 3): Bx, By, Bz
Magnitude = cdf_mfi.varget('Magnitude')  # Módulo do campo magnético

# Separar componentes do campo magnético
Bx = BGSEc[:, 0]
By = BGSEc[:, 1]
Bz = BGSEc[:, 2]


# === Extrair variáveis OMNI ===
epoch_raw_omni = cdf_omni.varget('Epoch')
epoch_omni = cdflib.cdfepoch.to_datetime(epoch_raw_omni)
epoch_np_omni = np.array(epoch_omni, dtype='datetime64[ns]')

AE_index = cdf_omni.varget('AE_INDEX')           # Índice AE OMNI 
SYMH_index = cdf_omni.varget('SYM_H')           # Índice DST OMNI 


# 3 TRATAMENTO DOS DADOS
fill_value = -1.00e+31

# === Limpar dados espúrios SWE ===
Np = np.where(Np == fill_value, np.nan, Np)
Vp = np.where(Vp == fill_value, np.nan, Vp)
Tpr = np.where(Tpr == fill_value, np.nan, Tpr)
alpha_ratio = np.where(alpha_ratio == fill_value, np.nan, alpha_ratio)

# === Limpar dados espúrios MFI ===
BGSEc[BGSEc == fill_value] = np.nan
Magnitude = np.where(Magnitude == fill_value, np.nan, Magnitude)

# === Limpar dados espúrios OMNI ===
AE_index = np.where(AE_index == fill_value, np.nan, AE_index)
SYMH_index = np.where(SYMH_index == fill_value, np.nan, SYMH_index)


# 4 TRANSFORMAR DADOS TRATADOS EM DATAFRAMES
# === DataFrame com dados SWE (64s) ===
df_swe = pd.DataFrame({
    'Np': Np,
    'Vp': Vp,
    'Tpr': Tpr,
    'alpha_ratio': alpha_ratio
}, index=pd.to_datetime(epoch_np_swe))

# === DataFrame com dados MFI (16s) ===
df_mfi = pd.DataFrame({
    'Bx': Bx,
    'By': By,
    'Bz': Bz,
    'Bmag': Magnitude
}, index=pd.to_datetime(epoch_np_mfi))

# === DataFrame com dados OMNI (1 min) ===
df_omni = pd.DataFrame({
    'AE_index': AE_index,
    'SYMH_index': SYMH_index,
}, index=pd.to_datetime(epoch_np_omni))


# 5 AJUSTAR A CADÊNCIA DOS DADOS
# === Reamostrar todos os dados para 1 minuto ===
df_swe_1min = df_swe.resample('1min').mean()
df_mfi_1min = df_mfi.resample('1min').mean()
# df_omni já está em 1min

# === Filtrar intervalo de tempo ===
df_swe_1min = df_swe_1min[(df_swe_1min.index >= start_time) & (df_swe_1min.index <= end_time)]
df_mfi_1min = df_mfi_1min[(df_mfi_1min.index >= start_time) & (df_mfi_1min.index <= end_time)]
df_omni = df_omni[(df_omni.index >= start_time) & (df_omni.index <= end_time)]

# === Interpolar dados faltantes (NaNs) com base no tempo ===
df_swe_1min = df_swe_1min.interpolate(method='time')
df_mfi_1min = df_mfi_1min.interpolate(method='time')
df_omni = df_omni.interpolate(method='time')



# 6 COMBINAR DADOS DOS INSTRUMENTOS EM UM DATAFRAME
# === Alinhar os dados usando merge_asof ===
# Garantir que estão ordenados
df_swe_1min = df_swe_1min.sort_index()
df_mfi_1min = df_mfi_1min.sort_index()
df_omni = df_omni.sort_index()

# Primeiro merge: SWE + MFI
df_temp = pd.merge_asof(
    df_swe_1min, df_mfi_1min,
    left_index=True, right_index=True,
    direction='nearest', tolerance=pd.Timedelta(seconds=30)
)

# Segundo merge: o resultado + OMNI
df_merged = pd.merge_asof(
    df_temp, df_omni,
    left_index=True, right_index=True,
    direction='nearest', tolerance=pd.Timedelta(seconds=30)
)


# 7 CRIAR OS GRÁFICOS
# === Definir tempo ===
epoch = df_merged.index  # índice datetime após merge

# === Criar subplots ===
fig, axs = plt.subplots(9, 1, figsize=(14, 18), sharex=True)
fig.subplots_adjust(hspace=0.2)

# === Plotar índice DST (SYMH_index) ===
axs[0].plot(epoch, df_merged['SYMH_index'], color='black')
axs[0].set_ylabel('SYMH (nT)')
axs[0].grid()
axs[0].tick_params(labelbottom=False)

# === Plotar índice AE (AE_index) ===
axs[1].plot(epoch, df_merged['AE_index'], color='black')
axs[1].set_ylabel('AE (nT)')
axs[1].grid()
axs[1].tick_params(labelbottom=False)

# === Plotar Vp ===
axs[2].plot(epoch, df_merged['Vp'], color='black')
axs[2].set_ylabel('Vp (km/s)')
axs[2].grid()
axs[2].tick_params(labelbottom=False)

# === Plotar Np ===
axs[3].plot(epoch, df_merged['Np'], color='black')
axs[3].set_ylabel('Np (cm⁻³)')
axs[3].grid()
axs[3].tick_params(labelbottom=False)

# === Plotar Tpr ===
axs[4].plot(epoch, df_merged['Tpr'], color='black')
axs[4].set_ylabel('Tpr (K)')
axs[4].grid()
axs[4].tick_params(labelbottom=False)

# Exibir Tpr em notação científica
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((3, 5))  # mostra notação científica se os valores forem 1e3 ou maiores
axs[4].yaxis.set_major_formatter(formatter)

# === Plotar |B| ===
axs[5].plot(epoch, df_merged['Bmag'], color='black')
axs[5].set_ylabel('|B| (nT)')
axs[5].grid()
axs[5].tick_params(labelbottom=False)

# === Plotar Bz ===
axs[6].plot(epoch, df_merged['Bz'], color='black')
axs[6].set_ylabel('Bz (nT)')
axs[6].grid()
axs[6].tick_params(labelbottom=False)

# === Plotar Bx e By no mesmo painel ===
axs[7].plot(epoch, df_merged['Bx'], color='black', label='Bx')
axs[7].plot(epoch, df_merged['By'], color='red', label='By')
axs[7].set_ylabel('B (nT)')
axs[7].grid()
axs[7].tick_params(labelbottom=False)
axs[7].legend(loc='upper right', fontsize='small')

# === Plotar alpha_ratio ===
axs[8].plot(epoch, df_merged['alpha_ratio'], color='black')
axs[8].set_ylabel('α ratio')
axs[8].grid()
axs[8].tick_params(labelbottom=True)
axs[8].set_xlabel('Time (UT)')
# Formatador personalizado:
locator = mdates.HourLocator(interval=3)
def custom_formatter(x, pos):
    dt = mdates.num2date(x)
    if dt.hour == 0 and dt.minute == 0:
        return dt.strftime('%m/%d %H:%M')
    else:
        return dt.strftime('%H:00')
axs[8].xaxis.set_major_locator(locator)
axs[8].xaxis.set_major_formatter(FuncFormatter(custom_formatter))


# === Ajustar layout e salvar a figura ===
plt.setp(axs[8].get_xticklabels(), rotation=45)
fig.align_ylabels(axs)  # Alinha os ylabels
plt.tight_layout()
plt.savefig('Figures/May30.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. Cargar dataset
df = pd.read_csv('data/dataset_final_1.csv', sep=';', decimal=',')
df.columns = df.columns.str.strip()  # Limpiar nombres de columnas

# 2. Crear diccionario de DataFrames por estación
dfs_estaciones = {
    cod: df_estacion.drop(columns='cod_est').reset_index(drop=True)
    for cod, df_estacion in df.groupby('cod_est')
}

# 3. Selección de estación Bernabéu
cod_bernabeu = 1006
cod_valdeacederas = 102
cod_begona = 1002
cod_parqueavendias = 708
bernabeu = dfs_estaciones[cod_bernabeu]

# 4. Información descriptiva
print("Head de bernabeu:")
print(bernabeu.head())

print("\nNombres de las variables:")
print(bernabeu.columns.tolist())

print("\nClases de las variables:")
print(bernabeu.dtypes)

print("\nDimensiones del DataFrame (filas, columnas):")
print(bernabeu.shape)

print("\nResumen estadístico:")
print(bernabeu.describe())

# 5. Análisis de correlación con 'ca_entradas'
correlaciones = bernabeu.corr(numeric_only=True)
correlaciones_entradas = correlaciones['ca_entradas'].drop('ca_entradas')

# 6. Filtrar variables con correlación significativa (|r| > 0.1)
umbral = 0.1
variables_significativas = correlaciones_entradas[correlaciones_entradas.abs() > umbral]
columnas_significativas = variables_significativas.index.tolist() + ['ca_entradas']
bernabeu_filtrado = bernabeu[columnas_significativas]

# 7. Heatmap de correlación (solo variables significativas)
plt.figure(figsize=(10, 8))
sns.heatmap(
    bernabeu_filtrado.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    annot_kws={"size": 4}
)
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.title("Matriz de correlación - Variables significativas (Bernabéu)", fontsize=14)
plt.tight_layout()
plt.show()

# 8. Cálculo de p-valores
cols = bernabeu_filtrado.columns
corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

for col1 in cols:
    for col2 in cols:
        r, p = pearsonr(bernabeu_filtrado[col1], bernabeu_filtrado[col2])
        corr_matrix.loc[col1, col2] = r
        pval_matrix.loc[col1, col2] = p

# 9. Heatmap de correlaciones estadísticamente significativas (p < 0.05)
plt.figure(figsize=(10, 8))
mask_insignificante = pval_matrix >= 0.05
sns.heatmap(
    corr_matrix,
    mask=mask_insignificante,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    annot_kws={"size": 6}
)
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.title("Correlaciones significativas (p < 0.05) - Bernabéu", fontsize=12)
plt.tight_layout()
plt.show()

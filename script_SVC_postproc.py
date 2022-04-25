"""
Descripción del script:
    1) Enlazar los ficheros .sig con los ids reales de los plots --> Función relate_index_with_file_name_GLOBAL
    2) Unir el dataframe obtenido en el paso 1 con el dataframe que tengo de cada plot con todos los atributos.
    3) Extraer los metadatos y datos espectrales con la librería specdal
    4) Obtener los resultados y las gráficas de los datos de calibración del SVC frente a la cámara MX
"""


#Generic imports
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from rasterstats import zonal_stats

#Local imports
from specdal import Collection, Spectrum


def relate_index_with_file_name_GLOBAL(datadir):
    # Extraer lista de rutas
    list_rutas = []
    df_global = pd.DataFrame(columns=['filas', 'columnas', 'id', 'file_names', 'tipo'])
    for folder_tipo in os.listdir(datadir)[0:2]:
        for folder in os.listdir(datadir + folder_tipo):
            # list_rutas.append(datadir+folder_tipo+'/'+folder+'/')
            df = relate_index_with_file_name(datadir, folder_tipo, folder)
            df_global = pd.concat([df_global, df])
            # df_global = df_global.concat(df)
    # Escribir en un CSV
    df_global.to_csv(datadir + 'files_and_ids_plot.csv')
    return df_global



def relate_index_with_file_name(datadir,tipo,folder):
    datadir = datadir + '/' + tipo + '/' + folder
    # Proceso
    fila_inicial = int(str(folder[14:16]))
    fila_final = int(str(folder[17:19]))
    columna = int(str(folder[7:9]))

    if fila_inicial > fila_final:
        step = -1
    else:
        step = 1

    filas = np.arange(fila_inicial, fila_final - 1, step, dtype='int')
    columnas = np.ones(len(filas), dtype='int') * columna
    # Convert to dataframe
    df_filas = pd.DataFrame(filas, columns=['filas'])
    df_columnas = pd.DataFrame(columnas, columns=['columnas'])
    # Merge dataframes
    df_data = df_filas.merge(df_columnas, how='left', left_index=True, right_index=True)
    # Calculate id with columns*1000 + rows
    df_data['id'] = df_data['columnas'] * 1000 + df_data['filas']

    # Asociar nombre fichero a la id
    files_names = []
    for f in sorted(os.listdir(datadir)):
        if f.endswith('.sig'):
            files_names.append(f)
    # Eliminar los target del panel en blanco y de los carteles informativos en el inicio y el fin
    del files_names[0:2]  # Target del panel y cartel informativo inicio
    del files_names[-1]  # Cartel informativo final

    # Añadir el nombre del fichero al dataframe
    df_data['file_names'] = files_names
    # Añadir el tipo
    df_data['tipo'] = tipo
    return df_data

def join_measures_with_attributes(att_dir):




if __name__ == "__main__":
    df_measures = relate_index_with_file_name_GLOBAL(datadir="Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_04_22/classified/")
    join_measures_with_attributes(att_dir="Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/01-attributes")


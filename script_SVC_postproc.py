"""
Descripción del script:
    1) Enlazar los ficheros .sig con los ids reales de los plots --> Función relate_index_with_file_name_GLOBAL
    2) Unir el dataframe obtenido en el paso 1 con el dataframe que tengo de cada plot con todos los atributos.
    3) Extraer los metadatos y datos espectrales con la librería specdal
    4) Obtener los resultados y las ¿gráficas? de los datos de calibración del SVC frente a la cámara MX
"""

#Generic imports
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sb
from rasterstats import zonal_stats

#Local imports
from specdal import Collection, Spectrum


def relate_index_with_file_name_GLOBAL(datadir):
    #Remove file if exists
    file_exists = exists(datadir + 'files_and_ids_plot.csv')
    if file_exists:
        os.remove(datadir + 'files_and_ids_plot.csv')
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
            files_names.append(f[0:-4])
    # Eliminar los target del panel en blanco y de los carteles informativos en el inicio y el fin
    del files_names[0:2]  # Target del panel y cartel informativo inicio
    del files_names[-1]  # Cartel informativo final

    # Añadir el nombre del fichero al dataframe
    df_data['file_name'] = files_names
    # Añadir el tipo
    df_data['tipo'] = tipo
    return df_data

def join_measures_with_attributes(att_dir_riego, att_dir_secano):
    """

    :param att_dir_riego: Path to riego-plots shapefile with attributes (ids, variety, etc.)
    :param att_dir_secano: Path to secano-plots shapefile with attributes (ids, variety, etc.)
    :return: dataframe merged with the measures of the SVC and attributes of the plots (riego and secano)
    """
    # Path for shapefile templates
    att_dir_riego = att_dir_riego
    att_dir_secano = att_dir_secano
    # Read shapefile with geopandas
    df_riego_att = gpd.read_file(att_dir_riego).set_index('id_real').rename(
        columns={"tipo": "tipo_plantilla", "fila_real": "fila_plantilla",
                 "columna_re": "columna_plantilla", "numero_var": "numero_variedad",
                 "nombre_var": "nombre_variedad"})
    df_secano_att = gpd.read_file(att_dir_secano).set_index('id_real').rename(
        columns={"tipo": "tipo_plantilla", "fila_real": "fila_plantilla",
                 "columna_re": "columna_plantilla", "numero_var": "numero_variedad",
                 "nombre_var": "nombre_variedad"})
    # Divide original df into riego and secano for merging
    df_measures_riego = df_measures.loc[df_measures['tipo'] == 'riego'].set_index('id')
    df_measures_secano = df_measures.loc[df_measures['tipo'] == 'secano'].set_index('id')
    # Merge dataframes
    df_riego = pd.merge(df_measures_riego, df_riego_att, how='inner', right_index=True, left_index=True)
    df_secano = pd.merge(df_measures_secano, df_secano_att, how='inner', right_index=True, left_index=True)
    #Concat both measures
    df_measures_att = pd.concat([df_riego, df_secano])
    return df_measures_att

def extract_metadata_and_data_SVC(data_path):
    datadir = data_path
    c = Collection(name='santaella')
    metadata = []
    for f in sorted(os.listdir(datadir)):
        if f.endswith('.sig'):
            spectrum = Spectrum(filepath=os.path.join(datadir, f))
            c.append(spectrum)
            df = pd.DataFrame([spectrum.metadata])
            metadata.append(df)
    df_metadata = pd.concat(metadata)

    return df_metadata, c.data

def export_metadata_to_csv(export_path, metadata, data, df_measures):
    # Exportar a un CSV (df_metadata y data)
    metadata.to_csv(export_path + 'metadata.csv', encoding='utf-8', index=False)
    data.to_csv(export_path + 'data.csv', encoding='utf-8')
    # SE UNEN LOS DATAFRAMES DE MEDIDAS SVC Y EL DE METADATOS PARA TENER SOLAMENTE LAS MEDIDAS DE LOS PLOTS EN UN CSV (metadata_filter.csv)
    df_measures_att = df_measures.reset_index().rename(columns={"index": "id"}).set_index('file_name')
    metadata = metadata.set_index('file_name')

    metadata_filter = pd.merge(metadata, df_measures_att, how='inner', left_index=True, right_index=True)
    metadata_filter = metadata_filter.drop(['file_names'], axis=1).reset_index()
    # Filtrar el dataset de data con los ficheros exclusivamente de las medidas
    data_filter = data[metadata_filter['file_name']]

    # Write to a CSV file
    data_filter.to_csv(export_path+'data_filter.csv', encoding='utf-8', index=False)
    metadata_filter.to_csv(export_path + 'metadata_filter.csv', encoding='utf-8', index=False)

    return metadata_filter, data_filter

def calibration_MX_and_SVC(metadata, data, indices_path, datadir, MX_filter_path):
    """ Interpolación de todos los indices del dataframe """
    init_WV = 338
    end_WV = 2519
    interval = 1
    new_index = pd.Index(np.arange(init_WV, end_WV, interval))
    df_interpolate = data.reindex(new_index).interpolate()

    """Lectura de los filtros de los canales de BLUE, GREEN, RED, REDEDGE, NIR."""
    redEdge_filters = pd.read_csv(MX_filter_path, sep=';', index_col=0)   ######TENGO QUE CAMBIAR ESTA RUTA DEL CSV!!!!!
    redEdge_filters_4_plot = redEdge_filters.replace("%", "", regex=True).astype(float)
    redEdge_filters_numeric = (redEdge_filters.replace("%", "", regex=True).astype(float)) / 100

    """ SUMATORIO DE LA RADIANCIA PARA CADA NIVEL DE TRANSMITANCIA """
    # Create empty lists for indexing values
    CH_blue = []
    CH_green = []
    CH_red = []
    CH_rededge = []
    CH_infrared = []

    # Iterate for each measure file ['220402_0922_R001_T002']
    for file in list(df_interpolate.columns):
        CH_blue_value_SVC = (redEdge_filters_numeric['Band 1'] * df_interpolate[file]).dropna().sum() / \
                            redEdge_filters_numeric['Band 1'].sum()
        CH_green_value_SVC = (redEdge_filters_numeric['Band 2'] * df_interpolate[file]).dropna().sum() / \
                             redEdge_filters_numeric['Band 2'].sum()
        CH_red_value_SVC = (redEdge_filters_numeric['Band 3'] * df_interpolate[file]).dropna().sum() / \
                           redEdge_filters_numeric['Band 3'].sum()
        CH_redEdge_value_SVC = (redEdge_filters_numeric['Band 5'] * df_interpolate[file]).dropna().sum() / \
                               redEdge_filters_numeric['Band 5'].sum()
        CH_infraRed_value_SVC = (redEdge_filters_numeric['Band 4'] * df_interpolate[file]).dropna().sum() / \
                                redEdge_filters_numeric['Band 4'].sum()

        # Append values to list
        CH_blue.append(CH_blue_value_SVC)
        CH_green.append(CH_green_value_SVC)
        CH_red.append(CH_red_value_SVC)
        CH_rededge.append(CH_redEdge_value_SVC)
        CH_infrared.append(CH_infraRed_value_SVC)

    # Create pandas dataframe with all channels
    df_svc = pd.DataFrame(columns=['CH_blue_SVC', 'CH_green_SVC', 'CH_red_SVC', 'CH_redEdge_SVC', 'CH_infrared_SVC'],
                          index=df_interpolate.columns)
    df_svc['CH_blue_SVC'] = CH_blue
    df_svc['CH_green_SVC'] = CH_green
    df_svc['CH_red_SVC'] = CH_red
    df_svc['CH_redEdge_SVC'] = CH_rededge
    df_svc['CH_infrared_SVC'] = CH_infrared

    """ EXTRAER LAS MEDIDAS DE LOS CANALES (MX-RedEdge) PARA CADA PLOT """
    svc_medidas_atributos = pd.read_csv('Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_04_22/metadata_filter.csv')
    svc_medidas_atributos = gpd.GeoDataFrame(svc_medidas_atributos, geometry=gpd.GeoSeries.from_wkt(svc_medidas_atributos.geometry))

    #Read tiff files for get MX index
    blue_tiff = 'Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices/blue/Santaella_20220421_MX_index_blue.tif'
    green_tiff = 'Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices/green/Santaella_20220421_MX_index_green.tif'
    red_tiff = 'Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices/red/Santaella_20220421_MX_index_red.tif'
    rededge_tiff = 'Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices/red_edge/Santaella_20220421_MX_index_red_edge.tif'
    nir_tiff = 'Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices/nir/Santaella_20220421_MX_index_nir.tif'

    stats_computed = 'mean max min median std'
    #Calculate zonal statistics of channels and join with shapefile atributes
    df_blue_stats_MX = pd.DataFrame(zonal_stats(svc_medidas_atributos, blue_tiff, stats = stats_computed)).rename(columns={"min": "MX_min_BLUE", "max": "MX_max_BLUE", "mean": "MX_mean_BLUE", "median": "MX_median_BLUE", "std": "MX_std_BLUE"})
    df_green_stats_MX = pd.DataFrame(zonal_stats(svc_medidas_atributos, green_tiff, stats = stats_computed)).rename(columns={"min": "MX_min_GREEN", "max": "MX_max_GREEN", "mean": "MX_mean_GREEN", "median": "MX_median_GREEN", "std": "MX_std_GREEN"})
    df_red_stats_MX = pd.DataFrame(zonal_stats(svc_medidas_atributos, red_tiff, stats = stats_computed)).rename(columns={"min": "MX_min_RED", "max": "MX_max_RED", "mean": "MX_mean_RED", "median": "MX_median_RED", "std": "MX_std_RED"})
    df_redegde_stats_MX = pd.DataFrame(zonal_stats(svc_medidas_atributos, rededge_tiff, stats = stats_computed)).rename(columns={"min": "MX_min_REDEDGE", "max": "MX_max_REDEDGE", "mean": "MX_mean_REDEDGE", "median": "MX_median_REDEDGE", "std": "MX_std_REDEDGE"})
    df_nir_stats_MX = pd.DataFrame(zonal_stats(svc_medidas_atributos, nir_tiff, stats = stats_computed)).rename(columns={"min": "MX_min_NIR", "max": "MX_max_NIR", "mean": "MX_mean_NIR", "median": "MX_median_NIR", "std": "MX_std_NIR"})

    df_stats = svc_medidas_atributos.join(df_blue_stats_MX).join(df_green_stats_MX).join(df_red_stats_MX).join(df_redegde_stats_MX).join(df_nir_stats_MX)

    """
    DESCRIPTION: Combinar los dataframe de SVC con el de la cámara MX
    INPUT:
        df_stats   --> MX camera
        df_svc  --> SVC
    OUTPUT:
        df_stats_full --> Combining both dataframes
    """

    # Set index for merging with the same index both
    df_stats = df_stats.reset_index().set_index('file_name')
    # Merging both dataframe
    df_stats_full = pd.merge(df_stats, df_svc, left_index=True, right_index=True)
    df_stats_full = df_stats_full.reset_index().set_index('id')
    # Write into a csv file
    df_stats_full.to_csv(datadir + '/' + 'df_stats_calibration_SVC_and_MX.csv')




if __name__ == "__main__":
    """                
    INPUT:
            1) Path of data classified
            2) Indices path
    OUTPUT: 
            1) metadata and spectral data of all measures
            2) metadata and spectral data filtered, just measures of plots
            3) 'files_and_ids_plot.csv' - Archivo que enlaza el fichero con el id del plot
            4) stats calibration of SVC and MX for plotting and compared data
    """
    # CONTROL LOOP.

    #Este es el único path que se modifica.
    datadir_classified_measures = "Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_04_22/classified/"
    datadir_full = datadir_classified_measures[0:-11]
    ############################################################################################################
    df_measures = relate_index_with_file_name_GLOBAL(datadir=datadir_classified_measures)
    df_measures_att = join_measures_with_attributes(att_dir_riego="Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/01-attributes/riego/riego_shapefile_full_atributes.shp",
                                  att_dir_secano="Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/01-attributes/secano/secano_shapefile_full_atributes.shp")
    metadata, data = extract_metadata_and_data_SVC(data_path=datadir_full)
    metadata_filter, data_filter = export_metadata_to_csv(datadir_full, metadata, data, df_measures_att)
    calibration_MX_and_SVC(metadata=metadata_filter, data=data_filter, indices_path='', datadir=datadir_full,
                           MX_filter_path='report_RedEdge_3_Filters_srs.csv')
    print('Algoritmo FINALIZADO correctamente')

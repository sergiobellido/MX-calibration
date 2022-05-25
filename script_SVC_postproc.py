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
import math
from scipy import stats
import seaborn as sb
from rasterstats import zonal_stats

#Local imports
from specdal import Collection, Spectrum


def relate_index_with_file_name_GLOBAL(datadir, numero_medidas_plot):
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
            df = relate_index_with_file_name(datadir, folder_tipo, folder, numero_medidas_plot)
            df_global = pd.concat([df_global, df])
            # df_global = df_global.concat(df)
    # Escribir en un CSV
    df_global.to_csv(datadir + 'files_and_ids_plot.csv')
    return df_global

def relate_index_with_file_name(datadir,tipo,folder, numero_medidas_plot):
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
    #Añadir el número de medidas realizadas por plot
    filas = np.repeat(filas, numero_medidas_plot)
    columnas = np.repeat(columnas, numero_medidas_plot)

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

def join_measures_with_attributes(att_dir_riego, att_dir_secano, df_measures):
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
    #df_measures_att['id'] = df_measures_att['columnas'] * 1000 +  df_measures_att['filas']
    return df_measures_att

def extract_metadata_and_data_SVC(data_path, measures_download_phone):
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
    c.stitch()  #Suaviza y elimina los duplicados en los cambios de bandas
    c.jump_correct(splices=[970], reference=0)  # Elimina el salto del VIS al SWIR1.
    c.interpolate()     #Interpolar a longitudes de onda de la unidad.
    if measures_download_phone:
        #Modificar el nombre del fichero, porque cambia si descargas los datos con el tlf o con el SVC
        df_metadata['file_name'] = df_metadata.loc[:,'file_name'].str[-14::]

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
    data_filter.to_csv(export_path+'data_filter.csv', encoding='utf-8', index=True)
    metadata_filter.to_csv(export_path + 'metadata_filter.csv', encoding='utf-8', index=True)

    return metadata_filter, data_filter

def calibration_MX_and_SVC(metadata, data, datadir_px4_project, datadir, MX_filter_path):
    """ Interpolación de todos los indices del dataframe """
    '''init_WV = 338
    end_WV = 2519
    interval = 1
    new_index = pd.Index(np.arange(init_WV, end_WV, interval))
    data = data.reset_index().rename(columns={'index': 'wavelength'})
    df_interpolate = data.reindex(new_index).interpolate()'''
    df_interpolate = data

    """Lectura de los filtros de los canales de BLUE, GREEN, RED, REDEDGE, NIR."""
    redEdge_filters = pd.read_csv(MX_filter_path, sep=';', index_col=0)
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
    df_svc = pd.DataFrame(columns=['blue_SVC', 'green_SVC', 'red_SVC', 'red_edge_SVC', 'nir_SVC'],
                          index=df_interpolate.columns)
    df_svc['blue_SVC'] = CH_blue
    df_svc['green_SVC'] = CH_green
    df_svc['red_SVC'] = CH_red
    df_svc['red_edge_SVC'] = CH_rededge
    df_svc['nir_SVC'] = CH_infrared
    # Añadir índices SVC (NOTE: nir is infrared)
    blue = df_svc['blue_SVC']
    green = df_svc['green_SVC']
    red = df_svc['red_SVC']
    red_edge = df_svc['red_edge_SVC']
    nir = df_svc['nir_SVC']
    #Calculate SVC all the index
    df_svc['ndvi_SVC'] = (nir - red) / (nir + red)
    df_svc['gndvi_SVC'] = (nir - green) / (nir + green)
    df_svc['ndre_SVC'] = (nir - red_edge) / (nir + red_edge)
    df_svc['tgi_SVC'] = green - 0.39 * red - 0.61 * blue
    df_svc['gli_SVC'] = ((green - red) + (green - blue)) / (2 * green + red + blue)
    df_svc['tcari_SVC'] = 3 * ((red_edge - red) - 0.2 * (red_edge - green) * (red_edge / red))
    df_svc['osavi_SVC'] = (1 + 0.16) * (nir - red) / (nir + red + 0.16)
    df_svc['sccci_SVC'] = ((nir - red_edge) / (nir + red_edge)) / ((nir - red) / (nir + red))

    """ EXTRAER LAS MEDIDAS DE LOS CANALES (MX-RedEdge) PARA CADA PLOT """
    svc_medidas_atributos = metadata  #Metadata filter.csv
    svc_medidas_atributos = gpd.GeoDataFrame(svc_medidas_atributos, geometry=svc_medidas_atributos.geometry)

    #Read tiff files for get MX index
    datadir_indices = datadir_px4_project + '4_index/' + 'indices/'
    indices_paths_MX = []
    for indice in os.listdir(datadir_indices):
        for file in os.listdir(datadir_indices + indice + '/'):
            if file.endswith('.tif'):
                path = datadir_indices + indice + '/' + file
                indices_paths_MX.append(path)
    indices_names_MX = os.listdir(datadir_indices)

    #Calcular zonal statistics y unir los dataframes
    stats_computed = 'mean max min median std'
    contador = 0
    for tif in indices_paths_MX:
        df = pd.DataFrame(zonal_stats(svc_medidas_atributos, tif, stats=stats_computed)).add_suffix('_' + indices_names_MX[contador] + '_MX')
        if contador == 0:
            df_stats_MX = svc_medidas_atributos.join(df)
        else:
            df_stats_MX = df_stats_MX.join(df)
        contador += 1

    """
    DESCRIPTION: Combinar los dataframe de SVC con el de la cámara MX
    INPUT:
        df_stats   --> MX camera
        df_svc  --> SVC
    OUTPUT:
        df_stats_full --> Combining both dataframes
    """

    # Set index for merging with the same index both
    df_stats_MX = df_stats_MX.reset_index().set_index('file_name')
    # Merging both dataframe
    df_stats_RAW = pd.merge(df_stats_MX, df_svc, left_index=True, right_index=True)
    df_stats_RAW = df_stats_RAW.reset_index().set_index('id')
    # Write into a csv file
    df_stats_RAW.to_csv(datadir + '/' + 'df_stats_calibration_SVC_and_MX.csv')
    return df_stats_RAW

def RMSE(df, p, x):
    return ((df[p] - df[x]) ** 2).mean() ** .5

def lregress(df, p, x):
    subset = df.dropna(subset=[p, x])
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset[x], subset[p])
    return (slope, intercept, r_value * r_value, p_value, std_err)

def validation_plot_SVC(x, y, data, title=None, x_label=None, y_label=None, alpha=.5, c=None, cmap=None, ax=None, size=40):
    slope, intercept, r2, p_value, std_err = lregress(data, y, x)
    rmse = RMSE(data, y, x)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if c is None:
        data.plot.scatter(x=x, y=y, marker='.', alpha=alpha, ax=ax, s=size)
    else:
        data.plot.scatter(x=x, y=y, marker='.', alpha=alpha, c=c, cmap=cmap, ax=ax, s=size)
    min_val = min(data[x].min(), data[y].min())
    max_val = max(data[x].max(), data[y].max())
    ax.plot([min_val, max_val], [min_val, max_val], c='black')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    val_text = "$r^2$: {r2:.5f}, slope: {slope:.2f}, intercept: {intercept:.5f}, RMSE: {rmse:.5f}".format(r2=r2, slope=slope, intercept=intercept, rmse=rmse)
    if title is None:
        ax.set_title("MX-drone ({x}) vs SVC ({y})\n{val}\n".format(x=x, y=y, val=val_text))
    else:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    return rmse, r2, slope, intercept

def join_stats_SVC_multiple_measures(df_stats_RAW):
    """
    ESTA FUNCIÓN ES PARA UNIR LAS MEDIDAS DEL SVC EN CASO DE REALIZAR MÁS DE UNA MEDIDA POR PLOT
    :return:
    """
    list_MX = []
    list_SVC = []
    for col_name in df_stats_RAW.columns:
        if col_name.endswith('_MX'):
            list_MX.append(col_name)
        if col_name.endswith('_SVC'):
            list_SVC.append(col_name)
    df_stats_grouped = df_stats_RAW[list_MX + list_SVC]
    #Agrupar y sacar la media en función del número de medidas repetidas en un plot
    df_stats_grouped = df_stats_grouped.groupby(level=0).mean()
    #Calcular la desviación estándar de las medidas del espectrómetro
    df_std_SVC_grouped = df_stats_RAW.groupby('id').std()
    df_std_SVC_grouped = df_std_SVC_grouped[list_SVC].add_prefix('std_')
    #Incorporar la desviación estándar en el dataframe.
    df_stats_grouped = pd.merge(df_stats_grouped, df_std_SVC_grouped, left_index=True, right_index=True)

    return df_stats_grouped

def graficas_vuelos(stats_computed,index,df_stats, title, errorbar, directory_svc):
    tipo_vuelo = title.split('-')[0]
    correccion_radiometrica = title.split('-')[1]
    index = index.split(" ")
    # Create dataframe for statistics table
    df_table = pd.DataFrame(columns=['tipo_vuelo', 'correccion_rad', 'indice', 'RMSE', 'r2', 'slope', 'intercept'])
    # Plotear con la función
    colors = ['b', 'g', 'r', 'k', 'purple']
    plot_stat = stats_computed + '_'
    contador_colors = 0
    columnas_plot = 2
    filas_plot = math.ceil(len(index) / 2)
    fig, axarr = plt.subplots(filas_plot, columnas_plot, figsize=(15, 15))
    axarr_fila = 0
    axarr_columna = 0
    for contador in range(len(index)):
        color = colors[contador_colors]
        #Plot graph
        [RMSE, r2, slope, intercept] = validation_plot_SVC(plot_stat + index[contador] + '_MX',
                                                                               index[contador] + '_SVC',
                                                                               df_stats,
                                                                               ax=axarr[axarr_fila][axarr_columna],
                                                                               alpha=1, c=color)
        if errorbar:
            #Plot errorbar
            axarr[axarr_fila][axarr_columna].errorbar(df_stats[plot_stat + index[contador] + '_MX'],
                                                      df_stats[index[contador] + '_SVC'],
                                                      xerr=df_stats['std_' + index[contador] + '_MX'],
                                                      yerr=df_stats['std_'+ index[contador] + '_SVC'],
                                                      fmt='o')
        # Asignar variables para la tabla estadística
        valores = [tipo_vuelo, correccion_radiometrica, index[contador], RMSE, r2, slope,
                   intercept]
        df_table.loc[len(df_table)] = valores

        # Actualizar el plot en el que pinto
        if axarr_columna == 0:
            axarr_columna = 1
        else:
            axarr_columna = 0
            axarr_fila += 1
        # Actualizar color de la gráfica
        if contador_colors == (len(colors) - 1):
            contador_colors = 0
        else:
            contador_colors += 1

    fig.suptitle(title)
    fig.tight_layout()
    try:
        plt.savefig(directory_svc + title + '.png')
        df_table.to_csv(directory_svc + title + '.csv', index=False)
        print('IMAGE SAVED, {}'.format(title))
        print('TABLE STATS SAVED, {}'.format(title))
    except Exception as ex:
        print('ERROR: Cannot save comparative figure, error type: {}'.format(ex))
        quit()



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
    datadir_classified_measures = "Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_05_17/classified/"
    datadir_full = datadir_classified_measures[0:-11]
    datadir_px4_project = "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/"
    numero_medidas_por_plot = 3
    measures_download_with_phone = True
    title = 'Cruzado - Camera and SunIrradiance - Fecha vuelo-17-05-2022'
    errorbar = False
    '''
    datadir_classified_measures = "Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_04_22 - copia/classified/"
    datadir_full = datadir_classified_measures[0:-11]
    datadir_px4_project = "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/"
    numero_medidas_por_plot = 1
    measures_download_with_phone = False
    title = 'Cruzado - Camera and SunIrradiance - Fecha vuelo-21-04-2022'
    '''
    ############################################################################################################
    df_measures = relate_index_with_file_name_GLOBAL(datadir=datadir_classified_measures, numero_medidas_plot=numero_medidas_por_plot)
    df_measures_att = join_measures_with_attributes(att_dir_riego="Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/01-attributes/riego/riego_shapefile_full_atributes.shp",
                                                    att_dir_secano="Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/01-attributes/secano/secano_shapefile_full_atributes.shp",
                                                    df_measures=df_measures)
    metadata, data = extract_metadata_and_data_SVC(data_path=datadir_full, measures_download_phone=measures_download_with_phone)
    metadata_filter, data_filter = export_metadata_to_csv(datadir_full, metadata, data, df_measures_att)
    df_stats_RAW = calibration_MX_and_SVC(metadata=metadata_filter, data=data_filter, datadir_px4_project=datadir_px4_project, datadir=datadir_full,
                           MX_filter_path='Z:/11-Projects/CERESTRES/99-Comparacion_SVC_MX/MX_tablas/report_RedEdge_3_Filters_srs.csv')

    df_stats_grouped = join_stats_SVC_multiple_measures(df_stats_RAW=df_stats_RAW)
    graficas_vuelos(stats_computed='median',
                    index="blue green red red_edge nir ndvi",
                    df_stats=df_stats_grouped,
                    title=title,
                    errorbar=errorbar,
                    directory_svc=datadir_full)
    #index="blue green red red_edge nir ndvi ndre tgi tcari osavi sccci"
    print('\n Algoritmo FINALIZADO correctamente \n')

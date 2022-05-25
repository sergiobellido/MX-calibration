import pandas as pd
import geopandas as gpd
import os
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import math
from scipy import stats
import argparse

'''
Description: Code for compare index in wheat plots of different flights.
Authors: Jose A. Jiménez-Berni & Sergio Bellido-Jiménez
Institution: IAS-CSIC
correspondence: berni@ias.csic.es or sbellido@ias.csic.es
'''

"""
Input:
    - directory_PR1:  Path of the directory where the index are located. Should end with "/indices"
    - directory_PR2:  Path of the directory where the index are located. Should end with "/indices"
    - index: Index you want to compute (i.e. blue, green, red, red_edge, nir, ndvi)
    - stats_computed: Statistical you want to compute. (i.e. min, max, count, mean, median, etc.)
Output:
    - Graphical plot with the evaluation of index of both flights.
"""

def RMSE(df, p, x):
    return ((df[p] - df[x]) ** 2).mean() ** .5


def lregress(df, p, x):
    subset = df.dropna(subset=[p, x])
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset[x], subset[p])
    return (slope, intercept, r_value * r_value, p_value, std_err)


def validation_plot_FLIGHTS(x, y, data, title=None, x_label=None, y_label=None, alpha=.5, c=None, cmap=None, ax=None, size=40):
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
        ax.set_title("drone ({x}) vs drone ({y})\n{val}\n".format(x=x, y=y, val=val_text))
    else:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)


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



class compare_flights():
    def __init__(self, directory_pr1, directory_pr2, title, index, stats_computed):
        self.directory_pr1 = directory_pr1
        self.directory_pr2 = directory_pr2
        self.title = title
        self.path_riego_shp = 'Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/00-raw/riego/plots_regadio.shp'
        self.path_secano_shp = 'Z:/11-Projects/CERESTRES/02-ensayo/Santaella/Shapefiles/00-raw/secano/plots_secano.shp'
        self.index = index
        self.stats_computed = stats_computed
        # Comprobar si existen los directorios
        if os.path.exists(self.directory_pr1) & os.path.exists(self.directory_pr2):
            try:
                # Extraer shapefiles RIEGO Y SECANO
                self.geometry_riego = gpd.read_file(self.path_riego_shp)
                self.geometry_riego['tipo'] = 'riego'
                self.geometry_secano = gpd.read_file(self.path_secano_shp)
                self.geometry_secano['tipo'] = 'secano'
                self.geometry_total = pd.concat([self.geometry_riego, self.geometry_secano])
            except Exception as ex:
                print('ERROR: Check geometry shapefile, error type: {}'. format(ex))
                quit()
        else:
            print('ERROR: Project directories do not exist...')
            quit()

        print("Starting comparison...")
        self.control_loop()

    def control_loop(self):
        # Obtener rutas de ficheros TIF
        if len(self.index) > 1:
            self.index = self.index.split()
        self.ficheros_tifs_proyecto_1 = []
        self.ficheros_tifs_proyecto_2 = []
        self.proyectos = [self.directory_pr1, self.directory_pr2]
        for proyecto in self.proyectos:
            for indice in self.index:
                ruta = proyecto + '/' + indice + '/'
                for file in os.listdir(ruta):
                    if file.endswith('.tif'):
                        if proyecto == self.proyectos[0]:
                            self.ficheros_tifs_proyecto_1.append(ruta + file)
                        else:
                            self.ficheros_tifs_proyecto_2.append(ruta + file)

        # Apply zonal statistics and join dataframes
            # Project 1
        contador = 0
        for tif in self.ficheros_tifs_proyecto_1:
            df = pd.DataFrame(zonal_stats(self.geometry_total, tif, stats=self.stats_computed)).add_suffix('_' + self.index[contador])
            if contador == 0:
                self.df_proyecto_1 = df
            else:
                self.df_proyecto_1 = self.df_proyecto_1.join(df)
            contador += 1

            # Project 2
        contador = 0
        for tif in self.ficheros_tifs_proyecto_2:
            df = pd.DataFrame(zonal_stats(self.geometry_total, tif, stats=self.stats_computed)).add_suffix('_' + self.index[contador])
            if contador == 0:
                self.df_proyecto_2 = df
            else:
                self.df_proyecto_2 = self.df_proyecto_2.join(df)
            contador += 1
        #Merge dataframes
        self.df_stats = pd.merge(self.df_proyecto_1, self.df_proyecto_2, left_index=True, right_index=True, suffixes=['_PR1', '_PR2'])

        # Plotear con la función
        colors = ['b', 'g', 'r', 'k', 'purple']
        plot_stat = str(self.stats_computed[0]) + '_'
        contador_colors = 0
        if len(self.index) > 1:
            columnas_plot = 2
            filas_plot = math.ceil(len(self.index) / 2)
        else:
            columnas_plot = 1
            filas_plot = 1

        fig, axarr = plt.subplots(filas_plot, columnas_plot, figsize=(15, 15))
        axarr_fila = 0
        axarr_columna = 0
        for contador in range(len(self.index)):
            color = colors[contador_colors]
            if len(self.index) > 1:
                validation_plot_FLIGHTS(plot_stat + self.index[contador] + '_PR1',
                                        plot_stat + self.index[contador] + '_PR2', self.df_stats,
                                        ax=axarr[axarr_fila][axarr_columna], alpha=1, c=color)
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
            else:
                validation_plot_FLIGHTS(plot_stat + str(self.index[0]) + '_PR1',
                                        plot_stat + str(self.index[0]) + '_PR2', self.df_stats,
                                        ax=axarr, alpha=1, c='blue')


        #fig.suptitle("PR1: {}, \n PR2: {} \n".format(self.directory_pr1, self.directory_pr2))
        fig.suptitle(self.title)
        fig.tight_layout()
        try:
            plt.savefig(self.title + '.png')
            print('IMAGE SAVED, {}'.format(self.title))
        except Exception as ex:
            print('ERROR: Cannot save comparative figure, error type: {}'.format(ex))
            quit()

    def __del__(self):
        print("Finished process")

class compare_flight_with_SVC():
    def __init__(self, directory_pr1, directory_svc_data, index, stats_computed, title_plot, errorbar):
        self.directory_pr1 = directory_pr1
        self.directory_svc_data = directory_svc_data
        self.index = index
        self.stats_computed = stats_computed
        self.title = title_plot
        self.tipo_vuelo = self.title.split('-')[0]
        self.correccion_radiometrica = self.title.split('-')[1]
        self.errorbar = errorbar
        self.ficheros_tifs_proyecto_1 = []
        # Comprobar si existen los directorios
        if os.path.exists(self.directory_pr1) & os.path.exists(self.directory_svc_data):
            try:
                #Extraer shapefiles RIEGO Y SECANO
                self.df_svc = pd.read_csv(self.directory_svc_data, index_col=0)
                self.raster_telas = pd.read_csv("C:/Users/Sergio/Downloads/poligonos_telas/telas_relacion.csv",
                                                index_col='file')
                self.gpd_svc = pd.merge(self.df_svc, self.raster_telas, left_index=True, right_index=True)
            except Exception as ex:
                print('ERROR: Check geometry shapefile, error type: {}'.format(ex))
                quit()
        else:
            print('ERROR: Project directories do not exist...')
            quit()
        print("Starting comparison...")
        self.control_loop()

    def control_loop(self):
        # Obtener rutas de ficheros TIF
        self.index = self.index.split()
        for indice in self.index:
            ruta = self.directory_pr1 + '/' + indice + '/'
            for file in os.listdir(ruta):
                if file.endswith('.tif'):
                    self.ficheros_tifs_proyecto_1.append(ruta + file)

        # Apply zonal statistics and join dataframes
        # Project 1
        contador = 0
        for tif in self.ficheros_tifs_proyecto_1:
            self.df = pd.DataFrame(zonal_stats(self.gpd_svc.geometry, tif, stats=self.stats_computed + ' std')).add_suffix('_' + self.index[contador]+ '_MX')
            if contador == 0:
                self.df_proyecto_1 = self.df
            else:
                self.df_proyecto_1 = self.df_proyecto_1.join(self.df)
            contador += 1
        self.df_proyecto_1.index = self.df_svc.index
        # Merge dataframes
        self.df_stats = pd.merge(self.df_svc, self.df_proyecto_1, left_index=True, right_index=True,
                                 suffixes=['_SVC', '_MX'])

        # Create dataframe for statistics table
        self.df_table = pd.DataFrame(columns=['tipo_vuelo','correccion_rad','indice','RMSE','r2','slope','intercept'])
        # Plotear con la función
        colors = ['b', 'g', 'r', 'k', 'purple']
        plot_stat = self.stats_computed + '_'
        contador_colors = 0
        columnas_plot = 2
        filas_plot = math.ceil(len(self.index) / 2)
        fig, axarr = plt.subplots(filas_plot, columnas_plot, figsize=(15, 15))
        axarr_fila = 0
        axarr_columna = 0
        for contador in range(len(self.index)):
            color = colors[contador_colors]
            [self.RMSE, self.r2, self.slope, self.intercept] = validation_plot_SVC(plot_stat + self.index[contador] + '_MX', self.index[contador] + '_SVC',
                            self.df_stats,
                            ax=axarr[axarr_fila][axarr_columna], alpha=1, c=color)
            if self.errorbar:
                # Plot errorbar
                axarr[axarr_fila][axarr_columna].errorbar(self.df_stats[plot_stat + self.index[contador] + '_MX'],
                                                          self.df_stats[self.index[contador] + '_SVC'],
                                                          xerr=self.df_stats['std_' + self.index[contador] + '_MX'],
                                                          fmt='o')

            #Asignar variables para la tabla estadística
            valores = [self.tipo_vuelo, self.correccion_radiometrica, self.index[contador], self.RMSE, self.r2, self.slope, self.intercept]
            self.df_table.loc[len(self.df_table)] = valores

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

        fig.suptitle(self.title)
        fig.tight_layout()
        try:
            plt.savefig(directory_svc + self.title + '.png')
            #self.df_table.to_csv(self.title + '.csv', index=False)
            print('IMAGE SAVED, {}'.format(self.title))
            #return self.df_table
        except Exception as ex:
            print('ERROR: Cannot save comparative figure, error type: {}'.format(ex))
            quit()

    def __del__(self):
        print('Finished process')



if __name__ == '__main__':
    comparar_SVC = True

    if comparar_SVC:
    #Comparar vuelo con SVC data

        titles = ['Cruzado - CameraOnly', 'Cruzado - Camera and SunIrradiance',
                  'Cruzado - Camera,SunIrradiance and Sun angle',
                  'Longitudinal - CameraOnly', 'Longitudinal - Camera and SunIrradiance',
                  'Longitudinal - Camera,SunIrradiance and Sun angle',
                  'Transversal - CameraOnly', 'Transversal - Camera and SunIrradiance',
                  'Transversal - Camera,SunIrradiance and Sun angle']

        directories_list = [
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_vuelo_dos_pasadas/Camera_only/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_vuelo_dos_pasadas/Camera_and_SunIrradiance/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_vuelo_dos_pasadas/Camera_SunIrradiance_SunAngle_using_DLS_IMU/indices",

            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_longitudinal/Camera_only/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_longitudinal/Camera_and_SunIrradiance/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_longitudinal/Camera_SunIrradiance_SunAngle_using_DLS_IMU/indices",

            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_transversal/Camera_only/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_transversal/Camera_and_SunIrradiance/indices",
            "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220517_MX/indices_analisis_SBJ/indices_transversal/Camera_SunIrradiance_SunAngle_using_DLS_IMU/indices"
        ]

        directory_svc = "Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_05_17/comparacion_tiposVuelos_vs_SVC/"
        # Call class
        contador_title = 0
        df_tablas_stats = pd.DataFrame(columns=['tipo_vuelo','correccion_rad','indice','RMSE','r2','slope','intercept'])
        for directory in directories_list:
            flight = compare_flight_with_SVC(directory_pr1=directory,
                                             directory_svc_data="Z:/11-Projects/CERESTRES/04-Raw/SVC_HR1024i/Santaella/2022_05_17/telas/df_svc_stats.csv",
                                             index="blue green red red_edge nir ndvi",
                                             stats_computed="median",
                                             errorbar = True,
                                             title_plot=titles[contador_title])
            #Concatenar los dataframes de las estadisticas
            df_tablas_stats = pd.concat([df_tablas_stats, flight.df_table])
            contador_title += 1
        df_tablas_stats.to_csv(directory_svc + 'global_validation_stats.csv', index=False)
        print('[GLOBAL PROCESS: Finished]')
    #Comparar vuelos
    else:
        #Para ejecutarlo a través de los argumentos o manual (argument=False)
        argument = False
        if argument:
            parser = argparse.ArgumentParser(
                description='Code for compare index in wheat plots of different flights')
            parser.add_argument('-pr1', '--dir_pr1', type=str, help='Path for project 1',
                                dest='directory_pr1', required=True)
            parser.add_argument('-pr2', '--dir_pr2', type=str, help='Path for project 2',
                                dest='directory_pr2', required=True)
            parser.add_argument('-i', '--index', type=str, help='"blue green red red_edge nir ndvi"',
                                default='blue green red red_edge nir ndvi', dest="index", required=False)
            parser.add_argument('-s', '--stats', type=str, help='"mean median std min max"',
                                default='median', dest="stats_computed", required=False)
            args = parser.parse_args()
            #Call class
            compare_flights(directory_pr1=args.directory_pr1,
                            directory_pr2=args.directory_pr2,
                            index=args.index,
                            stats_computed=args.stats_computed)
        else:
            #Comparar los vuelos, insertando las rutas e índices de manera manual
            titles = ['Cruzado(PR1) vs Longitudinal(PR2)',
                      'Cruzado(PR1) vs Transversal(PR2)',
                      'Longitudinal(PR1) vs Transversal(PR2)']

            directories_list_PR1 = [
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_cruzado/NoCorrection/indices",
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_cruzado/NoCorrection/indices",
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_longitudinal/NoCorrection/indices",
            ]
            directories_list_PR2 = [
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_longitudinal/NoCorrection/indices",
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_transversal/NoCorrection/indices",
                "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_Duet/indices_analisis_SBJ/vuelo_transversal/NoCorrection/indices"
            ]
            index = ['thermal_ir']
            stats_computed = ['median']
            for vuelo in range(len(directories_list_PR1)):
                flight_thermal = compare_flights(directory_pr1=directories_list_PR1[vuelo],
                                                 directory_pr2=directories_list_PR2[vuelo],
                                                 title=titles[vuelo],
                                                 index=index,
                                                 stats_computed=stats_computed)




# PROJECTS FOR TESTING:
#   Proyecto 1: "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/indices_analisis_SBJ/indices_vuelo_dos_pasadas/Camera_SunIrradiance_SunAngle_using_DLS_IMU/indices"
#   Proyecto 2: "Z:/11-Projects/CERESTRES/05-pix4d/Santaella_20220421_MX/4_index/indices"
# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Graficos:
    def __init__(self, data):
        """
        Clase para crear gráficos a partir de un conjunto de datos.

        :param data: Lista o Serie de pandas con los datos.
        """
        if not isinstance(data, (list, pd.Series)):
            raise ValueError("data debe ser una lista o una Serie de pandas.")
        self.data = pd.Series(data) if isinstance(data, list) else data

    def boxplot(self, columna=None, titulo="Boxplot"):
        """
        Crea y muestra un gráfico de boxplot de los datos.

        :param columna: Nombre de la columna a graficar, si es un DataFrame.
        :param titulo: Título del gráfico.
        """
        if columna:
            data = self.data[columna]
        else:
            data = self.data

        if data.empty:
            raise ValueError("No hay datos para graficar.")

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, vert=False)
        plt.title(titulo)
        plt.xlabel('Valor')
        plt.show()

    def histograma(self, columna=None, titulo="Histograma", bins=10, color='Blue'):
        """
        Crea y muestra un gráfico de histograma de los datos.

        :param columna: Nombre de la columna a graficar, si es un DataFrame.
        :param titulo: Título del gráfico.
        :param bins: Número de bins (intervalos) en el histograma.
        :param palette: Paleta de colores para el histograma.
        """
        if columna:
            data = self.data[columna]
        else:
            data = self.data

        if data.empty:
            raise ValueError("No hay datos para graficar.")

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color=color)
        plt.title(titulo)
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.show()
        

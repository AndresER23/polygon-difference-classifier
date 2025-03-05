class Estadisticas:
    def __init__(self, data):
        """
        Clase para calcular estadísticas básicas de un conjunto de datos.

        :param data: Lista o Serie de pandas con los datos.
        """
        self.data = data

    def maximo(self):
        """Devuelve el valor máximo de los datos."""
        return self.data.max()

    def minimo(self):
        """Devuelve el valor mínimo de los datos."""
        return self.data.min()

    def media(self):
        """Devuelve la media de los datos."""
        return self.data.mean()

    def varianza(self):
        """Devuelve la varianza de los datos."""
        return self.data.var()

    def detectar_atipicos(self):
        """
        Detecta y devuelve los valores atípicos (outliers) usando el método del rango intercuartílico (IQR).

        :return: Lista de valores atípicos.
        """
        if self.data.empty:
            return None

        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)
        atipicos = self.data[(self.data < (Q1 - (1.5 * IQR))) | (self.data > (Q3 + (1.5 * IQR)))]
        return atipicos.tolist()

    def resumen(self):
        """
        Devuelve un diccionario con el resumen de las estadísticas básicas.

        :return: Diccionario con máximo, mínimo, media y varianza.
        """
        return {
            "Máximo": self.maximo(),
            "Mínimo": self.minimo(),
            "Media": self.media(),
            "Varianza": self.varianza()
        }
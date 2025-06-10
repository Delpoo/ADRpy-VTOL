import sys
import os

# Agregar la carpeta raíz al sistema de rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ADRpy.analisis.Modulos.imputacion_correlacion.imputacion_correlacion import imputacion_correlacion
import pandas as pd
from ADRpy.analisis.Modulos.imputacion_correlacion import imputacion_correlacion


def test_imputacion_correlacion_basica():
    df, reporte = imputacion_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df.isna().any().any(), "Deberia imputar todos los valores faltantes"
    valor = df.loc[2, 'Potencia']
    assert abs(valor - 25.9691788448) < 2.0
    assert 'Confianza_cv' in reporte.columns

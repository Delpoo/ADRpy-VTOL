import os
import tempfile
import pandas as pd
import numpy as np
from ADRpy.analisis.Modulos.imputacion_correlacion.imputacion_correlacion import imputacion_correlacion

def test_imputacion_correlacion_basica():
    df, reporte = imputacion_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df.isna().any().any()
    assert 'Confianza_cv' in reporte.columns

def test_single_warning_per_missing_cell():
    data = pd.DataFrame({
        'Modelo': ['a', 'b', 'c'],
        'Misión': [1, 1, 1],
        'X': [1, 1, 1],
        'Y': [np.nan, np.nan, 5],
    })
    tmp = tempfile.mktemp(suffix='.xlsx')
    with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='data_frame_prueba', index=False)
    try:
        _, reporte = imputacion_correlacion(None, path=tmp)
        assert len(reporte) == data['Y'].isna().sum()
    finally:
        os.remove(tmp)

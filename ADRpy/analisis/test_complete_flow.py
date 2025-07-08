#!/usr/bin/env python3
"""
Test específico para verificar el flujo comple        add_normalized_imputation_point        add_normalized_imputation_points(
            fig=fig_2d,
            detalles_por_celda=detalles_por_celda,
            celda_key=celda_test,
            show_imputation_points=True,
            n_predictores_filter=1,
            global_ranges=None,
            modelos_por_celda=modelos_por_celda
        )       fig=fig_3d,
            detalles_por_celda=detalles_por_celda,
            celda_key=celda_test,
            show_imputation_points=True,
            n_predictores_filter=2,
            global_ranges=None,
            modelos_por_celda=modelos_por_celda
        )_normalized_imputation_points.
"""

import sys
import json
import logging
from pathlib import Path
import plotly.graph_objects as go

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Añadir módulos al path
sys.path.append(str(Path(__file__).parent / 'Modulos'))

def test_complete_flow():
    """Prueba el flujo completo como en la aplicación real."""
    
    print("🔧 TEST COMPLETO DEL FLUJO DE PUNTOS DE IMPUTACIÓN")
    print("=" * 55)
    
    # 1. Cargar datos reales
    json_path = Path(__file__).parent / "exports" / "celdas_con_imputacion.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)
    
    # Convertir a estructura esperada
    detalles_por_celda = {}
    for item in data_raw:
        if isinstance(item, dict) and 'celda_key' in item:
            celda_key = item['celda_key']
            detalles_por_celda[celda_key] = {
                'informacion_generica_celda': item.get('imputacion', {})
            }
    
    # Cargar datos de modelos (necesario para los rangos de normalización)
    try:
        with open('Results/modelos_por_celda.json', 'r') as f:
            modelos_por_celda = json.load(f)
            print("✅ Datos de modelos cargados exitosamente")
    except FileNotFoundError:
        print("❌ No se encontraron datos de modelos. Creando datos simulados...")
        # Crear datos simulados de modelo para testing
        modelos_por_celda = {
            'A7|Payload': [
                {
                    'confianza': 0.95,
                    'X_original': [[40.0, 7.0], [45.0, 8.0], [50.0, 9.0], [55.0, 10.0]],
                    'y_original': [55.0, 60.0, 65.0, 70.0],
                    'n_predictores': 2
                }
            ]
        }
        print("✅ Datos de modelos simulados creados")
    
    # 2. Seleccionar celda de prueba
    celda_test = "A7|Payload"
    print(f"\n📋 Probando con celda: {celda_test}")
    
    # 3. Importar función real
    from Analisis_modelos.plot_model_curves import add_normalized_imputation_points
    
    # 4. Crear figura 3D vacía
    fig_3d = go.Figure()
    
    print(f"\n🔧 Simulando llamada para gráfico 3D...")
    print(f"   - show_imputation_points: True")
    print(f"   - n_predictores_filter: 2")
    print(f"   - detalles_por_celda: ✅ disponible")
    print(f"   - modelos_por_celda: ✅ disponible ({celda_test in modelos_por_celda})")
    print(f"   - modelos_por_celda[{celda_test}]: {modelos_por_celda.get(celda_test, 'NO ENCONTRADO')}")
    
    # 5. Ejecutar función como en create_interactive_plot_3d
    try:
        add_normalized_imputation_points(
            fig=fig_3d,
            detalles_por_celda=detalles_por_celda,
            celda_key=celda_test,
            show_imputation_points=True,
            n_predictores_filter=2,  # Para gráficos 3D
            global_ranges=None,
            modelos_por_celda=modelos_por_celda
        )
        
        print(f"\n✅ Función ejecutada sin errores")
        print(f"📊 Trazas en la figura: {len(fig_3d.data)}")
        
        if len(fig_3d.data) > 0:
            print(f"🎯 ¡ÉXITO! Se añadieron trazas:")
            for i, trace in enumerate(fig_3d.data):
                print(f"   - Traza {i+1}: {getattr(trace, 'name', 'Sin nombre')}")
        else:
            print(f"❌ PROBLEMA: No se añadieron trazas a la figura")
            
    except Exception as e:
        print(f"\n❌ ERROR en la función: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
    
    # 6. Simular también gráfico 2D (debería fallar correctamente)
    print(f"\n🔧 Simulando llamada para gráfico 2D (debería dar 0 puntos)...")
    
    fig_2d = go.Figure()
    
    try:
        add_normalized_imputation_points(
            fig=fig_2d,
            detalles_por_celda=detalles_por_celda,
            celda_key=celda_test,
            show_imputation_points=True,
            n_predictores_filter=1,  # Para gráficos 2D
            global_ranges=None,
            modelos_por_celda=modelos_por_celda
        )
        
        print(f"✅ Función 2D ejecutada sin errores")
        print(f"📊 Trazas en figura 2D: {len(fig_2d.data)} (debería ser 0)")
        
    except Exception as e:
        print(f"❌ ERROR en gráfico 2D: {e}")

if __name__ == "__main__":
    test_complete_flow()

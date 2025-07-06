#!/usr/bin/env python3
"""
Test simple de la aplicación con la nueva estructura de datos
Solo para verificar que los datos se cargan correctamente
"""

import sys
import os
import json
from pathlib import Path

# Configurar el directorio de trabajo
work_dir = Path(r"c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Modulos\Analisis_modelos")
os.chdir(work_dir)
sys.path.insert(0, str(work_dir))

def test_basic_functionality():
    """Test básico de funcionalidad sin Dash"""
    print("🔧 TEST BÁSICO DE FUNCIONALIDAD")
    print("=" * 50)
    
    try:
        # Importar solo los módulos esenciales
        import data_loader
        import utils
        
        print("✅ Módulos básicos importados correctamente")
        
        # Ruta del JSON
        json_path = r"c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\modelos_completos_por_celda.json"
        
        # Test de carga de datos
        print(f"\n📁 Cargando datos desde: {json_path}")
        modelos_por_celda, detalles_por_celda = data_loader.load_models_data(json_path)
        
        print(f"✅ Datos cargados:")
        print(f"   - Celdas con modelos: {len(modelos_por_celda)}")
        print(f"   - Celdas con detalles: {len(detalles_por_celda)}")
        
        # Test de extracción de valores únicos
        unique_values = data_loader.extract_unique_values(modelos_por_celda)
        
        print(f"\n📊 Valores únicos extraídos:")
        print(f"   - Aeronaves: {unique_values['aeronaves']}")
        print(f"   - Tipos modelo: {unique_values['tipos_modelo']}")
        print(f"   - N predictores: {unique_values['n_predictores']}")
        
        # Test de filtrado básico
        aeronave_test = unique_values['aeronaves'][0] if unique_values['aeronaves'] else None
        parametros = data_loader.get_parametros_for_aeronave(modelos_por_celda, aeronave_test)
        
        print(f"\n🔍 Test de filtrado:")
        print(f"   - Aeronave ejemplo: {aeronave_test}")
        print(f"   - Parámetros disponibles: {parametros}")
        
        if parametros:
            parametro_test = parametros[0]
            modelos_filtrados = data_loader.filter_models(
                modelos_por_celda,
                aeronave=aeronave_test,
                parametro=parametro_test
            )
            
            celda_key = f"{aeronave_test}|{parametro_test}"
            modelos_celda = modelos_filtrados.get(celda_key, [])
            
            print(f"   - Modelos filtrados para {celda_key}: {len(modelos_celda)}")
            
            if modelos_celda:
                modelo_ejemplo = modelos_celda[0]
                print(f"   - Modelo ejemplo:")
                print(f"     * Tipo: {modelo_ejemplo.get('tipo')}")
                print(f"     * Predictores: {modelo_ejemplo.get('predictores')}")
                print(f"     * R2: {modelo_ejemplo.get('r2'):.3f}")
                print(f"     * MAPE: {modelo_ejemplo.get('mape'):.3f}")
                
                # Verificar campos esenciales
                campos_esenciales = ['coeficientes_originales', 'intercepto_original', 'ecuacion_string']
                for campo in campos_esenciales:
                    valor = modelo_ejemplo.get(campo)
                    tiene_valor = valor is not None and valor != ""
                    print(f"     * {campo}: {'✅' if tiene_valor else '❌'}")
                
                # Test del motor de normalización
                print(f"\n🔧 Test del motor de normalización:")
                try:
                    import normalization_engine
                    engine = normalization_engine.normalization_engine  # Usar la instancia global
                    
                    # Test de obtención de rangos
                    rangos_x, rango_y = engine.get_model_data_ranges(modelo_ejemplo)
                    
                    print(f"   ✅ Motor de normalización funciona")
                    print(f"   - Rangos X: {rangos_x}")
                    print(f"   - Rango Y: {rango_y}")
                    
                    # Test de datos de visualización
                    viz_data = engine.get_model_visualization_data(modelo_ejemplo)
                    print(f"   - Datos de visualización generados: {list(viz_data.keys())}")
                    
                except Exception as e:
                    print(f"   ❌ Error en motor de normalización: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n✅ TEST BÁSICO COMPLETADO EXITOSAMENTE")
        print(f"🎯 Sistema listo para Dash con nueva estructura JSON")
        return True
        
    except Exception as e:
        print(f"❌ Error en test básico: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print(f"\n🚀 Ahora puedes intentar ejecutar la aplicación Dash completa")
    else:
        print(f"\n❌ Hay problemas que necesitan resolverse antes de usar Dash")

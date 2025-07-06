#!/usr/bin/env python3
"""
Test script para verificar la nueva estructura JSON y carga de datos
"""

import sys
import os
from pathlib import Path
import json

# Agregar el directorio de módulos al path
current_dir = Path(__file__).parent
modulos_dir = current_dir / "Modulos" / "Analisis_modelos"
sys.path.insert(0, str(modulos_dir))

# Cambiar imports relativos por absolutos para test
sys.path.insert(0, str(current_dir / "Modulos"))

def test_data_loading():
    """Test básico de carga de datos"""
    print("🔧 TEST DE CARGA DE DATOS CON NUEVA ESTRUCTURA")
    print("=" * 50)
    
    # Ruta del JSON
    json_path = current_dir / "Results" / "modelos_completos_por_celda.json"
    
    if not json_path.exists():
        print(f"❌ Archivo JSON no encontrado: {json_path}")
        return False
    
    print(f"✅ Archivo JSON encontrado: {json_path}")
    
    try:
        # Simulación de load_models_data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON cargado exitosamente")
        
        # Adaptarse a la nueva estructura JSON
        modelos_por_celda = {}
        detalles_por_celda = {}
        
        # Nueva estructura: cada clave es una celda con sub-estructuras
        for celda_key, celda_data in data.items():
            if isinstance(celda_data, dict):
                # Extraer modelos de informacion_modelos_celda.modelos
                modelos_info = celda_data.get('informacion_modelos_celda', {})
                modelos_list = modelos_info.get('modelos', [])
                
                if modelos_list:
                    modelos_por_celda[celda_key] = modelos_list
                
                # Extraer detalles de informacion_generica_celda
                info_generica = celda_data.get('informacion_generica_celda', {})
                if info_generica:
                    detalles_por_celda[celda_key] = info_generica
        
        print(f"✅ Datos procesados:")
        print(f"   - Celdas con modelos: {len(modelos_por_celda)}")
        print(f"   - Celdas con detalles: {len(detalles_por_celda)}")
        
        # Test de extracción de valores únicos
        aeronaves = set()
        tipos_modelo = set()
        predictores = set()
        
        for celda_key, modelos in modelos_por_celda.items():
            # Parsear la clave de celda
            if '|' in celda_key:
                aeronave, parametro = celda_key.split('|', 1)
                aeronaves.add(aeronave)
            
            # Procesar cada modelo
            for modelo in modelos:
                if isinstance(modelo, dict):
                    # Tipos de modelo
                    tipo = modelo.get('tipo')
                    if tipo:
                        tipos_modelo.add(tipo)
                    
                    # Predictores individuales
                    pred_list = modelo.get('predictores', [])
                    if isinstance(pred_list, list):
                        predictores.update(pred_list)
        
        print(f"\n📊 VALORES ÚNICOS EXTRAÍDOS:")
        print(f"   - Aeronaves ({len(aeronaves)}): {sorted(list(aeronaves))}")
        print(f"   - Tipos modelo ({len(tipos_modelo)}): {sorted(list(tipos_modelo))}")
        print(f"   - Predictores ({len(predictores)}): {sorted(list(predictores))[:5]}...")  # Solo primeros 5
        
        # Test de un modelo específico
        if modelos_por_celda:
            primera_celda = list(modelos_por_celda.keys())[0]
            primer_modelo = modelos_por_celda[primera_celda][0]
            
            print(f"\n🔍 ANÁLISIS DE MODELO DE EJEMPLO:")
            print(f"   - Celda: {primera_celda}")
            print(f"   - Tipo: {primer_modelo.get('tipo')}")
            print(f"   - Predictores: {primer_modelo.get('predictores')}")
            print(f"   - N predictores: {primer_modelo.get('n_predictores')}")
            print(f"   - Coeficientes originales: {primer_modelo.get('coeficientes_originales')}")
            print(f"   - Intercepto original: {primer_modelo.get('intercepto_original')}")
            print(f"   - Ecuación string: {primer_modelo.get('ecuacion_string', '')[:50]}...")
            print(f"   - R2: {primer_modelo.get('r2')}")
            print(f"   - MAPE: {primer_modelo.get('mape')}")
            
            # Test de datos de entrenamiento
            datos_entrenamiento = primer_modelo.get('datos_entrenamiento', {})
            if datos_entrenamiento:
                print(f"   - Datos entrenamiento disponibles: ✅")
                print(f"     X_original: {len(datos_entrenamiento.get('X_original', []))} muestras")
                print(f"     y_original: {len(datos_entrenamiento.get('y_original', []))} muestras")
            else:
                print(f"   - Datos entrenamiento: ❌")
        
        # Test de detalles de imputación
        if detalles_por_celda:
            primera_celda_detalles = list(detalles_por_celda.keys())[0]
            detalles = detalles_por_celda[primera_celda_detalles]
            
            print(f"\n🔍 ANÁLISIS DE DETALLES DE IMPUTACIÓN:")
            print(f"   - Celda: {primera_celda_detalles}")
            
            for metodo in ['final', 'similitud', 'correlacion']:
                if metodo in detalles:
                    metodo_data = detalles[metodo]
                    if isinstance(metodo_data, dict) and metodo_data:
                        print(f"   - {metodo}: ✅")
                        print(f"     Valor imputado: {metodo_data.get('Valor imputado')}")
                        print(f"     Confianza: {metodo_data.get('Confianza')}")
                        print(f"     X_visualizacion: {metodo_data.get('X_visualizacion')}")
                    else:
                        print(f"   - {metodo}: ❌ (sin datos)")
        
        print(f"\n✅ TEST COMPLETADO EXITOSAMENTE")
        return True
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print(f"\n🎯 La nueva estructura JSON es compatible con el sistema de visualización")
    else:
        print(f"\n❌ Se requieren ajustes adicionales en el sistema")

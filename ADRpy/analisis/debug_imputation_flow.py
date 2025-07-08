#!/usr/bin/env python3
"""
Script de diagnóstico para auditar el flujo completo de puntos de imputación.
Rastrea paso a paso por qué los puntos no se visualizan en la aplicación Dash.
"""

import sys
import os
import logging
import json
import traceback
from pathlib import Path

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_imputation_flow.log')
    ]
)
logger = logging.getLogger(__name__)

# Añadir el directorio de módulos al path
sys.path.append(str(Path(__file__).parent / 'Modulos'))

def main():
    """Ejecuta el diagnóstico completo del flujo de puntos de imputación."""
    
    print("🔍 DIAGNÓSTICO COMPLETO DEL FLUJO DE PUNTOS DE IMPUTACIÓN")
    print("=" * 60)
    
    try:
        # 1. CARGAR DATOS
        print("\n1. CARGANDO DATOS...")
        print("-" * 30)
        
        # Cargar datos JSON
        json_path = Path(__file__).parent / "exports" / "celdas_con_imputacion.json"
        if not json_path.exists():
            print(f"❌ ERROR: No se encontró el archivo JSON: {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data_raw = json.load(f)
        
        print(f"✅ JSON cargado correctamente: {len(data_raw)} elementos")
        
        # Verificar estructura del JSON
        if isinstance(data_raw, list):
            print("📊 Estructura: Lista de elementos")
            
            # Convertir a diccionario por celda_key
            detalles_por_celda = {}
            for item in data_raw:
                if isinstance(item, dict) and 'celda_key' in item:
                    celda_key = item['celda_key']
                    # Simular la estructura esperada
                    detalles_por_celda[celda_key] = {
                        'informacion_generica_celda': item.get('imputacion', {})
                    }
                    print(f"   - {celda_key}: {list(item.get('imputacion', {}).keys())}")
            
            print(f"📋 Convertido a diccionario: {len(detalles_por_celda)} celdas")
        else:
            print("📊 Estructura: Diccionario")
            detalles_por_celda = data_raw
        
        # Seleccionar celda de ejemplo
        celda_ejemplo = None
        for celda_key in detalles_por_celda.keys():
            if "|" in celda_key:  # Formato esperado "aeronave|parametro"
                celda_ejemplo = celda_key
                break
        
        if not celda_ejemplo:
            print("❌ ERROR: No se encontró una celda con formato válido")
            return
        
        print(f"📋 Celda de ejemplo seleccionada: {celda_ejemplo}")
        
        # 2. VERIFICAR ESTRUCTURA DE DATOS
        print("\n2. VERIFICANDO ESTRUCTURA DE DATOS...")
        print("-" * 40)
        
        celda_data = detalles_por_celda[celda_ejemplo]
        print(f"📊 Estructura de la celda:")
        print(f"   - Claves principales: {list(celda_data.keys())}")
        
        # Verificar información genérica
        info_generica = celda_data.get('informacion_generica_celda', {})
        print(f"   - Información genérica: {list(info_generica.keys())}")
        
        # Verificar subdiccionarios de imputación
        subdics = ['similitud', 'correlacion', 'final']
        for subdic in subdics:
            subdic_data = info_generica.get(subdic, {})
            if subdic_data:
                print(f"   - {subdic}: ✅ (claves: {list(subdic_data.keys())})")
                # Mostrar un ejemplo de datos
                if 'Valor imputado' in subdic_data:
                    print(f"     * Valor imputado: {subdic_data['Valor imputado']}")
                if 'variable_independiente_1' in subdic_data:
                    print(f"     * Variable independiente 1: {subdic_data['variable_independiente_1']}")
                if 'variable_independiente_2' in subdic_data:
                    print(f"     * Variable independiente 2: {subdic_data['variable_independiente_2']}")
            else:
                print(f"   - {subdic}: ❌ (vacío o inexistente)")
        
        # 3. EXTRAER PUNTOS DE IMPUTACIÓN
        print("\n3. EXTRAYENDO PUNTOS DE IMPUTACIÓN...")
        print("-" * 40)
        
        from Analisis_modelos.plot_model_curves import extract_imputation_points
        
        # Extraer puntos sin filtro
        puntos_todos = extract_imputation_points(detalles_por_celda, celda_ejemplo)
        print(f"📈 Puntos extraídos (total): {len(puntos_todos)}")
        
        # Extraer puntos 2D
        puntos_2d = extract_imputation_points(detalles_por_celda, celda_ejemplo, n_predictores_filter=1)
        print(f"📈 Puntos extraídos (2D): {len(puntos_2d)}")
        
        # Extraer puntos 3D
        puntos_3d = extract_imputation_points(detalles_por_celda, celda_ejemplo, n_predictores_filter=2)
        print(f"📈 Puntos extraídos (3D): {len(puntos_3d)}")
        
        # Mostrar ejemplos de puntos extraídos
        if puntos_todos:
            punto_ejemplo = puntos_todos[0]
            print(f"\n📍 Ejemplo de punto extraído:")
            print(f"   - Método: {punto_ejemplo.get('subdic_name')}")
            print(f"   - X original: {punto_ejemplo.get('x_original')}")
            print(f"   - Y original: {punto_ejemplo.get('y_original')}")
            print(f"   - Valor imputado: {punto_ejemplo.get('valor_imputado')}")
            print(f"   - N predictores: {punto_ejemplo.get('n_predictores')}")
            print(f"   - Confianza: {punto_ejemplo.get('confianza')}")
        
        # 4. VERIFICAR MODELOS PARA NORMALIZACIÓN
        print("\n4. VERIFICANDO MODELOS PARA NORMALIZACIÓN...")
        print("-" * 50)
        print("⚠️  NOTA: Este JSON solo tiene datos de imputación, no modelos")
        print("⚠️  Los modelos deben estar en un archivo separado o en la aplicación")
        
        # Simulamos que no hay modelos en este JSON
        modelos = []
        modelos_2d = []
        modelos_3d = []
        
        print(f"📊 Modelos disponibles en JSON: {len(modelos)}")
        print("   - Este JSON solo contiene puntos de imputación")
        print("   - Los modelos deben cargarse desde la aplicación principal")
        
        # 5. PROBAR EXTRACCIÓN DE PUNTOS (REAL)
        print("\n5. PROBANDO EXTRACCIÓN DE PUNTOS (REAL)...")
        print("-" * 45)
        
        try:
            from Analisis_modelos.plot_model_curves import extract_imputation_points
            
            # Probar con la estructura real
            print("🧪 Probando extracción con estructura real...")
            puntos_extraidos = extract_imputation_points(detalles_por_celda, celda_ejemplo)
            print(f"   - Puntos extraídos: {len(puntos_extraidos)}")
            
            if puntos_extraidos:
                punto = puntos_extraidos[0]
                print(f"   - Ejemplo: {punto}")
            else:
                print("   - ❌ No se extrajeron puntos")
                
        except Exception as e:
            print(f"   - ❌ Error en extracción: {e}")
            print(f"   - Traceback: {traceback.format_exc()}")
        
        # 6. RESUMEN SIMPLIFICADO
        print("\n6. RESUMEN SIMPLIFICADO")
        print("-" * 25)
        
        print("📊 ESTRUCTURA DEL JSON:")
        print(f"   - Formato: Lista de {len(data_raw)} elementos")
        print(f"   - Celdas procesadas: {len(detalles_por_celda)}")
        print(f"   - Celda ejemplo: {celda_ejemplo}")
        
        celda_info = detalles_por_celda[celda_ejemplo]['informacion_generica_celda']
        print(f"   - Métodos en celda ejemplo: {list(celda_info.keys())}")
        
        for metodo, datos in celda_info.items():
            if isinstance(datos, dict):
                campos = list(datos.keys())
                print(f"     * {metodo}: {len(campos)} campos")
                if 'Valor imputado' in datos:
                    print(f"       - Valor imputado: {datos['Valor imputado']}")
                if 'variable_independiente_1' in datos:
                    print(f"       - Variable 1: {datos['variable_independiente_1']}")
                if 'variable_independiente_2' in datos:
                    print(f"       - Variable 2: {datos['variable_independiente_2']}")
        
        print("\n🎯 DIAGNÓSTICO CLAVE:")
        print("1. ✅ JSON cargado correctamente")
        print("2. ✅ Datos de imputación presentes")
        print("3. ⚠️  Falta integración con función extract_imputation_points")
        print("4. ⚠️  Falta validación con datos reales de la aplicación")
        
        print("\n📋 SIGUIENTES PASOS:")
        print("   1. Verificar que extract_imputation_points maneja esta estructura")
        print("   2. Probar con datos reales de la aplicación Dash")
        print("   3. Activar logging DEBUG en la aplicación")
        print("   4. Verificar que show_imputation_points=True en la UI")
        
    except Exception as e:
        logger.error(f"Error en diagnóstico: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ ERROR CRÍTICO: {e}")
        print(f"Ver debug_imputation_flow.log para detalles completos")

if __name__ == "__main__":
    main()

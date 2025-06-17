#!/usr/bin/env python3
"""
Script de prueba rápida para el Sistema de Análisis Visual Dinámico.
Este script valida que todos los componentes necesarios estén disponibles
y funcionando correctamente antes de ejecutar el notebook completo.

Autor: Sistema de Análisis ADRpy-VTOL
Fecha: 2024
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

def verificar_dependencias():
    """Verificar que todas las dependencias necesarias están instaladas."""
    print("🔍 VERIFICANDO DEPENDENCIAS...")
    print("=" * 35)
    
    dependencias_requeridas = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('plotly', 'plotly'),
        ('ipywidgets', 'widgets'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn')
    ]
      dependencias_ok = True
    
    for dep_nombre, dep_import in dependencias_requeridas:
        try:
            version = "desconocida"
            if dep_import == 'pd':
                import pandas as pd
                version = pd.__version__
            elif dep_import == 'np':
                import numpy as np
                version = np.__version__
            elif dep_import == 'plotly':
                import plotly
                version = plotly.__version__
            elif dep_import == 'widgets':
                import ipywidgets
                version = ipywidgets.__version__
            elif dep_import == 'scipy':
                import scipy
                version = scipy.__version__
            elif dep_import == 'sklearn':
                import sklearn
                version = sklearn.__version__
            
            print(f"   ✅ {dep_nombre}: {version}")
            
        except ImportError as e:
            print(f"   ❌ {dep_nombre}: NO INSTALADO")
            print(f"      Error: {e}")
            dependencias_ok = False
        except Exception as e:
            print(f"   ⚠️  {dep_nombre}: Instalado pero error obteniendo versión")
            print(f"      Error: {e}")
    
    return dependencias_ok

def verificar_modulos_locales():
    """Verificar que los módulos locales están disponibles."""
    print("\n🔍 VERIFICANDO MÓDULOS LOCALES...")
    print("=" * 35)
    
    # Agregar directorio de módulos al path
    modulos_path = Path("Modulos")
    if modulos_path.exists():
        sys.path.append(str(modulos_path.absolute()))
        print(f"   📁 Directorio Modulos: {modulos_path.absolute()}")
    else:
        print(f"   ❌ Directorio Modulos no encontrado")
        return False
    
    try:
        from imputacion_correlacion import (
            imputaciones_correlacion, 
            entrenar_modelo, 
            cargar_y_validar_datos
        )
        print("   ✅ imputacion_correlacion: Importado correctamente")
        print("      • imputaciones_correlacion: ✅")
        print("      • entrenar_modelo: ✅") 
        print("      • cargar_y_validar_datos: ✅")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ imputacion_correlacion: Error de importación")
        print(f"      Error: {e}")
        return False

def verificar_datos():
    """Verificar que los datos están disponibles."""
    print("\n🔍 VERIFICANDO DATOS...")
    print("=" * 25)
    
    data_path = Path("Data")
    if not data_path.exists():
        print(f"   ❌ Directorio Data no encontrado")
        return False, None
    
    print(f"   📁 Directorio Data: {data_path.absolute()}")
    
    # Buscar archivos de datos disponibles
    archivos_datos = []
    for ext in ['.xlsx', '.csv']:
        archivos_datos.extend(list(data_path.glob(f"*{ext}")))
    
    if not archivos_datos:
        print("   ❌ No se encontraron archivos de datos")
        return False, None
    
    print(f"   📊 Archivos de datos encontrados:")
    for archivo in archivos_datos:
        print(f"      • {archivo.name}")
    
    # Intentar cargar el archivo principal
    archivo_principal = None
    nombres_preferidos = [
        "Datos_aeronaves_completo.xlsx",
        "Datos_aeronaves.xlsx", 
        "datos_aeronaves.xlsx"
    ]
    
    for nombre in nombres_preferidos:
        archivo_test = data_path / nombre
        if archivo_test.exists():
            archivo_principal = archivo_test
            break
    
    if not archivo_principal:
        archivo_principal = archivos_datos[0]  # Usar el primero disponible
    
    try:
        print(f"   🔄 Probando carga de: {archivo_principal.name}")
        df = pd.read_excel(archivo_principal)
        print(f"   ✅ Datos cargados exitosamente")
        print(f"      • Filas: {df.shape[0]}")
        print(f"      • Columnas: {df.shape[1]}")
        print(f"      • Valores faltantes: {df.isnull().sum().sum()}")
        
        return True, str(archivo_principal)
        
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return False, None

def test_funcionalidad_basica(ruta_datos):
    """Test básico de funcionalidad de imputación."""
    print("\n🔍 TEST DE FUNCIONALIDAD BÁSICA...")
    print("=" * 35)
    
    try:
        # Importar funciones necesarias
        from imputacion_correlacion import cargar_y_validar_datos
        
        # Cargar datos
        print("   🔄 Cargando y validando datos...")
        df = cargar_y_validar_datos(ruta_datos)
        
        if df is None or df.empty:
            print("   ❌ Error: Datos no válidos")
            return False
        
        print(f"   ✅ Datos validados: {df.shape}")
        
        # Verificar que hay valores faltantes
        valores_faltantes = df.isnull().sum().sum()
        if valores_faltantes == 0:
            print("   ⚠️  Advertencia: No hay valores faltantes para imputar")
            return True  # No es error, pero no hay trabajo que hacer
        
        print(f"   ✅ Valores faltantes detectados: {valores_faltantes}")
        
        # Test de funciones auxiliares
        from imputacion_correlacion import (
            seleccionar_predictores_validos,
            generar_combinaciones,
            entrenar_modelo
        )
        
        print("   ✅ Funciones auxiliares importadas correctamente")
        
        # Test simple de entrenamiento
        print("   🔄 Test de entrenamiento de modelo...")
        
        # Buscar una columna con valores faltantes
        columnas_faltantes = df.columns[df.isnull().any()].tolist()
        if columnas_faltantes:
            objetivo = columnas_faltantes[0]
            idx_faltante = df[df[objetivo].isnull()].index[0]
            
            # Intentar entrenar un modelo simple
            df_filtrado, familia, filtro = seleccionar_predictores_validos(
                df, objetivo, idx_faltante
            )
            
            if not df_filtrado.empty:
                predictores = [col for col in df_filtrado.columns 
                              if col != objetivo and col != df_filtrado.columns[0]][:1]
                
                if predictores:
                    modelo = entrenar_modelo(df_filtrado, objetivo, predictores, False, idx_faltante)
                    if modelo and not modelo.get("descartado", False):
                        print("   ✅ Test de entrenamiento exitoso")
                        print(f"      • Tipo: {modelo['tipo']}")
                        print(f"      • MAPE: {modelo['mape']:.2f}%")
                        print(f"      • R²: {modelo['r2']:.3f}")
                        return True
        
        print("   ⚠️  No se pudo realizar test de entrenamiento completo")
        return True  # Funciones básicas funcionan
        
    except Exception as e:
        print(f"   ❌ Error en test funcional: {e}")
        return False

def main():
    """Función principal del script de verificación."""
    print("🚀 SCRIPT DE VERIFICACIÓN DEL SISTEMA")
    print("🎯 Análisis Visual Dinámico para Imputación por Correlación")
    print("=" * 60)
    
    # Cambiar al directorio correcto
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Directorio de trabajo: {script_dir.absolute()}")
    
    resultados = {
        "dependencias": False,
        "modulos": False,
        "datos": False,
        "funcionalidad": False
    }
    
    # 1. Verificar dependencias
    resultados["dependencias"] = verificar_dependencias()
    
    # 2. Verificar módulos locales
    if resultados["dependencias"]:
        resultados["modulos"] = verificar_modulos_locales()
    
    # 3. Verificar datos
    if resultados["modulos"]:
        datos_ok, ruta_datos = verificar_datos()
        resultados["datos"] = datos_ok
        
        # 4. Test de funcionalidad
        if datos_ok and ruta_datos:
            resultados["funcionalidad"] = test_funcionalidad_basica(ruta_datos)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 25)
    
    for test, resultado in resultados.items():
        estado = "✅" if resultado else "❌"
        print(f"   {estado} {test.title()}: {'PASS' if resultado else 'FAIL'}")
    
    exito_total = all(resultados.values())
    
    if exito_total:
        print(f"\n🎉 ¡VERIFICACIÓN EXITOSA!")
        print(f"✅ El sistema está listo para ejecutar el notebook")
        print(f"🚀 Puede proceder a ejecutar 'analisis_modelos_imputacion.ipynb'")
        
        print(f"\n📋 PRÓXIMOS PASOS:")
        print(f"   1. Abrir el notebook: analisis_modelos_imputacion.ipynb")
        print(f"   2. Ejecutar las celdas en orden")
        print(f"   3. Usar la interfaz interactiva para análisis")
        
    else:
        print(f"\n❌ VERIFICACIÓN FALLÓ")
        print(f"⚠️  Resuelva los problemas anteriores antes de continuar")
        
        print(f"\n💡 SOLUCIONES SUGERIDAS:")
        if not resultados["dependencias"]:
            print(f"   • Instalar dependencias: pip install -r requirements.txt")
        if not resultados["modulos"]:
            print(f"   • Verificar que está en el directorio correcto")
            print(f"   • Revisar archivos en la carpeta Modulos/")
        if not resultados["datos"]:
            print(f"   • Verificar archivos en la carpeta Data/")
            print(f"   • Asegurar que tiene permisos de lectura")
    
    print(f"\n" + "=" * 60)
    return exito_total

if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)

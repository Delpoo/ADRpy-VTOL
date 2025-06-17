#!/usr/bin/env python3
"""
Script simple de verificación para el Sistema de Análisis Visual Dinámico.
"""

def verificar_sistema():
    print("🚀 VERIFICACIÓN RÁPIDA DEL SISTEMA")
    print("=" * 40)
    
    # 1. Verificar dependencias básicas
    print("🔍 Verificando dependencias...")
    
    dependencias = ['pandas', 'numpy', 'plotly', 'ipywidgets', 'scipy', 'sklearn']
    
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - NO INSTALADO")
    
    # 2. Verificar módulos locales
    print("\n🔍 Verificando módulos locales...")
    
    import sys
    import os
    from pathlib import Path
    
    # Agregar módulos al path
    if Path("Modulos").exists():
        sys.path.append("Modulos")
        print("   ✅ Directorio Modulos encontrado")
    else:
        print("   ❌ Directorio Modulos NO encontrado")
        return False
    
    try:
        from imputacion_correlacion import imputaciones_correlacion
        print("   ✅ imputacion_correlacion importado")
    except ImportError as e:
        print(f"   ❌ Error importando imputacion_correlacion: {e}")
        return False
    
    # 3. Verificar datos
    print("\n🔍 Verificando datos...")
    
    if Path("Data").exists():
        archivos_datos = list(Path("Data").glob("*.xlsx"))
        if archivos_datos:
            print(f"   ✅ Encontrados {len(archivos_datos)} archivos de datos")
            for archivo in archivos_datos[:3]:  # Mostrar solo los primeros 3
                print(f"      • {archivo.name}")
        else:
            print("   ❌ No se encontraron archivos .xlsx en Data/")
            return False
    else:
        print("   ❌ Directorio Data NO encontrado")
        return False
    
    print("\n✅ VERIFICACIÓN COMPLETADA EXITOSAMENTE")
    print("🎯 El sistema está listo para usar")
    print("\n📋 PRÓXIMOS PASOS:")
    print("   1. Abrir: analisis_modelos_imputacion.ipynb")
    print("   2. Ejecutar todas las celdas en orden")
    print("   3. Usar la interfaz interactiva")
    
    return True

if __name__ == "__main__":
    exito = verificar_sistema()
    if not exito:
        print("\n❌ Hay problemas que resolver antes de continuar")
        print("💡 Consulte el README_ANALISIS.md para soluciones")

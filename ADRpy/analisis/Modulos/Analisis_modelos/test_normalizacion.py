"""
Script de prueba simple para verificar la funcionalidad de superposición normalizada
"""
import sys
import os

# Añadir el directorio de módulos al path
sys.path.append(r'c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Modulos\Analisis_modelos')

try:
    from data_loader import load_models_data, extract_unique_values, filter_models
    from plot_utils import create_interactive_plot, get_model_original_data, get_model_training_data
    
    print("✅ Módulos importados exitosamente")
    
    # Cargar datos
    archivo_json = r'c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\modelos_completos_por_celda.json'
    
    print("📊 Cargando datos...")
    modelos_por_celda, detalles_por_celda = load_models_data(archivo_json)
    print(f"✅ Cargados {len(modelos_por_celda)} celdas con modelos")
    
    # Obtener valores únicos
    valores_unicos = extract_unique_values(modelos_por_celda)
    print(f"🔍 Aeronaves disponibles: {valores_unicos['aeronaves']}")
    print(f"🔍 Parámetros disponibles: {valores_unicos['parametros']}")
    
    # Filtrar modelos
    modelos_filtrados = filter_models(
        modelos_por_celda=modelos_por_celda,
        tipos_modelo=['linear-1', 'log-1', 'poly-1'],
        n_predictores=[1]
    )
    print(f"✅ Modelos filtrados: {len(modelos_filtrados)} celdas")
    
    # Probar con una combinación específica
    aeronave_test = "A7"
    parametro_test = "payload"
    celda_key = f"{aeronave_test}|{parametro_test}"
    
    if celda_key in modelos_filtrados:
        modelos = modelos_filtrados[celda_key]
        modelos_1_pred = [m for m in modelos if isinstance(m, dict) and m.get('n_predictores', 0) == 1]
        
        print(f"\n🎯 Probando con: {aeronave_test} - {parametro_test}")
        print(f"📋 Modelos de 1 predictor: {len(modelos_1_pred)}")
        
        for i, modelo in enumerate(modelos_1_pred):
            predictor = modelo.get('predictores', ['N/A'])[0]
            tipo = modelo.get('tipo', 'unknown')
            print(f"  {i+1}. {predictor} ({tipo})")
            
            # Probar funciones auxiliares
            df_orig = get_model_original_data(modelo)
            df_train = get_model_training_data(modelo)
            
            print(f"      Datos orig.: {'✅' if df_orig is not None else '❌'}")
            print(f"      Datos entren.: {'✅' if df_train is not None else '❌'}")
        
        print(f"\n🎨 Generando gráfico normalizado...")
        
        # Crear el gráfico
        fig = create_interactive_plot(
            modelos_filtrados=modelos_filtrados,
            aeronave=aeronave_test,
            parametro=parametro_test,
            show_training_points=True,
            show_model_curves=True
        )
        
        print(f"✅ Gráfico generado exitosamente")
        print(f"📊 Título del eje X: {fig.layout.xaxis.title.text}")
        print(f"📊 Título del eje Y: {fig.layout.yaxis.title.text}")
        print(f"📊 Número de trazas: {len(fig.data)}")
        
        # Verificar que se aplicó la normalización correctamente
        if fig.layout.xaxis.title.text == "Input normalizado (por predictor)":
            print("✅ Normalización del eje X aplicada correctamente")
        else:
            print("❌ Problema con la normalización del eje X")
        
        # Mostrar información de las trazas
        for i, trace in enumerate(fig.data):
            print(f"  Traza {i+1}: {trace.name} ({trace.mode})")
        
    else:
        print(f"❌ No se encontraron modelos para {celda_key}")
        print(f"🔍 Claves disponibles: {list(modelos_filtrados.keys())}")
    
    print(f"\n🎉 Prueba completada exitosamente")

except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error general: {e}")
    import traceback
    traceback.print_exc()

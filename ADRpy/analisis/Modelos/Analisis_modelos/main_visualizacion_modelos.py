# Callback para mantener el estado de las vistas (preservar zoom y posición de cámara)
@app.callback(
    Output('view-state-store', 'data'),
    [Input('plot-tabs', 'value'),
     Input('main-plot', 'relayoutData')],
    [State('view-state-store', 'data')],
    prevent_initial_call=True
)
def update_view_state(selected_tab, relayout_data, current_state):
    """
    Mantiene el estado de las vistas para preservar zoom 2D y cámara 3D.
    """
    if current_state is None:
        current_state = {
            'current_tab': 'main-view',
            '3d_camera': None,
            '2d_zoom': None
        }
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_state
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'plot-tabs':
        # Cambio de pestaña
        current_state['current_tab'] = selected_tab
        
    elif trigger_id == 'main-plot' and relayout_data:
        # Actualización del gráfico
        current_tab = current_state.get('current_tab', 'main-view')
        
        if current_tab == '3d-view':
            # Guardar estado de cámara 3D
            if 'scene.camera' in relayout_data:
                current_state['3d_camera'] = relayout_data['scene.camera']
        else:
            # Guardar estado de zoom 2D
            zoom_data = {}
            for key in ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']:
                if key in relayout_data:
                    zoom_data[key] = relayout_data[key]
            if zoom_data:
                current_state['2d_zoom'] = zoom_data
    
    return current_state
# visualization.py
import folium

def create_parking_map(df_sensors, lat_col='lat', lon_col='lon', id_col='DeviceId', name_col='StreetName'):
    """
    Generate an interactive map with parking sensor locations, colored by zone type.
    """
    # Melbourne CBD coordinates for centering the map
    melbourne_coords = [-37.8136, 144.9631]
    # Create map    
    m = folium.Map(location=melbourne_coords, zoom_start=14, tiles='CartoDB dark_matter')
    
    color_map = {
        'Commercial': '#e74c3c',  
        'Residential': '#2ecc71',  
        'Center': '#3498db',       
        'General': '#95a5a6'          
    }

    # Legend HTML
    legend_html = '''
    <div style="
        position: fixed; 
        top: 20px; right: 20px; width: 140px; height: 130px; 
        background-color: rgba(255, 255, 255, 0.9); border: 2px solid grey; 
        z-index: 9999; font-size: 14px; padding: 8px; border-radius: 8px;
        transform: scale(1.15); transform-origin: top right;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
        <b style="color: #333; font-family: Arial;">Puntos de interés</b><br>
        <i style="background:#e74c3c; width: 12px; height: 12px; float: left; margin-top: 4px; margin-right: 8px; border-radius: 50%;"></i><span style="color: #333;">Comercial</span><br>
        <i style="background:#2ecc71; width: 12px; height: 12px; float: left; margin-top: 4px; margin-right: 8px; border-radius: 50%;"></i><span style="color: #333;">Residencial</span><br>
        <i style="background:#3498db; width: 12px; height: 12px; float: left; margin-top: 4px; margin-right: 8px; border-radius: 50%;"></i><span style="color: #333;">Centro</span><br>
        <i style="background:#95a5a6; width: 12px; height: 12px; float: left; margin-top: 4px; margin-right: 8px; border-radius: 50%;"></i><span style="color: #333;">Genérico</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add each sensor to the map
    for idx, row in df_sensors.iterrows():
        zone = row.get('ZoneType', 'General') # Default to 'General' if ZoneType is missing. This should not happen since posible Nones have already been filtered out
        color = color_map.get(zone, '#95a5a6')
        street = row.get(name_col, 'Unknown Street') # No names should be missing

        
        # HTML when hovering over a sensor
        tooltip_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Sensor:</b> {row[id_col]}<br>
            <b>Zona:</b> <span style="color: {color}; font-weight: bold;">{zone}</span><br>
            <b>Calle:</b> {street}
        </div>
        """
        
        # Add a circle marker for each sensor
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6, # Dot size
            color=color,
            weight=1, # Dot border thickness
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m)
        
    return m
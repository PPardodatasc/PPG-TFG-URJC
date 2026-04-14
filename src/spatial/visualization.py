# visualization.py
import folium

def create_parking_map(df_sensors, lat_col='lat', lon_col='lon', id_col='DeviceId', name_col='StreetName'):
    """
    Genera un mapa HTML interactivo usando Folium con los sensores clasificados.
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
    
    # Iterar sobre cada sensor y añadirlo al mapa
    for idx, row in df_sensors.iterrows():
        zone = row.get('ZoneType', 'General')
        color = color_map.get(zone, '#95a5a6')
        

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
            <h4 style="margin-bottom: 5px; color: {color};">{zone}</h4>
            <b>ID Sensor:</b> {row[id_col]}<br>
            <b>Calle:</b> {row[name_col]}
        </div>
        """
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6, # Dot size
            color=color,
            weight=1, # Dot border thickness
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Sensor {row[id_col]}" # Text on hover
        ).add_to(m)
        
    return m
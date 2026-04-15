# poi.py
import osmnx as ox
from tqdm import tqdm
import time

ox.settings.use_cache = True # Enable caching to speed up repeated queries to the same locations

def get_pois(lat: float, lon: float, radius=150):
    """
    Retrieve points of interest (POIs) from OpenStreetMap around a given location.
    Returns a dictionary with counts of different POI categories.
    """
    time.sleep(2) # avoid hitting API rate limits

    tags = {
        'amenity': True,  
        'shop': True,     
        'building': ['residential', 'commercial', 'retail', 'apartments'],
        'office': True
    }
    try:
        pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        
        if pois.empty:
            return {'amenities': 0, 'shops': 0, 'offices': 0, 'residential': 0}
            
        # Count POIs from each category
        residential = 0
        if 'building' in pois.columns:
            residential = pois['building'].isin(['residential', 'apartments']).sum()

        counts = {
            'amenities': pois['amenity'].notna().sum() if 'amenity' in pois.columns else 0,
            'shops': pois['shop'].notna().sum() if 'shop' in pois.columns else 0,
            'offices': pois['office'].notna().sum() if 'office' in pois.columns else 0,
            'residential': residential,
        }
        return counts

    except Exception as e:
        print(f"Error: {str(e)}")
        return {'amenities': 0, 'shops': 0, 'offices': 0, 'residential': 0}


def classify_zone(lat: float, lon: float, radius=150):
    """
    Classify a location based on nearby points of interest (POIs) from OpenStreetMap.
    Returns one of the following categories: 'Residential', 'Commercial', 'Center', or 'None'.
    """
    # Retrieve POIs around the given location
    pois = get_pois(lat, lon, radius=radius)

    if not pois:
        return 'None'

    residential = pois.get('residential', 0)
    shops = pois.get('shops', 0)
    amenities = pois.get('amenities', 0)
    offices = pois.get('offices', 0)

    # Zone classification
    if residential >= 2:
        return 'Residential'
    elif shops >= 15:
        return 'Commercial'
    elif amenities >= 80 or offices >= 5:
        return 'Center'
    # Fallback
    elif shops >= 5 or amenities >= 30:
        return 'Commercial'
    else:
        return 'General'


def get_zone_type(df_sensors, lat_col='lat', lon_col='lon'):
    """
    Classify each sensor in the dataframe based on nearby POIs from OpenStreetMap.
    """
    tqdm.pandas(desc="Progreso:")
    df_sensors['ZoneType'] = df_sensors.progress_apply(lambda row: classify_zone(row[lat_col], row[lon_col]), axis=1)
    return df_sensors
# poi.py
import osmnx as ox
from tqdm import tqdm
import time

ox.settings.use_cache = True

def get_pois(lat: float, lon: float, radius=100):
    """
    Retrieve points of interest (POIs) from OpenStreetMap around a given location.
    Returns a dictionary with counts of different POI categories.
    """
    time.sleep(5) # avoid hitting API rate limits

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
        print(f"Error at ({lat}, {lon}): {str(e)}")
        return {'amenities': 0, 'shops': 0, 'offices': 0, 'residential': 0}


def classify_zone(lat: float, lon: float, radius=100):
    """
    Classify a location based on a weighted scoring system of nearby POIs.
    """
    pois = get_pois(lat, lon, radius=radius)

    if not pois:
        return 'General'

    residential = pois.get('residential', 0)
    shops = pois.get('shops', 0)
    amenities = pois.get('amenities', 0)
    offices = pois.get('offices', 0)

    # Tag scoring system
    # Offices are weighted more heavily than shops, which are weighted more than generic amenities.
    # Residential buildings are weighted heavily because they represent many dwellings, while a single shop or office represents fewer people.
    commercial_score = (offices * 5) + (shops * 3) + amenities
    residential_score = residential * 10 

    # Central business district (CBD)
    if commercial_score > 120 or offices >= 10:
        return 'Center (CBD)'
        
    elif commercial_score > residential_score and commercial_score > 20:
        return 'Commercial'
        
    elif residential_score > commercial_score and residential > 0:
        return 'Residential'
        
    # Fallback: some local commercial activity but not enough to be classified as CBD
    elif commercial_score > 10:
        return 'Commercial'
        
    else:
        return 'General'


def get_zone_type(df_sensors, lat_col='lat', lon_col='lon'):
    """
    Classify each sensor in the dataframe based on nearby POIs from OpenStreetMap.
    """
    tqdm.pandas(desc="Clasificando Zonas (OSM):")
    df_sensors['ZoneType'] = df_sensors.progress_apply(lambda row: classify_zone(row[lat_col], row[lon_col]), axis=1)
    return df_sensors
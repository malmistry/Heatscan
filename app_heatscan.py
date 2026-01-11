
import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import pydeck as pdk
from pathlib import Path
from shapely.geometry import box, mapping
import rioxarray
import geopandas as gpd
import requests
import zipfile
import io
import os
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heat Impact Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# ---------------- PATHS (Malcolm: Update these paths as needed) ----------------
TW_DIR = Path("/content/drive/MyDrive/output/Tw_output_2015_2020_2025")
WBGT_DIR = Path("/content/drive/MyDrive/output/WBGT_output_2015_2020_2025")
POP_PATH = Path("/content/drive/MyDrive/Project/pop_ssp245_2050/ssp2_2050.nc")
CITIES_DATA_PATH = Path("/content/drive/MyDrive/cities15000.txt")

# ---------------- PRIORITY CITIES FOR AFRICA ----------------
# Malcolm: These are the 3 priority cities with their coordinates and bounding boxes.
# Format: name, lat, lon, bbox [xmin, ymin, xmax, ymax]
CITIES_CONFIG = [
    {
        "name": "Bangui",
        "country": "Central African Republic",
        "lat": 4.4,
        "lon": 18.5,
        "bbox": [18.442688, 4.304382, 18.644089, 4.549307]
    },
    {
        "name": "Abidjan",
        "country": "Côte d'Ivoire",
        "lat": 5.3,
        "lon": -4.0,
        "bbox": [-4.151940, 5.211219, -3.811289, 5.510966]
    },
    {
        "name": "Nouakchott",
        "country": "Mauritania",
        "lat": 18.1,
        "lon": -15.9,
        "bbox": [-16.0448, 17.9760, -15.8877, 18.1772]
    }
]

# ---------------- COUNTRY CONFIGURATION ----------------
COUNTRIES_CONFIG = [
    {
        "name": "Central African Republic",
        "iso_a3": "CAF",
        "bbox_fallback": [14.0, 2.0, 28.0, 11.0]
    },
    {
        "name": "Côte d'Ivoire",
        "iso_a3": "CIV",
        "bbox_fallback": [-9.0, 4.0, -3.0, 11.0]
    },
    {
        "name": "Mauritania",
        "iso_a3": "MRT",
        "bbox_fallback": [-17.0, 14.0, -4.0, 27.0]
    }
]

# ---------------- COLOR SCALES ----------------

# Figure 2 Color Combinations and Bins (Visual Replication)
FIG2_COLORS = [
    [225, 225, 225],  # 0-3 (Greyish)
    [255, 245, 204],  # 3-8
    [255, 230, 112],  # 8-24
    [255, 204, 51],   # 24-56
    [255, 175, 51],   # 56-112
    [255, 153, 51],   # 112-168
    [255, 111, 51],   # 168-240
    [255, 85, 0],     # 240-480
    [230, 40, 30],    # 480-720
    [200, 30, 20]     # >720
]
FIG2_LEVELS = [0, 3, 8, 24, 56, 112, 168, 240, 480, 720]

def get_fig2_color(value):
    """
    Map heat value to Figure 2 discrete color palette using specific bins.
    """
    if pd.isna(value) or value < 0:
        return [0, 0, 0, 0] # Transparent
    
    # Check bins
    for i in range(len(FIG2_LEVELS) - 1):
        if value >= FIG2_LEVELS[i] and value < FIG2_LEVELS[i+1]:
            # Add alpha 255 (Opaque)
            c = FIG2_COLORS[i]
            return [c[0], c[1], c[2], 255]
            
    # If >= last level (720)
    if value >= FIG2_LEVELS[-1]:
        c = FIG2_COLORS[-1]
        return [c[0], c[1], c[2], 255]
        
    return [0, 0, 0, 0]

def get_heat_color(value, max_value):
    """Returns RGBA color based on heat intensity."""
    if value is None or max_value is None or max_value == 0:
        return [128, 128, 128, 50]  # Transparent grey for no data
    
    if value <= 0:
        return [128, 128, 128, 50]  # Transparent grey for zero
    
    intensity = min(value / max_value, 1.0)
    
    if intensity < 0.5:
        r = 255
        g = int(255 - (intensity * 2) * 100)
        b = 0
    else:
        r = 255
        g = int(155 - ((intensity - 0.5) * 2) * 155)
        b = 0
    
    alpha = int(100 + intensity * 155)
    return [r, g, b, alpha]


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_population_data():
    """Load population raster data."""
    try:
        pop = xr.open_dataset(POP_PATH)["ssp2_2050"]
        
        # Malcolm: Fix dimension names for population data too, to be safe
        if 'lat' in pop.dims and 'lon' in pop.dims:
            pop = pop.rename({'lat': 'latitude', 'lon': 'longitude'})
        pop = pop.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        pop = pop.rio.write_crs("EPSG:4326")
        return pop
    except Exception as e:
        st.warning(f"Population data not loaded: {e}")
        return None


@st.cache_data
def load_heat_metric(_base_dir, timescale, metric, person_hours, year):
    """
    Load heat metric data from NetCDF files.
    """
    base_dir = Path(_base_dir)
    
    prefix = f"{timescale}_"
    if person_hours:
        var_name = f"{prefix}person_{metric}_hours"
    else:
        var_name = f"{prefix}{metric}_hours"
    
    nc_path = base_dir / f"{var_name}.nc"
    
    try:
        ds = xr.open_dataset(nc_path, decode_timedelta=False)
        da = ds[var_name]
        
        # Malcolm: CRITICAL FIX for "y dimension not found" error
        # NetCDFs often have dims named 'lat'/'lon'. rioxarray needs 'latitude'/'longitude' or explicit mapping.
        # We rename them to standard names to ensure rioxarray can clip them.
        if 'lat' in da.dims and 'lon' in da.dims:
            da = da.rename({'lat': 'latitude', 'lon': 'longitude'})
        
        # Explicitly set spatial dimensions to ensure rioxarray knows x and y
        if 'latitude' in da.dims and 'longitude' in da.dims:
            da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        else:
            # Fallback: try to use whatever dimensions exist if names are weird, 
            # but usually this means the data structure is unexpected.
            st.warning(f"Unexpected dimensions found: {list(da.dims)}")
            
        # Write CRS
        da = da.rio.write_crs("EPSG:4326")
        
        # Handle timedelta dtype if present
        if np.issubdtype(da.dtype, np.timedelta64):
            da = da.astype("timedelta64[h]").astype("float32")
        
        # Select the requested year
        da = da.sel(time=str(year), method="nearest")
        
        return da
    except FileNotFoundError:
        st.error(f"File not found: {nc_path}")
        return None
    except Exception as e:
        st.error(f"Error loading {nc_path}: {e}")
        return None


@st.cache_data
def load_world_cities(min_population=100000):
    """
    Load world cities with population >= min_population from GeoNames cities15000.txt.
    File format: tab-separated with columns:
    geonameid, name, asciiname, alternatenames, lat, lon, feature_class, feature_code,
    country_code, cc2, admin1, admin2, admin3, admin4, population, elevation, dem, timezone, modification_date
    """
    try:
        if not CITIES_DATA_PATH.exists():
            st.warning(f"Cities data file not found: {CITIES_DATA_PATH}")
            return None
        
        cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'lat', 'lon',
                'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1',
                'admin2', 'admin3', 'admin4', 'population', 'elevation', 'dem',
                'timezone', 'modification_date']
        
        df = pd.read_csv(CITIES_DATA_PATH, sep='\t', names=cols, encoding='utf-8',
                         dtype={'population': float, 'lat': float, 'lon': float},
                         on_bad_lines='skip')
        
        # Filter by population threshold
        df = df[df['population'] >= min_population].copy()
        
        # Create bbox for each city (0.25 degree padding around point for extraction)
        pad = 0.25
        df['bbox'] = df.apply(lambda r: [r['lon']-pad, r['lat']-pad, r['lon']+pad, r['lat']+pad], axis=1)
        
        return df[['geonameid', 'name', 'country_code', 'lat', 'lon', 'population', 'bbox']]
    except Exception as e:
        st.warning(f"Could not load GeoNames cities: {e}")
        return None


def extract_point_values(da, lats, lons):
    """
    Extract heat values at multiple lat/lon points using nearest neighbor interpolation.
    This is more efficient than clipping for many cities.
    """
    values = []
    for lat, lon in zip(lats, lons):
        try:
            val = float(da.sel(latitude=lat, longitude=lon, method='nearest').values)
            values.append(val if not np.isnan(val) else 0)
        except:
            values.append(0)
    return values


def extract_polygon_heat(da, geometry):
    """
    Extract total heat value within a polygon geometry.
    Used for country and continent boundaries.
    """
    try:
        if geometry is None:
            return 0
        da_clipped = da.rio.clip([geometry], drop=True, all_touched=True)
        heat_sum = float(da_clipped.sum(skipna=True))
        return heat_sum if heat_sum > 0 else 0
    except Exception as e:
        return 0

        return 0


def extract_gridded_data(da, stride=1):
    """
    Extract raw gridded data for visualization.
    Returns a DataFrame with lat, lon, and heat_hours.
    """
    try:
        # Downsample if needed to improve performance
        if stride > 1:
            da_sampled = da.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
        else:
            da_sampled = da
            
        # Convert to dataframe
        df = da_sampled.to_dataframe().reset_index()
        
        # Determine value column name (it might be the variable name or 'heat_hours')
        # xr.to_dataframe() often names the value column after the DataArray name
        val_col = None
        for col in df.columns:
            if col not in ['latitude', 'longitude', 'time', 'spatial_ref']:
                val_col = col
                break
        
        if val_col:
            df = df.rename(columns={val_col: 'heat_hours', 'latitude': 'lat', 'longitude': 'lon'})
            return df[['lat', 'lon', 'heat_hours']]
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error extracting gridded data: {e}")
        return pd.DataFrame()


def extract_city_metrics(da, pop, city_config):
    """Extract heat and population metrics for a city."""
    if da is None:
        return {"heat_hours": None, "population": None}
    
    try:
        # ISSUE 2 FIX - Expand bbox
        xmin, ymin, xmax, ymax = city_config["bbox"]
        pad = 0.5
        city_geom = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
        
        da_clipped = da.rio.clip([city_geom], drop=True, all_touched=True)
        
        # SANITY CHECK (Uncomment to debug)
        # st.write(f"DEBUG {city_config['name']} Heat Sum:", float(da_clipped.sum(skipna=True)))

        heat_sum = float(da_clipped.sum(skipna=True))
        
        pop_sum = None
        if pop is not None:
            try:
                pop_clipped = pop.rio.clip([city_geom], drop=True, all_touched=True)
                pop_sum = float(pop_clipped.sum(skipna=True))
            except Exception:
                pass
        
        return {
            "heat_hours": heat_sum if heat_sum > 0 else 0,
            "population": pop_sum
        }
    except Exception as e:
        st.error(f"Error extracting metrics for {city_config['name']}: {e}")
        return {"heat_hours": None, "population": None}


@st.cache_data
def load_country_boundaries():
    """Load country boundaries from Natural Earth data directly from URL"""
    try:
        # Direct URL to Natural Earth 1:110m countries data
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the .shp file
            shp_filename = None
            for file_name in zip_file.namelist():
                if file_name.endswith('.shp'):
                    shp_filename = file_name
                    break
            
            if shp_filename is None:
                raise Exception("No .shp file found in the zip archive")
            
            # Extract all files to a temporary directory
            temp_dir = "/tmp/ne_countries"
            os.makedirs(temp_dir, exist_ok=True)
            zip_file.extractall(temp_dir)
            
            # Load the shapefile
            shp_path = os.path.join(temp_dir, shp_filename)
            world = gpd.read_file(shp_path)
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
            
            return world
    except Exception as e:
        st.warning(f"Could not load Natural Earth boundaries: {e}")
        return None


def get_country_geometry_from_ne(country_name, world_gdf):
    """
    Get country geometry from Natural Earth data using country name
    """
    if world_gdf is None:
        return None
    
    try:
        # Try to find the country by name
        country_row = world_gdf[world_gdf['NAME'] == country_name]
        if len(country_row) > 0:
            return country_row.iloc[0]['geometry']
        else:
            # Try alternative names
            alt_names = {
                "Central African Republic": ["Central African Republic", "CAR", "Central African Rep."],
                "Côte d'Ivoire": ["Côte d'Ivoire", "Ivory Coast", "Cote d'Ivoire"],
                "Mauritania": ["Mauritania"]
            }
            
            for alt_name in alt_names.get(country_name, [country_name]):
                country_row = world_gdf[world_gdf['NAME'] == alt_name]
                if len(country_row) > 0:
                    return country_row.iloc[0]['geometry']
            
        return None
    except Exception as e:
        st.warning(f"Could not get geometry for {country_name}: {e}")
        return None


def extract_country_metrics(da, pop, country_config, world_gdf):
    """
    Extract heat and population metrics for a country using real boundaries from Natural Earth.
    """
    if da is None:
        return {"heat_hours": None, "population": None, "geometry": None}
    
    try:
        # Get real country geometry from Natural Earth or fallback to bbox
        country_geom = get_country_geometry_from_ne(country_config["name"], world_gdf)
        
        if country_geom is None:
            # Fallback to bounding box
            bbox = country_config["bbox_fallback"]
            country_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
            st.warning(f"Using fallback bbox for {country_config['name']}")
        
        # Clip heat data to country boundary
        da_clipped = da.rio.clip([country_geom], drop=True, all_touched=True)
        heat_sum = float(da_clipped.sum(skipna=True))
        
        # Clip population
        pop_sum = None
        if pop is not None:
            try:
                pop_clipped = pop.rio.clip([country_geom], drop=True, all_touched=True)
                pop_sum = float(pop_clipped.sum(skipna=True))
            except Exception as e:
                st.warning(f"Could not clip population for {country_config['name']}: {e}")
                pass
        
        return {
            "heat_hours": heat_sum if heat_sum > 0 else 0,
            "population": pop_sum,
            "geometry": mapping(country_geom)  # Convert shapely geometry to GeoJSON for PyDeck
        }
    except Exception as e:
        st.error(f"Error extracting metrics for {country_config['name']}: {e}")
        # Fallback to bbox
        try:
            bbox = country_config["bbox_fallback"]
            bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
            return {"heat_hours": 0, "population": None, "geometry": mapping(bbox_geom)}
        except:
            return {"heat_hours": None, "population": None, "geometry": None}


def get_bbox_polygon(bbox):
    """Convert bounding box to polygon coordinates for PyDeck."""
    xmin, ymin, xmax, ymax = bbox
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin]
    ]


# ---------------- MAIN APP ----------------
def main():
    st.title("🌡️ Heat Impact Dashboard")
    
    # ---------------- TABS ----------------
    tab_dashboard, tab_docs = st.tabs(["Dashboard", "Documentation"])
    
    with tab_dashboard:
        # ---------------- SIDEBAR ----------------
        st.sidebar.header("⚙️ Dashboard Settings")
        
        scenario = st.sidebar.selectbox("Climate Scenario", ["SSP2-RCP4.5"])
        
        metric_type = st.sidebar.radio(
            "Metric Type",
            ["Tw (Wet-bulb Temperature)", "WBGT (Wet Bulb Globe Temperature)"]
        )
        
        if "Tw" in metric_type:
            metric = st.sidebar.selectbox(
                "Impact Level",
                ["hot", "lethal"],
                format_func=lambda x: {"hot": "Hot Hours", "lethal": "Lethal Hours"}.get(x, x)
            )
            base_dir = TW_DIR
        else:
            metric = st.sidebar.selectbox(
                "Impact Level",
                ["mhs", "shs", "ehs"],
                format_func=lambda x: {
                    "mhs": "WBGT ≥30°C", "shs": "WBGT ≥33°C", "ehs": "WBGT ≥35°C"
                }.get(x, x)
            )
            base_dir = WBGT_DIR
        
        person_hours = st.sidebar.checkbox("Person × Hours", value=False)
        timescale = st.sidebar.selectbox("Aggregation Period", ["annual", "quarterly", "monthly"], format_func=lambda x: x.capitalize())
        year = st.sidebar.slider("Year", min_value=2015, max_value=2025, value=2025, step=5)
        
        # Updated Display Modes Order and Naming
        display_mode = st.sidebar.radio(
            "Display Mode",
            ["Global Gridded", "All Cities", "Priority Locations", "Countries", "Continents"],
            help="Global Gridded: Raw data | All Cities: ~4K cities >100K | Priority Locations: 3 African cities | Countries: National Aggregation | Continents: Regional Aggregation"
        )
        
        # ---------------- LOAD DATA ----------------
        with st.spinner("Loading heat data..."):
            da = load_heat_metric(str(base_dir), timescale, metric, person_hours, year)
            pop = load_population_data()
        
        if da is None:
            st.error("Could not load heat metric data.")
            st.stop()
        
        # Load country boundaries for Countries/Continents modes
        world_gdf = None
        if display_mode in ["Countries", "Continents"]:
            with st.spinner("Loading country boundaries..."):
                world_gdf = load_country_boundaries()
        
        # ---------------- COMPUTE METRICS ----------------
        with st.spinner("Computing metrics..."):
            is_point_mode = display_mode in ["All Cities", "Priority Locations"]
            is_grid_mode = display_mode == "Global Gridded"
            
            if display_mode == "Global Gridded":
                # Stride of 2 or 5 depending on resolution/performance needs
                # Using 4 to ensure responsiveness on Colab
                display_data = extract_gridded_data(da, stride=4)
                if len(display_data) > 0:
                     st.sidebar.success(f"Loaded {len(display_data):,} grid points")
                else:
                    st.sidebar.warning("No data found in grid")

            elif display_mode == "All Cities":
                cities_df = load_world_cities(min_population=100000)
                if cities_df is not None and len(cities_df) > 0:
                    heat_values = extract_point_values(da, cities_df['lat'].tolist(), cities_df['lon'].tolist())
                    city_data = []
                    for idx, (_, city) in enumerate(cities_df.iterrows()):
                        city_data.append({
                            "name": city["name"],
                            "country": city["country_code"],
                            "lat": city["lat"],
                            "lon": city["lon"],
                            "heat_hours": heat_values[idx],
                            "population": city["population"]
                        })
                    display_data = pd.DataFrame(city_data)
                    st.sidebar.success(f"Loaded {len(cities_df):,} cities")
                else:
                    st.sidebar.error("Could not load cities data")
                    display_data = pd.DataFrame()
            
            elif display_mode == "Priority Locations": # Renamed from Priority Cities
                city_data = []
                for city in CITIES_CONFIG:
                    metrics = extract_city_metrics(da, pop, city)
                    # For Priority Locations, we want to show them even if 0, 
                    # but extract_city_metrics might return None if error.
                    val = metrics["heat_hours"] if metrics["heat_hours"] is not None else 0
                    city_data.append({
                        "name": city["name"],
                        "country": city["country"],
                        "lat": city["lat"],
                        "lon": city["lon"],
                        "heat_hours": val,
                        "population": metrics["population"]
                    })
                display_data = pd.DataFrame(city_data)
                st.sidebar.info("Showing 3 priority locations")
            
            elif display_mode == "Countries":
                if world_gdf is not None:
                    country_data = []
                    progress_bar = st.progress(0)
                    total_countries = len(world_gdf)
                    for idx, (_, country) in enumerate(world_gdf.iterrows()):
                        heat_val = extract_polygon_heat(da, country.geometry)
                        country_data.append({
                            "name": country.get('NAME', country.get('ADMIN', 'Unknown')),
                            "continent": country.get('CONTINENT', 'Unknown'),
                            "heat_hours": heat_val,
                            "population": country.get('POP_EST', 0),
                            "geometry": mapping(country.geometry) if country.geometry else None
                        })
                        progress_bar.progress((idx + 1) / total_countries)
                    progress_bar.empty()
                    display_data = pd.DataFrame(country_data)
                    st.sidebar.success(f"Loaded {len(country_data)} countries")
                else:
                    display_data = pd.DataFrame()
            
            elif display_mode == "Continents":
                if world_gdf is not None:
                    continents_gdf = world_gdf.dissolve(by='CONTINENT', aggfunc='sum')
                    continent_data = []
                    for continent_name, continent_row in continents_gdf.iterrows():
                        heat_val = extract_polygon_heat(da, continent_row.geometry)
                        continent_data.append({
                            "name": continent_name,
                            "heat_hours": heat_val,
                            "population": continent_row.get('POP_EST', 0) if hasattr(continent_row, 'POP_EST') else 0,
                            "geometry": mapping(continent_row.geometry) if continent_row.geometry else None
                        })
                    display_data = pd.DataFrame(continent_data)
                    st.sidebar.success(f"Aggregated into {len(continent_data)} continents")
                else:
                    display_data = pd.DataFrame()
            
        # Color Scaling & Filtering
        st.caption(f"Processing {len(display_data)} locations...")
        
        # 1. Cleaning / Filtering
        if display_mode == "Priority Locations":
            # Keep everything for priority locations (even if 0 or None)
            pass
        elif display_mode == "Global Gridded":
            # For grid, keep 0s (land with no heat), drop NaNs (ocean)
            if "heat_hours" in display_data.columns:
                display_data = display_data.dropna(subset=["heat_hours"])
        else:
            # For others, filter out 0s to reduce noise
            if "heat_hours" in display_data.columns:
                display_data = display_data[display_data["heat_hours"] > 0].copy()
        
        # 2. Add display columns
        if not display_data.empty:
            valid_heat = display_data["heat_hours"]
            max_heat = valid_heat.max() if len(valid_heat) > 0 else 1
            
            if is_grid_mode:
                # Use discrete bins from Figure 2
                display_data["color"] = display_data["heat_hours"].apply(get_fig2_color)
            else:
                # Use continuous scaling for other modes
                display_data["color"] = display_data["heat_hours"].apply(lambda x: get_heat_color(x, max_heat))
                
            # Format numbers
            display_data["heat_display"] = display_data["heat_hours"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "No data")
            if "population" in display_data.columns:
                display_data["pop_display"] = display_data["population"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            else:
                 display_data["pop_display"] = "N/A"
        else:
            if display_mode != "Priority Locations":
                st.warning("No locations found with heat impact > 0 for this selection.")

        # ---------------- MAP LAYERS ----------------
        layers = []
        
        if is_grid_mode:
             if not display_data.empty:
                # Use ScatterplotLayer with large radius to simulate grid cells visually
                # Radius 30km approximates 0.5 degrees at equator
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    display_data,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=30000, 
                    pickable=True,
                    opacity=1.0,
                    stroked=False,
                    radius_min_pixels=1
                )
                layers.append(layer)
        
        elif is_point_mode:
            if not display_data.empty:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    display_data,
                    get_position="[lon, lat]",
                    get_radius=25000,
                    get_fill_color="color",
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    get_line_color=[255, 255, 255],
                    radius_min_pixels=5,
                    radius_max_pixels=30,
                )
                layers.append(layer)
        else:
            if not display_data.empty:
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    display_data,
                    opacity=0.7,
                    stroked=True,
                    filled=True,
                    get_line_color=[255, 255, 255],
                    get_fill_color="color",
                    pickable=True,
                    get_polygon="geometry"
                )
                layers.append(layer)
        
        view_state = pdk.ViewState(latitude=9.0, longitude=5.0, zoom=3.0, pitch=0)
        
        metric_label = "Lethal Hours" if metric == "lethal" else "Hot Hours" if metric == "hot" else f"WBGT {metric.upper()} Hours"
        if person_hours:
            metric_label = f"Person × {metric_label}"
        
        # Tooltip Logic (Fixed country display)
        # Use python f-string logic that PyDeck understands.
        # {name} in pydeck string means look up property "name".
        # We need to ensure columns exist. grid mode might not have name/country.
        
        if is_grid_mode:
            tooltip = {
                "html": f"""
                    <div style='padding: 8px; font-family: Arial, sans-serif;'>
                        <b>{metric_label}:</b> {{heat_hours}}<br/>
                    </div>
                """,
                "style": {"backgroundColor": "#1a1a2e", "color": "white", "border": "1px solid #444"}
            }
        else:
            # For points/polygons
            tooltip_html = f"<div style='padding: 8px; font-family: Arial, sans-serif;'><b style='font-size: 14px;'>{{name}}</b><br/>"
            
            # Only add country row if in Point/Priority mode
            if is_point_mode:
                 tooltip_html += "<span style='color: #aaa;'>{country}</span><br/><br/>"
            elif display_mode == "Countries":
                 tooltip_html += "<span style='color: #aaa;'>{continent}</span><br/><br/>"
                 
            tooltip_html += f"<b>{metric_label}:</b> {{heat_display}}<br/>"
            tooltip_html += "<b>Population:</b> {pop_display}<br/></div>"
            
            tooltip = {
                "html": tooltip_html,
                "style": {"backgroundColor": "#1a1a2e", "color": "white", "border": "1px solid #444"}
            }
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        )
        
        st.pydeck_chart(deck, use_container_width=True)
        
        # ---------------- DATA TABLE & EXPORT ----------------
        st.subheader("📊 Data Summary")
        
        if not display_data.empty and not is_grid_mode:
            if is_point_mode:
                cols = ["name", "country", "lat", "lon", "heat_display", "pop_display"]
                new_cols = ["City", "Country", "Latitude", "Longitude", metric_label, "Population"]
            else:
                cols = ["name", "heat_display", "pop_display"]
                new_cols = ["Location", metric_label, "Population"]
                if "continent" in display_data.columns:
                    cols.insert(1, "continent")
                    new_cols.insert(1, "Continent")
            
            # select existing columns only
            display_df = display_data[[c for c in cols if c in display_data.columns]].copy()
            if len(display_df.columns) == len(new_cols):
                 display_df.columns = new_cols
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export
            export_df = display_data.copy()
            # Add Metrics
            export_df["Year"] = year
            export_df["Period"] = timescale.capitalize()
            export_df["Metric Type"] = metric_label
            
            csv_data = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            filename = f"heat_impact_{year}_{metric}.csv"
            href = f'<a href="data:text/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.3rem; font-weight: 500;">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.caption("Click the button above to download the summary table as CSV")
        elif is_grid_mode:
            st.info("Data table not available for gridded view. Use 'All Cities' to see listing.")

    with tab_docs:
        st.markdown("""
        ## Heat Impact Metrics Documentation
        
        ### Metric Definitions
        
        **1. Tw (Wet-bulb Temperature - Units °C)**
        The temperature a parcel of air would have if it were cooled to saturation (100% relative humidity) by the evaporation of water into it, with the latent heat supplied by the parcel. Tw accounts only for air temperature and humidity, and is thus an indoor heat stress metric.
        - **Hot Hours**: Hours where Tw exceeds 30.6 °C. Following Vecellio et al., 2023, this threshold is referred to as uncompensable heat or human survivability limit. Also referred to as the theoretical physiological limit to heat adaptation. Note that this threshold is for an average healthy young adult, and the threshold would be lower for vulnerable people. The threshold also reduces linearly at locations with air temperature exceeding 40°C (referred to as dry hot regions). See Vecellio et al., 2023 for further details.
        - **Lethal Hours**: Hours where Tw > 30.6°C for at least 6 consecutive hours in a day. Can lead to heatstroke even for an average healthy adult wearing light clothing and at rest.
        
        **2. WBGT (Wet Bulb Globe Temperature  - Units °C)**
        A composite outdoor heat stress metric used to estimate the effect of temperature, humidity, wind speed, and solar radiation on humans. Commonly used as a heat stress metric in assessing labour productivity. Following Brimicombe et al., 2023 and Mishra et al., 2025, the following three definitions are used for computing the unworkable hours. 
        - **MHS hours (WBGT ≥ 30°C & < 33 °C)**: Moderate Heat Stress. High intensity labour conducted by acclimatized adults in light clothing should be moderated with frequent breaks.
        - **SHS hours (WBGT ≥ 33°C & < 35 °C)**: Severe Heat Stress. Same assumptions as in MHS, but  work activities should be severely limited or stopped
        - **EHS hours (WBGT ≥ 35°C)**: Extreme Heat Stress. Lethal or extreme heat stress. No outdoor physical activity should be undertaken and health conditions of individuals to be monitored, as this threshold when exceeded can be fatal.
        
        ### Methodology
        The Tw and WBGT global gridded data at 0.25 deg resolution (~30km x 30km at the equator) from CMIP6 downscaled models covering 2015-2100 in the SSP245 scenario at 3-hourly time steps, as well as the SSP2 projected population for 2050 at a global gridded scale of 0.125 deg (~15km x 15km at the equator), were both provided by the authors of the Vecellio et al., 2023 study. The sub-daily files for each year were processed to derive weekly, monthly, quarterly and annual number of hours exceeding the above threshold.
        
        - **Person-Hours**: The metric multiplied by the local population count, representing location-specific total human exposure at the above sub-annual/annual timescales.
        - **Cities**: Aggregated at city locations (GeoNames database).
        - **National/Regional**: Spatially aggregated over administrative boundaries.
        - **References**: (i) Brimicombe et al., 2023: https://doi.org/10.1029/2022GH000701, (ii) Mishra et al., 2025: https://doi.org/10.1029/2025EF006167, Vecellio et al., 2023: : https://doi.org/10.1073/pnas.2305427120

        """)


if __name__ == "__main__":
    main()





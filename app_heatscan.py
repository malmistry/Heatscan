
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heat Impact Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# ---------------- PATHS (Malcolm: Update these paths as needed) ----------------
TW_DIR = Path("/home/lshmm22/Projects/WB/Data/Heatscan_and_SP_work/Tw_hours/output_2015_2100_5yrly_interval")
WBGT_DIR = Path("/home/lshmm22/Projects/WB/Data/Heatscan_and_SP_work/WBGT_hours/output_2015_2100_5yrly_interval")
#POP_PATH = Path("/home/lshmm22/Projects/WB/Data/Heatscan_and_SP_work/population/ssp2_2050.nc")
#https://github.com/malmistry/Heatscan/tree/main/data
CITIES_DATA_PATH = Path("/home/lshmm22/Projects/WB/Data/Heatscan_and_SP_work/cities/cities15000.txt")

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
    #year = st.sidebar.slider("Year", min_value=2015, max_value=2025, value=2025, step=5)
    year = st.sidebar.slider("Year", min_value=2025, max_value=2100, value=2025, step=5)
    display_mode = st.sidebar.radio("City Representation", ["Points", "Boundaries"])
    
    # ---------------- LOAD DATA ----------------
    with st.spinner("Loading heat data..."):
        da = load_heat_metric(str(base_dir), timescale, metric, person_hours, year)
        pop = load_population_data()
    
    if da is None:
        st.error("Could not load heat metric data.")
        st.stop()
    
    # Load country boundaries for boundaries mode
    world_gdf = None
    if display_mode == "Boundaries":
        with st.spinner("Loading country boundaries..."):
            world_gdf = load_country_boundaries()
    
    # ---------------- COMPUTE METRICS ----------------
    with st.spinner("Computing metrics..."):
        if display_mode == "Points":
            # Load all cities with population > 100K from GeoNames
            cities_df = load_world_cities(min_population=100000)
            
            if cities_df is not None and len(cities_df) > 0:
                # Extract heat values efficiently using point sampling
                heat_values = extract_point_values(da, cities_df['lat'].tolist(), cities_df['lon'].tolist())
                
                city_data = []
                for idx, (_, city) in enumerate(cities_df.iterrows()):
                    city_data.append({
                        "name": city["name"],
                        "country": city["country_code"],
                        "lat": city["lat"],
                        "lon": city["lon"],
                        "bbox": city["bbox"],
                        "polygon": get_bbox_polygon(city["bbox"]),
                        "heat_hours": heat_values[idx],
                        "population": city["population"]
                    })
                display_data = pd.DataFrame(city_data)
                st.sidebar.success(f"Loaded {len(cities_df):,} cities with population > 100K")
            else:
                # Fallback to hardcoded cities if GeoNames fails
                city_data = []
                for city in CITIES_CONFIG:
                    metrics = extract_city_metrics(da, pop, city)
                    city_data.append({
                        "name": city["name"],
                        "country": city["country"],
                        "lat": city["lat"],
                        "lon": city["lon"],
                        "bbox": city["bbox"],
                        "polygon": get_bbox_polygon(city["bbox"]),
                        "heat_hours": metrics["heat_hours"],
                        "population": metrics["population"]
                    })
                display_data = pd.DataFrame(city_data)
                st.sidebar.warning("Using fallback cities (3 priority cities)")
        else:
            # Use manual country config with Natural Earth boundaries
            country_data = []
            for country in COUNTRIES_CONFIG:
                metrics = extract_country_metrics(da, pop, country, world_gdf)
                country_data.append({
                    "name": country["name"],
                    "heat_hours": metrics["heat_hours"],
                    "population": metrics["population"],
                    "geometry": metrics["geometry"]
                })
            display_data = pd.DataFrame(country_data)
        
        # Color Scaling
        valid_heat = display_data["heat_hours"].dropna()
        max_heat = valid_heat.max() if len(valid_heat) > 0 else 1
        display_data["color"] = display_data["heat_hours"].apply(lambda x: get_heat_color(x, max_heat))
        display_data["heat_display"] = display_data["heat_hours"].apply(lambda x: f"{x:,.0f}" if x and x > 0 else "No data")
        display_data["pop_display"] = display_data["population"].apply(lambda x: f"{x:,.0f}" if x and x > 0 else "N/A")
    
    # ---------------- MAP LAYERS ----------------
    layers = []
    
    if display_mode == "Points":
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
            radius_min_pixels=15,
            radius_max_pixels=60,
        )
        layers.append(layer)
    else:
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
    
    view_state = pdk.ViewState(latitude=9.0, longitude=5.0, zoom=3.5, pitch=0)
    
    metric_label = "Lethal Hours" if metric == "lethal" else "Hot Hours" if metric == "hot" else f"WBGT {metric.upper()} Hours"
    if person_hours:
        metric_label = f"Person × {metric_label}"
    
    tooltip = {
        "html": f"""
            <div style='padding: 8px; font-family: Arial, sans-serif;'>
                <b style='font-size: 14px;'>{{name}}</b><br/>
                {'<span style=\'color: #aaa;\'>{{country}}</span><br/><br/>' if display_mode == 'Points' else ''}
                <b>{metric_label}:</b> {{heat_display}}<br/>
                <b>Population:</b> {{pop_display}}<br/>
            </div>
        """,
        "style": {"backgroundColor": "#1a1a2e", "color": "white", "border": "1px solid #444"}
    }
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # ---------------- DATA TABLE ----------------
    st.subheader("📊 Data Summary")
    if display_mode == "Points":
        display_df = display_data[["name", "country", "lat", "lon", "heat_display", "pop_display"]].copy()
        display_df.columns = ["City", "Country", "Latitude", "Longitude", metric_label, "Population"]
        # Export dataframe with raw numeric values
        export_df = display_data[["name", "country", "lat", "lon", "heat_hours", "population"]].copy()
        export_df.columns = ["City", "Country", "Latitude", "Longitude", metric_label, "Population"]
    else:
        display_df = display_data[["name", "heat_display", "pop_display"]].copy()
        display_df.columns = ["Country", metric_label, "Population"]
        # Export dataframe with raw numeric values
        export_df = display_data[["name", "heat_hours", "population"]].copy()
        export_df.columns = ["Country", metric_label, "Population"]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # CSV Download button
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"heat_impact_{year}_{metric}.csv",
        mime="text/csv",
        help="Download the summary table as a CSV file"
    )

if __name__ == "__main__":
    main()


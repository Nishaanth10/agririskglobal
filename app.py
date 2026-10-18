import math
from datetime import date

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from folium.plugins import Draw
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from streamlit_folium import st_folium


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriRisk Global",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriRisk Global")
st.caption(
    "Global cropland analysis, weather-risk screening, "
    "and stochastic simulation"
)

LAND_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up area",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map"
)
POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "selected_point": (13.0827, 80.2707),
    "weather_data": None,
    "weather_location": None,
    "cropland_result": None,
    "cropland_location": None,
    "selected_polygon": None,
    "polygon_result": None,
    "polygon_signature": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_longitude(lon):
    return ((float(lon) + 180) % 360) - 180


def location_key(lat, lon):
    return (round(float(lat), 5), round(float(lon), 5))


def worldcover_tile_name(lat, lon):
    south = math.floor(lat / 3) * 3
    west = math.floor(lon / 3) * 3
    ns = "N" if south >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(south):02d}{ew}{abs(west):03d}"


def iter_polygon_coordinates(geometry):
    """Yield coordinate pairs from Polygon or MultiPolygon GeoJSON."""
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        for ring in coords:
            for point in ring:
                yield float(point[0]), float(point[1])

    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for point in ring:
                    yield float(point[0]), float(point[1])

    else:
        raise ValueError(
            "Please draw a polygon or rectangle on the map."
        )


def approximate_polygon_center(geometry):
    points = list(iter_polygon_coordinates(geometry))
    if not points:
        raise ValueError("The drawn polygon has no coordinates.")

    # A simple vertex-average center for the NASA POWER proxy.
    # This is not an exact area-weighted polygon centroid.
    lat = sum(p[1] for p in points) / len(points)
    lon = sum(p[0] for p in points) / len(points)

    return float(lat), normalize_longitude(lon)


def polygon_bounds(geometry):
    points = list(iter_polygon_coordinates(geometry))
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return min(lons), min(lats), max(lons), max(lats)


def tile_url(tile):
    filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    return f"{WORLDCOVER_BASE}/{filename}"


# --------------------------------------------------
# SINGLE-POINT WORLDCOVER LOOKUP
# --------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_worldcover_class(lat, lon):
    lat = float(lat)
    lon = normalize_longitude(lon)

    if not -90 <= lat <= 90:
        raise ValueError("Invalid latitude.")

    tile = worldcover_tile_name(lat, lon)
    url = tile_url(tile)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="1",
    ):
        with rasterio.open(url) as dataset:
            pixel = next(dataset.sample([(lon, lat)], masked=True))

            if np.ma.is_masked(pixel[0]):
                code = None
            else:
                code = int(pixel[0])

    return {
        "code": code,
        "name": LAND_CLASSES.get(code, "Unknown / no data"),
        "cropland": code == 40,
        "tile": tile,
        "url": url
    }


# --------------------------------------------------
# POLYGON CROPLAND PERCENTAGE
# --------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def calculate_polygon_cropland(geometry_json):
    """
    Count actual WorldCover pixels inside a drawn polygon.
    Geometry is GeoJSON in longitude/latitude (EPSG:4326).
    """
    geometry = geometry_json
    min_lon, min_lat, max_lon, max_lat = polygon_bounds(geometry)

    if max_lat <= -90 or min_lat >= 90:
        raise ValueError("Polygon is outside valid latitude bounds.")

    min_lat = max(min_lat, -90)
    max_lat = min(max_lat, 90)

    if max_lon - min_lon > 180:
        raise ValueError(
            "Polygons crossing the antimeridian are not supported yet. "
            "Please draw a polygon on one side of the date line."
        )

    total_valid = 0
    total_cropland = 0
    total_pixels = 0
    tiles_read = []

    south_tiles = range(
        math.floor(min_lat / 3) * 3,
        math.floor(max_lat / 3) * 3 + 1,
        3
    )
    west_tiles = range(
        math.floor(min_lon / 3) * 3,
        math.floor(max_lon / 3) * 3 + 1,
        3
    )

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="1",
    ):
        for south in south_tiles:
            for west in west_tiles:
                tile = worldcover_tile_name(
                    south + 0.01, west + 0.01
                )
                url = tile_url(tile)

                # Intersect polygon bounding box with this 3x3 degree tile.
                left = max(min_lon, west)
                bottom = max(min_lat, south)
                right = min(max_lon, west + 3)
                top = min(max_lat, south + 3)

                if left >= right or bottom >= top:
                    continue

                try:
                    with rasterio.open(url) as dataset:
                        # The WorldCover tile grid is geographic.
                        # Transform polygon if a tile uses another CRS.
                        from rasterio.warp import transform_geom

                        raster_geometry = geometry
                        if dataset.crs and dataset.crs.to_epsg() != 4326:
                            raster_geometry = transform_geom(
                                "EPSG:4326",
                                dataset.crs,
                                geometry
                            )

                        if dataset.crs and dataset.crs.to_epsg() == 4326:
                            window = from_bounds(
                                left, bottom, right, top,
                                dataset.transform
                            )
                        else:
                            # For non-geographic tiles, use a transformed
                            # geometry's bounds to construct the read window.
                            from rasterio.features import bounds as geom_bounds
                            gleft, gbottom, gright, gtop = geom_bounds(
                                raster_geometry
                            )
                            window = from_bounds(
                                gleft, gbottom, gright, gtop,
                                dataset.transform
                            )

                        window = window.round_offsets().round_lengths()
                        window = window.intersection(
                            Window(0, 0, dataset.width, dataset.height)
                        )

                        if window.width <= 0 or window.height <= 0:
                            continue

                        data = dataset.read(1, window=window, masked=True)
                        transform = dataset.window_transform(window)

                        inside = geometry_mask(
                            [raster_geometry],
                            out_shape=data.shape,
                            transform=transform,
                            invert=True,
                            all_touched=False
                        )

                        valid = inside & ~np.ma.getmaskarray(data)
                        values = np.asarray(data.data)

                        # WorldCover 0 is treated as no-data.
                        valid &= values != 0

                        valid_count = int(valid.sum())
                        crop_count = int(
                            np.logical_and(valid, values == 40).sum()
                        )

                        total_valid += valid_count
                        total_cropland += crop_count
                        total_pixels += int(inside.sum())
                        tiles_read.append(tile)

                except RasterioIOError:
                    # A tile with no available data may fail to open.
                    # Other tiles can still contribute to the estimate.
                    continue

    if total_valid == 0:
        raise RuntimeError(
            "No valid WorldCover pixels were read. "
            "Check the polygon and your internet connection."
        )

    percentage = 100.0 * total_cropland / total_valid

    return {
        "cropland_pixels": total_cropland,
        "valid_pixels": total_valid,
        "inside_pixels": total_pixels,
        "cropland_percentage": percentage,
        "noncropland_percentage": 100.0 - percentage,
        "tiles": sorted(set(tiles_read)),
    }


# --------------------------------------------------
# NASA POWER WEATHER
# --------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(lat, lon, start_date, end_date):
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": normalize_longitude(lon),
        "latitude": float(lat),
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC"
    }

    response = requests.get(
        POWER_URL, params=params, timeout=90
    )
    response.raise_for_status()
    payload = response.json()
    parameters = payload["properties"]["parameter"]

    rainfall = parameters["PRECTOTCORR"]
    temperature = parameters["T2M"]

    rows = []
    for day, rain in rainfall.items():
        temp = temperature.get(day)

        if rain is None or temp is None:
            continue
        if rain <= -900 or temp <= -900:
            continue

        rows.append({
            "Date": pd.to_datetime(day, format="%Y%m%d"),
            "Rainfall (mm/day)": float(rain),
            "Temperature (°C)": float(temp)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("NASA POWER returned no valid weather records.")

    return df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# WEATHER RISK SCREENING INDEX + DATA-FITTED STATES
# --------------------------------------------------

def classify_weather_states(df):
    """Classify observed days into favorable, watch, and adverse states.

    Thresholds are estimated from the selected weather history. This is a
    weather-stress proxy, not a crop-specific biological threshold.
    """
    data = df.copy().sort_values("Date").reset_index(drop=True)
    rain = data["Rainfall (mm/day)"].clip(lower=0)
    temp = data["Temperature (°C)"]

    rain_q25 = float(rain.quantile(0.25))
    rain_q50 = float(rain.quantile(0.50))
    temp_q70 = float(temp.quantile(0.70))
    temp_q85 = float(temp.quantile(0.85))

    # State 0 = favorable, 1 = watch, 2 = adverse.
    # Adverse includes unusually dry OR unusually hot days.
    states = []
    for r, t in zip(rain, temp):
        is_adverse = (r <= rain_q25) or (t >= temp_q85)
        is_favorable = (r >= rain_q50) and (t < temp_q70)
        if is_adverse:
            states.append(2)
        elif is_favorable:
            states.append(0)
        else:
            states.append(1)

    data["Weather state"] = states
    return data, {
        "rain_q25": rain_q25,
        "rain_q50": rain_q50,
        "temp_q70": temp_q70,
        "temp_q85": temp_q85,
    }


def fit_weather_transition_matrix(states, smoothing=1.0):
    """Fit transition counts and posterior-mean transition probabilities.

    A symmetric Dirichlet prior with concentration ``smoothing`` prevents
    zero-probability transitions. Raw counts are returned for uncertainty
    estimation during simulation.
    """
    counts = np.zeros((3, 3), dtype=float)
    values = np.asarray(states, dtype=int)
    for current, following in zip(values[:-1], values[1:]):
        counts[current, following] += 1

    posterior_alpha = counts + float(smoothing)
    posterior_mean = posterior_alpha / posterior_alpha.sum(axis=1, keepdims=True)
    return posterior_mean, counts, posterior_alpha


def estimate_weather_risk(df):
    """Descriptive index and data-fitted state model, not crop-failure odds."""
    data, thresholds = classify_weather_states(df)
    rain = data["Rainfall (mm/day)"].clip(lower=0)
    temp = data["Temperature (°C)"]

    # Relative anomalies use this selected period as the baseline.
    rain_median = float(rain.median())
    temp_median = float(temp.median())
    rain_deficit = float((rain <= thresholds["rain_q25"]).mean())
    hot_fraction = float((temp >= thresholds["temp_q85"]).mean())

    # This index is intentionally labeled as a screening score.
    score = 100 * (0.60 * rain_deficit + 0.40 * hot_fraction)
    score = float(np.clip(score, 0, 100))

    if score < 25:
        category = "Lower relative weather-stress indicator"
    elif score < 50:
        category = "Moderate relative weather-stress indicator"
    elif score < 75:
        category = "Elevated relative weather-stress indicator"
    else:
        category = "High relative weather-stress indicator"

    matrix, transition_counts, transition_alpha = fit_weather_transition_matrix(
        data["Weather state"].to_numpy(), smoothing=1.0
    )
    counts = data["Weather state"].value_counts(normalize=True).reindex(
        [0, 1, 2], fill_value=0
    )

    monthly = data.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly = monthly.groupby("Month").agg(
        rainfall=("Rainfall (mm/day)", "sum"),
        mean_temp=("Temperature (°C)", "mean"),
        adverse_days=("Weather state", lambda x: int((x == 2).sum())),
        days=("Weather state", "size"),
    ).reset_index()

    return {
        "score": score,
        "category": category,
        "rain_deficit": rain_deficit,
        "hot_fraction": hot_fraction,
        "monthly": monthly,
        "classified": data,
        "thresholds": thresholds,
        "transition_matrix": matrix,
        "transition_counts": transition_counts,
        "transition_alpha": transition_alpha,
        "state_fractions": counts.to_numpy(dtype=float),
        "rain_median": rain_median,
        "temp_median": temp_median,
    }


def simulate_weather_risk(
    transition_alpha,
    historical_data,
    initial_state,
    horizon_days,
    n_simulations,
    adverse_fraction_threshold=0.30,
    adverse_streak_threshold=5,
    seed=None,
):
    """Bayesian Markov Monte Carlo with weather-value resampling.

    Each simulated path draws a transition matrix from row-wise Dirichlet
    posteriors, so uncertainty in estimated transition probabilities is
    represented. Weather values are resampled from historical observations
    in the same state and nearby calendar months to retain rough seasonality.

    The event probability is for a provisional adverse-weather rule, not crop
    failure. With short records, posterior and empirical-resampling uncertainty
    remain substantial.
    """
    rng = np.random.default_rng(seed)
    historical = historical_data.copy().sort_values("Date").reset_index(drop=True)
    historical["Month"] = historical["Date"].dt.month
    historical["Rainfall (mm/day)"] = historical["Rainfall (mm/day)"].clip(lower=0)

    # Precompute conditional empirical pools for each state and month.
    # A +/- 1 month window helps when only a few years are available.
    pools = {}
    for state in range(3):
        for month in range(1, 13):
            month_distance = np.minimum(
                (historical["Month"].to_numpy() - month) % 12,
                (month - historical["Month"].to_numpy()) % 12,
            )
            mask = (
                (historical["Weather state"].to_numpy() == state)
                & (month_distance <= 1)
            )
            subset = historical.loc[mask, ["Rainfall (mm/day)", "Temperature (°C)"]]
            if subset.empty:
                subset = historical.loc[
                    historical["Weather state"] == state,
                    ["Rainfall (mm/day)", "Temperature (°C)"]
                ]
            if subset.empty:
                subset = historical[["Rainfall (mm/day)", "Temperature (°C)"]]
            pools[(state, month)] = subset.to_numpy(dtype=float)

    # Begin the forecast after the last observation's calendar month.
    last_month = int(historical["Date"].iloc[-1].month)
    states = np.zeros((n_simulations, horizon_days), dtype=np.int8)
    rainfall_paths = np.zeros((n_simulations, horizon_days), dtype=float)
    temperature_paths = np.zeros((n_simulations, horizon_days), dtype=float)

    for sim in range(n_simulations):
        # Posterior uncertainty: each row is a Dirichlet draw over next states.
        transition = np.vstack([
            rng.dirichlet(transition_alpha[row]) for row in range(3)
        ])
        previous_state = int(initial_state)
        for day in range(horizon_days):
            month = ((last_month - 1 + day + 1) % 12) + 1
            state = int(rng.choice(3, p=transition[previous_state]))
            states[sim, day] = state
            previous_state = state
            sampled = pools[(state, month)]
            observation = sampled[rng.integers(0, len(sampled))]
            rainfall_paths[sim, day] = max(0.0, float(observation[0]))
            temperature_paths[sim, day] = float(observation[1])

    adverse = states == 2
    adverse_days = adverse.sum(axis=1)
    adverse_fraction = adverse_days / horizon_days

    longest_run = np.zeros(n_simulations, dtype=int)
    current_run = np.zeros(n_simulations, dtype=int)
    for day in range(horizon_days):
        current_run = np.where(adverse[:, day], current_run + 1, 0)
        longest_run = np.maximum(longest_run, current_run)

    # Additional weather-impact indicators from simulated physical variables.
    historical_rain = historical["Rainfall (mm/day)"].to_numpy(dtype=float)
    historical_temp = historical["Temperature (°C)"].to_numpy(dtype=float)
    rain_q10 = float(np.quantile(historical_rain, 0.10))
    temp_q90 = float(np.quantile(historical_temp, 0.90))
    dry_days = (rainfall_paths <= rain_q10).sum(axis=1)
    hot_days = (temperature_paths >= temp_q90).sum(axis=1)
    historical_mean_daily_rain = float(np.mean(historical_rain))
    low_rain_cutoff = max(
        1.0, historical_mean_daily_rain * horizon_days * 0.70
    )
    low_rain_event = rainfall_paths.sum(axis=1) < low_rain_cutoff

    # Event: state-based adverse exposure or a prolonged adverse run.
    event = (adverse_fraction >= adverse_fraction_threshold) | (
        longest_run >= adverse_streak_threshold
    )
    event_count = int(event.sum())
    p_hat = event_count / n_simulations

    # Wilson score interval: more reliable than a normal interval near 0 or 1.
    z = 1.96
    denom = 1 + z * z / n_simulations
    center = (p_hat + z * z / (2 * n_simulations)) / denom
    half = z * np.sqrt(
        p_hat * (1 - p_hat) / n_simulations
        + z * z / (4 * n_simulations**2)
    ) / denom

    return {
        "event_probability": float(p_hat * 100),
        "ci_lower": float(max(0.0, center - half) * 100),
        "ci_upper": float(min(1.0, center + half) * 100),
        "adverse_days_mean": float(adverse_days.mean()),
        "adverse_days_p90": float(np.quantile(adverse_days, 0.90)),
        "event_count": event_count,
        "n_simulations": int(n_simulations),
        "adverse_days": adverse_days,
        "event_flags": event,
        "dry_days_mean": float(dry_days.mean()),
        "hot_days_mean": float(hot_days.mean()),
        "rainfall_total_mean": float(rainfall_paths.sum(axis=1).mean()),
        "temperature_mean": float(temperature_paths.mean()),
        "longest_adverse_run_p90": float(np.quantile(longest_run, 0.90)),
        "low_rain_path_fraction": float(low_rain_event.mean() * 100),
    }


# --------------------------------------------------
# MAP AND SELECTION
# --------------------------------------------------

st.subheader("1. Select a region")

lat, lon = st.session_state.selected_point

show_worldcover = st.checkbox(
    "Show ESA WorldCover overlay",
    value=False
)

m = folium.Map(
    location=[lat, lon],
    zoom_start=8,
    tiles="OpenStreetMap",
    control_scale=True
)

if show_worldcover:
    folium.WmsTileLayer(
        url="https://services.terrascope.be/wms/v2",
        layers="WORLDCOVER_2021_MAP",
        name="ESA WorldCover 2021",
        fmt="image/png",
        transparent=True,
        overlay=True,
        opacity=0.65,
        attr="ESA WorldCover 2021"
    ).add_to(m)

folium.Marker(
    [lat, lon],
    tooltip="Selected point",
    icon=folium.Icon(color="green", icon="leaf", prefix="fa")
).add_to(m)

Draw(
    export=False,
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True
    }
).add_to(m)

folium.LayerControl().add_to(m)

map_data = st_folium(
    m,
    height=480,
    width=None,
    key="agririsk_map",
    returned_objects=["last_clicked", "all_drawings"]
)

clicked = map_data.get("last_clicked")
if clicked:
    new_lat = float(clicked["lat"])
    new_lon = normalize_longitude(clicked["lng"])
    if location_key(new_lat, new_lon) != location_key(lat, lon):
        st.session_state.selected_point = (new_lat, new_lon)
        st.session_state.weather_data = None
        st.session_state.weather_location = None
        st.session_state.cropland_result = None
        st.session_state.cropland_location = None
        st.rerun()

drawings = map_data.get("all_drawings") or []
polygon_feature = None

for feature in reversed(drawings):
    geometry = feature.get("geometry") or {}
    if geometry.get("type") in ("Polygon", "MultiPolygon"):
        polygon_feature = feature
        break

if polygon_feature:
    geometry = polygon_feature["geometry"]
    signature = repr(geometry)
    if signature != st.session_state.polygon_signature:
        st.session_state.selected_polygon = geometry
        st.session_state.polygon_result = None
        st.session_state.polygon_signature = signature

lat, lon = st.session_state.selected_point

st.write(f"**Selected point:** {lat:.5f}, {lon:.5f}")

if st.session_state.selected_polygon:
    st.success("Polygon detected. You can calculate its cropland percentage.")
else:
    st.info("Draw a polygon or rectangle using the map's drawing toolbar.")


# --------------------------------------------------
# POLYGON CROPLAND ANALYSIS
# --------------------------------------------------

st.subheader("2. Calculate cropland percentage")

if st.button("🌱 Calculate polygon cropland percentage", type="primary"):
    if not st.session_state.selected_polygon:
        st.warning("Draw a polygon or rectangle on the map first.")
    else:
        try:
            with st.spinner(
                "Reading ESA WorldCover raster pixels inside the polygon..."
            ):
                result = calculate_polygon_cropland(
                    st.session_state.selected_polygon
                )
                st.session_state.polygon_result = result
                st.session_state.polygon_signature = repr(
                    st.session_state.selected_polygon
                )
        except Exception as e:
            st.error(f"Cropland calculation failed: {e}")

poly_result = st.session_state.polygon_result

if poly_result:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Cropland percentage",
        f"{poly_result['cropland_percentage']:.2f}%"
    )
    c2.metric(
        "Cropland pixels",
        f"{poly_result['cropland_pixels']:,}"
    )
    c3.metric(
        "Valid pixels",
        f"{poly_result['valid_pixels']:,}"
    )

    st.progress(
        min(1.0, max(0.0, poly_result["cropland_percentage"] / 100))
    )

    st.write(
        f"Non-cropland share: "
        f"{poly_result['noncropland_percentage']:.2f}%"
    )
    st.caption(
        "Percentage is based on valid WorldCover pixels inside the polygon. "
        "The dataset represents 2021 land cover at nominal 10 m resolution; "
        "it does not confirm active cultivation in the current year."
    )
    with st.expander("Raster processing details"):
        st.write("Tiles read:", ", ".join(poly_result["tiles"]))
        st.write("Pixels inside polygon:", f"{poly_result['inside_pixels']:,}")


# --------------------------------------------------
# WEATHER PERIOD AND REGIONAL WEATHER
# --------------------------------------------------

st.subheader("3. Regional weather analysis")

date_col1, date_col2 = st.columns(2)
with date_col1:
    start_date = st.date_input(
        "Start date",
        value=date(2023, 1, 1),
        key="start_date"
    )
with date_col2:
    end_date = st.date_input(
        "End date",
        value=date(2025, 12, 31),
        key="end_date"
    )

weather_point = st.session_state.selected_point
weather_label = "Selected point"

if st.session_state.selected_polygon:
    try:
        weather_point = approximate_polygon_center(
            st.session_state.selected_polygon
        )
        weather_label = "Approximate polygon center"
    except Exception:
        pass

wlat, wlon = weather_point
st.write(
    f"**Weather sampling location ({weather_label}):** "
    f"{wlat:.5f}, {wlon:.5f}"
)
st.caption(
    "NASA POWER provides point/grid weather data. This is a center-point "
    "proxy for the polygon, not a spatial average over the whole region."
)

if start_date > end_date:
    st.error("Start date must be before end date.")
else:
    if st.button("Fetch regional NASA POWER weather"):
        try:
            with st.spinner("Fetching weather history..."):
                df = get_weather(wlat, wlon, start_date, end_date)
                st.session_state.weather_data = df
                st.session_state.weather_location = (
                    location_key(wlat, wlon), start_date, end_date
                )
        except Exception as e:
            st.error(f"Weather request failed: {e}")

expected_weather_key = (
    location_key(wlat, wlon), start_date, end_date
)
df = st.session_state.weather_data
weather_valid = (
    df is not None
    and st.session_state.weather_location == expected_weather_key
)

if weather_valid:
    total_rain = df["Rainfall (mm/day)"].sum()
    mean_temp = df["Temperature (°C)"].mean()
    rainy_days = int((df["Rainfall (mm/day)"] >= 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total rainfall", f"{total_rain:.1f} mm")
    c2.metric("Mean temperature", f"{mean_temp:.1f} °C")
    c3.metric("Days with ≥1 mm rain", rainy_days)

    st.markdown("#### Daily rainfall")
    st.line_chart(df.set_index("Date")[["Rainfall (mm/day)"]])

    st.markdown("#### Daily temperature")
    st.line_chart(df.set_index("Date")[["Temperature (°C)"]])

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download weather CSV",
        data=df.to_csv(index=False),
        file_name="agririsk_regional_weather.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# REGIONAL WEATHER RISK
# --------------------------------------------------

st.subheader("4. Regional weather-risk probability model")
st.write(
    "This version learns three weather states and their transition "
    "probabilities from the selected historical weather series. The resulting "
    "probability estimates an adverse-weather scenario, not crop failure."
)

if weather_valid:
    risk = estimate_weather_risk(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Relative weather-stress index", f"{risk['score']:.1f} / 100")
    c2.metric("Historically adverse days", f"{risk['state_fractions'][2] * 100:.1f}%")
    c3.metric("High-temperature days", f"{risk['hot_fraction'] * 100:.1f}%")
    st.write(f"**Screening category:** {risk['category']}")

    st.markdown("#### Data-derived weather states")
    state_table = pd.DataFrame({
        "State": ["Favorable", "Watch", "Adverse"],
        "Observed share (%)": risk["state_fractions"] * 100,
        "Definition": [
            "Rainfall at/above period median and temperature below 70th percentile",
            "Neither favorable nor adverse under the selected-period thresholds",
            "Rainfall at/below 25th percentile OR temperature at/above 85th percentile",
        ]
    })
    st.dataframe(state_table, use_container_width=True, hide_index=True)

    st.caption(
        f"Thresholds fitted from this selected period: rainfall 25th percentile "
        f"{risk['thresholds']['rain_q25']:.2f} mm/day; temperature 70th percentile "
        f"{risk['thresholds']['temp_q70']:.1f} °C; temperature 85th percentile "
        f"{risk['thresholds']['temp_q85']:.1f} °C. These are relative local-history "
        "thresholds, not crop-specific agronomic limits."
    )

    st.markdown("#### Fitted Markov transition probabilities")
    matrix_df = pd.DataFrame(
        risk["transition_matrix"] * 100,
        index=["Current: Favorable", "Current: Watch", "Current: Adverse"],
        columns=["Next: Favorable (%)", "Next: Watch (%)", "Next: Adverse (%)"]
    )
    st.dataframe(matrix_df.round(1), use_container_width=True)
    st.caption(
        "Rows are estimated from consecutive observed daily states with a "
        "Dirichlet(1,1,1) prior. The simulation draws a new transition matrix "
        "from the corresponding Dirichlet posterior for each Monte Carlo path, "
        "representing uncertainty in the transition probabilities."
    )

    st.markdown("#### Monthly regional weather")
    st.dataframe(risk["monthly"], use_container_width=True, hide_index=True)
else:
    st.info("Fetch regional weather data first to fit the probability model.")


# --------------------------------------------------
# STOCHASTIC MONTE CARLO WEATHER SCENARIOS
# --------------------------------------------------

st.subheader("5. Monte Carlo adverse-weather probability")
st.write(
    "The simulator samples a transition matrix from its Dirichlet posterior for "
    "each path, generates state sequences, and resamples observed rainfall and "
    "temperature from nearby calendar months. It estimates how often simulated "
    "periods cross a provisional adverse-weather rule."
)

if weather_valid:
    risk = estimate_weather_risk(df)
    last_state = int(risk["classified"]["Weather state"].iloc[-1])
    state_names = {0: "Favorable", 1: "Watch", 2: "Adverse"}

    mc1, mc2 = st.columns(2)
    with mc1:
        number_simulations = st.slider(
            "Monte Carlo paths", min_value=500, max_value=10000,
            value=3000, step=500
        )
    with mc2:
        probability_horizon = st.selectbox(
            "Forecast horizon", [30, 60, 90], index=2,
            format_func=lambda x: f"{x} days"
        )

    st.write(f"Starting state: **{state_names[last_state]}**")
    st.write(
        "Provisional event rule: at least 30% adverse days OR a run of "
        "5 consecutive adverse days. You can adjust these assumptions below."
    )
    rule1, rule2 = st.columns(2)
    with rule1:
        adverse_fraction_threshold = st.slider(
            "Adverse-day threshold (%)", 10, 60, 30, 5
        ) / 100.0
    with rule2:
        adverse_streak_threshold = st.slider(
            "Consecutive adverse days", 3, 10, 5
        )

    if st.button("Estimate adverse-weather probability", type="primary"):
        result = simulate_weather_risk(
            risk["transition_alpha"],
            risk["classified"],
            last_state,
            int(probability_horizon),
            int(number_simulations),
            adverse_fraction_threshold=adverse_fraction_threshold,
            adverse_streak_threshold=adverse_streak_threshold,
        )

        a, b, c = st.columns(3)
        a.metric(
            "Adverse-weather scenario frequency",
            f"{result['event_probability']:.1f}%",
            help="Monte Carlo frequency under the fitted weather-state model; not crop-failure probability."
        )
        b.metric(
            "Mean adverse days",
            f"{result['adverse_days_mean']:.1f} / {probability_horizon}"
        )
        c.metric(
            "90th percentile adverse days",
            f"{result['adverse_days_p90']:.0f} days"
        )

        st.caption(
            f"95% Wilson interval for event frequency: {result['ci_lower']:.1f}%–"
            f"{result['ci_upper']:.1f}%. The simulation also samples transition "
            "matrices from their Dirichlet posteriors and resamples rainfall/temperature "
            "from historical observations in nearby calendar months. This interval is "
            "not a guarantee of forecast calibration."
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Mean simulated rainfall", f"{result['rainfall_total_mean']:.1f} mm")
        d2.metric("Mean simulated temperature", f"{result['temperature_mean']:.1f} °C")
        d3.metric("90th percentile adverse streak", f"{result['longest_adverse_run_p90']:.0f} days")

        d4, d5, d6 = st.columns(3)
        d4.metric("Mean very-dry days", f"{result['dry_days_mean']:.1f}")
        d5.metric("Mean very-hot days", f"{result['hot_days_mean']:.1f}")
        d6.metric("Low-total-rainfall paths", f"{result['low_rain_path_fraction']:.1f}%")

        fig, ax = plt.subplots()
        ax.hist(result["adverse_days"], bins=25)
        ax.set_xlabel(f"Adverse days in {probability_horizon}-day simulation")
        ax.set_ylabel("Number of simulated paths")
        ax.set_title("Monte Carlo distribution of adverse-weather days")
        st.pyplot(fig)
        plt.close(fig)

        summary = pd.DataFrame({
            "Metric": [
                "Forecast horizon (days)",
                "Monte Carlo paths",
                "Paths crossing adverse-weather rule",
                "Estimated adverse-weather scenario frequency (%)",
                "Mean adverse days",
                "90th percentile adverse days",
                "Adverse-day threshold (%)",
                "Consecutive adverse-day threshold",
                "95% Wilson interval lower (%)",
                "95% Wilson interval upper (%)",
                "Mean simulated rainfall total (mm)",
                "Mean simulated temperature (°C)",
                "Mean very-dry days",
                "Mean very-hot days",
            ],
            "Value": [
                probability_horizon,
                result["n_simulations"],
                result["event_count"],
                round(result["event_probability"], 2),
                round(result["adverse_days_mean"], 2),
                round(result["adverse_days_p90"], 2),
                round(adverse_fraction_threshold * 100, 1),
                adverse_streak_threshold,
                round(result["ci_lower"], 2),
                round(result["ci_upper"], 2),
                round(result["rainfall_total_mean"], 2),
                round(result["temperature_mean"], 2),
                round(result["dry_days_mean"], 2),
                round(result["hot_days_mean"], 2),
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download probability simulation summary",
            data=summary.to_csv(index=False),
            file_name="agririsk_probability_summary.csv",
            mime="text/csv"
        )

    st.warning(
        "Interpret this as the model-estimated frequency of crossing a chosen "
        "adverse-weather rule under a Bayesian Markov/empirical-resampling model. "
        "It is NOT the probability of crop failure. Real crop-failure probability "
        "requires crop type, planting date/growth stage, soil and irrigation data, "
        "historical yield/outcome labels, and out-of-sample validation."
    )
else:
    st.info("Fetch weather data first to enable the fitted Markov and Monte Carlo model.")

st.divider()
st.caption(
    "Data sources: ESA WorldCover 2021 v200 and NASA POWER. "
    "AgriRisk Global is a research prototype."
)
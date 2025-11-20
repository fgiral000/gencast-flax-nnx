
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cartopy.io.shapereader import natural_earth, Reader

# ==============================================================================
# 1. CONFIGURATION & DATA LOADING
# ==============================================================================
FILE_PATH = "inference_plots_30steps/rollout_30steps.nc"
FPS = 3
FRAME_DURATION = int(1000 / FPS)  # ms

print(f"Loading {FILE_PATH}...")
ds = xr.open_dataset(FILE_PATH)

# Extract variables (squeeze batch if exists)
# We perform basic unit conversions for better readability
data_map = {
    "temp": {
        "data": ds["2m_temperature"].isel(batch=0) - 273.15,  # K -> C
        "title": "2m Temperature (°C)",
        "cmap": "Turbo",
        "range": [-30, 45]
    },
    "msl": {
        "data": ds["mean_sea_level_pressure"].isel(batch=0) / 100.0, # Pa -> hPa
        "title": "Mean Sea Level Pressure (hPa)",
        "cmap": "Viridis",
        "range": [980, 1030] # Adjust based on your data min/max
    },
    "u10": {
        "data": ds["10m_u_component_of_wind"].isel(batch=0),
        "title": "10m U-Wind (m/s)",
        "cmap": "RdBu_r", # Diverging: Blue(-), White(0), Red(+)
        "range": [-20, 20]
    },
    "v10": {
        "data": ds["10m_v_component_of_wind"].isel(batch=0),
        "title": "10m V-Wind (m/s)",
        "cmap": "RdBu_r",
        "range": [-20, 20]
    }
}

# ==============================================================================
# 2. PRE-PROCESSING (Cyclic Wrapping & 3D Projection)
# ==============================================================================
print("Pre-processing coordinates...")

# Get Lat/Lon from one of the arrays
sample_da = data_map["temp"]["data"]
lat = sample_da.lat.values
lon = sample_da.lon.values

# Fix Seam: Create a new longitude array that wraps 360 -> 0
# We will pad the DATA arrays later
lon_cyclic = np.append(lon, lon[0] + 360) 
n_lat, n_lon_cyclic = len(lat), len(lon_cyclic)

# Generate 3D Sphere Mesh
Lon, Lat = np.meshgrid(np.deg2rad(lon_cyclic), np.deg2rad(lat))
R = 1.0
X = R * np.cos(Lat) * np.cos(Lon)
Y = R * np.cos(Lat) * np.sin(Lon)
Z = R * np.sin(Lat)

# Prepare Data Arrays (Pad the longitude dimension for all vars)
processed_values = {}
for key, info in data_map.items():
    # pad width: ((time, 0), (lat, 0), (lon, 1)) -> adds 1 column to end of lon
    # Assuming shape is (time, lat, lon)
    vals = info["data"].values
    # Check shape to ensure we pad correctly. 
    # If shape is (30, 73, 144), we pad axis 2.
    vals_padded = np.pad(vals, ((0,0), (0,0), (0,1)), mode='wrap')
    
    # Fix the wrap manually if 'wrap' mode didn't grab index 0 exactly how we want
    # (np.pad wrap usually works, but explicit assignment ensures 360=0)
    vals_padded[:, :, -1] = vals_padded[:, :, 0]
    
    processed_values[key] = vals_padded

num_steps = processed_values["temp"].shape[0]

# ==============================================================================
# 3. COASTLINES GENERATOR
# ==============================================================================
def get_coastlines_3d():
    print("Generating high-quality 3D coastlines...")
    cl_x, cl_y, cl_z = [], [], []
    # Use 110m for speed, or '50m' for better quality
    shp = natural_earth(category='physical', name='coastline', resolution='110m')
    for geom in Reader(shp).geometries():
        if geom.geom_type == 'MultiLineString': geoms = geom.geoms
        else: geoms = [geom]
        for line in geoms:
            lons, lats = np.array(line.coords).T
            r_c = 1.015 # Float slightly above surface
            xc = r_c * np.cos(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))
            yc = r_c * np.cos(np.deg2rad(lats)) * np.sin(np.deg2rad(lons))
            zc = r_c * np.sin(np.deg2rad(lats))
            cl_x.extend(xc.tolist() + [None])
            cl_y.extend(yc.tolist() + [None])
            cl_z.extend(zc.tolist() + [None])
    return cl_x, cl_y, cl_z

cx, cy, cz = get_coastlines_3d()
coastline_trace = go.Scatter3d(
    x=cx, y=cy, z=cz, mode="lines",
    line=dict(color="black", width=2), hoverinfo="skip", showlegend=False
)

# ==============================================================================
# 4. BUILD FIGURE & FRAMES
# ==============================================================================
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=[data_map[k]["title"] for k in ["temp", "msl", "u10", "v10"]],
    horizontal_spacing=0.05, vertical_spacing=0.08
)

frames = []
print(f"Building {num_steps} frames (approx {FPS} FPS)...")

# Lighting settings for the "Earthkit" glow
lighting = dict(ambient=0.65, diffuse=0.5, specular=0.2, roughness=0.5)

for t in range(num_steps):
    
    # Camera Position (Rotating)
    angle = np.deg2rad((t / num_steps) * 360)
    eye = dict(x=1.8*np.sin(angle), y=1.8*np.cos(angle), z=0.6)
    
    frame_data = []
    
    # Order: Top-Left, Top-Right, Bot-Left, Bot-Right
    keys = ["temp", "msl", "u10", "v10"]
    
    for key in keys:
        # Add Surface
        frame_data.append(
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=processed_values[key][t],
                colorscale=data_map[key]["cmap"],
                cmin=data_map[key]["range"][0],
                cmax=data_map[key]["range"][1],
                lighting=lighting,
                showscale=False  # Hide colorbars to save space
            )
        )
        # Add Coastlines (Must be added to every subplot in every frame)
        frame_data.append(coastline_trace)

    frames.append(go.Frame(
        data=frame_data,
        name=str(t),
        layout=dict(
            scene =dict(camera=dict(eye=eye)),
            scene2=dict(camera=dict(eye=eye)),
            scene3=dict(camera=dict(eye=eye)),
            scene4=dict(camera=dict(eye=eye))
        )
    ))

# ==============================================================================
# 5. FINAL LAYOUT & EXPORT
# ==============================================================================
# Add initial data (Frame 0)
initial_traces = frames[0].data
# Map traces to subplots: 2 traces per subplot (Surface + Coastline)
# Indices: 0,1 -> (1,1); 2,3 -> (1,2); 4,5 -> (2,1); 6,7 -> (2,2)
subplot_indices = [(1,1), (1,2), (2,1), (2,2)]

for i, (row, col) in enumerate(subplot_indices):
    fig.add_trace(initial_traces[i*2], row=row, col=col)   # Surface
    fig.add_trace(initial_traces[i*2+1], row=row, col=col) # Coastline

fig.update_layout(
    title="GenCast Weather Rollout (30 Steps)",
    height=1000, width=1300,
    paper_bgcolor="#050505", # Dark background
    font=dict(color="white"),
    
    # Clean scenes
    scene =dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
    scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
    scene3=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
    scene4=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),

    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=0.98,
        buttons=[dict(
            label="▶ Play Animation",
            method="animate",
            args=[None, dict(frame=dict(duration=FRAME_DURATION, redraw=True), 
                             fromcurrent=True, mode='immediate')]
        )]
    )]
)

fig.frames = frames

output_file = "gencast_dashboard_3fps.html"
print(f"Saving to {output_file}...")
fig.write_html(output_file)
print("Done!")




# # training/3d_visualization_v2.py
# import xarray as xr
# import numpy as np
# import plotly.graph_objects as go
# import cartopy.feature as cfeature
# from cartopy.io.shapereader import natural_earth, Reader

# # ------------------------------------------------------------------
# # 1. Load & Preprocess Data
# # ------------------------------------------------------------------
# ds = xr.open_dataset("inference_plots_30steps/rollout_30steps.nc")
# da = ds["2m_temperature"].squeeze(drop=True) - 273.15  # K -> C

# # --- FIX 1: THE SEAM PROBLEM ---
# # We must wrap the data around the globe (connect 360 back to 0)
# # Concatenate the first longitude column to the end of the array
# da_cyclic = da.pad(lon=(0, 1), mode='wrap') 
# # Ensure the last lon value is actually correct (e.g., 360.0)
# # If your lon is 0..359, the new last one should be 360.
# new_lon = da_cyclic.lon.values
# new_lon[-1] = new_lon[0] + 360 if new_lon[0] >= 0 else new_lon[0] + 360

# values = da_cyclic.values  # Shape: (time, lat, lon_padded)
# lat = da.lat.values
# lon = new_lon

# # ------------------------------------------------------------------
# # 2. Coordinate Transform (Lat/Lon -> 3D Cartesian)
# # ------------------------------------------------------------------
# def sph2cart(lat_deg, lon_deg, radius=1.0):
#     """Convert lat/lon degrees to 3D cartesian coordinates."""
#     lon_rad = np.deg2rad(lon_deg)
#     lat_rad = np.deg2rad(lat_deg)
#     x = radius * np.cos(lat_rad) * np.cos(lon_rad)
#     y = radius * np.cos(lat_rad) * np.sin(lon_rad)
#     z = radius * np.sin(lat_rad)
#     return x, y, z

# # Create mesh for the Surface
# Lon, Lat = np.meshgrid(lon, lat)
# X_globe, Y_globe, Z_globe = sph2cart(Lat, Lon, radius=1.0)

# # ------------------------------------------------------------------
# # 3. Generate Coastlines (The "Earthkit" Look)
# # ------------------------------------------------------------------
# # We extract coastline segments from Natural Earth and project them to 3D
# print("Generating 3D coastlines...")
# coastlines_x, coastlines_y, coastlines_z = [], [], []

# # Get 110m (low res) or 50m (med res) coastlines
# shp_path = natural_earth(category='physical', name='coastline', resolution='110m')
# reader = Reader(shp_path)

# for geometry in reader.geometries():
#     if geometry.geom_type == 'MultiLineString':
#         geoms = geometry.geoms
#     else:
#         geoms = [geometry]
        
#     for line in geoms:
#         # Extract xy
#         lons, lats = np.array(line.coords).T
        
#         # Convert to 3D
#         # IMPORTANT: Slightly larger radius (1.01) so lines float ABOVE the surface
#         xc, yc, zc = sph2cart(lats, lons, radius=1.01) 
        
#         coastlines_x.extend(xc.tolist() + [None]) # 'None' breaks the line in Plotly
#         coastlines_y.extend(yc.tolist() + [None])
#         coastlines_z.extend(zc.tolist() + [None])

# # Create the immutable coastline trace
# coastline_trace = go.Scatter3d(
#     x=coastlines_x, y=coastlines_y, z=coastlines_z,
#     mode="lines",
#     line=dict(color="black", width=2), # Earthkit uses dark borders
#     hoverinfo="skip",
#     name="Coastlines"
# )

# # ------------------------------------------------------------------
# # 4. Build Frames
# # ------------------------------------------------------------------
# frames = []
# print(f"Building {values.shape[0]} frames...")

# for t in range(values.shape[0]):
#     # --- FIX 2: LIGHTING & COLORS ---
#     # Earthkit uses a "glowing" look. We achieve this by high ambient light
#     # and the "Turbo" colormap (standard for weather now).
#     surf = go.Surface(
#         x=X_globe, y=Y_globe, z=Z_globe,
#         surfacecolor=values[t],
#         colorscale="Turbo", # Vibrant, distinct colors (replaces RdBu)
#         cmin=-30, cmax=40,  # Tighten range to make colors pop
#         colorbar=dict(thickness=20, len=0.5, x=0.9, title="Temp (°C)"),
        
#         # Lighting: High ambient = "Self-illuminated" look
#         lighting=dict(
#             ambient=0.7,    # High ambient makes it look like it's glowing
#             diffuse=0.5, 
#             specular=0.1,   # Low specular reduces plastic reflection
#             roughness=0.5,
#             fresnel=0.2
#         ),
#         name=f"Frame {t}"
#     )
    
#     # Each frame needs the surface AND the coastlines
#     # (Note: In Plotly frames, it's often better to keep static elements in 'data' 
#     # and only update dynamic ones, but for 3D rotation sometimes explicit is safer)
#     frames.append(go.Frame(data=[surf, coastline_trace], name=str(t)))

# # ------------------------------------------------------------------
# # 5. Layout & Animation
# # ------------------------------------------------------------------
# # Initial data (Frame 0)
# fig = go.Figure(data=[frames[0].data[0], coastline_trace], frames=frames)

# fig.update_layout(
#     title="Earthkit-Style Reproduction",
#     scene=dict(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         zaxis=dict(visible=False),
#         aspectmode="data",
#         camera=dict(
#             eye=dict(x=1.6, y=1.6, z=0.8), # Angled slightly down
#             projection=dict(type="orthographic") # Makes it look like a telescope view
#         ),
#         dragmode="orbit"
#     ),
#     paper_bgcolor="#050505", # Deep dark background
#     margin=dict(l=0, r=0, t=50, b=0),
    
#     # Animation settings
#     updatemenus=[dict(
#         type="buttons",
#         showactive=False,
#         x=0.1, y=0.1,
#         buttons=[dict(label="Play",
#                       method="animate",
#                       args=[None, dict(frame=dict(duration=50, redraw=True), 
#                                        fromcurrent=True)])]
#     )]
# )

# # ------------------------------------------------------------------
# # 6. Rotation Logic (Optional: Modify camera per frame)
# # ------------------------------------------------------------------
# # If you want the camera to rotate WHILE the animation plays, you calculate
# # the eye position for each frame and inject it into the frames list.
# n = len(frames)
# for i in range(n):
#     angle = np.deg2rad(i / n * 360)
#     # Rotate camera around Z
#     cam_x = 1.8 * np.sin(angle)
#     cam_y = 1.8 * np.cos(angle)
    
#     # Update the layout part of the frame
#     frames[i].layout = dict(
#         scene=dict(camera=dict(eye=dict(x=cam_x, y=cam_y, z=0.8)))
#     )

# # ------------------------------------------------------------------
# # 7. Save
# # ------------------------------------------------------------------
# fig.write_html("earthkit_style_globe.html")
# print("Saved interactive HTML.")

# # Only uncomment if kaleido is installed
# # fig.write_image("earthkit_style.gif", width=800, height=800)
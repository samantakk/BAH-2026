import ee
import geemap
import os

# Initialize GEE
ee.Initialize(project='firstproject-499806')

# Define area of interest — small patch over Assam, NER India
aoi = ee.Geometry.Rectangle([91.5, 26.0, 91.7, 26.2])

# Get Sentinel-2 collection
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(aoi) \
    .filterDate('2023-01-01', '2023-06-30') \
    .select(['B2', 'B3', 'B4', 'B8'])

# Image 1: cleanest available image (ground truth reference)
clean_image = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

# Image 2: real cloudy image (for visual demo in pitch)
cloudy_real = s2.filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

os.makedirs("D:/cloud-removal/data/bands", exist_ok=True)

print("Downloading clean reference image...")
geemap.ee_export_image(
    clean_image,
    filename="D:/cloud-removal/data/bands/clean_reference.tif",
    scale=30,
    region=aoi,
    file_per_band=False
)

print("Downloading real cloudy image...")
geemap.ee_export_image(
    cloudy_real,
    filename="D:/cloud-removal/data/bands/real_cloudy.tif",
    scale=30,
    region=aoi,
    file_per_band=False
)

print("Done! Check D:/cloud-removal/data/bands/")
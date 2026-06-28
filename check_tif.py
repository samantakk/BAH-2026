import rasterio

with rasterio.open('data/bands/clean_reference.tif') as src:
    print('Bands:', src.count)
    print('Size:', src.width, 'x', src.height)
    print('CRS:', src.crs)
    data = src.read()
    print('Data shape:', data.shape)
    print('Min/Max values:', data.min(), data.max())
# general imports
import os
import numpy as np
from tqdm import tqdm
import warnings

# image imports
from rasterio import open as ropen
from rasterio.mask import mask
from skimage import exposure

# geometry imports
from geopandas import GeoDataFrame as gdf
from shapely.geometry import Polygon

tqdm.pandas()
warnings.filterwarnings("ignore")

class RasterTiler:
    def __init__(self, raster_filepath, tile_outdir) -> None:

        self.raster_filepath = self.check_raster_filepath(raster_filepath=raster_filepath)
        self.raster_root, self.raster_filename = os.path.split(self.raster_filepath)
        self.raster_name, self.raster_ext = os.path.splitext(self.raster_filename)

        self.tile_outdir = self.check_tile_outdir(tile_outdir=tile_outdir)
        self.tile_metadata_name = self.raster_name + '_tile_metadata.geojson'
        self.tile_metadata_outpath = os.path.join(self.tile_outdir, self.tile_metadata_name)

    def tile_raster(
        self, 
        tile_size: int = 640,
        overlap_ratio: float = 0.0,
        save_tiles: bool = False, 
        save_edges: bool = False
        ):
        """ 
        Read the raster and calculate tiles, always save metadata
        optionally: save tiles to disk, save edges (these will create overlapping 
        tiles on the edges but will ensure the entire raster is covered by a tile)
        """
        with ropen(self.raster_filepath, 'r') as src:
            tiles, is_edge = self.calculate_tile_bboxes(
                raster_height=src.height,
                raster_width=src.width,
                tile_size=tile_size,
                overlap_ratio=overlap_ratio,
                include_edge=save_edges
            )
            geo_tiles_dataframe = gdf(
                {
                    'raster_source': [os.path.abspath(self.raster_filepath)] * len(tiles),
                    'tile_name': ['_'.join([str(t) for t in tile]) for tile in tiles],
                    'tile_ext': [self.raster_ext] * len(tiles),
                    'is_edge': is_edge,
                    'pixel_xmin': [t[0] for t in tiles],
                    'pixel_ymin': [t[1] for t in tiles],
                    'pixel_xmax': [t[2] for t in tiles],
                    'pixel_ymax': [t[3] for t in tiles],
                    'geometry': [self.geo_tile(src, tile) for tile in tiles]
                },
                crs=src.crs.to_string()
            )
            geo_tiles_dataframe.to_file(self.tile_metadata_outpath, driver='GeoJSON')
            if not save_tiles:
                return geo_tiles_dataframe
            else:
                geo_tiles_dataframe.progress_apply(
                    self.process_tile,
                    args=[self.tile_outdir, src],
                    axis=1
                )
                return geo_tiles_dataframe
                
    @staticmethod
    def check_raster_filepath(raster_filepath):
        """ 
        Make sure the filepath to the raster exists.
        """
        assert os.path.exists(raster_filepath), f'{raster_filepath} does not exist!'
        return raster_filepath

    @staticmethod
    def check_tile_outdir(tile_outdir):
        """ 
        If the tile output directory does not exist, create it.
        """
        if not os.path.exists(tile_outdir):
            os.makedirs(tile_outdir)
        return tile_outdir
    
    @staticmethod
    def process_tile(row, tile_outdir, src):
        """
        Extract the tile from the full raster, preprocess to 'uint8', then save to file.
        """
        def preprocess_tile(tile):
            """
            Rescale the intensity of an image. Mask out where pixels equal zero to avoid rescale issues.
            """
            preprocessed_tile_placeholder = np.zeros_like(tile)
            masked_tile = tile[tile != 0]
            p1, p2 = np.percentile(masked_tile, (0.5, 99.5))
            preprocessed_tile = exposure.rescale_intensity(
                image=masked_tile, 
                in_range=(p1, p2), 
                out_range='uint8'
            )
            preprocessed_tile_placeholder[tile != 0] = preprocessed_tile
            return preprocessed_tile_placeholder
        
        tile_filename = row.raster_name + '_' + row.tile_name + row.tile_ext
        tile_outpath = os.path.join(tile_outdir, tile_filename)
        geometry = row.geometry
        tile, transform = mask(src, [geometry], crop=True, filled=False)
        data = tile.data
        data = data[-1:] # HARDCODE: UNNECESSARY UNLESS YOU HAVE MULTIDIMENSIONAL ARRAYS
        preprocessed_data = preprocess_tile(data)
        profile = src.profile
        profile.update(transform=transform, count=1, width=preprocessed_data.shape[2], height=preprocessed_data.shape[1])
        with ropen(tile_outpath, "w", **profile) as dst:
            dst.write(preprocessed_data)

    @staticmethod
    def calculate_tile_bboxes(
        raster_height: int,
        raster_width: int,
        tile_size: int = 640,
        overlap_ratio: float = 0.0,
        include_edge: bool = False,
        ):
        """
        Calculate a list of [xmin, ymin, xmax, ymax] for all tiles that cover an image.
        """
        tile_bboxes = []
        is_edge = []
        y_max = y_min = 0
        y_overlap = int(overlap_ratio * tile_size)
        x_overlap = int(overlap_ratio * tile_size)
        while y_max < raster_height:
            x_min = x_max = 0
            y_max = y_min + tile_size
            while x_max < raster_width:
                x_max = x_min + tile_size
                if y_max > raster_height or x_max > raster_width:
                    if include_edge:
                        xmax = min(raster_width, x_max)
                        ymax = min(raster_height, y_max)
                        xmin = max(0, xmax - tile_size)
                        ymin = max(0, ymax - tile_size)
                        tile_bboxes.append([xmin, ymin, xmax, ymax])
                        is_edge.append(True)
                else:
                    tile_bboxes.append([x_min, y_min, x_max, y_max])
                    is_edge.append(False)
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return tile_bboxes, is_edge

    @staticmethod
    def geo_tile(src, tile):
        """
        Precise conversion of pixel coordinates to geographic coordinates for each tile.
        """
        xmin, ymin = src.xy(tile[1],tile[0], 'center')
        xmax, ymax = src.xy(tile[3]-1,tile[2]-1, 'center')
        return Polygon.from_bounds(xmin, ymax, xmax, ymin)

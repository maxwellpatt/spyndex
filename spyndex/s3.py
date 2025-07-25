"""AWS S3 integration for spyndex spectral analysis

This module provides functionality to seamlessly integrate AWS S3 data storage
with spyndex spectral index computation capabilities, including parallel processing
and efficient data handling for large-scale remote sensing workflows.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import boto3
    import numpy as np
    import rasterio
    from botocore.exceptions import ClientError, NoCredentialsError

    S3_AVAILABLE = True
except ImportError as e:
    S3_AVAILABLE = False
    _IMPORT_ERROR = e

import dask.array as da
import pandas as pd
import xarray as xr
from dask import delayed

from . import computeIndex, computeKernel

# Set up logging
logger = logging.getLogger(__name__)


def _check_s3_dependencies():
    """Check if S3 dependencies are available"""
    if not S3_AVAILABLE:
        raise ImportError(
            f"S3 functionality requires additional dependencies. "
            f"Install with: pip install spyndex[s3]. "
            f"Missing dependency error: {_IMPORT_ERROR}"
        )


class S3Config:
    """Configuration management for S3 connections"""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.endpoint_url = endpoint_url
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

    def create_client(self):
        """Create boto3 S3 client with current configuration"""
        _check_s3_dependencies()

        try:
            # Use profile-based session if profile is specified
            if self.profile_name:
                session = boto3.Session(profile_name=self.profile_name)
                client = session.client("s3", endpoint_url=self.endpoint_url)
            else:
                # Use explicit credentials
                client = boto3.client(
                    "s3",
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url,
                )
            # Test connection
            client.list_buckets()
            return client
        except NoCredentialsError:
            raise ValueError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables or pass them explicitly."
            )
        except ClientError as e:
            raise ConnectionError(f"Failed to connect to S3: {e}")


class S3DataLoader:
    """Handle S3 data loading and caching operations"""

    def __init__(self, config: S3Config, cache_dir: Optional[str] = None):
        self.config = config
        self.client = config.create_client()
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "spyndex_s3_cache"
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        pattern: Optional[str] = None,
        max_keys: Optional[int] = None,
    ) -> List[str]:
        """List S3 objects matching criteria"""
        objects = []
        paginator = self.client.get_paginator("list_objects_v2")

        page_iterator = paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
            PaginationConfig={"MaxItems": max_keys} if max_keys else {},
        )

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if pattern is None or pattern in key:
                        objects.append(key)

        logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
        return objects

    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Download a single file from S3"""
        if local_path is None:
            # Generate cache path
            cache_key = key.replace("/", "_")
            local_path = str(self.cache_dir / cache_key)

        if use_cache and os.path.exists(local_path):
            logger.debug(f"Using cached file: {local_path}")
            return local_path

        try:
            logger.debug(f"Downloading s3://{bucket}/{key} to {local_path}")
            self.client.download_file(bucket, key, local_path)
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise

    def download_files_parallel(
        self, bucket: str, keys: List[str], max_workers: int = 4, use_cache: bool = True
    ) -> Dict[str, str]:
        """Download multiple files in parallel"""
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(
                    self.download_file, bucket, key, use_cache=use_cache
                ): key
                for key in keys
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    local_path = future.result()
                    results[key] = local_path
                    logger.debug(f"Downloaded {key}")
                except Exception as e:
                    logger.error(f"Failed to download {key}: {e}")
                    results[key] = None

        return results

    def load_raster(self, local_path: str) -> Tuple[np.ndarray, Any]:
        """Load raster data using rasterio"""
        _check_s3_dependencies()

        try:
            with rasterio.open(local_path) as src:
                data = src.read()
                profile = src.profile
                return data, profile
        except Exception as e:
            logger.error(f"Failed to load raster {local_path}: {e}")
            raise


class S3SpectralProcessor:
    """Combine S3 data loading with spyndex spectral processing"""

    def __init__(self, config: S3Config, cache_dir: Optional[str] = None):
        self.loader = S3DataLoader(config, cache_dir)

    def process_single_tile(
        self,
        bucket: str,
        key: str,
        band_mapping: Dict[str, int],
        indices: List[str],
        scale_factor: float = 1 / 255.0,
        **index_params,
    ) -> Dict[str, np.ndarray]:
        """Process a single raster tile"""
        local_path = self.loader.download_file(bucket, key)
        data, profile = self.loader.load_raster(local_path)

        # Scale data if needed
        if scale_factor != 1.0:
            data = data.astype(np.float32) * scale_factor

        # Extract bands according to mapping
        bands = {}
        for band_name, band_idx in band_mapping.items():
            if band_idx < data.shape[0]:
                bands[band_name] = data[band_idx]
            else:
                logger.warning(
                    f"Band index {band_idx} not found in data with shape {data.shape}"
                )

        # Combine with any additional parameters
        params = {**bands, **index_params}

        # Compute indices
        results = {}
        for index_name in indices:
            try:
                result = computeIndex(index_name, params)
                results[index_name] = result
                logger.debug(f"Computed {index_name} for tile {key}")
            except Exception as e:
                logger.error(f"Failed to compute {index_name} for tile {key}: {e}")
                results[index_name] = None

        return results

    def process_tiles_parallel(
        self,
        bucket: str,
        keys: List[str],
        band_mapping: Dict[str, int],
        indices: List[str],
        max_workers: int = 4,
        scale_factor: float = 1 / 255.0,
        **index_params,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Process multiple tiles in parallel"""
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(
                    self.process_single_tile,
                    bucket,
                    key,
                    band_mapping,
                    indices,
                    scale_factor,
                    **index_params,
                ): key
                for key in keys
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    tile_results = future.result()
                    results[key] = tile_results
                    logger.info(f"Processed tile {key}")
                except Exception as e:
                    logger.error(f"Failed to process tile {key}: {e}")
                    results[key] = None

        return results

    def create_mosaic(
        self,
        tile_results: Dict[str, Dict[str, np.ndarray]],
        mosaic_method: str = "horizontal",
    ) -> Dict[str, np.ndarray]:
        """Create mosaics from tile results"""
        if not tile_results:
            return {}

        # Get list of indices from first successful tile
        sample_tile = next(
            (results for results in tile_results.values() if results is not None), None
        )
        if sample_tile is None:
            logger.error("No successful tile processing results found")
            return {}

        indices = list(sample_tile.keys())
        mosaics = {}

        for index_name in indices:
            valid_tiles = [
                results[index_name]
                for results in tile_results.values()
                if results is not None and results.get(index_name) is not None
            ]

            if not valid_tiles:
                logger.warning(f"No valid tiles found for index {index_name}")
                continue

            if mosaic_method == "horizontal":
                mosaic = np.concatenate(valid_tiles, axis=1)
            elif mosaic_method == "vertical":
                mosaic = np.concatenate(valid_tiles, axis=0)
            else:
                logger.error(f"Unknown mosaic method: {mosaic_method}")
                continue

            mosaics[index_name] = mosaic
            logger.info(f"Created mosaic for {index_name} with shape {mosaic.shape}")

        return mosaics

    def process_bluesky_rgb_tiles(
        self,
        bucket: str,
        base_prefix: str,
        tile_indices: List[int],
        indices: List[str] = None,
        max_workers: int = 4,
    ) -> Dict[str, np.ndarray]:
        """Process BlueSky RGB tiles specifically"""
        if indices is None:
            indices = ["TGI", "GLI", "VARI", "ExG", "ExR", "MGRVI", "RGBVI"]

        # Generate S3 keys for tiles
        keys = [f"{base_prefix}/rgb_tile_10000_{idx}.tif" for idx in tile_indices]

        # RGB band mapping (typical order: R, G, B)
        band_mapping = {"R": 0, "G": 1, "B": 2}

        # Process tiles
        tile_results = self.process_tiles_parallel(
            bucket=bucket,
            keys=keys,
            band_mapping=band_mapping,
            indices=indices,
            max_workers=max_workers,
            scale_factor=1 / 255.0,
        )

        # Create horizontal mosaic
        mosaics = self.create_mosaic(tile_results, mosaic_method="horizontal")

        return mosaics


# Convenience functions
def create_s3_config(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: str = "us-east-1",
    profile_name: Optional[str] = None,
) -> S3Config:
    """Create S3 configuration"""
    return S3Config(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
        profile_name=profile_name,
    )


def process_s3_spectral_data(
    bucket: str,
    keys: List[str],
    band_mapping: Dict[str, int],
    indices: List[str],
    config: Optional[S3Config] = None,
    max_workers: int = 4,
    **kwargs,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    High-level function to process spectral data from S3

    Args:
        bucket: S3 bucket name
        keys: List of S3 object keys
        band_mapping: Map of band names to array indices (e.g., {"R": 0, "G": 1, "B": 2})
        indices: List of spectral indices to compute
        config: S3 configuration (uses env vars if None)
        max_workers: Number of parallel workers
        **kwargs: Additional parameters for index computation

    Returns:
        Dictionary mapping tile keys to computed indices
    """
    if config is None:
        config = create_s3_config()

    processor = S3SpectralProcessor(config)
    return processor.process_tiles_parallel(
        bucket, keys, band_mapping, indices, max_workers, **kwargs
    )

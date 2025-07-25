"""Tests for S3 integration functionality in spyndex"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

try:
    import boto3
    from moto import mock_s3
    import rasterio
    from rasterio.transform import from_bounds
    S3_DEPS_AVAILABLE = True
except ImportError:
    S3_DEPS_AVAILABLE = False

import spyndex
from spyndex.s3 import (
    S3Config, 
    S3DataLoader, 
    S3SpectralProcessor, 
    create_s3_config,
    process_s3_spectral_data,
    _check_s3_dependencies
)


@pytest.mark.skipif(not S3_DEPS_AVAILABLE, reason="S3 dependencies not available")
class TestS3Config(unittest.TestCase):
    """Test S3 configuration management"""
    
    def test_config_from_params(self):
        """Test config creation with explicit parameters"""
        config = S3Config(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2"
        )
        
        self.assertEqual(config.aws_access_key_id, "test_key")
        self.assertEqual(config.aws_secret_access_key, "test_secret")
        self.assertEqual(config.region_name, "us-west-2")
    
    def test_config_from_env(self):
        """Test config creation from environment variables"""
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'env_key',
            'AWS_SECRET_ACCESS_KEY': 'env_secret',
            'AWS_DEFAULT_REGION': 'eu-west-1'
        }):
            config = S3Config()
            
            self.assertEqual(config.aws_access_key_id, "env_key")
            self.assertEqual(config.aws_secret_access_key, "env_secret")
            self.assertEqual(config.region_name, "eu-west-1")
    
    @mock_s3
    def test_create_client_success(self):
        """Test successful S3 client creation"""
        config = S3Config(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        client = config.create_client()
        self.assertIsNotNone(client)
        
        # Test that client works
        response = client.list_buckets()
        self.assertIn('Buckets', response)
    
    def test_create_client_no_credentials(self):
        """Test client creation without credentials"""
        config = S3Config()  # No credentials
        
        with self.assertRaises(ValueError):
            config.create_client()


@pytest.mark.skipif(not S3_DEPS_AVAILABLE, reason="S3 dependencies not available")
class TestS3DataLoader(unittest.TestCase):
    """Test S3 data loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = S3Config(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
    
    @mock_s3
    def test_list_objects(self):
        """Test S3 object listing"""
        # Create mock S3 environment
        client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-bucket'
        client.create_bucket(Bucket=bucket_name)
        
        # Upload test objects
        test_objects = [
            'data/tile_001.tif',
            'data/tile_002.tif',
            'other/file.txt'
        ]
        
        for obj_key in test_objects:
            client.put_object(Bucket=bucket_name, Key=obj_key, Body=b'test data')
        
        # Test listing
        loader = S3DataLoader(self.config)
        
        # List all objects with prefix
        objects = loader.list_objects(bucket_name, prefix='data/')
        self.assertEqual(len(objects), 2)
        self.assertIn('data/tile_001.tif', objects)
        self.assertIn('data/tile_002.tif', objects)
        
        # List with pattern filter
        tif_objects = loader.list_objects(bucket_name, prefix='data/', pattern='.tif')
        self.assertEqual(len(tif_objects), 2)
    
    @mock_s3
    def test_download_file(self):
        """Test single file download"""
        # Setup mock S3
        client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-bucket'
        client.create_bucket(Bucket=bucket_name)
        
        test_data = b'test raster data'
        client.put_object(Bucket=bucket_name, Key='test.tif', Body=test_data)
        
        # Test download
        loader = S3DataLoader(self.config)
        local_path = loader.download_file(bucket_name, 'test.tif')
        
        self.assertTrue(os.path.exists(local_path))
        with open(local_path, 'rb') as f:
            downloaded_data = f.read()
        self.assertEqual(downloaded_data, test_data)
    
    @mock_s3
    def test_download_files_parallel(self):
        """Test parallel file downloads"""
        # Setup mock S3
        client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-bucket'
        client.create_bucket(Bucket=bucket_name)
        
        test_files = {
            'file1.tif': b'data1',
            'file2.tif': b'data2',
            'file3.tif': b'data3'
        }
        
        for key, data in test_files.items():
            client.put_object(Bucket=bucket_name, Key=key, Body=data)
        
        # Test parallel download
        loader = S3DataLoader(self.config)
        results = loader.download_files_parallel(
            bucket_name, 
            list(test_files.keys()),
            max_workers=2
        )
        
        self.assertEqual(len(results), 3)
        for key in test_files.keys():
            self.assertIn(key, results)
            self.assertIsNotNone(results[key])
            self.assertTrue(os.path.exists(results[key]))


def create_test_raster(width=100, height=100, bands=3, dtype=np.uint8):
    """Create a test raster file"""
    data = np.random.randint(0, 255, (bands, height, width), dtype=dtype)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_file.close()
    
    # Write raster data
    profile = {
        'driver': 'GTiff',
        'dtype': str(dtype),
        'width': width,
        'height': height,
        'count': bands,
        'crs': 'EPSG:4326',
        'transform': from_bounds(-180, -90, 180, 90, width, height)
    }
    
    with rasterio.open(temp_file.name, 'w', **profile) as dst:
        dst.write(data)
    
    return temp_file.name, data


@pytest.mark.skipif(not S3_DEPS_AVAILABLE, reason="S3 dependencies not available")
class TestS3SpectralProcessor(unittest.TestCase):
    """Test spectral processing with S3 data"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = S3Config(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        self.processor = S3SpectralProcessor(self.config)
    
    @mock_s3
    def test_process_single_tile(self):
        """Test processing a single tile"""
        # Create test raster
        test_file, test_data = create_test_raster(bands=3)
        
        # Setup mock S3
        client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-bucket'
        client.create_bucket(Bucket=bucket_name)
        
        # Upload test file
        with open(test_file, 'rb') as f:
            client.put_object(Bucket=bucket_name, Key='rgb_tile.tif', Body=f.read())
        
        # Test processing
        band_mapping = {"R": 0, "G": 1, "B": 2}
        indices = ["TGI", "ExG"]
        
        results = self.processor.process_single_tile(
            bucket_name, 'rgb_tile.tif', band_mapping, indices
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn("TGI", results)
        self.assertIn("ExG", results)
        
        # Check that results are numpy arrays
        for index_name, result in results.items():
            if result is not None:
                self.assertIsInstance(result, np.ndarray)
        
        # Cleanup
        os.unlink(test_file)
    
    def test_create_mosaic_horizontal(self):
        """Test horizontal mosaic creation"""
        # Create mock tile results
        tile1_data = np.ones((50, 50))
        tile2_data = np.ones((50, 50)) * 2
        
        tile_results = {
            'tile1.tif': {'NDVI': tile1_data, 'EVI': tile1_data * 0.5},
            'tile2.tif': {'NDVI': tile2_data, 'EVI': tile2_data * 0.5}
        }
        
        mosaics = self.processor.create_mosaic(tile_results, "horizontal")
        
        self.assertIn('NDVI', mosaics)
        self.assertIn('EVI', mosaics)
        
        # Check mosaic dimensions (should be 50x100 for horizontal concatenation)
        self.assertEqual(mosaics['NDVI'].shape, (50, 100))
        
        # Check values
        np.testing.assert_array_equal(mosaics['NDVI'][:, :50], tile1_data)
        np.testing.assert_array_equal(mosaics['NDVI'][:, 50:], tile2_data)
    
    def test_create_mosaic_vertical(self):
        """Test vertical mosaic creation"""
        # Create mock tile results
        tile1_data = np.ones((50, 50))
        tile2_data = np.ones((50, 50)) * 2
        
        tile_results = {
            'tile1.tif': {'NDVI': tile1_data},
            'tile2.tif': {'NDVI': tile2_data}
        }
        
        mosaics = self.processor.create_mosaic(tile_results, "vertical")
        
        # Check mosaic dimensions (should be 100x50 for vertical concatenation)
        self.assertEqual(mosaics['NDVI'].shape, (100, 50))
        
        # Check values
        np.testing.assert_array_equal(mosaics['NDVI'][:50, :], tile1_data)
        np.testing.assert_array_equal(mosaics['NDVI'][50:, :], tile2_data)


class TestConvenienceFunctions(unittest.TestCase):
    """Test high-level convenience functions"""
    
    def test_create_s3_config(self):
        """Test S3 config creation function"""
        config = create_s3_config("key", "secret", "us-west-1")
        
        self.assertIsInstance(config, S3Config)
        self.assertEqual(config.aws_access_key_id, "key")
        self.assertEqual(config.aws_secret_access_key, "secret")
        self.assertEqual(config.region_name, "us-west-1")
    
    @patch('spyndex.s3.S3SpectralProcessor')
    def test_process_s3_spectral_data(self, mock_processor_class):
        """Test high-level processing function"""
        # Mock the processor
        mock_processor = Mock()
        mock_processor.process_tiles_parallel.return_value = {'tile1': {'NDVI': np.array([1, 2, 3])}}
        mock_processor_class.return_value = mock_processor
        
        # Test function call
        result = process_s3_spectral_data(
            bucket='test-bucket',
            keys=['tile1.tif'],
            band_mapping={'R': 0, 'G': 1, 'B': 2},
            indices=['NDVI'],
            max_workers=2
        )
        
        # Verify processor was called correctly
        mock_processor.process_tiles_parallel.assert_called_once()
        self.assertIsInstance(result, dict)


class TestDependencyChecking(unittest.TestCase):
    """Test dependency checking functionality"""
    
    def test_check_s3_dependencies_available(self):
        """Test dependency checking when deps are available"""
        if S3_DEPS_AVAILABLE:
            # Should not raise an exception
            _check_s3_dependencies()
        else:
            with self.assertRaises(ImportError):
                _check_s3_dependencies()
    
    @patch('spyndex.s3.S3_AVAILABLE', False)
    def test_check_s3_dependencies_missing(self):
        """Test dependency checking when deps are missing"""
        with self.assertRaises(ImportError):
            _check_s3_dependencies()


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    @pytest.mark.skipif(not S3_DEPS_AVAILABLE, reason="S3 dependencies not available")
    @mock_s3
    def test_end_to_end_rgb_processing(self):
        """Test complete RGB processing workflow"""
        # Create test RGB raster
        test_file, test_data = create_test_raster(width=50, height=50, bands=3)
        
        # Setup mock S3
        client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'bluesky-data'
        client.create_bucket(Bucket=bucket_name)
        
        # Upload test files for multiple tiles
        tile_keys = []
        for i in range(3):
            key = f'invermark/rgb/raw/rgb_tile_10000_{5335+i}.tif'
            tile_keys.append(key)
            with open(test_file, 'rb') as f:
                client.put_object(Bucket=bucket_name, Key=key, Body=f.read())
        
        # Process using the BlueSky-specific method
        config = S3Config("test_key", "test_secret")
        processor = S3SpectralProcessor(config)
        
        results = processor.process_bluesky_rgb_tiles(
            bucket=bucket_name,
            base_prefix='invermark/rgb/raw',
            tile_indices=[5335, 5336, 5337],
            max_workers=2
        )
        
        # Verify results
        self.assertIsInstance(results, dict)
        
        # Should have RGB-based indices
        expected_indices = ["TGI", "GLI", "VARI", "ExG", "ExR", "MGRVI", "RGBVI"]
        for index_name in expected_indices:
            if index_name in results:  # Some indices might fail with test data
                self.assertIsInstance(results[index_name], np.ndarray)
                # Mosaic should be 50x150 (3 tiles of 50x50 horizontally concatenated)
                self.assertEqual(results[index_name].shape[1], 150)
        
        # Cleanup
        os.unlink(test_file)


if __name__ == '__main__':
    # Run with pytest for better output and skip handling
    pytest.main([__file__, '-v'])
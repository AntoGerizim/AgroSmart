import cdsapi
import os
from datetime import datetime, date
import json
import time
import zipfile
import glob


class AgERA5QuarterlyDownloader:
    def __init__(self, base_directory="./agera5_data", location=None):
        """
        Initialize downloader with quarterly time splitting and ZIP handling

        Args:
            base_directory (str): Base directory for all downloads
            location (dict): Location dictionary with coordinates and name
        """
        self.base_directory = base_directory
        self.client = cdsapi.Client()

        # Default location
        self.location = location or {
            'area': [27.2, 88.4, 27.2, 88.4],  # Single point for Jorethang, Sikkim
            'location_name': 'Jorethang_Sikkim'
        }

        # PyFAO56 required variables with correct statistics handling
        self.variables_config = {
            '2m_temperature': {
                'dir_name': 'temperature_2m',
                'description': '2m Air Temperature',
                'statistics': ['24_hour_mean', '24_hour_maximum', '24_hour_minimum']
            },
            '2m_dewpoint_temperature': {
                'dir_name': 'dewpoint_2m',
                'description': '2m Dewpoint Temperature',
                'statistics': ['24_hour_mean']  # Only available statistic
            },
            'solar_radiation_flux': {
                'dir_name': 'solar_radiation',
                'description': 'Solar Radiation Flux',
                'statistics': None  # No statistics - omit field from request
            },
            'precipitation_flux': {
                'dir_name': 'precipitation',
                'description': 'Precipitation Flux',
                'statistics': None  # No statistics - omit field from request
            },
            '10m_wind_speed': {
                'dir_name': 'wind_speed_10m',
                'description': '10m Wind Speed',
                'statistics': ['24_hour_mean']  # No statistics - omit field from request
            }
        }

        # Define quarterly time periods
        self.quarters = {
            'Q1': {'months': ['01', '02', '03'], 'description': 'Jan-Mar'},
            'Q2': {'months': ['04', '05', '06'], 'description': 'Apr-Jun'},
            'Q3': {'months': ['07', '08', '09'], 'description': 'Jul-Sep'},
            'Q4': {'months': ['10', '11', '12'], 'description': 'Oct-Dec'}
        }

        # Create directory structure
        self._create_directories()

    def _create_directories(self):
        """Create organized directory structure with quarterly subdirectories"""
        os.makedirs(self.base_directory, exist_ok=True)

        for variable, config in self.variables_config.items():
            var_dir = os.path.join(self.base_directory, config['dir_name'])
            os.makedirs(var_dir, exist_ok=True)

            # Create quarterly subdirectories within each variable directory
            for quarter in self.quarters.keys():
                quarter_dir = os.path.join(var_dir, quarter)
                os.makedirs(quarter_dir, exist_ok=True)

                # Create extracted subdirectory for ZIP contents
                extracted_dir = os.path.join(quarter_dir, 'extracted')
                os.makedirs(extracted_dir, exist_ok=True)

        # Create logs directory
        logs_dir = os.path.join(self.base_directory, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        print(f"📁 Created quarterly directory structure in: {self.base_directory}")

    def set_location_by_point(self, latitude, longitude, location_name, buffer=0.05):
        """Set location using single point with buffer"""
        north = latitude + buffer
        south = latitude - buffer
        east = longitude + buffer
        west = longitude - buffer

        self.location = {
            'area': [north, west, south, east],
            'location_name': location_name.replace(' ', '_').replace(',', '')
        }

        print(f"✅ Location set: {location_name}")
        print(f"   Center: {latitude}°N, {longitude}°E")
        print(f"   Area: [{north:.2f}, {west:.2f}, {south:.2f}, {east:.2f}]")

    def _is_zip_file(self, file_path):
        """Check if file is actually a ZIP archive"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'PK\x03\x04'  # ZIP file signature
        except Exception:
            return False

    def _extract_zip_file(self, zip_path):
        """
        Extract ZIP file and return path to NetCDF file

        Args:
            zip_path (str): Path to ZIP file

        Returns:
            str: Path to extracted NetCDF file, or None if extraction failed
        """
        try:
            # Create extraction directory
            extract_dir = zip_path.replace('.nc', '_extracted')
            os.makedirs(extract_dir, exist_ok=True)

            print(f"📦 Extracting ZIP file: {os.path.basename(zip_path)}")

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = zip_ref.namelist()

            print(f"   → Extracted {len(extracted_files)} files")

            # Find NetCDF files in extracted content
            nc_files = [f for f in extracted_files if f.endswith('.nc')]

            if nc_files:
                nc_path = os.path.join(extract_dir, nc_files[0])
                file_size = os.path.getsize(nc_path) / (1024 * 1024)
                print(f"   → Found NetCDF: {nc_files[0]} ({file_size:.1f} MB)")
                return nc_path
            else:
                print(f"   ❌ No NetCDF files found in ZIP")
                return None

        except Exception as e:
            print(f"   ❌ ZIP extraction failed: {str(e)}")
            return None

    def _validate_netcdf_file(self, file_path):
        """
        Validate NetCDF file by trying to open it with xarray

        Args:
            file_path (str): Path to NetCDF file

        Returns:
            bool: True if file is valid and readable
        """
        try:
            import xarray as xr

            # Try different engines
            engines = ['netcdf4', 'h5netcdf', 'scipy']

            for engine in engines:
                try:
                    ds = xr.open_dataset(file_path, engine=engine)
                    ds.close()
                    print(f"   ✅ File validated with {engine} engine")
                    return True
                except Exception:
                    continue

            print(f"   ❌ File failed validation with all engines")
            return False

        except ImportError:
            print(f"   ⚠️  xarray not available for validation")
            return True  # Assume valid if can't validate

    def download_quarter(self, variable, year, quarter, retry_count=3, delay_between_requests=5):
        """
        Download single variable for a specific quarter with conditional statistics handling

        Args:
            variable (str): Variable name (e.g., '2m_temperature')
            year (int): Year to download
            quarter (str): Quarter ('Q1', 'Q2', 'Q3', 'Q4')
            retry_count (int): Number of retry attempts on failure
            delay_between_requests (int): Seconds to wait between requests
        """
        if variable not in self.variables_config:
            print(f"❌ Unknown variable: {variable}")
            return None

        if quarter not in self.quarters:
            print(f"❌ Invalid quarter: {quarter}. Use Q1, Q2, Q3, or Q4")
            return None

        config = self.variables_config[variable]
        quarter_info = self.quarters[quarter]

        # Create paths
        var_dir = os.path.join(self.base_directory, config['dir_name'])
        quarter_dir = os.path.join(var_dir, quarter)

        # Generate filename (initially assuming it might be ZIP)
        base_filename = f"{variable}_{self.location['location_name']}_{year}_{quarter}"
        download_path = os.path.join(quarter_dir, f"{base_filename}.nc")

        # Check if valid NetCDF already exists
        if os.path.exists(download_path) and not self._is_zip_file(download_path):
            if self._validate_netcdf_file(download_path):
                file_size = os.path.getsize(download_path) / (1024 * 1024)
                print(f"⏭️  Valid NetCDF exists: {os.path.basename(download_path)} ({file_size:.1f} MB)")
                return download_path

        # Check if extracted NetCDF already exists
        extracted_path = os.path.join(quarter_dir, f"{base_filename}_extracted")
        existing_nc = glob.glob(os.path.join(extracted_path, "*.nc"))
        if existing_nc and self._validate_netcdf_file(existing_nc[0]):
            file_size = os.path.getsize(existing_nc[0]) / (1024 * 1024)
            print(f"⏭️  Extracted NetCDF exists: {os.path.basename(existing_nc[0])} ({file_size:.1f} MB)")
            return existing_nc[0]

        # Prepare request for specific quarter
        days = [f"{d:02d}" for d in range(1, 32)]  # All possible days

        # Build base request
        request = {
            'variable': [variable],
            'year': [str(year)],
            'month': quarter_info['months'],  # Only months for this quarter
            'day': days,
            'area': self.location['area'],
            'version': '2_0'
        }

        # CONDITIONAL: Only include statistics if they are defined (not None)
        stats = config['statistics']
        if stats is not None:
            request['statistic'] = stats
            stats_info = f" with statistics: {', '.join(stats)}"
        else:
            stats_info = " (no statistics - using default)"

        print(f"⬇️  Downloading {config['description']} - {year} {quarter} ({quarter_info['description']})")
        print(f"   Months: {', '.join(quarter_info['months'])}{stats_info}")
        print(f"   Output: {download_path}")

        # Attempt download with retries
        for attempt in range(retry_count):
            try:
                self.client.retrieve(
                    'sis-agrometeorological-indicators',
                    request,
                    download_path
                )

                print(f"✅ Download completed: {os.path.basename(download_path)}")

                # Check if downloaded file is ZIP
                if self._is_zip_file(download_path):
                    print(f"📦 Downloaded file is ZIP format")
                    extracted_nc = self._extract_zip_file(download_path)

                    if extracted_nc and self._validate_netcdf_file(extracted_nc):
                        file_size = os.path.getsize(extracted_nc) / (1024 * 1024)
                        print(f"✅ Successfully extracted and validated NetCDF ({file_size:.1f} MB)")

                        # Log successful download
                        self._log_download(variable, year, quarter, extracted_nc, "success_extracted")

                        # Add delay between requests
                        if delay_between_requests > 0:
                            time.sleep(delay_between_requests)

                        return extracted_nc
                    else:
                        print(f"❌ Failed to extract or validate NetCDF from ZIP")
                        return None

                else:
                    # Direct NetCDF file
                    if self._validate_netcdf_file(download_path):
                        file_size = os.path.getsize(download_path) / (1024 * 1024)
                        print(f"✅ Successfully downloaded NetCDF ({file_size:.1f} MB)")

                        # Log successful download
                        self._log_download(variable, year, quarter, download_path, "success_direct")

                        # Add delay between requests
                        if delay_between_requests > 0:
                            time.sleep(delay_between_requests)

                        return download_path
                    else:
                        print(f"❌ Downloaded file failed validation")
                        return None

            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{retry_count} failed for {variable} {year} {quarter}: {str(e)}")

                # Log failed attempt
                self._log_download(variable, year, quarter, download_path, f"failed_attempt_{attempt + 1}: {str(e)}")

                if attempt < retry_count - 1:
                    wait_time = delay_between_requests * (attempt + 1)
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ All attempts failed for {variable} {year} {quarter}")
                    return None

    def download_variable_yearly(self, variable, year, delay_between_quarters=10):
        """
        Download all 4 quarters for a specific variable and year

        Args:
            variable (str): Variable name
            year (int): Year to download
            delay_between_quarters (int): Seconds to wait between quarters
        """
        print(f"\n📊 Downloading {self.variables_config[variable]['description']} for {year} (4 quarters)")

        quarterly_files = []

        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            result = self.download_quarter(variable, year, quarter)

            if result:
                quarterly_files.append(result)

            # Wait between quarters
            if delay_between_quarters > 0 and quarter != 'Q4':  # Don't wait after last quarter
                print(f"   Waiting {delay_between_quarters} seconds before next quarter...")
                time.sleep(delay_between_quarters)

        print(f"   📋 Completed {len(quarterly_files)}/4 quarters for {variable} {year}")
        return quarterly_files

    def download_all_variables_yearly(self, year_list, delay_between_variables=15, delay_between_quarters=10):
        """
        Download all variables for specified years, split into quarters with conditional statistics

        Args:
            year_list (list): List of years to download
            delay_between_variables (int): Seconds to wait between different variables
            delay_between_quarters (int): Seconds to wait between quarters of same variable
        """
        total_downloads = len(self.variables_config) * len(year_list) * 4  # 4 quarters per year
        completed = 0
        failed = 0

        print(f"🚀 Starting quarterly download with conditional statistics handling")
        print(f"   Variables: {len(self.variables_config)}")
        print(f"   Years: {len(year_list)}")
        print(f"   Total expected downloads: {total_downloads}")
        print(f"   Location: {self.location['location_name']}")

        # Show statistics configuration
        print(f"\n📊 Variables Configuration:")
        for var_name, config in self.variables_config.items():
            stats_str = str(config['statistics']) if config['statistics'] else "None (default)"
            print(f"   • {config['description']}: {stats_str}")

        for year in year_list:
            print(f"\n📅 === Processing Year {year} ===")

            for variable, config in self.variables_config.items():
                print(f"\n📊 Processing: {config['description']} - {year}")

                quarterly_results = self.download_variable_yearly(
                    variable, year, delay_between_quarters
                )

                # Update counters
                completed += len(quarterly_results)
                failed += (4 - len(quarterly_results))  # 4 quarters expected

                print(f"   Progress: {completed + failed}/{total_downloads} ({completed} success, {failed} failed)")

                # Wait between different variables
                if delay_between_variables > 0:
                    print(f"   Waiting {delay_between_variables} seconds before next variable...")
                    time.sleep(delay_between_variables)

        print(f"\n🎉 Download session complete!")
        print(f"   ✅ Successful: {completed}/{total_downloads}")
        print(f"   ❌ Failed: {failed}/{total_downloads}")

        return completed, failed

    def merge_quarters_to_annual(self, variable, year, keep_quarterly_files=True):
        """
        Merge 4 quarterly files into a single annual file using xarray

        Args:
            variable (str): Variable name
            year (int): Year to merge
            keep_quarterly_files (bool): Whether to keep original quarterly files
        """
        try:
            import xarray as xr
        except ImportError:
            print("❌ xarray not installed. Install with: pip install xarray")
            return None

        config = self.variables_config[variable]
        var_dir = os.path.join(self.base_directory, config['dir_name'])

        # Collect quarterly files (including extracted ones)
        quarterly_paths = []
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            quarter_dir = os.path.join(var_dir, quarter)

            # Check for direct NetCDF
            direct_nc = os.path.join(quarter_dir, f"{variable}_{self.location['location_name']}_{year}_{quarter}.nc")
            if os.path.exists(direct_nc) and not self._is_zip_file(direct_nc):
                quarterly_paths.append(direct_nc)
                continue

            # Check for extracted NetCDF
            extracted_dir = os.path.join(quarter_dir,
                                         f"{variable}_{self.location['location_name']}_{year}_{quarter}_extracted")
            extracted_files = glob.glob(os.path.join(extracted_dir, "*.nc"))
            if extracted_files:
                quarterly_paths.append(extracted_files[0])
                continue

            print(f"❌ Missing quarterly file for {quarter}")
            return None

        if len(quarterly_paths) != 4:
            print(f"❌ Found only {len(quarterly_paths)}/4 quarterly files for {variable} {year}")
            return None

        print(f"🔗 Merging 4 quarters into annual file for {variable} {year}...")

        try:
            # Load and concatenate quarterly datasets
            datasets = []
            engines = ['netcdf4', 'h5netcdf', 'scipy']

            for path in quarterly_paths:
                ds_loaded = False
                for engine in engines:
                    try:
                        ds = xr.open_dataset(path, engine=engine)
                        datasets.append(ds)
                        ds_loaded = True
                        break
                    except Exception:
                        continue

                if not ds_loaded:
                    print(f"❌ Could not load {os.path.basename(path)}")
                    for ds in datasets:
                        ds.close()
                    return None

            # Concatenate and sort by time
            annual_ds = xr.concat(datasets, dim='time')
            annual_ds = annual_ds.sortby('time')  # Ensure chronological order

            # Save merged annual file
            annual_filename = f"{variable}_{self.location['location_name']}_{year}_annual.nc"
            annual_path = os.path.join(var_dir, annual_filename)
            annual_ds.to_netcdf(annual_path)

            # Close datasets
            for ds in datasets:
                ds.close()
            annual_ds.close()

            file_size = os.path.getsize(annual_path) / (1024 * 1024)
            print(f"✅ Created annual file: {annual_filename} ({file_size:.1f} MB)")

            return annual_path

        except Exception as e:
            print(f"❌ Error merging quarters: {str(e)}")
            return None

    def _log_download(self, variable, year, quarter, file_path, status):
        """Log download attempts and results"""
        log_file = os.path.join(self.base_directory, 'logs', 'download_log.json')

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'year': str(year),
            'quarter': quarter,
            'file_path': file_path,
            'status': status,
            'location': self.location
        }

        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []

        logs.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def test_netcdf_reading(self):
        """Test reading all downloaded NetCDF files"""
        print("\n🧪 Testing NetCDF file reading...")

        try:
            import xarray as xr
        except ImportError:
            print("❌ xarray not available for testing")
            return

        engines = ['netcdf4', 'h5netcdf', 'scipy']
        tested_files = 0
        successful_reads = 0

        for variable, config in self.variables_config.items():
            var_dir = os.path.join(self.base_directory, config['dir_name'])

            # Find all NetCDF files for this variable
            nc_files = []

            # Check quarterly directories
            for quarter in self.quarters.keys():
                quarter_dir = os.path.join(var_dir, quarter)

                # Direct NetCDF files
                direct_files = glob.glob(os.path.join(quarter_dir, "*.nc"))
                nc_files.extend([f for f in direct_files if not self._is_zip_file(f)])

                # Extracted NetCDF files
                extracted_files = glob.glob(os.path.join(quarter_dir, "*_extracted", "*.nc"))
                nc_files.extend(extracted_files)

            # Annual files
            annual_files = glob.glob(os.path.join(var_dir, "*_annual.nc"))
            nc_files.extend(annual_files)

            print(f"\n📊 Testing {config['description']} ({len(nc_files)} files)")

            for nc_file in nc_files:
                tested_files += 1
                file_readable = False

                for engine in engines:
                    try:
                        ds = xr.open_dataset(nc_file, engine=engine)
                        file_size = os.path.getsize(nc_file) / (1024 * 1024)
                        print(f"   ✅ {os.path.basename(nc_file)} ({file_size:.1f} MB) - {engine}")
                        ds.close()
                        successful_reads += 1
                        file_readable = True
                        break
                    except Exception:
                        continue

                if not file_readable:
                    print(f"   ❌ {os.path.basename(nc_file)} - No working engine")

        print(f"\n📋 Test Summary:")
        print(f"   Total files tested: {tested_files}")
        print(f"   Successfully readable: {successful_reads}")
        print(f"   Failed to read: {tested_files - successful_reads}")

    def list_downloaded_data(self):
        """List all downloaded files organized by variable and quarter"""
        print(f"\n📋 Downloaded Data Summary for {self.location['location_name']}:")

        total_files = 0
        total_size = 0

        for variable, config in self.variables_config.items():
            var_dir = os.path.join(self.base_directory, config['dir_name'])

            print(f"\n📊 {config['description']}:")

            # Count quarterly files
            quarterly_count = 0
            for quarter in self.quarters.keys():
                quarter_dir = os.path.join(var_dir, quarter)
                if os.path.exists(quarter_dir):
                    # Direct NetCDF files (non-ZIP)
                    direct_files = [f for f in os.listdir(quarter_dir)
                                    if f.endswith('.nc') and not self._is_zip_file(os.path.join(quarter_dir, f))]

                    # ZIP files
                    zip_files = [f for f in os.listdir(quarter_dir)
                                 if f.endswith('.nc') and self._is_zip_file(os.path.join(quarter_dir, f))]

                    # Extracted NetCDF files
                    extracted_files = glob.glob(os.path.join(quarter_dir, "*_extracted", "*.nc"))

                    quarterly_count += len(direct_files) + len(zip_files) + len(extracted_files)

                    if direct_files or zip_files or extracted_files:
                        print(f"   {quarter} ({self.quarters[quarter]['description']}):")

                        for file in direct_files:
                            file_path = os.path.join(quarter_dir, file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            total_size += file_size
                            print(f"     • {file} ({file_size:.1f} MB) [NetCDF]")

                        for file in zip_files:
                            file_path = os.path.join(quarter_dir, file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            total_size += file_size
                            print(f"     • {file} ({file_size:.1f} MB) [ZIP]")

                        for file in extracted_files:
                            file_size = os.path.getsize(file) / (1024 * 1024)
                            total_size += file_size
                            print(f"     • {os.path.basename(file)} ({file_size:.1f} MB) [Extracted]")

            # Count annual files
            annual_files = [f for f in os.listdir(var_dir) if f.endswith('_annual.nc')] if os.path.exists(
                var_dir) else []
            if annual_files:
                print(f"   Annual files: {len(annual_files)}")
                for file in sorted(annual_files):
                    file_path = os.path.join(var_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    total_size += file_size
                    print(f"     • {file} ({file_size:.1f} MB)")

            total_files += quarterly_count + len(annual_files)

        print(f"\n📁 Total: {total_files} files, {total_size:.1f} MB")
        print(f"📍 Base directory: {self.base_directory}")

    def show_directory_structure(self):
        """Show the created directory structure"""
        print(f"\n🏗️  Quarterly Directory Structure with Conditional Statistics:")
        print(f"{self.base_directory}/")

        for variable, config in self.variables_config.items():
            stats_info = f"[stats: {config['statistics']}]" if config['statistics'] else "[no stats]"
            print(f"├── {config['dir_name']}/          # {config['description']} {stats_info}")
            for quarter, info in self.quarters.items():
                print(f"│   ├── {quarter}/               # {info['description']}")
                print(f"│   │   ├── *.nc               # Downloaded files (may be ZIP)")
                print(f"│   │   └── *_extracted/       # Extracted NetCDF files")
            print(f"│   └── *_annual.nc         # Merged annual files")

        print(f"└── logs/                    # Download logs and metadata")


# Example usage
def main():
    """Example usage with conditional statistics handling"""

    # Initialize downloader
    downloader = AgERA5QuarterlyDownloader(base_directory="./weather_data_conditional_stats")

    # Set your location
    downloader.set_location_by_point(27.21, 88.42, "Jorethang_Sikkim_Farm")

    # Show directory structure and configuration
    downloader.show_directory_structure()

    years_to_download = [2023]  # Test with one year

    completed, failed = downloader.download_all_variables_yearly(
        year_list=years_to_download,
        delay_between_variables=15,
        delay_between_quarters=10
    )

    # Test NetCDF reading
    downloader.test_netcdf_reading()

    # List downloaded data
    downloader.list_downloaded_data()

if __name__ == "__main__":
    main()


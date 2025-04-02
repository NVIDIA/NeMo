# import pytest
# import time
# from unittest.mock import patch, MagicMock
# import io
# import logging

# # Import the class to be tested
# from nemo.common.ckpt.fs.ngc import NGCFileSystem

# # Import exceptions from the underlying library for mocking
# # Assuming these exceptions exist in nvidia_ngc.errors
# # If not, we might need to adjust based on the actual library structure
# try:
#     from nvidia_ngc.errors import NGCApiKeyNotFound, NGCException
# except ImportError:
#     # Define dummy exceptions if nvidia-ngc-client is not installed
#     # This allows basic test structure setup even without the dependency
#     class NGCApiKeyNotFound(Exception):
#         pass

#     class NGCException(Exception):
#         pass


# # Define a fixture to mock the NGCClient for most tests
# @pytest.fixture
# def mock_ngc_client():
#     # Use autospec=True if NGCClient has a clear spec, helps catch signature mismatches
#     # We might need to mock specific methods within tests depending on the scenario
#     with patch("nemo.common.ckpt.fs.ngc.NGCClient", autospec=True) as mock_client_class:
#         mock_instance = mock_client_class.return_value
#         # You might pre-configure common mock behaviors here if needed
#         # e.g., mock_instance.list_files.return_value = []
#         yield mock_instance  # The mock *instance* used by NGCFileSystem


# def test_init_success_defaults(mock_ngc_client):
#     """Test successful initialization with default arguments."""
#     fs = NGCFileSystem()
#     # Assert NGCClient was instantiated (called)
#     mock_ngc_client.__class__.assert_called_once_with()  # Checks __init__ call on the class
#     assert fs.client is mock_ngc_client
#     assert fs._info_cache_ttl == 60  # Default TTL


# def test_init_success_with_args(mock_ngc_client):
#     """Test successful init passing api_key, org, team."""
#     api_key = "test_key"
#     org = "test_org"
#     team = "test_team"
#     ttl = 120
#     fs = NGCFileSystem(api_key=api_key, org=org, team=team, info_cache_ttl=ttl)

#     # Assert NGCClient was instantiated with the correct arguments
#     mock_ngc_client.__class__.assert_called_once_with(
#         api_key=api_key, org=org, team=team
#     )
#     assert fs.client is mock_ngc_client
#     assert fs._info_cache_ttl == ttl


# def test_init_failure_api_key_not_found():
#     """Test init failure on NGCApiKeyNotFound."""
#     with patch(
#         "nemo.common.ckpt.fs.ngc.NGCClient",
#         side_effect=NGCApiKeyNotFound("Test Key Not Found"),
#     ) as mock_client_class:
#         with pytest.raises(NGCApiKeyNotFound):
#             NGCFileSystem()
#         mock_client_class.assert_called_once_with()  # Ensure it was attempted


# def test_init_failure_other_ngc_exception():
#     """Test init failure on other NGCException."""
#     custom_error_message = "Some other NGC configuration error"
#     with patch(
#         "nemo.common.ckpt.fs.ngc.NGCClient",
#         side_effect=NGCException(custom_error_message),
#     ) as mock_client_class:
#         with pytest.raises(NGCException, match=custom_error_message):
#             NGCFileSystem()
#         mock_client_class.assert_called_once_with()  # Ensure it was attempted


# # === Protocol Stripping Tests (_strip_protocol) ===


# @pytest.fixture
# def fs_instance(mock_ngc_client):
#     """Provides a basic NGCFileSystem instance for tests."""
#     # We use the mock_ngc_client fixture implicitly here
#     return NGCFileSystem()


# @pytest.mark.parametrize(
#     "path_in, expected_path_out",
#     [
#         ("ngc://my/path", "my/path"),
#         ("ngc://another/path/to/resource:v1", "another/path/to/resource:v1"),
#         ("no/protocol/path", "no/protocol/path"),
#         ("", ""),
#         # Test with lists
#         (
#             ["ngc://list/path1", "path2", "ngc://list/path3"],
#             ["list/path1", "path2", "list/path3"],
#         ),
#         ([], []),
#     ],
# )
# def test_strip_protocol(fs_instance, path_in, expected_path_out):
#     """Test the _strip_protocol method."""
#     assert fs_instance._strip_protocol(path_in) == expected_path_out


# @pytest.fixture
# def mock_ngc_client_class():
#     """Mocks the NGCClient class itself."""
#     with patch("nemo.common.ckpt.fs.ngc.NGCClient", autospec=True) as MockClientClass:
#         yield MockClientClass


# @pytest.fixture
# def mock_ngc_client_instance(mock_ngc_client_class):
#     """Provides the mocked instance returned by the mocked class."""
#     # Configure default successful return values for common methods if needed
#     mock_instance = mock_ngc_client_class.return_value
#     # Example: mock_instance.list_files.return_value = []
#     # Example: mock_instance.get_file_info.return_value = {...}
#     yield mock_instance


# @pytest.fixture
# def fs_instance(mock_ngc_client_instance):
#     """Provides a basic NGCFileSystem instance initialized with the mocked client instance."""
#     # mock_ngc_client_instance fixture ensures NGCClient() returns the mock
#     return NGCFileSystem()



# class TestNGCFileSystemInit:
#     """Tests for NGCFileSystem initialization."""

#     def test_init_success_defaults(
#         self, mock_ngc_client_class, mock_ngc_client_instance
#     ):
#         """Test successful initialization with default arguments."""
#         fs = NGCFileSystem()
#         mock_ngc_client_class.assert_called_once_with()  # Check class constructor call
#         assert fs.client is mock_ngc_client_instance  # Check instance assignment
#         assert fs._info_cache_ttl == 60  # Default TTL

#     def test_init_success_with_args(
#         self, mock_ngc_client_class, mock_ngc_client_instance
#     ):
#         """Test successful init passing api_key, org, team."""
#         api_key = "test_key"
#         org = "test_org"
#         team = "test_team"
#         ttl = 120
#         fs = NGCFileSystem(api_key=api_key, org=org, team=team, info_cache_ttl=ttl)

#         mock_ngc_client_class.assert_called_once_with(
#             api_key=api_key, org=org, team=team
#         )
#         assert fs.client is mock_ngc_client_instance
#         assert fs._info_cache_ttl == ttl

#     def test_init_failure_api_key_not_found(self, mock_ngc_client_class):
#         """Test init failure on NGCApiKeyNotFound."""
#         mock_ngc_client_class.side_effect = NGCApiKeyNotFound("Test Key Not Found")
#         with pytest.raises(NGCApiKeyNotFound):
#             NGCFileSystem()
#         mock_ngc_client_class.assert_called_once_with()  # Ensure it was attempted

#     def test_init_failure_other_ngc_exception(self, mock_ngc_client_class):
#         """Test init failure on other NGCException."""
#         custom_error_message = "Some other NGC configuration error"
#         mock_ngc_client_class.side_effect = NGCException(custom_error_message)
#         with pytest.raises(NGCException, match=custom_error_message):
#             NGCFileSystem()
#         mock_ngc_client_class.assert_called_once_with()  # Ensure it was attempted


# class TestNGCFileSystemUtils:
#     """Tests for utility methods like _strip_protocol."""

#     @pytest.mark.parametrize(
#         "path_in, expected_path_out",
#         [
#             ("ngc://my/path", "my/path"),
#             ("ngc://another/path/to/resource:v1", "another/path/to/resource:v1"),
#             ("no/protocol/path", "no/protocol/path"),
#             ("", ""),
#             (
#                 ["ngc://list/path1", "path2", "ngc://list/path3"],
#                 ["list/path1", "path2", "list/path3"],
#             ),
#             ([], []),
#         ],
#     )
#     def test_strip_protocol(self, fs_instance, path_in, expected_path_out):
#         """Test the _strip_protocol method."""
#         assert fs_instance._strip_protocol(path_in) == expected_path_out

#     # Test for unstrip_protocol which is implicitly used but good to have explicit if needed
#     @pytest.mark.parametrize(
#         "path_in, expected_path_out",
#         [
#             ("my/path", "ngc://my/path"),
#             ("another/path/to/resource:v1", "ngc://another/path/to/resource:v1"),
#             ("", "ngc://"),  # Assuming this is the expected behavior for empty string
#         ],
#     )
#     def test_unstrip_protocol(self, fs_instance, path_in, expected_path_out):
#         """Test the unstrip_protocol method."""
#         assert fs_instance.unstrip_protocol(path_in) == expected_path_out


# class TestNGCFileSystemLs:
#     """Tests for the ls() method."""

#     @pytest.fixture
#     def sample_ls_data(self):
#         # Sample data returned by mock client.list_files
#         return [
#             {
#                 "path": "my/org/model/file1.txt",
#                 "size": 100,
#                 "lastModified": "2023-01-01T00:00:00Z",
#             },
#             {
#                 "path": "my/org/model/subdir/file2.bin",
#                 "size": 2048,
#                 "lastModified": "2023-01-02T10:30:00Z",
#             },
#             {
#                 "path": "my/org/model/config.yaml",
#                 "size": 512,
#                 "lastModified": "2023-01-01T05:00:00Z",
#             },
#         ]

#     def test_ls_detail_false(
#         self, fs_instance, mock_ngc_client_instance, sample_ls_data
#     ):
#         """Test ls(detail=False) returns list of full names."""
#         ngc_path_in = "ngc://my/org/model"
#         stripped_path = "my/org/model"
#         mock_ngc_client_instance.list_files.return_value = sample_ls_data

#         result = fs_instance.ls(ngc_path_in, detail=False)

#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert result == [
#             "ngc://my/org/model/file1.txt",
#             "ngc://my/org/model/subdir/file2.bin",
#             "ngc://my/org/model/config.yaml",
#         ]
#         # Check dircache population (names only needed for detail=False result)
#         assert ngc_path_in in fs_instance.dircache
#         assert len(fs_instance.dircache[ngc_path_in]) == len(sample_ls_data)
#         assert (
#             fs_instance.dircache[ngc_path_in][0]["name"]
#             == "ngc://my/org/model/file1.txt"
#         )

#     def test_ls_detail_true(
#         self, fs_instance, mock_ngc_client_instance, sample_ls_data
#     ):
#         """Test ls(detail=True) returns list of dictionaries."""
#         ngc_path_in = "ngc://my/org/model"
#         stripped_path = "my/org/model"
#         mock_ngc_client_instance.list_files.return_value = sample_ls_data

#         result = fs_instance.ls(ngc_path_in, detail=True)

#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert len(result) == 3
#         # Check structure of one dictionary
#         assert result[0] == {
#             "name": "ngc://my/org/model/file1.txt",
#             "size": 100,
#             "type": "file",  # Assuming 'file' type is default
#             "lastModified": "2023-01-01T00:00:00Z",
#         }
#         assert result[1]["name"] == "ngc://my/org/model/subdir/file2.bin"
#         assert result[1]["size"] == 2048
#         # Check dircache population
#         assert ngc_path_in in fs_instance.dircache
#         assert (
#             fs_instance.dircache[ngc_path_in] == result
#         )  # Should store detailed results

#     def test_ls_empty(self, fs_instance, mock_ngc_client_instance):
#         """Test ls() returns empty list when client.list_files returns empty."""
#         ngc_path_in = "ngc://my/org/empty_model"
#         stripped_path = "my/org/empty_model"
#         mock_ngc_client_instance.list_files.return_value = []

#         result_detail = fs_instance.ls(ngc_path_in, detail=True)
#         result_no_detail = fs_instance.ls(
#             ngc_path_in, detail=False
#         )  # Call again to check cache use

#         mock_ngc_client_instance.list_files.assert_called_once_with(
#             stripped_path
#         )  # Should only be called once
#         assert result_detail == []
#         assert result_no_detail == []
#         assert fs_instance.dircache[ngc_path_in] == []  # Cache should store empty list

#     def test_ls_not_found_error(self, fs_instance, mock_ngc_client_instance):
#         """Test ls() raises FileNotFoundError for 'not found' NGCException."""
#         ngc_path_in = "ngc://my/org/nonexistent"
#         stripped_path = "my/org/nonexistent"
#         # Simulate the specific error string check in the code
#         mock_ngc_client_instance.list_files.side_effect = NGCException(
#             "Resource not found."
#         )

#         with pytest.raises(FileNotFoundError, match=f"Path not found: {ngc_path_in}"):
#             fs_instance.ls(ngc_path_in)

#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert (
#             ngc_path_in not in fs_instance.dircache
#         )  # Cache should not be populated on error

#     def test_ls_other_ngc_error(self, fs_instance, mock_ngc_client_instance):
#         """Test ls() raises original NGCException for other errors."""
#         ngc_path_in = "ngc://my/org/error_path"
#         stripped_path = "my/org/error_path"
#         error_message = "Some other API error"
#         mock_ngc_client_instance.list_files.side_effect = NGCException(error_message)

#         with pytest.raises(NGCException, match=error_message):
#             fs_instance.ls(ngc_path_in)

#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert ngc_path_in not in fs_instance.dircache

#     def test_ls_dircache_usage(
#         self, fs_instance, mock_ngc_client_instance, sample_ls_data
#     ):
#         """Test dircache: ls() called twice, API called once."""
#         ngc_path_in = "ngc://my/org/cached_model"
#         stripped_path = "my/org/cached_model"
#         mock_ngc_client_instance.list_files.return_value = sample_ls_data

#         # First call - populates cache
#         result1 = fs_instance.ls(ngc_path_in, detail=True)
#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert len(result1) == len(sample_ls_data)
#         assert ngc_path_in in fs_instance.dircache

#         # Second call - should use cache
#         result2 = fs_instance.ls(ngc_path_in, detail=True)
#         # Assert API was NOT called again
#         mock_ngc_client_instance.list_files.assert_called_once()  # Still only once
#         assert result1 == result2  # Results should be identical

#         # Third call - detail=False, should also use cache but format output
#         result3 = fs_instance.ls(ngc_path_in, detail=False)
#         mock_ngc_client_instance.list_files.assert_called_once()  # Still only once
#         assert len(result3) == len(sample_ls_data)
#         assert result3[0] == fs_instance.unstrip_protocol(sample_ls_data[0]["path"])

#     def test_ls_dircache_refresh(
#         self, fs_instance, mock_ngc_client_instance, sample_ls_data
#     ):
#         """Test refresh=True bypasses dircache."""
#         ngc_path_in = "ngc://my/org/refresh_model"
#         stripped_path = "my/org/refresh_model"
#         mock_ngc_client_instance.list_files.return_value = sample_ls_data

#         # First call
#         fs_instance.ls(ngc_path_in)
#         mock_ngc_client_instance.list_files.assert_called_once_with(stripped_path)
#         assert ngc_path_in in fs_instance.dircache

#         # Second call with refresh=True
#         new_data = [
#             {
#                 "path": "my/org/refresh_model/new_file.txt",
#                 "size": 50,
#                 "lastModified": "...",
#             }
#         ]
#         mock_ngc_client_instance.list_files.return_value = (
#             new_data  # Change return value for second call
#         )
#         result_refresh = fs_instance.ls(ngc_path_in, refresh=True, detail=True)

#         # Assert API was called again (total 2 times)
#         assert mock_ngc_client_instance.list_files.call_count == 2
#         mock_ngc_client_instance.list_files.assert_called_with(
#             stripped_path
#         )  # Check last call args
#         assert len(result_refresh) == 1
#         assert result_refresh[0]["name"] == "ngc://my/org/refresh_model/new_file.txt"
#         # Check cache was updated
#         assert fs_instance.dircache[ngc_path_in] == result_refresh


# class TestNGCFileSystemInfo:
#     """Tests for the info() method."""

#     ngc_path = "ngc://my/org/resource"
#     stripped_path = "my/org/resource"

#     @pytest.fixture
#     def mock_time(self):
#         """Mock time.time() to control cache expiry."""
#         initial_time = time.time()
#         with patch("time.time") as mock_time_func:
#             # Use a list to allow modification within the test
#             current_time = [initial_time]
#             mock_time_func.side_effect = lambda: current_time[0]
#             yield current_time  # Yield the list to allow advancing time

#     def test_info_file_success(self, fs_instance, mock_ngc_client_instance):
#         """Test info() when get_file_info returns file data."""
#         file_info_data = {"size": 1024, "lastModified": "2023-01-01", "id": "file1"}
#         mock_ngc_client_instance.get_file_info.return_value = file_info_data

#         result = fs_instance.info(self.ngc_path)

#         mock_ngc_client_instance.get_file_info.assert_called_once_with(
#             self.stripped_path
#         )
#         mock_ngc_client_instance.list_files.assert_not_called()
#         expected = {
#             "name": self.ngc_path,
#             "size": 1024,
#             "type": "file",
#             "lastModified": "2023-01-01",
#             "id": "file1",
#         }
#         assert result == expected
#         assert self.stripped_path in fs_instance._info_cache
#         assert fs_instance._info_cache[self.stripped_path] == expected

#     def test_info_directory_heuristic(self, fs_instance, mock_ngc_client_instance):
#         """Test info() determines 'directory' when get_file_info returns no size."""
#         file_info_data = {"size": None, "lastModified": "2023-01-01", "id": "dir1"}
#         mock_ngc_client_instance.get_file_info.return_value = file_info_data

#         result = fs_instance.info(self.ngc_path)

#         mock_ngc_client_instance.get_file_info.assert_called_once_with(
#             self.stripped_path
#         )
#         mock_ngc_client_instance.list_files.assert_not_called()
#         expected = {
#             "name": self.ngc_path,
#             "size": None,
#             "type": "directory",
#             "lastModified": "2023-01-01",
#             "id": "dir1",
#         }
#         assert result == expected
#         assert self.stripped_path in fs_instance._info_cache

#     def test_info_directory_fallback(self, fs_instance, mock_ngc_client_instance):
#         """Test info() falls back to list_files for directory check."""
#         # get_file_info fails (e.g., path is a true directory/model root)
#         mock_ngc_client_instance.get_file_info.side_effect = NGCException(
#             "Cannot get info for this type"
#         )
#         # list_files succeeds (even if empty)
#         mock_ngc_client_instance.list_files.return_value = []

#         result = fs_instance.info(self.ngc_path)

#         mock_ngc_client_instance.get_file_info.assert_called_once_with(
#             self.stripped_path
#         )
#         mock_ngc_client_instance.list_files.assert_called_once_with(self.stripped_path)
#         expected = {
#             "name": self.ngc_path,
#             "size": None,
#             "type": "directory",
#         }
#         assert result == expected
#         assert self.stripped_path in fs_instance._info_cache

#     def test_info_not_found_both_fail(self, fs_instance, mock_ngc_client_instance):
#         """Test info() raises FileNotFoundError if get_file_info and list_files fail."""
#         mock_ngc_client_instance.get_file_info.side_effect = NGCException(
#             "Primary not found"
#         )
#         mock_ngc_client_instance.list_files.side_effect = NGCException(
#             "Fallback not found"
#         )  # Match string check

#         with pytest.raises(FileNotFoundError, match=f"Path not found: {self.ngc_path}"):
#             fs_instance.info(self.ngc_path)

#         mock_ngc_client_instance.get_file_info.assert_called_once_with(
#             self.stripped_path
#         )
#         mock_ngc_client_instance.list_files.assert_called_once_with(self.stripped_path)
#         assert self.stripped_path not in fs_instance._info_cache

#     def test_info_other_exception(self, fs_instance, mock_ngc_client_instance):
#         """Test info() raises original error if get_file_info fails with non-'not found'."""
#         error_message = "Some other API error"
#         mock_ngc_client_instance.get_file_info.side_effect = NGCException(error_message)

#         with pytest.raises(NGCException, match=error_message) as exc_info:
#             fs_instance.info(self.ngc_path)

#         # Ensure the raised exception is the one from get_file_info
#         assert exc_info.value is mock_ngc_client_instance.get_file_info.side_effect
#         mock_ngc_client_instance.get_file_info.assert_called_once_with(
#             self.stripped_path
#         )
#         # list_files should not be called if the first error wasn't "not found" related
#         # (Adjust based on exact implementation error handling)
#         # In current implementation, list_files *is* called as a fallback even for other errors
#         # mock_ngc_client_instance.list_files.assert_not_called()
#         # Let's test the actual behavior where it tries list_files too:
#         mock_ngc_client_instance.list_files.side_effect = NGCException(
#             "Fallback error irrelevant"
#         )
#         with pytest.raises(NGCException, match=error_message):
#             fs_instance.info(self.ngc_path)
#         mock_ngc_client_instance.list_files.assert_called_once_with(self.stripped_path)

#     def test_info_cache_usage(self, fs_instance, mock_ngc_client_instance):
#         """Test info() uses the cache on subsequent calls."""
#         file_info_data = {"size": 100, "type": "file", "name": self.ngc_path}
#         mock_ngc_client_instance.get_file_info.return_value = file_info_data
#         result1 = fs_instance.info(self.ngc_path)  # Call 1 - fetches
#         mock_ngc_client_instance.get_file_info.assert_called_once()
#         result2 = fs_instance.info(self.ngc_path)  # Call 2 - uses cache
#         mock_ngc_client_instance.get_file_info.assert_called_once()  # Still 1 call
#         assert result1 == result2

#     def test_info_cache_expiry(self, fs_instance, mock_ngc_client_instance, mock_time):
#         """Test info() cache expires after TTL."""
#         file_info_data = {"size": 100, "type": "file", "name": self.ngc_path}
#         mock_ngc_client_instance.get_file_info.return_value = file_info_data

#         # Call 1 - fetches and caches at time T
#         result1 = fs_instance.info(self.ngc_path)
#         mock_ngc_client_instance.get_file_info.assert_called_once()
#         assert self.stripped_path in fs_instance._info_cache

#         # Advance time past TTL (fs_instance has TTL=1s)
#         mock_time[0] += 2  # Advance time by 2 seconds

#         # Call 2 - should fetch again
#         new_info_data = {"size": 200, "type": "file", "name": self.ngc_path}
#         mock_ngc_client_instance.get_file_info.return_value = new_info_data
#         result2 = fs_instance.info(self.ngc_path)

#         assert mock_ngc_client_instance.get_file_info.call_count == 2
#         assert result2["size"] == 200  # Ensure new data is returned
#         assert result1 != result2

#     def test_info_refresh(self, fs_instance, mock_ngc_client_instance):
#         """Test info(refresh=True) bypasses the cache."""
#         file_info_data = {"size": 100, "type": "file", "name": self.ngc_path}
#         mock_ngc_client_instance.get_file_info.return_value = file_info_data
#         result1 = fs_instance.info(self.ngc_path)  # Call 1 - fetches
#         mock_ngc_client_instance.get_file_info.assert_called_once()
#         assert result1["size"] == 100

#         new_info_data = {"size": 200, "type": "file", "name": self.ngc_path}
#         mock_ngc_client_instance.get_file_info.return_value = new_info_data
#         result2 = fs_instance.info(self.ngc_path, refresh=True)  # Call 2 - refresh
#         assert mock_ngc_client_instance.get_file_info.call_count == 2
#         assert result2["size"] == 200


# class TestNGCFileSystemExistence:
#     """Tests for exists(), isdir(), isfile()."""

#     ngc_path = "ngc://test/path"
#     stripped_path = "test/path"

#     @patch.object(
#         NGCFileSystem, "info"
#     )  # Mock the info method directly for these tests
#     def test_exists_true(self, mock_info, fs_instance):
#         mock_info.return_value = {"name": self.ngc_path, "type": "file", "size": 100}
#         assert fs_instance.exists(self.ngc_path) is True
#         mock_info.assert_called_once_with(self.ngc_path)  # Check info was called

#     @patch.object(NGCFileSystem, "info")
#     def test_exists_false_not_found(self, mock_info, fs_instance):
#         mock_info.side_effect = FileNotFoundError(f"Path not found: {self.ngc_path}")
#         assert fs_instance.exists(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_exists_false_other_error(self, mock_info, fs_instance, caplog):
#         mock_info.side_effect = NGCException("Some other error")
#         with caplog.at_level(logging.WARNING):
#             assert fs_instance.exists(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)
#         assert (
#             f"Error checking existence for {self.ngc_path}: Some other error"
#             in caplog.text
#         )

#     @patch.object(NGCFileSystem, "info")
#     def test_isdir_true(self, mock_info, fs_instance):
#         mock_info.return_value = {
#             "name": self.ngc_path,
#             "type": "directory",
#             "size": None,
#         }
#         assert fs_instance.isdir(self.ngc_path) is True
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_isdir_false_for_file(self, mock_info, fs_instance):
#         mock_info.return_value = {"name": self.ngc_path, "type": "file", "size": 100}
#         assert fs_instance.isdir(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_isdir_false_not_found(self, mock_info, fs_instance):
#         mock_info.side_effect = FileNotFoundError()
#         assert fs_instance.isdir(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_isfile_true(self, mock_info, fs_instance):
#         mock_info.return_value = {"name": self.ngc_path, "type": "file", "size": 100}
#         assert fs_instance.isfile(self.ngc_path) is True
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_isfile_false_for_dir(self, mock_info, fs_instance):
#         mock_info.return_value = {
#             "name": self.ngc_path,
#             "type": "directory",
#             "size": None,
#         }
#         assert fs_instance.isfile(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)

#     @patch.object(NGCFileSystem, "info")
#     def test_isfile_false_not_found(self, mock_info, fs_instance):
#         mock_info.side_effect = FileNotFoundError()
#         assert fs_instance.isfile(self.ngc_path) is False
#         mock_info.assert_called_once_with(self.ngc_path)

#     # Test that underlying info cache is used
#     @patch.object(NGCFileSystem, "_fetch_info")  # Mock the actual fetcher this time
#     def test_existence_checks_use_info_cache(self, mock_fetch, fs_instance):
#         mock_fetch.return_value = {"name": self.ngc_path, "type": "file", "size": 100}
#         assert fs_instance.exists(self.ngc_path) is True
#         mock_fetch.assert_called_once()  # First call fetches
#         assert fs_instance.isfile(self.ngc_path) is True
#         mock_fetch.assert_called_once()  # Second call should use cache
#         assert fs_instance.isdir(self.ngc_path) is False
#         mock_fetch.assert_called_once()  # Third call should use cache


# class TestNGCFileSystemOpen:
#     """Tests for the _open() method."""

#     ngc_path = "ngc://my/org/file.bin"
#     stripped_path = "my/org/file.bin"

#     def test_open_success_bytes(self, fs_instance, mock_ngc_client_instance, caplog):
#         """Test _open when download_file returns bytes."""
#         file_content = b"This is file content."
#         mock_ngc_client_instance.download_file.return_value = file_content
#         with caplog.at_level(logging.WARNING):
#             f = fs_instance._open(self.ngc_path, mode="rb")

#         assert "Downloading entire file content into memory" in caplog.text
#         mock_ngc_client_instance.download_file.assert_called_once_with(
#             self.stripped_path
#         )
#         assert isinstance(f, io.BytesIO)
#         assert f.read() == file_content
#         f.close()  # Good practice

#     def test_open_success_seekable_stream(
#         self, fs_instance, mock_ngc_client_instance, caplog
#     ):
#         """Test _open when download_file returns a seekable stream."""
#         mock_stream = io.BytesIO(b"Stream content.")
#         mock_stream.seek(
#             0, io.SEEK_END
#         )  # Move pointer to end to check if seek(0) works
#         mock_ngc_client_instance.download_file.return_value = mock_stream

#         with caplog.at_level(logging.WARNING):
#             f = fs_instance._open(self.ngc_path, mode="rb")

#         assert (
#             "Downloading entire file content into memory" in caplog.text
#         )  # Still logs warning
#         mock_ngc_client_instance.download_file.assert_called_once_with(
#             self.stripped_path
#         )
#         assert f is mock_stream  # Should return the original stream
#         assert f.tell() == 0  # Should have been seeked to start
#         assert f.read() == b"Stream content."
#         f.close()

#     def test_open_success_non_seekable_stream(
#         self, fs_instance, mock_ngc_client_instance, caplog
#     ):
#         """Test _open when download_file returns a non-seekable stream."""
#         # Create a mock stream that lacks seek but has read
#         mock_stream = MagicMock(spec=io.RawIOBase)  # Use spec for attribute checking
#         mock_stream.read.return_value = b"Non-seekable."
#         mock_stream.seek = MagicMock(
#             side_effect=io.UnsupportedOperation
#         )  # Explicitly fail seek
#         # Simulate BinaryIO protocol partially
#         mock_stream.__enter__ = lambda: mock_stream
#         mock_stream.__exit__ = lambda exc_type, exc_val, exc_tb: None

#         mock_ngc_client_instance.download_file.return_value = mock_stream

#         with caplog.at_level(logging.WARNING):
#             f = fs_instance._open(self.ngc_path, mode="rb")

#         assert "Downloading entire file content into memory" in caplog.text
#         assert (
#             f"Stream for {self.ngc_path} is not seekable." in caplog.text
#         )  # Check seek warning
#         assert (
#             f"Stream for {self.ngc_path} might not fully implement BinaryIO."
#             in caplog.text
#         )  # Type warning
#         mock_ngc_client_instance.download_file.assert_called_once_with(
#             self.stripped_path
#         )
#         assert f is mock_stream
#         # Call read to ensure it works
#         assert f.read() == b"Non-seekable."
#         # mock_stream.close() # If close is mocked/needed

#     def test_open_not_found_error(self, fs_instance, mock_ngc_client_instance):
#         """Test _open raises FileNotFoundError."""
#         mock_ngc_client_instance.download_file.side_effect = NGCException(
#             "File not found"
#         )
#         with pytest.raises(FileNotFoundError, match=f"File not found: {self.ngc_path}"):
#             fs_instance._open(self.ngc_path, mode="rb")
#         mock_ngc_client_instance.download_file.assert_called_once_with(
#             self.stripped_path
#         )

#     def test_open_other_ngc_error(self, fs_instance, mock_ngc_client_instance):
#         """Test _open raises other NGCException."""
#         error_message = "Download quota exceeded"
#         mock_ngc_client_instance.download_file.side_effect = NGCException(error_message)
#         with pytest.raises(NGCException, match=error_message):
#             fs_instance._open(self.ngc_path, mode="rb")
#         mock_ngc_client_instance.download_file.assert_called_once_with(
#             self.stripped_path
#         )

#     def test_open_invalid_mode(self, fs_instance):
#         """Test _open raises ValueError for mode != 'rb'."""
#         with pytest.raises(ValueError, match="Only binary read mode .* is supported"):
#             fs_instance._open(self.ngc_path, mode="r")
#         with pytest.raises(ValueError):
#             fs_instance._open(self.ngc_path, mode="wb")

#     def test_open_unexpected_download_type(self, fs_instance, mock_ngc_client_instance):
#         """Test _open raises TypeError for unexpected download_file return type."""
#         mock_ngc_client_instance.download_file.return_value = (
#             12345  # Not bytes or stream
#         )
#         with pytest.raises(
#             TypeError, match=f"Unexpected type from download_file for {self.ngc_path}"
#         ):
#             fs_instance._open(self.ngc_path, mode="rb")


# class TestNGCFileSystemCat:
#     """Tests for the cat_file() method."""

#     ngc_path = "ngc://my/org/cat_file.txt"
#     file_content = b"Content for cat."

#     @patch.object(NGCFileSystem, "_open")  # Mock _open for cat tests
#     def test_cat_file_success(self, mock_open, fs_instance):
#         """Test successful cat_file operation."""
#         mock_file = io.BytesIO(self.file_content)
#         # Mock the context manager behavior of _open
#         mock_open.return_value.__enter__.return_value = mock_file
#         mock_open.return_value.__exit__.return_value = None

#         result = fs_instance.cat_file(self.ngc_path)

#         assert result == self.file_content
#         mock_open.assert_called_once_with(self.ngc_path, mode="rb")

#     def test_cat_file_with_start(self, fs_instance):
#         """Test cat_file raises NotImplementedError with start."""
#         with pytest.raises(NotImplementedError, match="Partial reads .* not supported"):
#             fs_instance.cat_file(self.ngc_path, start=10)

#     def test_cat_file_with_end(self, fs_instance):
#         """Test cat_file raises NotImplementedError with end."""
#         with pytest.raises(NotImplementedError, match="Partial reads .* not supported"):
#             fs_instance.cat_file(self.ngc_path, end=50)

#     @patch.object(NGCFileSystem, "_open")
#     def test_cat_file_not_found(self, mock_open, fs_instance):
#         """Test cat_file propagates FileNotFoundError from _open."""
#         mock_open.side_effect = FileNotFoundError(f"File not found: {self.ngc_path}")
#         with pytest.raises(FileNotFoundError):
#             fs_instance.cat_file(self.ngc_path)
#         mock_open.assert_called_once_with(self.ngc_path, mode="rb")

#     @patch.object(NGCFileSystem, "_open")
#     def test_cat_file_other_exception(self, mock_open, fs_instance):
#         """Test cat_file propagates other exceptions from _open."""
#         mock_open.side_effect = NGCException("Some API error")
#         with pytest.raises(NGCException):
#             fs_instance.cat_file(self.ngc_path)
#         mock_open.assert_called_once_with(self.ngc_path, mode="rb")


# class TestNGCFileSystemCacheClear:
#     """Tests for the _clear_cache() method."""

#     # Helper to populate caches
#     def _populate_caches(self, fs_instance):
#         fs_instance._info_cache = {"path1": {"type": "file"}, "path2": {"type": "dir"}}
#         fs_instance._info_cache_expiry = {
#             "path1": time.time() + 60,
#             "path2": time.time() + 60,
#         }
#         fs_instance.dircache = {
#             "ngc://dir1": [{"name": "ngc://dir1/file"}],
#             "ngc://dir2": [],
#         }

#     @patch("fsspec.utils.invalidate_cache")  # Mock fsspec utility if needed
#     def test_clear_cache_all(self, mock_invalidate, fs_instance):
#         """Test _clear_cache() clears all internal caches."""
#         self._populate_caches(fs_instance)
#         assert fs_instance._info_cache  # Ensure not empty before
#         assert fs_instance.dircache

#         fs_instance._clear_cache()

#         assert not fs_instance._info_cache
#         assert not fs_instance._info_cache_expiry
#         assert not fs_instance.dircache
#         # invalidate_cache should have been called on the dircache before clearing
#         mock_invalidate.assert_called_once_with(
#             fs_instance.dircache, None
#         )  # Check it was called

#     @patch("fsspec.utils.invalidate_cache")
#     def test_clear_cache_specific_path(self, mock_invalidate, fs_instance):
#         """Test _clear_cache(path) clears only entries for that path."""
#         self._populate_caches(fs_instance)
#         path_to_clear = "ngc://dir1"
#         stripped_path_to_clear = "path1"  # Assume path1 maps to dir1 for info cache key

#         # Add specific entry to info cache matching path_to_clear's stripped version
#         fs_instance._info_cache[stripped_path_to_clear] = {"type": "file"}
#         fs_instance._info_cache_expiry[stripped_path_to_clear] = time.time() + 60

#         fs_instance._clear_cache(path_to_clear)

#         # Info cache for the specific path should be gone
#         assert stripped_path_to_clear not in fs_instance._info_cache
#         assert stripped_path_to_clear not in fs_instance._info_cache_expiry
#         # Other info cache entries should remain
#         assert "path2" in fs_instance._info_cache

#         # Dircache entry for the specific path should be gone (via invalidate_cache)
#         mock_invalidate.assert_called_once_with(fs_instance.dircache, path_to_clear)
#         # NOTE: We only mock invalidate_cache, we don't check the dircache dict directly here
#         # unless invalidate_cache is configured to modify it in the mock.
#         # If invalidate_cache is not mocked, we would check:
#         # assert path_to_clear not in fs_instance.dircache
#         assert "ngc://dir2" in fs_instance.dircache  # Other dircache entries remain

import logging
import io
import time
from typing import Any, Dict, List, Optional, Union, BinaryIO

from fsspec.spec import AbstractFileSystem
from fsspec.utils import stringify_path, invalidate_cache
from fsspec.registry import register_implementation
from nvidia_ngc import NGCClient
from nvidia_ngc.errors import NGCApiKeyNotFound, NGCException


logger = logging.getLogger(__name__)


class NGCFileSystem(AbstractFileSystem):
    """Provides an fsspec interface to NVIDIA NGC.

    This filesystem allows accessing models, datasets, and other resources
    stored in NVIDIA's NGC registry using fsspec-compatible methods.

    Authentication is handled by the underlying `nvidia_ngc.NGCClient`,
    which typically uses the NGC CLI configuration or environment variables.

    Authentication Setup:
        The `NGCFileSystem` requires authentication with NVIDIA NGC. There are
        several ways to provide your NGC API key:

        1.  **NGC CLI Configuration (Recommended):** The easiest method is to configure
            the NGC CLI. If you haven't already, install the CLI and run:
            ```bash
            ngc config set
            ```
            Follow the prompts to enter your API key. The key will be stored securely
            (typically in `~/.ngc/config`), and `NGCFileSystem` (via the underlying
            `nvidia-ngc-client`) will automatically detect and use it. You might also
            need to configure your default organization (`ngc config set org <your_org>`)
            and team (`ngc config set team <your_team>`) if applicable.

        2.  **Environment Variable:** You can set the `NGC_API_KEY` environment variable:
            ```bash
            export NGC_API_KEY="<your_ngc_api_key_here>"
            ```
            The filesystem will pick up this variable if the CLI configuration is not found.

        3.  **Direct Instantiation (Less Common):** You can pass the key directly when
            creating an instance, though this is less common when relying on fsspec's
            automatic registration:
            ```python
            # Not the typical usage pattern with fsspec auto-registration
            fs = NGCFileSystem(api_key="<your_ngc_api_key_here>")
            ```

    Basic Usage (with fsspec):
        Once authentication is set up and this module is imported, `fsspec` can
        use the `ngc://` protocol directly:

        ```python
        import fsspec
        import nemo.common.ckpt.fs.ngc # Ensure the filesystem is registered

        # List files in an NGC model repository
        files = fsspec.ls("ngc://nvidia/nemo/megatron_gpt_345m/checkpoints")
        print(files)

        # Check if a file exists
        exists = fsspec.exists("ngc://nvidia/nemo/megatron_gpt_345m/config.yaml")
        print(f"Config exists: {exists}")

        # Open and read a file (Warning: downloads entire file to memory first!)
        try:
            with fsspec.open("ngc://nvidia/nemo/megatron_gpt_345m/tokenizer.model", "rb") as f:
                tokenizer_data = f.read(512) # Read first 512 bytes
                print(f"Read {len(tokenizer_data)} bytes from tokenizer.")
        except FileNotFoundError:
            print("Tokenizer file not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        ```

    Examples:
        >>> fs = NGCFileSystem()
        >>> fs.ls("ngc://nvidia/nemo/megatron_gpt_345m")
        # Lists files within the specified NGC resource (results cached)

        >>> fs.info("ngc://nvidia/nemo/megatron_gpt_345m/config.yaml")
        # Gets file metadata (result cached)

        >>> with fs.open("ngc://nvidia/nemo/megatron_gpt_345m/model_weights.ckpt", "rb") as f:
        >>>     # Warning: This reads the entire file into memory first!
        >>>     weights = f.read(1024) # Read first 1KB

    Attributes:
        client: An instance of `nvidia_ngc.NGCClient` used for API interactions.
        protocol (str): The protocol string used for this filesystem ("ngc").
    """

    protocol = "ngc"

    def __init__(
        self,
        api_key: Optional[str] = None,
        org: Optional[str] = None,
        team: Optional[str] = None,
        info_cache_ttl: int = 60,  # Add TTL for info cache (seconds)
        **kwargs: Any,
    ):
        """Initializes the NGCFileSystem.

        Args:
            api_key (Optional[str]): NGC API key. If None, the client attempts
                to find it via NGC CLI config or environment variables.
            org (Optional[str]): NGC organization. Defaults to the default organization
                configured in the NGC CLI if not provided.
            team (Optional[str]): NGC team. Defaults to no team context if not provided.
            info_cache_ttl (int): Time-to-live in seconds for the info cache. Defaults to 60.
            **kwargs: Additional keyword arguments passed to AbstractFileSystem.
        """
        super().__init__(**kwargs)
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._info_cache_expiry: Dict[str, float] = {}
        self._info_cache_ttl = info_cache_ttl
        self.dircache: Dict[str, List[Dict[str, Any]]] = {}  # Enable fsspec dircache

        try:
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if org:
                client_kwargs["org"] = org
            if team:
                client_kwargs["team"] = team
            self.client = NGCClient(**client_kwargs)
            logger.info("NGCClient initialized successfully.")
        except NGCApiKeyNotFound:
            logger.error(
                "NGC API Key not found. Please configure the NGC CLI or "
                "set the NGC_API_KEY environment variable."
            )
            raise
        except NGCException as e:
            logger.error(f"Failed to initialize NGCClient: {e}")
            raise

    def _strip_protocol(self, path: Union[str, List[str]]) -> Union[str, List[str]]:
        """Removes the 'ngc://' protocol prefix from a path or list of paths."""
        # Use inherited method for consistency if available, otherwise implement
        # return super()._strip_protocol(path) # If base class provides it reliably
        path = stringify_path(path)
        protocol_prefix = f"{self.protocol}://"
        if isinstance(path, list):
            return [
                p[len(protocol_prefix) :] if p.startswith(protocol_prefix) else p
                for p in path
            ]
        elif isinstance(path, str) and path.startswith(protocol_prefix):
            return path[len(protocol_prefix) :]
        return path

    def _clear_cache(self, path: Optional[str] = None):
        """Clears internal caches.

        Args:
            path: If provided, only clear cache entries related to this path.
                  If None, clear all caches.
        """
        if path:
            norm_path = self._strip_protocol(path)
            self._info_cache.pop(norm_path, None)
            self._info_cache_expiry.pop(norm_path, None)
            invalidate_cache(self.dircache, path)  # Use fsspec utility
        else:
            self._info_cache.clear()
            self._info_cache_expiry.clear()
            self.dircache.clear()

    def _fetch_info(self, ngc_path: str) -> Dict[str, Any]:
        """Internal method to fetch info, bypassing cache."""
        full_path = self.unstrip_protocol(ngc_path)
        try:
            # Prefer get_file_info as it might be more direct for files
            file_info = self.client.get_file_info(ngc_path)
            file_type = (
                "file" if file_info.get("size") is not None else "directory"
            )  # Heuristic

            info = {
                "name": full_path,
                "size": file_info.get("size"),
                "type": file_type,
                "lastModified": file_info.get("lastModified"),
                "id": file_info.get("id"),
                # Add other useful fields if available from get_file_info
            }
            logger.debug(f"Fetched info for {full_path} via get_file_info")
            return info

        except NGCException as e_info:
            # If get_file_info fails (e.g., path is a dataset/model root, or doesn't exist),
            # try list_files as a fallback to check for directory-like existence.
            # Handle specific 'not found' exceptions if the library provides them.
            # Example: except (ModelNotFoundException, DatasetNotFoundException, ...)
            if "not found" in str(e_info).lower():  # Generic check
                raise FileNotFoundError(f"Path not found: {full_path}") from e_info

            try:
                # Check if listing works - indicates a directory-like structure
                # Note: This still makes a second API call in this fallback case.
                files = self.client.list_files(ngc_path)
                # If list_files succeeds, even if empty, treat as a directory.
                logger.debug(
                    f"Path {full_path} behaves like a directory (list_files succeeded)."
                )
                return {
                    "name": full_path,
                    "size": None,
                    "type": "directory",
                    # Add other relevant details if list_files provides them
                }
            except NGCException as e_list:
                logger.error(
                    f"Failed to get info for {full_path}: Primary error={e_info}, Fallback list error={e_list}"
                )
                # Prefer raising the original error if listing also fails
                if "not found" in str(e_list).lower():
                    raise FileNotFoundError(f"Path not found: {full_path}") from e_list
                raise e_info  # Re-raise the original, potentially more specific, error

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Get information about a single file or resource in NGC.

        Results are cached in memory based on `info_cache_ttl`.

        Args:
            path (str): The NGC path.
            **kwargs: Refresh cache if kwargs['refresh'] is True.

        Returns:
            Dict[str, Any]: File information (name, size, type, etc.).

        Raises:
            FileNotFoundError: If the path does not exist.
            NGCException: For other NGC API errors.
        """
        refresh = kwargs.pop("refresh", False)
        ngc_path = self._strip_protocol(path)
        now = time.time()

        if (
            not refresh
            and ngc_path in self._info_cache
            and now < self._info_cache_expiry.get(ngc_path, 0)
        ):
            logger.debug(f"Using cached info for {path}")
            return self._info_cache[ngc_path].copy()  # Return copy to prevent mutation

        logger.debug(f"Fetching info for {path} (refresh={refresh})")
        info_result = self._fetch_info(ngc_path)  # Calls the actual API

        # Update cache
        self._info_cache[ngc_path] = info_result
        self._info_cache_expiry[ngc_path] = now + self._info_cache_ttl

        return info_result.copy()  # Return copy

    def ls(
        self, path: str, detail: bool = False, refresh: bool = False, **kwargs: Any
    ) -> List[Union[str, Dict[str, Any]]]:
        """List files and directories at the given NGC path.

        Uses fsspec's dircache for caching.

        Args:
            path (str): The NGC path (e.g., "ngc://org/team/model:version").
            detail (bool): If True, returns dictionaries with file details.
                           If False, returns file names (full paths).
            refresh (bool): If True, bypass the cache and fetch fresh listing.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            List[Union[str, Dict[str, Any]]]: List of file names or dictionaries.

        Raises:
            FileNotFoundError: If the path does not exist in NGC.
            NGCException: For other NGC API errors.
        """
        ngc_path = self._strip_protocol(path)

        # Use fsspec's caching mechanism
        if not refresh and path in self.dircache:
            logger.debug(f"Using dircache for {path}")
            cached_result = self.dircache[path]
            # Ensure detail level matches request
            if detail:
                return cached_result  # Assuming dircache stores detailed results
            else:
                return [d["name"] for d in cached_result]

        logger.debug(f"Fetching ls for {path} (refresh={refresh})")
        try:
            # Note: list_files might only list files within a resource version.
            files = self.client.list_files(ngc_path)
            # No need for the extra exists() check here. If list_files fails
            # with a "not found", the exception handling below catches it.
            # If it succeeds with an empty list, it's an empty resource.

        except NGCException as e:
            logger.error(f"Failed to list files for {path}: {e}")
            # Example: Catch specific exceptions if available
            # except (ModelNotFoundException, DatasetNotFoundException) as e_nf:
            #     raise FileNotFoundError(f"Path not found: {path}") from e_nf
            if "not found" in str(e).lower():
                raise FileNotFoundError(f"Path not found: {path}") from e
            raise  # Re-raise other NGC errors

        # Construct fsspec-compatible detail dictionaries
        detailed_files = []
        for f in files:
            full_path = self.unstrip_protocol(f["path"])
            # Try to get size from list_files, default to None
            size = f.get("size")
            # If size is missing, *maybe* call info for that specific file? Costly!
            # For now, leave size as potentially None from ls.
            info_dict = {
                "name": full_path,
                "size": size,
                "type": "file",  # Assume file from list_files
                "lastModified": f.get("lastModified"),
            }
            detailed_files.append(info_dict)

            # Optionally update info cache from ls results if trustworthy
            # file_ngc_path = self._strip_protocol(full_path)
            # if file_ngc_path not in self._info_cache or time.time() >= self._info_cache_expiry.get(file_ngc_path, 0):
            #    self._info_cache[file_ngc_path] = info_dict
            #    self._info_cache_expiry[file_ngc_path] = time.time() + self._info_cache_ttl

        # Update dircache (always store detailed version)
        self.dircache[path] = detailed_files

        if detail:
            return detailed_files
        else:
            return [d["name"] for d in detailed_files]

    def exists(self, path: str, **kwargs: Any) -> bool:
        """Check if a path exists in NGC. Uses cached info."""
        try:
            self.info(path, **kwargs)  # Leverage info() which uses caching
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(
                f"Error checking existence for {path}: {e}", exc_info=False
            )  # Less noisy log
            return False

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Optional[int] = None,  # Keep for fsspec compatibility
        autocommit: bool = True,  # Keep for fsspec compatibility
        cache_options: Optional[Dict[str, Any]] = None,  # Keep for fsspec compatibility
        **kwargs: Any,
    ) -> BinaryIO:
        """Open an NGC file for reading (binary mode only).

        WARNING: This currently downloads the ENTIRE file into memory first due
        to limitations in the underlying `NGCClient.download_file` (as assumed).
        This is inefficient for large files. Check for future `nvidia-ngc-client`
        updates that might support streaming downloads.

        Args:
            path (str): The NGC file path.
            mode (str): Must be "rb".
            block_size: Ignored by this implementation.
            autocommit: Ignored by this implementation (read-only).
            cache_options: Ignored by this implementation (fsspec handles caching externally).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            BinaryIO: A file-like object (typically `io.BytesIO`) containing the downloaded file content.

        Raises:
            ValueError: If mode is not "rb".
            FileNotFoundError: If the file does not exist.
            NGCException: For NGC API errors during download.
        """
        if mode != "rb":
            raise ValueError("Only binary read mode ('rb') is supported for NGC.")

        ngc_path = self._strip_protocol(path)
        try:
            # === CRITICAL PERFORMANCE BOTTLENECK ===
            # download_file likely reads the whole file into memory.
            # Investigate nvidia_ngc.NGCClient for streaming alternatives.
            # If a streaming method exists, replace this section.
            logger.warning(
                f"Opening {path}: Downloading entire file content into memory."
            )
            file_content_or_stream = self.client.download_file(ngc_path)
            logger.debug(f"Downloaded file {path} from NGC.")

            # Ensure we return a consistent BinaryIO interface
            if isinstance(file_content_or_stream, bytes):
                file_obj = io.BytesIO(file_content_or_stream)
                logger.debug(f"Wrapped downloaded bytes for {path} in io.BytesIO.")
                return file_obj
            elif hasattr(file_content_or_stream, "read"):
                # If it's already file-like (e.g., could be tempfile from client)
                if hasattr(file_content_or_stream, "seek"):
                    try:
                        file_content_or_stream.seek(0)
                    except io.UnsupportedOperation:
                        logger.warning(f"Stream for {path} is not seekable.")
                logger.debug(f"Returning existing file-like object for {path}.")
                # Ensure it conforms to BinaryIO protocol if possible
                if not isinstance(file_content_or_stream, BinaryIO):
                    logger.warning(
                        f"Stream for {path} might not fully implement BinaryIO."
                    )
                return file_content_or_stream  # type: ignore[return-value]
            else:
                # Should not happen based on expected NGCClient behavior
                raise TypeError(
                    f"Unexpected type from download_file for {path}: {type(file_content_or_stream)}"
                )

        except NGCException as e:
            logger.error(f"Failed to open/download file {path}: {e}")
            # Example: Catch specific exceptions if available
            # except (ModelNotFoundException, ...) as e_nf:
            if "not found" in str(e).lower():
                raise FileNotFoundError(f"File not found: {path}") from e
            raise

    def cat_file(
        self,
        path: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> bytes:
        """Read the complete content or a byte range of a file from NGC.

        Note: Byte range requests (start/end) are currently NOT supported due
        to the underlying download mechanism fetching the whole file. Providing
        start/end will raise NotImplementedError.

        Args:
            path (str): The NGC file path.
            start (Optional[int]): Start byte position (NOT SUPPORTED).
            end (Optional[int]): End byte position (NOT SUPPORTED).
            **kwargs: Additional arguments passed to `_open`.

        Returns:
            bytes: The content of the file.

        Raises:
            NotImplementedError: If start or end arguments are provided.
            FileNotFoundError: If the file does not exist.
            NGCException: For underlying API errors.
        """
        if start is not None or end is not None:
            logger.error("Partial reads with start/end are not supported for NGC.")
            raise NotImplementedError(
                "Partial reads with start/end are not supported for NGCFileSystem."
            )

        # _open already downloads the full content, so just read it all.
        with self._open(path, mode="rb", **kwargs) as f:
            return f.read()  # Reads from the in-memory BytesIO

    def isdir(self, path: str) -> bool:
        """Check if the path is a directory-like entity in NGC. Uses cached info."""
        try:
            # Leverage info() which uses caching
            return self.info(path)["type"] == "directory"
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(
                f"Error checking isdir for {path}: {e}", exc_info=False
            )  # Less noisy log
            return False

    def isfile(self, path: str) -> bool:
        """Check if the path is a file in NGC. Uses cached info."""
        try:
            # Leverage info() which uses caching
            return self.info(path)["type"] == "file"
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(
                f"Error checking isfile for {path}: {e}", exc_info=False
            )  # Less noisy log
            return False


register_implementation("ngc", NGCFileSystem)

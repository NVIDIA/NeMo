from typing import Optional

from loguru import logger
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.websocket_server import (
    WebsocketServerCallbacks,
    WebsocketServerInputTransport,
    WebsocketServerOutputTransport,
    WebsocketServerParams,
)

from nemo.collections.voice_agent.pipecat.transports.base_transport import TransportParams

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    """Configuration parameters for WebSocket server transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for message encoding/decoding.
        session_timeout: Timeout in seconds for client sessions.
    """

    add_wav_header: bool = False
    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None


class WebsocketServerTransport(BaseTransport):
    """WebSocket server transport for bidirectional real-time communication.

    Provides a complete WebSocket server implementation with separate input and
    output transports, client connection management, and event handling for
    real-time audio and data streaming applications.
    """

    def __init__(
        self,
        params: WebsocketServerParams,
        host: str = "localhost",
        port: int = 8765,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the WebSocket server transport.

        Args:
            params: WebSocket server configuration parameters.
            host: Host address to bind the server to. Defaults to "localhost".
            port: Port number to bind the server to. Defaults to 8765.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: Optional[WebsocketServerInputTransport] = None
        self._output: Optional[WebsocketServerOutputTransport] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> WebsocketServerInputTransport:
        """Get the input transport for receiving client data.

        Returns:
            The WebSocket server input transport instance.
        """
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = WebsocketServerOutputTransport(self, self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connection events."""
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnection events."""
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_session_timeout(self, websocket):
        """Handle client session timeout events."""
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        """Handle WebSocket server ready events."""
        await self._call_event_handler("on_websocket_ready")

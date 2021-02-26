# NMT Server Getting Started
1. Install latest grpc (pip install grpc grpcio-tools)
2. Create a models directory and copy nemo models there. Note that models should be named `<source>-<target>.nemo`
   where `<source>` and `<target>` are both two letter language codes, e.g. `en-es.nemo`.
3. Start the server, explicitly loading each model: 
   ```
   python server.py --model models/en-es.nemo --model models/es-en.nemo --model models/En-Ja.nemo
   ```

## Notes
Port can be overridden with `--port` flag. Default is 50052. Beam decoder parameters can also be set at server start time. See `--help` for more details.

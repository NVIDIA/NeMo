import onnx
import onnx_graphsurgeon as gs

def main():
    onnx_model = onnx.load("/home/dgalvez/scratch/code/asr/nemo_conformer_benchmark/NeMo/examples/asr/export/transducer/encoder-temp_rnnt.onnx")
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        if "layers.0" not in node.name and "pre_encode" not in node.name:
            print("Remove:", node.name)

    graph.cleanup()
    graph.toposort()

if __name__ == "__main__":
    main()

# Try using polygraphy surgeon prune, with the ouput being... "%/layers.1/norm_out/LayerNormalization_output_0"

# polygraphy surgeon extract /home/dgalvez/scratch/code/asr/nemo_conformer_benchmark/NeMo/examples/asr/export/transducer/encoder-temp_rnnt.onnx --inputs audio_signal:auto:auto,length:auto:auto --outputs /layers.1/norm_out/LayerNormalization_output_0:auto -o encoder_subgraph.onnx

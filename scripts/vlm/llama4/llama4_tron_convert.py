from nemo.tron.converter.llama import HFLlamaImporter

if __name__ == '__main__':
    hf_model_name = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    output_path = f"/opt/checkpoints/tron/{hf_model_name}"
    print(f"Importing model {hf_model_name} to {output_path}...")
    importer = HFLlamaImporter(
        hf_model_name,
        output_path=f"/opt/checkpoints/tron/{hf_model_name}",
    )
    importer.apply()
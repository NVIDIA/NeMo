#!/bin/bash

# Directory containing the dataset
DATASET_DIR="./FluxBrushNetDataset_train"
OUTPUT_DIR="./train_files"
TARFILE_DIR="./tarfiles_train"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TARFILE_DIR"

# Define the extensions to include
EXTENSIONS=(".new_caption" ".original_key" ".pooled_prompt_embeds" ".prompt_embeds" ".text_ids" ".caption" ".filtered_segmentation" ".aesthetic_score" ".url" ".jpg")

MIN_VALUE=0 #Index of Image file to resume packing
START_INDEX=0 #Index of jpg files to start searching for the resuming point

# Find the last completed tar index and resume from the last processed single mask
last_tar=$(ls "$TARFILE_DIR"/batch_*.tar 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -n 1 | sed 's/^0*//')
tar_index=$((last_tar + 1))
processed_files=$(( (last_tar) * 1000 ))  # Calculate number of processed single mask files
echo $last_tar is found
# Collect all .jpg files as sample identifiers
mapfile -t JPG_FILES < <(find "$DATASET_DIR" -type f -name "*.jpg" | sort)

file_count=0
temp_list=()


# Start loop from specified index
for ((i=START_INDEX; i<${#JPG_FILES[@]}; i++)); do
    jpg="${JPG_FILES[i]}"
    sample_name="$(basename "$jpg" .jpg)"

    # Extract numerical part and check against MIN_VALUE
    num_value=$(echo "$sample_name" | grep -o -E '[0-9]+' | head -n1 | sed 's/^0*//')
    if [[ -n "$num_value" && "$num_value" -lt "$MIN_VALUE" ]]; then
	echo skipped $sample_name
        continue
    fi

    echo $sample_name

    mask_files=("$DATASET_DIR/$sample_name.single_mask_*")
    mask_index=0

    for mask in ${mask_files[@]}; do
        if [[ -f "$mask" ]]; then
            # Skip already processed files when resuming
            #if [[ $processed_files -gt 0 ]]; then
		#echo $processed_files
                #((processed_files--))
                #continue
            #fi

            new_mask_name="$OUTPUT_DIR/${sample_name}_$mask_index.single_mask"
            cp "$mask" "$new_mask_name"
            sample_files=("$new_mask_name")

            for ext in "${EXTENSIONS[@]}"; do
                file="$DATASET_DIR/$sample_name$ext"
                if [[ -f "$file" ]]; then
                    new_file="$OUTPUT_DIR/${sample_name}_$mask_index$ext"
                    cp "$file" "$new_file"
                    sample_files+=("$new_file")
                fi
            done

            temp_list+=("${sample_files[@]}")
            ((mask_index++))
            ((file_count++))

            # Create a tar file every 1000 files
            if [[ $file_count -eq 1000 ]]; then
                tar -cf "$TARFILE_DIR/batch_$(printf "%04d" $tar_index).tar" "${temp_list[@]}"
		echo tarfile $tar_index finished
                ((tar_index++))
                file_count=0
                temp_list=()
            fi
        fi
    done

done

# Tar remaining files if any exist
if [[ ${#temp_list[@]} -gt 0 ]]; then
    tar -cf "$TARFILE_DIR/batch_$(printf "%04d" $tar_index).tar" "${temp_list[@]}"
fi
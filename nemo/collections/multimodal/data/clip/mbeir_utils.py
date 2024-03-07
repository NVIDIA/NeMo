import json
import os
from multiprocessing import Lock, Manager, Pool, cpu_count

from PIL import Image
from torchvision import transforms

# Mapping of dataset names to IDs
DATASET_IDS = {
    "VisualNews": 0,
    "Fashion200K": 1,
    "WebQA": 2,
    "EDIS": 3,
    "NIGHTS": 4,
    "OVEN": 5,
    "INFOSEEK": 6,
    "FashionIQ": 7,
    "CIRR": 8,
    "MSCOCO": 9,
}

MBEIR_TASK = {
    "text -> image": 0,
    "text -> text": 1,
    "text -> image,text": 2,
    "image -> text": 3,
    "image -> image": 4,
    "image -> text,image": 5,  # This is not a valid task For now, we will ignore this task
    "image,text -> text": 6,
    "image,text -> image": 7,
    "image,text -> image,text": 8,
}

MBEIR_DATASET_TO_DOMAIN = {
    "VisualNews": "news",
    "Fashion200K": "fashion",
    "WebQA": "wiki",
    "EDIS": "news",
    "NIGHTS": "common",
    "OVEN": "wiki",
    "INFOSEEK": "wiki",
    "FashionIQ": "fashion",
    "CIRR": "common",
    "MSCOCO": "common",
}

IMAGE_SHORT_SIDE = 256  # The shortest side of the image is resized to this size

DATASET_CAN_NUM_UPPER_BOUND = 10000000  # Maximum number of candidates per dataset
DATASET_QUERY_NUM_UPPER_BOUND = 500000  # Maximum number of queries per dataset

# Define the resize transform
resize_transform = transforms.Resize(IMAGE_SHORT_SIDE)


def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id


def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id


def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


def get_dataset_id(dataset_name):
    """Get the dataset ID from the dataset name."""
    return DATASET_IDS.get(dataset_name, None)


def get_dataset_name(id):
    dataset_id = int(id.split(":")[0])
    for name, id_ in DATASET_IDS.items():
        if id_ == dataset_id:
            return name
    return None


def get_mbeir_task_id(source_modality, target_modality):
    """Get the MBEIR task ID using source and target modalities."""
    task_name = f"{source_modality} -> {target_modality}"
    return MBEIR_TASK.get(task_name, None)


def get_mbeir_task_name(task_id):
    """Get the MBEIR task name from the task ID."""
    for name, id_ in MBEIR_TASK.items():
        if id_ == task_id:
            return name
    return None


def get_mbeir_query_modality_cand_modality_from_task_id(task_id):
    for name, id_ in MBEIR_TASK.items():
        if id_ == task_id:
            return name.split(" -> ")


def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s


def resize_and_convert_image_to_jpg(image_path):
    try:
        with Image.open(image_path) as img:
            # Check for palette image and convert to RGBA if necessary (like some GIF images)
            if img.mode == "P":
                img = img.convert("RGBA")

            img = img.convert("RGB")  # Convert the image to RGB format (necessary for GIF and other non-RGB formats)

            # The shortest side is resized to IMAGE_SHORT_SIDE pixels. So we can apply data augmentation later
            img_resized = resize_transform(img)

            # Convert and save the image in JPG format
            save_path = os.path.splitext(image_path)[0] + ".jpg"
            img_resized.save(save_path, "JPEG")

            # Delete the original image after processing
            if save_path != image_path:
                os.remove(image_path)
            return True

    except Exception as e:
        print(f"Error processing {image_path}. Invalid or corrupted image.  Message: {e}")
        os.remove(image_path)
        print(f"Removed {image_path}")
        return False


def is_valid_image(img_path):
    """Check if the image at the given path is valid, can be opened, and is in JPG format."""
    try:
        # Try opening the image
        with Image.open(img_path) as img:
            if img.format != "JPEG":
                print(f"Image at {img_path} is not in JPEG format. Found: {img.format}")
                return False
            return True
    except Exception as e:
        print(f"Invalid or corrupted image at {img_path}: {e}")
        return False


def parallel_process_image_file(image_path):
    success = resize_and_convert_image_to_jpg(image_path)
    if not success:
        return 1
    return 0


def parallel_process_image_directory(images_dir, num_processes=cpu_count()):
    """
    Resize and convert all images in the given directory to JPG format.
    This function will delete corrupt images and original images after resizing.
    Multiple processes are used to speed up the process.

    Here is a sample directory structure and os.walk() will traverse the directory recursively:
    ├── images_dir
    │   ├── xxx.jpg
    │   ├── xxx.png
    │   ├── folder 1
    │   │   ├── xxx.jpg
    │   │   ├── xxx.png
    │   │   ├── folder 2
    │   │   │   ├── ...
    """
    all_image_paths = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, file))
    print(f"Number of images found: {len(all_image_paths)}")

    # Create a Pool of workers, defaulting to one per CPU core
    with Pool(num_processes) as p:
        results = p.map(parallel_process_image_file, all_image_paths)
        corrupted_files_count = sum(results)
    print(f"Number of corrupted files: {corrupted_files_count}")


def save_list_as_jsonl(data, filename, mode="w"):
    with open(filename, mode) as file:
        for entry in data:
            json_str = json.dumps(entry)
            file.write(json_str + "\n")


def load_jsonl_as_list(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            entry = json.loads(line.strip())  # strip() removes the newline character
            data.append(entry)
    return data


def aggregate_candidates_for_mbeir_format_dataset(data, print_duplicate=True):
    mapping = {}
    for entry in data:
        query_txt = entry["query_txt"]
        query_img_path = entry["query_img_path"]
        query_modality = entry["query_modality"]
        pos_cand_list = entry["pos_cand_list"]
        neg_cand_list = entry["neg_cand_list"]
        key = f"{query_txt}-{query_img_path}-{query_modality}"

        if key not in mapping:
            mapping[key] = {
                "qid": entry["qid"],
                "query_txt": query_txt,
                "query_img_path": query_img_path,
                "query_modality": query_modality,
                "query_src_content": entry["query_src_content"],
                "pos_cand_list": [],
                "neg_cand_list": [],
            }
        else:
            if print_duplicate:
                print(f"Duplicate entry found:")
                print(f"\tKey: {key}")
                print(f"\tEntry 1: {mapping[key]}")
                print(f"\tEntry 2: {entry}")
        # Check duplicates in pos_cand_list
        for did in pos_cand_list:
            if did not in mapping[key]["pos_cand_list"]:
                mapping[key]["pos_cand_list"].append(did)
            else:
                if print_duplicate:
                    print(f"Duplicate positive candidate found: {did}")
        # Check duplicates in neg_cand_list
        for did in neg_cand_list:
            if did not in mapping[key]["neg_cand_list"]:
                mapping[key]["neg_cand_list"].append(did)
            else:
                if print_duplicate:
                    print(f"Duplicate negative candidate found: {did}")
    return list(mapping.values())


def count_entries_in_file(filename):
    """Load and return data from a JSON or JSONL file based on file extension."""
    if filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))
    elif filename.endswith(".json"):
        with open(filename, "r") as file:
            data = json.load(file)
    else:
        raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")
    return len(data), data


def count_mbeir_format_pool_entries_based_on_modality(filename):
    """Load pool data from a JSONL file and count based on 'modality' field."""
    data = []
    assert filename.endswith(".jsonl"), "Only JSONL files are supported."
    with open(filename, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    total_entries = len(data)

    # Count based on 'modality' field
    modality_count = {}
    for entry in data:
        modality = entry.get("modality", None)
        if modality:
            modality_count[modality] = modality_count.get(modality, 0) + 1
    return total_entries, modality_count, data


def check_duplicates_in_mbeir_format_cand_pool(data, print_duplicate=True):
    """Check and return duplicate entries based on text, image, and modality."""
    # Check ids are unique
    did_set = set()
    seen_candidates = {}
    duplicates = []
    for entry in data:
        did = entry.get("did")
        assert did not in did_set, f"Duplicate id found: {did}"
        did_set.add(did)
        doc_key = generate_mbeir_format_doc_key(entry)
        if doc_key in seen_candidates:
            seen_cand = seen_candidates[doc_key]
            if print_duplicate:
                print(f"Duplicate candidate found: {entry} and {seen_cand}")
            duplicates.append(entry)
        seen_candidates[doc_key] = entry
    return duplicates


def check_duplicates(data, attribute):
    """Check and return duplicate entries based on the specified attribute."""
    seen_values = set()
    duplicates = []

    for entry in data:
        value = entry.get(attribute, None)

        # Skip entries with None or empty values for the attribute
        if not value:
            continue

        if value in seen_values:
            duplicates.append(entry)
        seen_values.add(value)

    return duplicates


def generate_mbeir_format_doc_key(entry):
    """Output: txt-img_path-modality"""
    txt = entry.get("txt", "")
    img_path = entry.get("img_path", "")
    modality = entry.get("modality", "")
    assert txt or img_path, f"Either txt or img_path must be present.{entry}"

    doc_key_parts = [part for part in [txt, img_path, modality] if part]
    doc_key = "-".join(doc_key_parts)
    return doc_key


def load_mbeir_format_pool_file_as_dict(pool_file_path, doc_key_to_content=False, key_type="mbeir_converted_key"):
    """
    Load the candidate pool file into a dictionary.
    {doc_key : did} or {doc_key : entry}
    """
    pool_dict = {}
    assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."

    with open(pool_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())

            if key_type == "mbeir_converted_key":
                doc_key = generate_mbeir_format_doc_key(entry)
            elif key_type == "did":
                doc_key = entry["did"]
            else:
                raise ValueError(f"Invalid key_type: {key_type}")

            assert doc_key not in pool_dict, f"Duplicate doc_key found: {doc_key}"

            if doc_key_to_content:
                pool_dict[doc_key] = entry
            else:
                pool_dict[doc_key] = entry["did"]
    return pool_dict


def load_mbeir_format_query_file_as_dict(query_file_path, doc_key_to_content=False, key_type="mbeir_converted_key"):
    """
    Load the query data file into a dictionary.
    {qid : did} or {qid : entry}
    """
    query_dict = {}
    assert query_file_path.endswith(".jsonl"), "Only JSONL files are supported."

    with open(query_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())

            if key_type == "mbeir_converted_key":
                doc_key = generate_mbeir_format_doc_key(entry)
            elif key_type == "qid":
                doc_key = entry["qid"]
            else:
                raise ValueError(f"Invalid key_type: {key_type}")

            assert doc_key not in query_dict, f"Duplicate doc_key found: {doc_key}"

            if doc_key_to_content:
                query_dict[doc_key] = entry
            else:
                query_dict[doc_key] = entry["qid"]
    return query_dict


def get_modality_stats(data, cand_pool_dict):
    modality_combinations = [
        "text-image",
        "text-text",
        "text-image,text",
        "image-text",
        "image-image",
        "image,text-text",
        "image,text-image",
        "image,text-image,text",
    ]

    modality_counts = {
        "pos": {
            comb: {"count": 0, "queries": [], "unique_queries": set(), "unique_dids": set()}
            for comb in modality_combinations
        },
        "neg": {
            comb: {"count": 0, "queries": [], "unique_queries": set(), "unique_dids": set()}
            for comb in modality_combinations
        },
    }

    # Initialize counts for computing average word lengths
    total_query_txt_words = 0
    total_pos_txt_words = 0
    total_neg_txt_words = 0
    total_entries = 0  # For computing the average
    total_pos_cand_count = 0
    total_neg_cand_count = 0
    total_query_question_type_count = {qtype: 0 for qtype in ["String", "Numerical", "Time"]}  # For INFOSEEK dataset

    for entry in data:
        qid = entry["qid"]
        query_modality = entry["query_modality"]

        # Count words for query_txt
        query_txt = entry.get("query_txt") or ""
        total_query_txt_words += len(query_txt.split())
        total_entries += 1

        # Count the candidates for pos_cand_list and neg_cand_list
        total_pos_cand_count += len(entry.get("pos_cand_list", []))
        total_neg_cand_count += len(entry.get("neg_cand_list", []))

        # For INFOSEEK dataset
        query_src_content = entry.get("query_src_content", None)
        if query_src_content:
            query_src_content = json.loads(query_src_content)
        query_question_type = query_src_content.get("question_type", None) if query_src_content else None
        if query_question_type:
            total_query_question_type_count[query_question_type] += 1

        for cand_type in ["pos_cand_list", "neg_cand_list"]:
            translated_cand_type = "pos" if cand_type == "pos_cand_list" else "neg"
            cand_list = entry.get(cand_type, [])

            golden_task_modality = ""
            for idx, did in enumerate(cand_list):
                cand = cand_pool_dict[did]
                cand_modality = cand["modality"]
                combined_modality = query_modality + "-" + cand_modality

                if cand_type == "pos_cand_list":
                    if idx == 0:
                        golden_task_modality = combined_modality
                    else:
                        assert (
                            golden_task_modality == combined_modality
                        ), "Golden task modality does not match with the combined modality"

                if cand_type == "pos_cand_list":
                    cand_txt = cand.get("txt") or ""
                    total_pos_txt_words += len(cand_txt.split())
                elif cand_type == "neg_cand_list":
                    cand_txt = cand.get("txt") or ""
                    total_neg_txt_words += len(cand_txt.split())

                # Check if combined_modality exists in our dictionary
                if combined_modality in modality_counts[translated_cand_type]:
                    modality_counts[translated_cand_type][combined_modality]["count"] += 1
                    modality_counts[translated_cand_type][combined_modality]["unique_queries"].add(qid)
                    modality_counts[translated_cand_type][combined_modality]["unique_dids"].add(cand["did"])

                    if idx == 0:
                        modality_counts[translated_cand_type][combined_modality]["queries"].append(qid)

    stats = {
        "pos": {
            "examples": {
                modality: data["count"] for modality, data in modality_counts["pos"].items() if data["count"] > 0
            },
            "queries": {
                modality: len(data["queries"])
                for modality, data in modality_counts["pos"].items()
                if len(data["queries"]) > 0
            },
            "unique_queries": {
                modality: len(data["unique_queries"])
                for modality, data in modality_counts["pos"].items()
                if len(data["unique_queries"]) > 0
            },
            "unique_docs": {
                modality: len(data["unique_dids"])
                for modality, data in modality_counts["pos"].items()
                if len(data["unique_dids"]) > 0
            },
            "avg_pos_cand_count": total_pos_cand_count / total_entries,
            "avg_txt_words": total_pos_txt_words / total_pos_cand_count,
        },
        "neg": {
            "examples": {
                modality: data["count"] for modality, data in modality_counts["neg"].items() if data["count"] > 0
            },
            "queries": {
                modality: len(data["queries"])
                for modality, data in modality_counts["neg"].items()
                if len(data["queries"]) > 0
            },
            "unique_queries": {
                modality: len(data["unique_queries"])
                for modality, data in modality_counts["neg"].items()
                if len(data["unique_queries"]) > 0
            },
            "unique_docs": {
                modality: len(data["unique_dids"])
                for modality, data in modality_counts["neg"].items()
                if len(data["unique_dids"]) > 0
            },
            "avg_neg_cand_count": total_neg_cand_count / total_entries,
            "avg_txt_words": total_neg_txt_words / total_neg_cand_count if total_neg_cand_count > 0 else 0,
        },
        "avg_query_txt_words": total_query_txt_words / total_entries,
    }

    # For INFOSEEK dataset
    if any(total_query_question_type_count[key] > 0 for key in ["String", "Numerical", "Time"]):
        stats["total_query_question_type_count"] = total_query_question_type_count
    return stats


def print_mbeir_format_dataset_stats(data, cand_pool_dict):
    stats = get_modality_stats(data, cand_pool_dict)
    for category, data in stats.items():
        if category == "avg_query_txt_words":  # Handle 'avg_query_txt_words' separately
            print(f"--- {category.upper()} ---")
            print(f"\t{data:.1f}")
            continue
        print(f"--- {category.upper()} ---")
        for section, values in data.items():
            print(f"{section}:")
            if isinstance(values, dict):  # Check if the value is a dictionary
                for modality, count in values.items():
                    print(f"\t{modality}: {count}")
            else:
                print(f"\t{values:.1f}")  # Print the value as a float with 2 decimal places


def print_mbeir_format_cand_pool_stats(candidate_pool_path, print_duplicate=True):
    total_entries, modality_counts, _data = count_mbeir_format_pool_entries_based_on_modality(candidate_pool_path)
    print(f"Total number of entries in {candidate_pool_path}: {total_entries}")
    print(f"Modality counts: {modality_counts}")
    duplicates = check_duplicates_in_mbeir_format_cand_pool(_data, print_duplicate=print_duplicate)
    print(f"Number of duplicates: {len(duplicates)}")
    if duplicates:
        print(f"Sample duplicates: {duplicates[:5]}")


def save_and_print_mbeir_format_dataset_stats(data, data_file_path, cand_pool_file_path):
    # Save to the new JSONL file
    save_list_as_jsonl(data, data_file_path)

    # Print stats
    total_entries, _data = count_entries_in_file(data_file_path)
    print(f"Saved dataset to {data_file_path}")
    print(f"Total number of entries in {data_file_path}: {total_entries}")
    assert os.path.exists(cand_pool_file_path), f"File {cand_pool_file_path} does not exist"
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(cand_pool_file_path, doc_key_to_content=True, key_type="did")
    print_mbeir_format_dataset_stats(_data, cand_pool_dict)

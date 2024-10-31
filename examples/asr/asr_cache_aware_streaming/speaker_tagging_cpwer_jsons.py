import json
import argparse
import logging

def process_session_data(hyp_json_path, src_json_path):
    hyp_json_lines = open(hyp_json_path).readlines()
    src_json_lines = open(src_json_path).readlines()
    # sessions = json.loads(json_data_path)
    hyp_sessions = json.loads("".join(hyp_json_lines))
    src_sessions = json.loads("".join(src_json_lines))
    total_count = len(hyp_sessions)
    total_improved = 0.0
    total_non_improve = 0.0
    improved_count = 0
    for session_id, session_info in hyp_sessions.items():
        src_cpwer = src_sessions[session_id]['error_rate']
        hyp_cpwer = hyp_sessions[session_id]['error_rate']
        if hyp_cpwer < src_cpwer:
            # print(f"------------------->>> {session_id}: {hyp_cpwer:.4f} < {src_cpwer:.4f}")
            print(f"improved: {hyp_cpwer - src_cpwer:.4f}, session_id: {session_id}") 
            total_improved += (hyp_cpwer - src_cpwer)
            improved_count += 1
        else:
            total_non_improve += (hyp_cpwer - src_cpwer)
            pass
        
    logging.info(f"total_improved: {total_improved:.4f}, total_non_improve: {total_non_improve}")
    logging.info(f"total_count: {total_count}, improved_count: {improved_count} non_improve_count: {total_count - improved_count}")
    print(f"total_improved: {total_improved:.4f}, total_non_improve: {total_non_improve}")
    print(f"total_count: {total_count}, improved_count: {improved_count} non_improve_count: {total_count - improved_count}")
    return total_improved + total_non_improve 


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Process JSON paths.')

    # Add the arguments
    parser.add_argument('--hyp_json_path', type=str, required=True, help='Path to the hypothesis JSON file')
    parser.add_argument('--src_json_path', type=str, required=True, help='Path to the source JSON file')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    hyp_json_path = args.hyp_json_path
    src_json_path = args.src_json_path

    process_session_data(hyp_json_path, src_json_path)
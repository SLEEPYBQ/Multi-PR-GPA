#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Average metrics from three experiment runs')
    parser.add_argument('--model', type=str, required=True, help='Base model name (e.g., deepseek-v3)')
    parser.add_argument('--base_dir', type=str, default='New_Experiment', help='Base directory containing experiment folders')
    return parser.parse_args()

def average_csv_files(model, base_dir, file_name):
    """Average the specified CSV file across three runs."""
    dataframes = []
    
    # Read CSV files from all three runs
    for run in range(1, 4):
        file_path = os.path.join(base_dir, f"{model}-{run}", "ablation_dialogue", file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)
        else:
            print(f"Warning: File {file_path} does not exist")
    
    if not dataframes:
        print(f"Error: No data found for {file_name}")
        return None
    
    if file_name == "metrics_by_round.csv":
        # For metrics_by_round.csv, we can simply average corresponding rows across dataframes
        result_df = pd.DataFrame({'Rounds': dataframes[0]['Rounds']})
        
        # Average numeric columns
        numeric_cols = [col for col in dataframes[0].columns if col != 'Rounds']
        for col in numeric_cols:
            result_df[col] = np.mean([df[col].values for df in dataframes], axis=0)
    
    elif file_name == "metrics_by_agent_type.csv":
        # For metrics_by_agent_type.csv, we need to handle different Agent_Types
        # First, merge all dataframes and then group by Agent_Type and Rounds
        all_data = pd.concat(dataframes, ignore_index=True)
        
        # Group by Agent_Type and Rounds, then calculate mean for each group
        result_df = all_data.groupby(['Agent_Type', 'Rounds'], as_index=False).mean()
    
    else:
        print(f"Error: Unsupported file type {file_name}")
        return None
    
    return result_df

def main():
    args = parse_arguments()
    model = args.model
    base_dir = args.base_dir
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir, model, "ablation_dialogue")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process metrics_by_round.csv
    metrics_by_round = average_csv_files(model, base_dir, "metrics_by_round.csv")
    if metrics_by_round is not None:
        output_file = os.path.join(output_dir, "metrics_by_round.csv")
        metrics_by_round.to_csv(output_file, index=False, float_format='%.3f')
        print(f"Saved averaged metrics_by_round.csv to {output_file}")
    
    # Process metrics_by_agent_type.csv
    metrics_by_agent_type = average_csv_files(model, base_dir, "metrics_by_agent_type.csv")
    if metrics_by_agent_type is not None:
        output_file = os.path.join(output_dir, "metrics_by_agent_type.csv")
        metrics_by_agent_type.to_csv(output_file, index=False, float_format='%.3f')
        print(f"Saved averaged metrics_by_agent_type.csv to {output_file}")

if __name__ == "__main__":
    main()
import os
import argparse
from dialogue_parser import DialogueParser
from personality_evaluator import PersonalityEvaluator
from experiment_runner import ExperimentRunner

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run multi-round personality experiment.')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--dir', type=str, default='.', help='Directory with input files')
    parser.add_argument('--pattern', type=str, default='*_dialogue.txt', help='File pattern')
    parser.add_argument('--personality', type=str, help='Filter by a specific personality type (e.g. "尽责性")')
    parser.add_argument('--ground-truth', type=str, help='CSV path for ground truth scores')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding')
    parser.add_argument('--api-key', type=str, default='sk-zI6Q67cPAsBZUdDX6tCl4ZCuJ7x5A34lc2UruLVaepWhIrtl', help='LLM API key')
    parser.add_argument('--api-base', type=str, default='https://api.openai-proxy.org/v1', help='LLM API base')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano', help='LLM model')
    parser.add_argument('--workers', type=int, default=16, help='Parallel workers')
    parser.add_argument('--ablation', type=str, choices=['dialogue'], default='dialogue', 
                        help='Ablation study type (only dialogue mode supported now)')
    
    args = parser.parse_args()
    
    # 收集文件
    if args.file and os.path.exists(args.file):
        files = [args.file]
    else:
        import glob
        files = glob.glob(os.path.join(args.dir, args.pattern))
    
    if not files:
        print("No files found!")
        return
    
    
    # 初始化
    dialogue_parser = DialogueParser(encoding=args.encoding)
    personality_evaluator = PersonalityEvaluator(
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model,
        temperature=0.0
    )
    runner = ExperimentRunner(dialogue_parser, personality_evaluator)
    
    # 运行
    results = runner.run_experiment(
        files=files,
        personality_type=args.personality,
        ground_truth_file=args.ground_truth,
        output_dir=args.output,
        verbose=args.verbose,
        max_workers=args.workers,
        ablation_choice=args.ablation
    )
    print("Experiment completed.")

if __name__ == "__main__":
    main()
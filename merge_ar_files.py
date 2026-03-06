import os
import argparse
import glob
from tqdm import tqdm
from array_record.python.array_record_module import ArrayRecordReader
from array_record.python.array_record_module import ArrayRecordWriter

def main():
    parser = argparse.ArgumentParser(description="Merge ArrayRecord files to reduce file count (e.g. for Kaggle limits)")
    parser.add_argument("--input-dir", required=True, help="Directory containing original .ar files")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged .ar files")
    parser.add_argument("--split", default="train", help="Prefix of the files, e.g., 'train'")
    parser.add_argument("--shards-out", type=int, default=128, help="Number of final .ar files you want (Kaggle max 1000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Find all input files and sort them deterministically
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ar")))
    num_inputs = len(input_files)
    
    if num_inputs == 0:
        print(f"No .ar files found in {args.input_dir}")
        return
        
    print(f"Found {num_inputs} .ar files. Merging into {args.shards_out} files...")
    
    files_per_shard = max(1, num_inputs // args.shards_out)
    current_out_idx = 0
    writer = None
    
    for in_idx, file in enumerate(tqdm(input_files, desc="Processing .ar files")):
        target_out_idx = min(in_idx // files_per_shard, args.shards_out - 1)
        
        if writer is None or target_out_idx > current_out_idx:
            if writer is not None:
                writer.close()
                
            current_out_idx = target_out_idx
            out_file = os.path.join(args.output_dir, f"{args.split}-{current_out_idx:05d}-of-{args.shards_out:05d}.ar")
            writer = ArrayRecordWriter(out_file, options="")
            
        reader = ArrayRecordReader(file)
        
        try:
            # Fallback 1: num_records()
            if hasattr(reader, "num_records"):
                n = reader.num_records()
                for i in range(n):
                    writer.write(reader.read(i))
            # Fallback 2: NumRecords() (C++ style)
            elif hasattr(reader, "NumRecords"):
                n = reader.NumRecords()
                for i in range(n):
                    writer.write(reader.read(i))
            # Fallback 3: Sequence behavior
            elif hasattr(reader, "__len__"):
                for i in range(len(reader)):
                    writer.write(reader[i])
            # Fallback 4: Iterator behavior
            else:
                for record in reader:
                    writer.write(record)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            raise
                
    if writer is not None:
        writer.close()
        
    print(f"Successfully merged {num_inputs} files into {args.shards_out} final .ar files in {args.output_dir}")

if __name__ == "__main__":
    main()

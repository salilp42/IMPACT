from pathlib import Path
from impact.data.loaders import TaoWuLoader, NeuroconLoader

def process_taowu():
    """Process the Tao Wu dataset."""
    base_dir = Path("/path/to/taowu/data")
    output_dir = Path("/path/to/output/taowu")
    
    loader = TaoWuLoader(base_dir, output_dir)
    loader.process_dataset()

def process_neurocon():
    """Process the Neurocon dataset."""
    base_dir = Path("/path/to/neurocon/data")
    output_dir = Path("/path/to/output/neurocon")
    
    loader = NeuroconLoader(base_dir, output_dir)
    loader.process_dataset()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process fMRI datasets for IMPACT")
    parser.add_argument("dataset", choices=["taowu", "neurocon"], help="Dataset to process")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    
    args = parser.parse_args()
    
    if args.dataset == "taowu":
        loader = TaoWuLoader(args.data_dir, args.output_dir)
    else:
        loader = NeuroconLoader(args.data_dir, args.output_dir)
    
    loader.process_dataset() 
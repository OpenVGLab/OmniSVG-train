#!/usr/bin/env python3
"""
SVG Data Processing Script
Supports both single file and batch directory processing.
"""

import os
import argparse
import subprocess
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox


def preprocess_svg(input_path: str, output_path: str) -> bool:
    """
    Simplify SVG syntax using picosvg, removing groups and transforms.
    
    Args:
        input_path: Path to input SVG file
        output_path: Path to save preprocessed SVG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w") as output_file:
            subprocess.run(["picosvg", input_path], stdout=output_file, check=True)
        print(f"Preprocessed SVG saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


def process_svg(
    svg: SVG,
    scale: float,
    width: int,
    height: int,
    simplify: bool = False,
    max_dist: int = 5
) -> SVG:
    """
    Apply transformations to SVG.
    
    Args:
        svg: SVG object to process
        scale: Zoom scale factor
        width: Target width for normalization
        height: Target height for normalization
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        
    Returns:
        Processed SVG object
    """
    svg.zoom(scale)
    svg.normalize(Bbox(width, height))
    
    if simplify:
        svg.simplify_arcs()
        svg.simplify_heuristic()
        svg.split(max_dist=max_dist)
    
    return svg


def process_single_file(
    input_path: str,
    output_path: str,
    scale: float,
    width: int,
    height: int,
    simplify: bool,
    max_dist: int
) -> bool:
    """
    Process a single SVG file.
    
    Args:
        input_path: Path to input SVG file
        output_path: Path to save processed SVG
        scale: Zoom scale factor
        width: Target width
        height: Target height
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Preprocess with picosvg
        if not preprocess_svg(input_path, output_path):
            print(f"Skipping {input_path}: preprocessing failed")
            return False

        # Check if output file is empty
        if os.path.getsize(output_path) == 0:
            print(f"Skipping {output_path}: file is empty")
            os.remove(output_path)
            return False

        # Load and process SVG
        svg = SVG.load_svg(output_path)
        svg = process_svg(svg, scale, width, height, simplify=simplify, max_dist=max_dist)

        # Save processed SVG
        svg.save_svg(output_path)
        print(f"Saved processed SVG to {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def process_directory(
    input_dir: str,
    output_dir: str,
    scale: float,
    width: int,
    height: int,
    simplify: bool,
    max_dist: int
) -> tuple:
    """
    Process all SVG files in a directory.
    
    Args:
        input_dir: Input directory containing SVG files
        output_dir: Output directory for processed files
        scale: Zoom scale factor
        width: Target width
        height: Target height
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith(".svg"):
                continue
                
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            if process_single_file(
                input_path, output_path,
                scale, width, height, simplify, max_dist
            ):
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="SVG Data Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python process_svg.py --input file.svg --output processed.svg
  
  # Process directory
  python process_svg.py --input_dir ./svgs --output_dir ./output
  
  # Process with simplification
  python process_svg.py --input_dir ./svgs --output_dir ./output --simplify
        """
    )
    
    # Input/Output options
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Single SVG file to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for single file processing"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing SVG files for batch processing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for batch processing"
    )
    
    # Processing options
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="SVG zoom scale factor (default: 1.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="Output SVG width (default: 200)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=200,
        help="Output SVG height (default: 200)"
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=5,
        help="Max path length before splitting (default: 5)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable path simplification"
    )

    args = parser.parse_args()

    # Validate arguments
    single_mode = args.input is not None
    batch_mode = args.input_dir is not None
    
    if not single_mode and not batch_mode:
        parser.error("Please specify --input for single file or --input_dir for batch processing")
    
    if single_mode and batch_mode:
        parser.error("Cannot use both --input and --input_dir simultaneously")
    
    if single_mode:
        # Single file processing
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_processed{ext}"
        
        success = process_single_file(
            args.input, args.output,
            args.scale, args.width, args.height,
            args.simplify, args.max_dist
        )
        
        if success:
            print(f"\nProcessing complete: {args.output}")
        else:
            print(f"\nProcessing failed: {args.input}")
            exit(1)
    
    else:
        # Batch processing
        if args.output_dir is None:
            args.output_dir = f"{args.input_dir}_processed"
        
        success, failure = process_directory(
            args.input_dir, args.output_dir,
            args.scale, args.width, args.height,
            args.simplify, args.max_dist
        )
        
        print(f"\nBatch processing complete:")
        print(f"  Success: {success}")
        print(f"  Failed:  {failure}")


if __name__ == "__main__":
    main()
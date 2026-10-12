"""Reproduce the 2021--2025 experiment from Excel or normalized annual files."""
import argparse
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source',help='Folder containing the five year folders and extracted XLS/XLSX workbooks')
    group.add_argument('--normalized',help='Folder containing raw_observations_2021.csv.gz through raw_observations_2025.csv.gz')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output).resolve()
    input_dir=Path(args.source or args.normalized).resolve()
    if not input_dir.is_dir():p.error('Input folder does not exist')
    if output==input_dir or output in input_dir.parents:p.error('Output must not be the input folder or an ancestor of it')
    if (output/'manifest.json').exists():p.error('Completed results already exist: choose a fresh output folder')
    os.environ['ALMATY_OUTPUT']=str(output)
    if args.source:
        os.environ['ALMATY_SOURCE']=str(input_dir)
        os.environ['ALMATY_RAW']=str(output/'raw')
        for year in range(2021,2026):
            if not (input_dir/str(year)).is_dir():p.error(f'Missing year folder: {year}')
        import import_sources
        import_sources.main()
    else:
        os.environ['ALMATY_RAW']=str(input_dir)
        for year in range(2021,2026):
            if not (input_dir/f'raw_observations_{year}.csv.gz').is_file():p.error(f'Missing normalized year: {year}')
    import analysis
    analysis.main()

if __name__=='__main__':main()

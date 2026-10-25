"""Run the complete Instamart Copilot pipeline from raw data to agent demonstrations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from part_a import run as run_a
from part_b import run as run_b
from part_c import run as run_c
from part_d import run as run_d
from validate_data import main as validate_data

def main():
    print('\n=== PART A: DATA FOUNDATION ==='); run_a(); validate_data()
    print('\n=== PART B: SEGMENTATION & LAPSE MODEL ==='); run_b()
    print('\n=== PART C: TICKET TRIAGE ==='); run_c()
    print('\n=== PART D: POLICY RAG & REFUND AGENT ==='); run_d()
    print('\nPIPELINE FINISHED')
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Compare two npy files containing numerical data."""

import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two npy files")
    parser.add_argument("file1", type=str, help="First npy file to compare")
    parser.add_argument("file2", type=str, help="Second npy file to compare")
    
    args = parser.parse_args()
    
    data1 = np.load(args.file1).flatten()
    data2 = np.load(args.file2).flatten()
    
    # Real part - compare absolute values (allow sign differences)
    real1, real2 = data1.real, data2.real
    abs_diff_real = np.abs(np.abs(real1) - np.abs(real2))
    max_abs_real = np.maximum(np.abs(real1), np.abs(real2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff_real = abs_diff_real / max_abs_real
        rel_diff_real = np.nan_to_num(rel_diff_real, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Imaginary part - compare absolute values (allow sign differences)
    imag1, imag2 = data1.imag, data2.imag
    abs_diff_imag = np.abs(np.abs(imag1) - np.abs(imag2))
    max_abs_imag = np.maximum(np.abs(imag1), np.abs(imag2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff_imag = abs_diff_imag / max_abs_imag
        rel_diff_imag = np.nan_to_num(rel_diff_imag, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Print top 10 largest relative differences for real part
    top_indices_real = np.argsort(rel_diff_real)[-10:][::-1]
    print("Top 10 largest relative differences (REAL part, comparing |val|):")
    for idx in top_indices_real:
        print(f"  idx {idx}: rel_diff={rel_diff_real[idx]:.16e}, val1={real1[idx]:.16e}, val2={real2[idx]:.16e}")
    
    print()
    
    # Print top 10 largest relative differences for imaginary part
    top_indices_imag = np.argsort(rel_diff_imag)[-10:][::-1]
    print("Top 10 largest relative differences (IMAG part, comparing |val|):")
    for idx in top_indices_imag:
        print(f"  idx {idx}: rel_diff={rel_diff_imag[idx]:.16e}, val1={imag1[idx]:.16e}, val2={imag2[idx]:.16e}")

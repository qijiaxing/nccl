# NCCL EP

# Low Latency (LL) perf

GPUs: 6kd GPUs x 4

=== Summary (Low Latency, across 4 ranks) ===

--- Host-observed performance ---
Dispatch (BF16):  avg=79.94 us, min=74.05 us, max=85.47 us
                  throughput: avg=77.78 GB/s, min=75.32 GB/s (rank 2), max=80.20 GB/s (rank 1)
Combine (BF16):   avg=113.29 us, min=107.55 us, max=119.65 us
                  throughput: avg=54.88 GB/s, min=53.84 GB/s (rank 1), max=55.88 GB/s (rank 2)
Total (D+C):      avg=199.21 us, min=195.97 us, max=204.90 us
                  throughput: avg=62.42 GB/s, min=62.34 GB/s (rank 2), max=62.40 GB/s (rank 3)

--- Kernel-only performance ---
Dispatch:    avg=76.72 us, min=73.87 us, max=79.55 us
                  throughput: avg=81.04 GB/s, min=84.17 GB/s, max=78.16 GB/s
Combine:     avg=108.98 us, min=107.85 us, max=110.20 us
                  throughput: avg=57.05 GB/s, min=57.65 GB/s, max=56.42 GB/s

Byte counts: dispatch=6.22 MB (BF16), combine=6.22 MB (BF16), selections=759

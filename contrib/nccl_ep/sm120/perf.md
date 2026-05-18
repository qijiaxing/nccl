# NCCL EP

# Low Latency (LL) perf

##  6kd, 4 GPUS, dispatch fp8

```
=== Summary (Low Latency, across 4 ranks) ===

--- Host-observed performance ---
Dispatch (FP8):  avg=79.99 us, min=76.26 us, max=85.54 us
                  throughput: avg=40.23 GB/s, min=39.01 GB/s (rank 2), max=41.02 GB/s (rank 1)
Combine (BF16):   avg=113.04 us, min=107.42 us, max=119.81 us
                  throughput: avg=55.01 GB/s, min=54.24 GB/s (rank 3), max=56.11 GB/s (rank 2)
Total (D+C):      avg=198.92 us, min=195.46 us, max=203.74 us
                  throughput: avg=47.43 GB/s, min=47.36 GB/s (rank 3), max=47.45 GB/s (rank 1)

--- Kernel-only performance ---
Dispatch:    avg=76.81 us, min=74.90 us, max=79.39 us
                  throughput: avg=41.90 GB/s, min=42.96 GB/s, max=40.53 GB/s
Combine:     avg=107.30 us, min=105.14 us, max=109.20 us
                  throughput: avg=57.95 GB/s, min=59.14 GB/s, max=56.94 GB/s

Byte counts: dispatch=3.22 MB (FP8), combine=6.22 MB (BF16), selections=759

=== Setup Timing (across 4 ranks) ===
ncclEpCreateGroup:   avg=90.41 ms, min=90.41 ms, max=90.41 ms
Handle creation:     avg=0.01 ms, min=0.01 ms, max=0.01 ms

[JQ] LL combine smem size: 69 KB
```

## 6kd, 4 GPUs, dispatch bf16

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

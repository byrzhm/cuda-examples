# Devices

## [0] `NVIDIA GeForce RTX 4090 D`
* SM Version: 890 (PTX Version: 520)
* Number of SMs: 114
* SM Default Clock Rate: 2520 MHz
* Global Memory: 23724 MiB Free / 24080 MiB Total
* Global Memory Bus Peak: 1008 GB/sec (384-bit DDR @10501MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 73728 KiB
* Maximum Active Blocks: 24/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

## [1] `NVIDIA GeForce RTX 4070 SUPER`
* SM Version: 890 (PTX Version: 520)
* Number of SMs: 56
* SM Default Clock Rate: 2640 MHz
* Global Memory: 11684 MiB Free / 11874 MiB Total
* Global Memory Bus Peak: 504 GB/sec (192-bit DDR @10501MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 49152 KiB
* Maximum Active Blocks: 24/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  [1/112] sum_naive_bench [Device=0 N=1024]
Warn: Current measurement timed out (15.00s) while over noise threshold (7.23% > 0.50%)
Warn: Current measurement timed out (15.00s) before accumulating min_time (0.37s < 0.50s)
Pass: Cold: 0.004208ms GPU, 0.026369ms CPU, 0.37s total GPU, 15.00s total wall, 87402x 
Pass: Batch: 0.002474ms GPU, 0.50s total GPU, 0.50s total wall, 202066x
Run:  [2/112] sum_naive_bench [Device=0 N=4096]
Pass: Cold: 0.008130ms GPU, 0.028746ms CPU, 0.50s total GPU, 9.50s total wall, 61504x 
Pass: Batch: 0.006417ms GPU, 0.50s total GPU, 0.50s total wall, 77923x
Run:  [3/112] sum_naive_bench [Device=0 N=16384]
Pass: Cold: 0.024077ms GPU, 0.044552ms CPU, 0.50s total GPU, 2.95s total wall, 20768x 
Pass: Batch: 0.022451ms GPU, 0.50s total GPU, 0.50s total wall, 22272x
Run:  [4/112] sum_naive_bench [Device=0 N=65536]
Pass: Cold: 0.088277ms GPU, 0.108819ms CPU, 0.50s total GPU, 1.11s total wall, 5665x 
Pass: Batch: 0.086253ms GPU, 0.50s total GPU, 0.50s total wall, 5797x
Run:  [5/112] sum_naive_bench [Device=0 N=262144]
Pass: Cold: 0.343147ms GPU, 0.363942ms CPU, 0.50s total GPU, 0.65s total wall, 1458x 
Pass: Batch: 0.341661ms GPU, 0.51s total GPU, 0.51s total wall, 1507x
Run:  [6/112] sum_naive_bench [Device=0 N=1048576]
Pass: Cold: 1.366000ms GPU, 1.387537ms CPU, 0.50s total GPU, 0.54s total wall, 367x 
Pass: Batch: 1.363134ms GPU, 0.52s total GPU, 0.52s total wall, 383x
Run:  [7/112] sum_naive_bench [Device=0 N=4194304]
Pass: Cold: 5.451582ms GPU, 5.473736ms CPU, 0.50s total GPU, 0.51s total wall, 92x 
Pass: Batch: 5.449003ms GPU, 0.52s total GPU, 0.52s total wall, 96x
Run:  [8/112] sum_naive_bench [Device=0 N=16777216]
Pass: Cold: 21.795677ms GPU, 21.819935ms CPU, 0.50s total GPU, 0.50s total wall, 23x 
Pass: Batch: 21.792330ms GPU, 0.52s total GPU, 0.52s total wall, 24x
Run:  [9/112] sum_naive_bench [Device=1 N=1024]
Warn: Current measurement timed out (15.00s) while over noise threshold (12.01% > 0.50%)
Warn: Current measurement timed out (15.00s) before accumulating min_time (0.31s < 0.50s)
Pass: Cold: 0.003728ms GPU, 0.022098ms CPU, 0.31s total GPU, 15.00s total wall, 81974x 
Pass: Batch: 0.002411ms GPU, 0.50s total GPU, 0.50s total wall, 207344x
Run:  [10/112] sum_naive_bench [Device=1 N=4096]
Pass: Cold: 0.008054ms GPU, 0.026705ms CPU, 0.50s total GPU, 10.91s total wall, 62080x 
Pass: Batch: 0.006479ms GPU, 0.50s total GPU, 0.50s total wall, 77176x
Run:  [11/112] sum_naive_bench [Device=1 N=16384]
Pass: Cold: 0.024515ms GPU, 0.043007ms CPU, 0.50s total GPU, 3.35s total wall, 20400x 
Pass: Batch: 0.022860ms GPU, 0.50s total GPU, 0.50s total wall, 21873x
Run:  [12/112] sum_naive_bench [Device=1 N=65536]
Pass: Cold: 0.090486ms GPU, 0.109033ms CPU, 0.50s total GPU, 1.22s total wall, 5536x 
Pass: Batch: 0.088229ms GPU, 0.50s total GPU, 0.50s total wall, 5668x
Run:  [13/112] sum_naive_bench [Device=1 N=262144]
Pass: Cold: 0.352704ms GPU, 0.371406ms CPU, 0.50s total GPU, 0.68s total wall, 1418x 
Pass: Batch: 0.349883ms GPU, 0.52s total GPU, 0.52s total wall, 1476x
Run:  [14/112] sum_naive_bench [Device=1 N=1048576]
Pass: Cold: 1.399228ms GPU, 1.418396ms CPU, 0.50s total GPU, 0.54s total wall, 358x 
Pass: Batch: 1.396654ms GPU, 0.52s total GPU, 0.52s total wall, 374x
Run:  [15/112] sum_naive_bench [Device=1 N=4194304]
Pass: Cold: 5.584853ms GPU, 5.604424ms CPU, 0.50s total GPU, 0.51s total wall, 90x 
Pass: Batch: 5.583719ms GPU, 0.52s total GPU, 0.52s total wall, 94x
Run:  [16/112] sum_naive_bench [Device=1 N=16777216]
Pass: Cold: 22.333035ms GPU, 22.353247ms CPU, 0.51s total GPU, 0.52s total wall, 23x 
Pass: Batch: 22.330027ms GPU, 0.54s total GPU, 0.54s total wall, 24x
Run:  [17/112] sum_v0_bench [Device=0 N=1024]
Pass: Cold: 0.010019ms GPU, 0.020523ms CPU, 0.50s total GPU, 6.87s total wall, 49920x 
Run:  [18/112] sum_v0_bench [Device=0 N=4096]
Pass: Cold: 0.009883ms GPU, 0.020390ms CPU, 0.50s total GPU, 6.98s total wall, 50592x 
Run:  [19/112] sum_v0_bench [Device=0 N=16384]
Pass: Cold: 0.009910ms GPU, 0.020447ms CPU, 0.50s total GPU, 6.94s total wall, 50464x 
Run:  [20/112] sum_v0_bench [Device=0 N=65536]
Pass: Cold: 0.011035ms GPU, 0.021551ms CPU, 0.50s total GPU, 6.10s total wall, 45312x 
Run:  [21/112] sum_v0_bench [Device=0 N=262144]
Pass: Cold: 0.018853ms GPU, 0.029450ms CPU, 0.50s total GPU, 3.44s total wall, 26528x 
Run:  [22/112] sum_v0_bench [Device=0 N=1048576]
Pass: Cold: 0.021783ms GPU, 0.032363ms CPU, 0.50s total GPU, 2.90s total wall, 22960x 
Run:  [23/112] sum_v0_bench [Device=0 N=4194304]
Pass: Cold: 0.055161ms GPU, 0.066174ms CPU, 0.50s total GPU, 1.34s total wall, 9072x 
Run:  [24/112] sum_v0_bench [Device=0 N=16777216]
Pass: Cold: 0.178278ms GPU, 0.189159ms CPU, 0.72s total GPU, 1.04s total wall, 4016x 
Run:  [25/112] sum_v0_bench [Device=1 N=1024]
Pass: Cold: 0.009046ms GPU, 0.018903ms CPU, 0.50s total GPU, 8.97s total wall, 55280x 
Run:  [26/112] sum_v0_bench [Device=1 N=4096]
Pass: Cold: 0.008819ms GPU, 0.018644ms CPU, 0.50s total GPU, 9.25s total wall, 56704x 
Run:  [27/112] sum_v0_bench [Device=1 N=16384]
Pass: Cold: 0.009274ms GPU, 0.019073ms CPU, 0.50s total GPU, 8.70s total wall, 53920x 
Run:  [28/112] sum_v0_bench [Device=1 N=65536]
Pass: Cold: 0.011861ms GPU, 0.021638ms CPU, 0.50s total GPU, 6.59s total wall, 42160x 
Run:  [29/112] sum_v0_bench [Device=1 N=262144]
Pass: Cold: 0.018636ms GPU, 0.028462ms CPU, 0.50s total GPU, 4.06s total wall, 26832x 
Run:  [30/112] sum_v0_bench [Device=1 N=1048576]
Pass: Cold: 0.031186ms GPU, 0.041169ms CPU, 0.50s total GPU, 2.46s total wall, 16048x 
Run:  [31/112] sum_v0_bench [Device=1 N=4194304]
Pass: Cold: 0.088352ms GPU, 0.098321ms CPU, 0.50s total GPU, 1.11s total wall, 5664x 
Run:  [32/112] sum_v0_bench [Device=1 N=16777216]
Pass: Cold: 0.301814ms GPU, 0.311734ms CPU, 0.70s total GPU, 0.92s total wall, 2304x 
Run:  [33/112] sum_v1_bench [Device=0 N=1024]
Pass: Cold: 0.010127ms GPU, 0.020676ms CPU, 0.50s total GPU, 6.79s total wall, 49376x 
Run:  [34/112] sum_v1_bench [Device=0 N=4096]
Pass: Cold: 0.009832ms GPU, 0.020301ms CPU, 0.50s total GPU, 7.00s total wall, 50864x 
Run:  [35/112] sum_v1_bench [Device=0 N=16384]
Pass: Cold: 0.009976ms GPU, 0.020466ms CPU, 0.50s total GPU, 6.89s total wall, 50128x 
Run:  [36/112] sum_v1_bench [Device=0 N=65536]
Pass: Cold: 0.010322ms GPU, 0.020813ms CPU, 0.50s total GPU, 6.61s total wall, 48448x 
Run:  [37/112] sum_v1_bench [Device=0 N=262144]
Pass: Cold: 0.019204ms GPU, 0.029748ms CPU, 0.50s total GPU, 3.39s total wall, 26048x 
Run:  [38/112] sum_v1_bench [Device=0 N=1048576]
Pass: Cold: 0.021058ms GPU, 0.031959ms CPU, 0.50s total GPU, 3.03s total wall, 23744x 
Run:  [39/112] sum_v1_bench [Device=0 N=4194304]
Pass: Cold: 0.054413ms GPU, 0.065352ms CPU, 0.50s total GPU, 1.27s total wall, 9200x 
Run:  [40/112] sum_v1_bench [Device=0 N=16777216]
Pass: Cold: 0.185420ms GPU, 0.196522ms CPU, 0.50s total GPU, 0.67s total wall, 2704x 
Run:  [41/112] sum_v1_bench [Device=1 N=1024]
Pass: Cold: 0.008922ms GPU, 0.018670ms CPU, 0.50s total GPU, 9.14s total wall, 56048x 
Run:  [42/112] sum_v1_bench [Device=1 N=4096]
Pass: Cold: 0.008854ms GPU, 0.018614ms CPU, 0.50s total GPU, 9.24s total wall, 56480x 
Run:  [43/112] sum_v1_bench [Device=1 N=16384]
Pass: Cold: 0.009013ms GPU, 0.018767ms CPU, 0.50s total GPU, 9.06s total wall, 55488x 
Run:  [44/112] sum_v1_bench [Device=1 N=65536]
Pass: Cold: 0.011190ms GPU, 0.020911ms CPU, 0.50s total GPU, 7.03s total wall, 44688x 
Run:  [45/112] sum_v1_bench [Device=1 N=262144]
Pass: Cold: 0.018706ms GPU, 0.028375ms CPU, 0.50s total GPU, 4.05s total wall, 26736x 
Run:  [46/112] sum_v1_bench [Device=1 N=1048576]
Pass: Cold: 0.029530ms GPU, 0.039387ms CPU, 0.50s total GPU, 2.55s total wall, 16944x 
Run:  [47/112] sum_v1_bench [Device=1 N=4194304]
Pass: Cold: 0.084566ms GPU, 0.094534ms CPU, 0.50s total GPU, 1.07s total wall, 5920x 
Run:  [48/112] sum_v1_bench [Device=1 N=16777216]
Pass: Cold: 0.283950ms GPU, 0.293989ms CPU, 0.50s total GPU, 0.62s total wall, 1776x 
Run:  [49/112] sum_v2_bench [Device=0 N=1024]
Pass: Cold: 0.009774ms GPU, 0.020545ms CPU, 0.50s total GPU, 7.03s total wall, 51168x 
Run:  [50/112] sum_v2_bench [Device=0 N=4096]
Pass: Cold: 0.009592ms GPU, 0.020373ms CPU, 0.50s total GPU, 7.18s total wall, 52128x 
Run:  [51/112] sum_v2_bench [Device=0 N=16384]
Pass: Cold: 0.009572ms GPU, 0.020587ms CPU, 0.50s total GPU, 7.23s total wall, 52240x 
Run:  [52/112] sum_v2_bench [Device=0 N=65536]
Pass: Cold: 0.010284ms GPU, 0.021458ms CPU, 0.50s total GPU, 6.68s total wall, 48624x 
Run:  [53/112] sum_v2_bench [Device=0 N=262144]
Pass: Cold: 0.014920ms GPU, 0.025813ms CPU, 0.50s total GPU, 4.37s total wall, 33520x 
Run:  [54/112] sum_v2_bench [Device=0 N=1048576]
Pass: Cold: 0.021262ms GPU, 0.031951ms CPU, 0.50s total GPU, 2.98s total wall, 23520x 
Run:  [55/112] sum_v2_bench [Device=0 N=4194304]
Pass: Cold: 0.056219ms GPU, 0.067090ms CPU, 0.50s total GPU, 1.23s total wall, 8896x 
Run:  [56/112] sum_v2_bench [Device=0 N=16777216]
Pass: Cold: 0.167743ms GPU, 0.178746ms CPU, 0.50s total GPU, 0.69s total wall, 2992x 
Run:  [57/112] sum_v2_bench [Device=1 N=1024]
Pass: Cold: 0.008909ms GPU, 0.018867ms CPU, 0.50s total GPU, 9.15s total wall, 56128x 
Run:  [58/112] sum_v2_bench [Device=1 N=4096]
Pass: Cold: 0.008808ms GPU, 0.018793ms CPU, 0.50s total GPU, 9.28s total wall, 56784x 
Run:  [59/112] sum_v2_bench [Device=1 N=16384]
Pass: Cold: 0.009156ms GPU, 0.019147ms CPU, 0.50s total GPU, 8.87s total wall, 54624x 
Run:  [60/112] sum_v2_bench [Device=1 N=65536]
Pass: Cold: 0.012039ms GPU, 0.021922ms CPU, 0.50s total GPU, 6.47s total wall, 41536x 
Run:  [61/112] sum_v2_bench [Device=1 N=262144]
Pass: Cold: 0.018702ms GPU, 0.028573ms CPU, 0.50s total GPU, 4.05s total wall, 26736x 
Run:  [62/112] sum_v2_bench [Device=1 N=1048576]
Pass: Cold: 0.029062ms GPU, 0.038948ms CPU, 0.50s total GPU, 2.58s total wall, 17216x 
Run:  [63/112] sum_v2_bench [Device=1 N=4194304]
Pass: Cold: 0.086385ms GPU, 0.096475ms CPU, 0.50s total GPU, 1.06s total wall, 5792x 
Run:  [64/112] sum_v2_bench [Device=1 N=16777216]
Pass: Cold: 0.282546ms GPU, 0.292501ms CPU, 0.50s total GPU, 0.61s total wall, 1776x 
Run:  [65/112] sum_v3_bench [Device=0 N=1024]
Pass: Cold: 0.009400ms GPU, 0.020181ms CPU, 0.50s total GPU, 7.34s total wall, 53200x 
Run:  [66/112] sum_v3_bench [Device=0 N=4096]
Pass: Cold: 0.009485ms GPU, 0.020364ms CPU, 0.50s total GPU, 7.30s total wall, 52720x 
Run:  [67/112] sum_v3_bench [Device=0 N=16384]
Pass: Cold: 0.009549ms GPU, 0.020464ms CPU, 0.50s total GPU, 7.26s total wall, 52368x 
Run:  [68/112] sum_v3_bench [Device=0 N=65536]
Pass: Cold: 0.009729ms GPU, 0.020610ms CPU, 0.50s total GPU, 7.06s total wall, 51408x 
Run:  [69/112] sum_v3_bench [Device=0 N=262144]
Pass: Cold: 0.014219ms GPU, 0.024910ms CPU, 0.50s total GPU, 4.60s total wall, 35168x 
Run:  [70/112] sum_v3_bench [Device=0 N=1048576]
Pass: Cold: 0.019367ms GPU, 0.030194ms CPU, 0.50s total GPU, 3.27s total wall, 25824x 
Run:  [71/112] sum_v3_bench [Device=0 N=4194304]
Pass: Cold: 0.044692ms GPU, 0.055715ms CPU, 0.50s total GPU, 1.45s total wall, 11200x 
Run:  [72/112] sum_v3_bench [Device=0 N=16777216]
Pass: Cold: 0.134337ms GPU, 0.145493ms CPU, 0.50s total GPU, 0.74s total wall, 3728x 
Run:  [73/112] sum_v3_bench [Device=1 N=1024]
Pass: Cold: 0.008542ms GPU, 0.018476ms CPU, 0.50s total GPU, 9.60s total wall, 58544x 
Run:  [74/112] sum_v3_bench [Device=1 N=4096]
Pass: Cold: 0.008549ms GPU, 0.018514ms CPU, 0.50s total GPU, 9.60s total wall, 58496x 
Run:  [75/112] sum_v3_bench [Device=1 N=16384]
Pass: Cold: 0.009103ms GPU, 0.019101ms CPU, 0.50s total GPU, 8.89s total wall, 54928x 
Run:  [76/112] sum_v3_bench [Device=1 N=65536]
Pass: Cold: 0.011767ms GPU, 0.021702ms CPU, 0.50s total GPU, 6.64s total wall, 42496x 
Run:  [77/112] sum_v3_bench [Device=1 N=262144]
Pass: Cold: 0.017880ms GPU, 0.027869ms CPU, 0.50s total GPU, 4.23s total wall, 27968x 
Run:  [78/112] sum_v3_bench [Device=1 N=1048576]
Pass: Cold: 0.026904ms GPU, 0.036850ms CPU, 0.50s total GPU, 2.76s total wall, 18592x 
Run:  [79/112] sum_v3_bench [Device=1 N=4194304]
Pass: Cold: 0.075350ms GPU, 0.085353ms CPU, 0.50s total GPU, 1.14s total wall, 6640x 
Run:  [80/112] sum_v3_bench [Device=1 N=16777216]
Pass: Cold: 0.248746ms GPU, 0.258555ms CPU, 0.50s total GPU, 0.63s total wall, 2016x 
Run:  [81/112] sum_v4_bench [Device=0 N=1024]
Pass: Cold: 0.009693ms GPU, 0.020277ms CPU, 0.50s total GPU, 7.08s total wall, 51584x 
Run:  [82/112] sum_v4_bench [Device=0 N=4096]
Pass: Cold: 0.009638ms GPU, 0.020214ms CPU, 0.50s total GPU, 7.12s total wall, 51888x 
Run:  [83/112] sum_v4_bench [Device=0 N=16384]
Pass: Cold: 0.009732ms GPU, 0.020615ms CPU, 0.50s total GPU, 7.13s total wall, 51392x 
Run:  [84/112] sum_v4_bench [Device=0 N=65536]
Pass: Cold: 0.009808ms GPU, 0.020625ms CPU, 0.50s total GPU, 7.00s total wall, 50992x 
Run:  [85/112] sum_v4_bench [Device=0 N=262144]
Pass: Cold: 0.013577ms GPU, 0.024206ms CPU, 0.50s total GPU, 4.83s total wall, 36832x 
Run:  [86/112] sum_v4_bench [Device=0 N=1048576]
Pass: Cold: 0.019266ms GPU, 0.029795ms CPU, 0.50s total GPU, 3.27s total wall, 25952x 
Run:  [87/112] sum_v4_bench [Device=0 N=4194304]
Pass: Cold: 0.044178ms GPU, 0.055008ms CPU, 0.50s total GPU, 1.48s total wall, 11328x 
Run:  [88/112] sum_v4_bench [Device=0 N=16777216]
Pass: Cold: 0.117861ms GPU, 0.128793ms CPU, 0.50s total GPU, 0.78s total wall, 4256x 
Run:  [89/112] sum_v4_bench [Device=1 N=1024]
Pass: Cold: 0.008262ms GPU, 0.017970ms CPU, 0.50s total GPU, 9.97s total wall, 60528x 
Run:  [90/112] sum_v4_bench [Device=1 N=4096]
Pass: Cold: 0.008283ms GPU, 0.017970ms CPU, 0.50s total GPU, 9.94s total wall, 60384x 
Run:  [91/112] sum_v4_bench [Device=1 N=16384]
Pass: Cold: 0.008553ms GPU, 0.018205ms CPU, 0.50s total GPU, 9.56s total wall, 58464x 
Run:  [92/112] sum_v4_bench [Device=1 N=65536]
Pass: Cold: 0.009251ms GPU, 0.018913ms CPU, 0.50s total GPU, 8.74s total wall, 54064x 
Run:  [93/112] sum_v4_bench [Device=1 N=262144]
Pass: Cold: 0.018024ms GPU, 0.027819ms CPU, 0.50s total GPU, 4.23s total wall, 27744x 
Run:  [94/112] sum_v4_bench [Device=1 N=1048576]
Pass: Cold: 0.026673ms GPU, 0.036733ms CPU, 0.50s total GPU, 2.79s total wall, 18752x 
Run:  [95/112] sum_v4_bench [Device=1 N=4194304]
Pass: Cold: 0.075413ms GPU, 0.085425ms CPU, 0.50s total GPU, 1.14s total wall, 6640x 
Run:  [96/112] sum_v4_bench [Device=1 N=16777216]
Pass: Cold: 0.215674ms GPU, 0.225586ms CPU, 0.50s total GPU, 0.65s total wall, 2319x 
Run:  [97/112] sum_v5_bench [Device=0 N=1024]
Pass: Cold: 0.009738ms GPU, 0.020376ms CPU, 0.50s total GPU, 7.06s total wall, 51360x 
Run:  [98/112] sum_v5_bench [Device=0 N=4096]
Pass: Cold: 0.009889ms GPU, 0.020609ms CPU, 0.50s total GPU, 6.94s total wall, 50576x 
Run:  [99/112] sum_v5_bench [Device=0 N=16384]
Pass: Cold: 0.009569ms GPU, 0.020293ms CPU, 0.50s total GPU, 7.20s total wall, 52256x 
Run:  [100/112] sum_v5_bench [Device=0 N=65536]
Pass: Cold: 0.009649ms GPU, 0.020405ms CPU, 0.50s total GPU, 7.11s total wall, 51824x 
Run:  [101/112] sum_v5_bench [Device=0 N=262144]
Pass: Cold: 0.012568ms GPU, 0.023123ms CPU, 0.50s total GPU, 5.27s total wall, 39792x 
Run:  [102/112] sum_v5_bench [Device=0 N=1048576]
Pass: Cold: 0.022202ms GPU, 0.032624ms CPU, 0.50s total GPU, 2.86s total wall, 22528x 
Run:  [103/112] sum_v5_bench [Device=0 N=4194304]
Pass: Cold: 0.042849ms GPU, 0.053560ms CPU, 0.50s total GPU, 1.57s total wall, 11680x 
Run:  [104/112] sum_v5_bench [Device=0 N=16777216]
Pass: Cold: 0.115382ms GPU, 0.126296ms CPU, 0.50s total GPU, 0.79s total wall, 4336x 
Run:  [105/112] sum_v5_bench [Device=1 N=1024]
Pass: Cold: 0.008380ms GPU, 0.018142ms CPU, 0.50s total GPU, 9.82s total wall, 59680x 
Run:  [106/112] sum_v5_bench [Device=1 N=4096]
Pass: Cold: 0.008507ms GPU, 0.018272ms CPU, 0.50s total GPU, 9.66s total wall, 58784x 
Run:  [107/112] sum_v5_bench [Device=1 N=16384]
Pass: Cold: 0.008867ms GPU, 0.018568ms CPU, 0.50s total GPU, 9.22s total wall, 56400x 
Run:  [108/112] sum_v5_bench [Device=1 N=65536]
Pass: Cold: 0.009277ms GPU, 0.018848ms CPU, 0.50s total GPU, 8.68s total wall, 53904x 
Run:  [109/112] sum_v5_bench [Device=1 N=262144]
Pass: Cold: 0.018568ms GPU, 0.028080ms CPU, 0.50s total GPU, 4.07s total wall, 26944x 
Run:  [110/112] sum_v5_bench [Device=1 N=1048576]
Pass: Cold: 0.026167ms GPU, 0.036258ms CPU, 0.50s total GPU, 2.88s total wall, 19120x 
Run:  [111/112] sum_v5_bench [Device=1 N=4194304]
Pass: Cold: 0.074331ms GPU, 0.084164ms CPU, 0.50s total GPU, 1.16s total wall, 6736x 
Run:  [112/112] sum_v5_bench [Device=1 N=16777216]
Pass: Cold: 0.218063ms GPU, 0.227728ms CPU, 0.50s total GPU, 0.65s total wall, 2304x 
```

# Benchmark Results

## sum_naive_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  |  Noise  |  GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWUtil | Samples | Batch GPU  |
|----------|---------|------------|---------|------------|-------|----------|--------------|--------|---------|------------|
|     1024 |  87402x |  26.369 us | 106.63% |   4.208 us | 7.23% | 243.348M | 974.342 MB/s |  0.10% | 202066x |   2.474 us |
|     4096 |  61504x |  28.746 us |   4.75% |   8.130 us | 3.52% | 503.784M |   2.016 GB/s |  0.20% |  77923x |   6.417 us |
|    16384 |  20768x |  44.552 us |   2.14% |  24.077 us | 2.05% | 680.496M |   2.722 GB/s |  0.27% |  22272x |  22.451 us |
|    65536 |   5665x | 108.819 us |   1.02% |  88.277 us | 0.41% | 742.394M |   2.970 GB/s |  0.29% |   5797x |  86.253 us |
|   262144 |   1458x | 363.942 us |   0.29% | 343.147 us | 0.08% | 763.942M |   3.056 GB/s |  0.30% |   1507x | 341.661 us |
|  1048576 |    367x |   1.388 ms |   0.16% |   1.366 ms | 0.03% | 767.625M |   3.071 GB/s |  0.30% |    383x |   1.363 ms |
|  4194304 |     92x |   5.474 ms |   0.03% |   5.452 ms | 0.01% | 769.374M |   3.077 GB/s |  0.31% |     96x |   5.449 ms |
| 16777216 |     23x |  21.820 ms |   0.02% |  21.796 ms | 0.00% | 769.750M |   3.079 GB/s |  0.31% |     24x |  21.792 ms |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil | Samples | Batch GPU  |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|---------|------------|
|     1024 |  81974x |  22.098 us | 60.57% |   3.728 us | 12.01% | 274.698M |   1.100 GB/s |  0.22% | 207344x |   2.411 us |
|     4096 |  62080x |  26.705 us |  3.07% |   8.054 us |  4.41% | 508.558M |   2.035 GB/s |  0.40% |  77176x |   6.479 us |
|    16384 |  20400x |  43.007 us |  3.94% |  24.515 us |  1.06% | 668.339M |   2.674 GB/s |  0.53% |  21873x |  22.860 us |
|    65536 |   5536x | 109.033 us |  0.79% |  90.486 us |  0.53% | 724.269M |   2.897 GB/s |  0.57% |   5668x |  88.229 us |
|   262144 |   1418x | 371.406 us |  0.27% | 352.704 us |  0.14% | 743.240M |   2.973 GB/s |  0.59% |   1476x | 349.883 us |
|  1048576 |    358x |   1.418 ms |  0.07% |   1.399 ms |  0.03% | 749.396M |   2.998 GB/s |  0.59% |    374x |   1.397 ms |
|  4194304 |     90x |   5.604 ms |  0.02% |   5.585 ms |  0.01% | 751.014M |   3.004 GB/s |  0.60% |     94x |   5.584 ms |
| 16777216 |     23x |  22.353 ms |  0.01% |  22.333 ms |  0.01% | 751.229M |   3.005 GB/s |  0.60% |     24x |  22.330 ms |

## sum_v0_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  49920x |  20.523 us |  5.21% |  10.019 us |  8.40% | 102.210M | 410.438 MB/s |  0.04% |
|     4096 |  50592x |  20.390 us |  4.98% |   9.883 us |  8.89% | 414.430M |   1.664 GB/s |  0.17% |
|    16384 |  50464x |  20.447 us |  4.79% |   9.910 us |  8.25% |   1.653G |   6.639 GB/s |  0.66% |
|    65536 |  45312x |  21.551 us | 12.66% |  11.035 us | 24.67% |   5.939G |  23.849 GB/s |  2.37% |
|   262144 |  26528x |  29.450 us |  3.44% |  18.853 us |  4.90% |  13.905G |  55.836 GB/s |  5.54% |
|  1048576 |  22960x |  32.363 us |  4.56% |  21.783 us |  6.20% |  48.137G | 193.299 GB/s | 19.17% |
|  4194304 |   9072x |  66.174 us |  5.16% |  55.161 us |  6.10% |  76.038G | 305.340 GB/s | 30.29% |
| 16777216 |   4016x | 189.159 us |  7.82% | 178.278 us |  8.32% |  94.107G | 377.899 GB/s | 37.49% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  55280x |  18.903 us |  9.18% |   9.046 us | 17.30% | 113.198M | 454.562 MB/s |  0.09% |
|     4096 |  56704x |  18.644 us |  8.25% |   8.819 us | 15.90% | 464.431M |   1.865 GB/s |  0.37% |
|    16384 |  53920x |  19.073 us | 11.61% |   9.274 us | 22.54% |   1.767G |   7.094 GB/s |  1.41% |
|    65536 |  42160x |  21.638 us | 15.85% |  11.861 us | 28.44% |   5.525G |  22.188 GB/s |  4.40% |
|   262144 |  26832x |  28.462 us |  4.77% |  18.636 us |  6.70% |  14.066G |  56.485 GB/s | 11.21% |
|  1048576 |  16048x |  41.169 us |  7.21% |  31.186 us |  9.34% |  33.623G | 135.019 GB/s | 26.79% |
|  4194304 |   5664x |  98.321 us |  3.53% |  88.352 us |  3.93% |  47.472G | 190.631 GB/s | 37.82% |
| 16777216 |   2304x | 311.734 us |  3.26% | 301.814 us |  3.39% |  55.588G | 223.220 GB/s | 44.29% |

## sum_v1_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|-------|------------|--------|----------|--------------|--------|
|     1024 |  49376x |  20.676 us | 4.30% |  10.127 us |  7.49% | 101.111M | 406.025 MB/s |  0.04% |
|     4096 |  50864x |  20.301 us | 4.35% |   9.832 us |  7.89% | 416.600M |   1.673 GB/s |  0.17% |
|    16384 |  50128x |  20.466 us | 4.46% |   9.976 us |  8.08% |   1.642G |   6.595 GB/s |  0.65% |
|    65536 |  48448x |  20.813 us | 8.80% |  10.322 us | 16.81% |   6.349G |  25.495 GB/s |  2.53% |
|   262144 |  26048x |  29.748 us | 3.78% |  19.204 us |  5.10% |  13.650G |  54.815 GB/s |  5.44% |
|  1048576 |  23744x |  31.959 us | 2.78% |  21.058 us |  3.67% |  49.794G | 199.952 GB/s | 19.83% |
|  4194304 |   9200x |  65.352 us | 6.79% |  54.413 us |  8.19% |  77.082G | 309.534 GB/s | 30.70% |
| 16777216 |   2704x | 196.522 us | 0.90% | 185.420 us |  0.98% |  90.482G | 363.344 GB/s | 36.04% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  56048x |  18.670 us |  5.75% |   8.922 us | 10.20% | 114.772M | 460.880 MB/s |  0.09% |
|     4096 |  56480x |  18.614 us |  5.36% |   8.854 us |  9.73% | 462.633M |   1.858 GB/s |  0.37% |
|    16384 |  55488x |  18.767 us |  5.65% |   9.013 us | 10.14% |   1.818G |   7.300 GB/s |  1.45% |
|    65536 |  44688x |  20.911 us | 15.85% |  11.190 us | 29.08% |   5.857G |  23.518 GB/s |  4.67% |
|   262144 |  26736x |  28.375 us |  3.35% |  18.706 us |  3.94% |  14.014G |  56.273 GB/s | 11.16% |
|  1048576 |  16944x |  39.387 us |  4.78% |  29.530 us |  6.23% |  35.508G | 142.589 GB/s | 28.29% |
|  4194304 |   5920x |  94.534 us |  3.30% |  84.566 us |  3.61% |  49.598G | 199.168 GB/s | 39.51% |
| 16777216 |   1776x | 293.989 us |  1.23% | 283.950 us |  1.25% |  59.085G | 237.263 GB/s | 47.07% |

## sum_v2_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  51168x |  20.545 us |  4.39% |   9.774 us |  7.65% | 104.769M | 420.714 MB/s |  0.04% |
|     4096 |  52128x |  20.373 us |  4.67% |   9.592 us |  8.51% | 427.009M |   1.715 GB/s |  0.17% |
|    16384 |  52240x |  20.587 us |  4.76% |   9.572 us |  8.27% |   1.712G |   6.873 GB/s |  0.68% |
|    65536 |  48624x |  21.458 us |  9.28% |  10.284 us | 19.13% |   6.373G |  25.590 GB/s |  2.54% |
|   262144 |  33520x |  25.813 us | 14.63% |  14.920 us | 25.28% |  17.570G |  70.556 GB/s |  7.00% |
|  1048576 |  23520x |  31.951 us |  3.08% |  21.262 us |  3.57% |  49.316G | 198.036 GB/s | 19.64% |
|  4194304 |   8896x |  67.090 us |  4.10% |  56.219 us |  4.80% |  74.606G | 299.591 GB/s | 29.72% |
| 16777216 |   2992x | 178.746 us |  1.06% | 167.743 us |  0.93% | 100.017G | 401.631 GB/s | 39.84% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  56128x |  18.867 us |  7.74% |   8.909 us | 14.41% | 114.942M | 461.566 MB/s |  0.09% |
|     4096 |  56784x |  18.793 us |  7.60% |   8.808 us | 13.70% | 465.047M |   1.867 GB/s |  0.37% |
|    16384 |  54624x |  19.147 us | 10.14% |   9.156 us | 19.88% |   1.789G |   7.186 GB/s |  1.43% |
|    65536 |  41536x |  21.922 us | 16.70% |  12.039 us | 29.92% |   5.443G |  21.859 GB/s |  4.34% |
|   262144 |  26736x |  28.573 us |  3.37% |  18.702 us |  4.71% |  14.017G |  56.285 GB/s | 11.17% |
|  1048576 |  17216x |  38.948 us |  3.63% |  29.062 us |  4.46% |  36.080G | 144.885 GB/s | 28.74% |
|  4194304 |   5792x |  96.475 us |  2.64% |  86.385 us |  2.95% |  48.554G | 194.973 GB/s | 38.68% |
| 16777216 |   1776x | 292.501 us |  0.94% | 282.546 us |  0.90% |  59.379G | 238.443 GB/s | 47.31% |

## sum_v3_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  53200x |  20.181 us |  4.47% |   9.400 us |  7.47% | 108.941M | 436.190 MB/s |  0.04% |
|     4096 |  52720x |  20.364 us |  4.62% |   9.485 us |  7.45% | 431.823M |   1.728 GB/s |  0.17% |
|    16384 |  52368x |  20.464 us |  4.79% |   9.549 us |  8.88% |   1.716G |   6.863 GB/s |  0.68% |
|    65536 |  51408x |  20.610 us |  7.03% |   9.729 us | 14.36% |   6.736G |  26.946 GB/s |  2.67% |
|   262144 |  35168x |  24.910 us | 15.43% |  14.219 us | 27.41% |  18.436G |  73.745 GB/s |  7.32% |
|  1048576 |  25824x |  30.194 us |  3.17% |  19.367 us |  4.08% |  54.141G | 216.564 GB/s | 21.48% |
|  4194304 |  11200x |  55.715 us |  2.26% |  44.692 us |  2.36% |  93.849G | 375.396 GB/s | 37.24% |
| 16777216 |   3728x | 145.493 us |  0.79% | 134.337 us |  0.69% | 124.889G | 499.557 GB/s | 49.55% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  58544x |  18.476 us |  8.65% |   8.542 us | 17.15% | 119.879M | 479.983 MB/s |  0.10% |
|     4096 |  58496x |  18.514 us |  8.30% |   8.549 us | 16.24% | 479.145M |   1.917 GB/s |  0.38% |
|    16384 |  54928x |  19.101 us | 12.26% |   9.103 us | 24.84% |   1.800G |   7.200 GB/s |  1.43% |
|    65536 |  42496x |  21.702 us | 16.73% |  11.767 us | 30.47% |   5.570G |  22.278 GB/s |  4.42% |
|   262144 |  27968x |  27.869 us |  3.58% |  17.880 us |  4.76% |  14.661G |  58.644 GB/s | 11.63% |
|  1048576 |  18592x |  36.850 us |  3.37% |  26.904 us |  4.14% |  38.974G | 155.897 GB/s | 30.93% |
|  4194304 |   6640x |  85.353 us |  1.43% |  75.350 us |  1.57% |  55.664G | 222.657 GB/s | 44.17% |
| 16777216 |   2016x | 258.555 us |  0.85% | 248.746 us |  0.85% |  67.447G | 269.789 GB/s | 53.52% |

## sum_v4_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  51584x |  20.277 us |  4.45% |   9.693 us |  7.65% | 105.642M | 422.980 MB/s |  0.04% |
|     4096 |  51888x |  20.214 us |  4.32% |   9.638 us |  7.50% | 424.995M |   1.700 GB/s |  0.17% |
|    16384 |  51392x |  20.615 us |  5.03% |   9.732 us |  8.04% |   1.684G |   6.735 GB/s |  0.67% |
|    65536 |  50992x |  20.625 us |  4.16% |   9.808 us |  7.41% |   6.682G |  26.727 GB/s |  2.65% |
|   262144 |  36832x |  24.206 us | 14.72% |  13.577 us | 26.98% |  19.308G |  77.233 GB/s |  7.66% |
|  1048576 |  25952x |  29.795 us |  2.96% |  19.266 us |  3.44% |  54.425G | 217.700 GB/s | 21.60% |
|  4194304 |  11328x |  55.008 us |  3.34% |  44.178 us |  3.96% |  94.940G | 379.762 GB/s | 37.67% |
| 16777216 |   4256x | 128.793 us |  0.78% | 117.861 us |  0.65% | 142.348G | 569.391 GB/s | 56.48% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|-------|------------|--------|----------|--------------|--------|
|     1024 |  60528x |  17.970 us | 4.89% |   8.262 us |  9.64% | 123.943M | 496.257 MB/s |  0.10% |
|     4096 |  60384x |  17.970 us | 5.49% |   8.283 us | 10.51% | 494.534M |   1.979 GB/s |  0.39% |
|    16384 |  58464x |  18.205 us | 6.82% |   8.553 us | 13.44% |   1.916G |   7.663 GB/s |  1.52% |
|    65536 |  54064x |  18.913 us | 6.91% |   9.251 us | 13.37% |   7.085G |  28.339 GB/s |  5.62% |
|   262144 |  27744x |  27.819 us | 4.30% |  18.024 us |  5.94% |  14.544G |  58.176 GB/s | 11.54% |
|  1048576 |  18752x |  36.733 us | 3.27% |  26.673 us |  3.91% |  39.312G | 157.249 GB/s | 31.20% |
|  4194304 |   6640x |  85.425 us | 1.39% |  75.413 us |  1.17% |  55.618G | 222.470 GB/s | 44.14% |
| 16777216 |   2319x | 225.586 us | 0.56% | 215.674 us |  0.44% |  77.790G | 311.158 GB/s | 61.73% |

## sum_v5_bench

### [0] NVIDIA GeForce RTX 4090 D

|    N     | Samples |  CPU Time  | Noise  |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|--------|------------|--------|----------|--------------|--------|
|     1024 |  51360x |  20.376 us |  4.60% |   9.738 us |  7.51% | 105.152M | 421.018 MB/s |  0.04% |
|     4096 |  50576x |  20.609 us |  5.78% |   9.889 us |  9.06% | 414.197M |   1.657 GB/s |  0.16% |
|    16384 |  52256x |  20.293 us |  4.51% |   9.569 us |  7.86% |   1.712G |   6.849 GB/s |  0.68% |
|    65536 |  51824x |  20.405 us |  5.49% |   9.649 us | 10.23% |   6.792G |  27.167 GB/s |  2.69% |
|   262144 |  39792x |  23.123 us | 13.14% |  12.568 us | 24.40% |  20.858G |  83.431 GB/s |  8.28% |
|  1048576 |  22528x |  32.624 us |  3.86% |  22.202 us |  5.06% |  47.228G | 188.913 GB/s | 18.74% |
|  4194304 |  11680x |  53.560 us |  3.51% |  42.849 us |  4.24% |  97.886G | 391.546 GB/s | 38.84% |
| 16777216 |   4336x | 126.296 us |  1.77% | 115.382 us |  1.89% | 145.406G | 581.623 GB/s | 57.70% |
### [1] NVIDIA GeForce RTX 4070 SUPER

|    N     | Samples |  CPU Time  | Noise |  GPU Time  | Noise  |  Elem/s  | GlobalMem BW | BWUtil |
|----------|---------|------------|-------|------------|--------|----------|--------------|--------|
|     1024 |  59680x |  18.142 us | 6.07% |   8.380 us | 11.01% | 122.202M | 489.285 MB/s |  0.10% |
|     4096 |  58784x |  18.272 us | 5.85% |   8.507 us | 11.16% | 481.500M |   1.926 GB/s |  0.38% |
|    16384 |  56400x |  18.568 us | 7.59% |   8.867 us | 13.08% |   1.848G |   7.392 GB/s |  1.47% |
|    65536 |  53904x |  18.848 us | 8.75% |   9.277 us | 16.59% |   7.064G |  28.257 GB/s |  5.61% |
|   262144 |  26944x |  28.080 us | 7.34% |  18.568 us | 10.81% |  14.118G |  56.473 GB/s | 11.20% |
|  1048576 |  19120x |  36.258 us | 7.01% |  26.167 us |  9.61% |  40.072G | 160.287 GB/s | 31.80% |
|  4194304 |   6736x |  84.164 us | 1.98% |  74.331 us |  2.12% |  56.428G | 225.711 GB/s | 44.78% |
| 16777216 |   2304x | 227.728 us | 1.00% | 218.063 us |  1.03% |  76.937G | 307.749 GB/s | 61.06% |

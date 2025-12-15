# Dinov3-LMM
在运行代码前，请将项目根目录加入 PYTHONPATH：
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
对于cuda allocator碎片化可能导致的OOM解决办法：
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
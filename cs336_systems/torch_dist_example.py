
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from multiprocessing import Manager


def setup(rank, world_size, backend_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend_type, rank=rank, world_size=world_size)

def benchmark_dp(rank, config: dict, out: dict):
    # cpu + gloo or gpu + nccl
    backend_type = config['backend_type']
    # float32 data element number, 1MB / 4MB / 100MB / 1GB
    data_size = config['data_size']
    # worker number 
    num_proc = config['num_proc']
    # wramup iter
    warmup_iters = config.get('warmup_iters', 5)
    # formal iter
    formal_iters = config.get('formal_iters', 5)

    is_gpu = backend_type == 'gpu'
    device = 'cuda' if is_gpu else 'cpu'
    setup(rank, num_proc, 'nccl' if is_gpu else 'gloo')

    # per iter ops
    def func(iter_num, rank) -> float:
        # functions run on every worker
        # 4 bytes per element for float32 
        start_time = datetime.now()
        if is_gpu:
            # set visble device
            torch.cuda.set_device(rank)
        data = torch.randn((1, data_size // 4), device=device)
        print(f'iter [{iter_num}] rank [{rank}] start all reduce with {data_size} bytes')
        dist.all_reduce(data, async_op=False)
        if is_gpu:
            torch.cuda.synchronize()
        print(f'iter [{iter_num}] rank [{rank}] finshed all reduce')
        duration_ms = torch.tensor((datetime.now().timestamp() - start_time.timestamp()), dtype = torch.float32, device=device)
        duration_ms_list = [torch.zeros_like(duration_ms, device=device) for _ in range(num_proc)]
        dist.all_gather(duration_ms_list, duration_ms, )
        if rank == 0:
            # only print result on first rank
            # print('benchmark result:', duration_ms_list[0])
            return torch.mean(duration_ms_list[0]).cpu().item()
        else:
            return 0
    
    for i in range(warmup_iters):
        print('start warmup iters: ')
        func(i, rank)
    
    result_list = []
    for i in range(formal_iters):
        print('start formal iters: ')
        result_list.append(func(i, rank))

    # return {
    #     'backend_type':backend_type,
    #     'data_size':data_size,
    #     'num_proc':num_proc,
    #     'warmup_iters': warmup_iters,
    #     'formal_iters': formal_iters,
    #     'result': sum(result_list) / len(result_list)
    # }
    if rank == 0:
        out['result'] = sum(result_list) / len(result_list)


def main():
    C_MB = 1024 * 1024
    # cpu + gloo or gpu + nccl
    backend_type_list = ['cpu']
    # float32 data element number, 1MB / 4MB / 100MB / 1GB
    # bytes number
    data_size_list = [1 * C_MB, 4 * C_MB, 100 * C_MB, 1024 * C_MB]
    # data_size_list = [1 * C_MB]
    num_proc_list = [2, 4, 6]
    warmup_iters = 5
    formal_iters = 5
    result_list = []
    for backend_type in backend_type_list:
        for data_size in data_size_list:
            for num_proc in num_proc_list:
            # config also act as result dict
                with Manager() as manager:
                    out = manager.dict()
                    config = {
                        'backend_type':backend_type,
                        'data_size':data_size,
                        'num_proc':num_proc,
                        'warmup_iters': warmup_iters,
                        'formal_iters': formal_iters
                    }
                    mp.spawn(fn=benchmark_dp, args=(config, out), nprocs=num_proc, join=True) # type: ignore
                    result = {
                        **config,
                        'result': out['result']
                    }
                    print('result: ', result)
                    result_list.append(result)
    with open('comm_bench.json', 'w') as f:
        import json
        json.dump(result_list, f)

if __name__ == "__main__":
    main()

    
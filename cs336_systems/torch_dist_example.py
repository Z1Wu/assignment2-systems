
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
    # TODO return deivce based on backend type and world size

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

class testModule(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.w.weight)
    
    def forward(self, x):
        return torch.nn.functional.sigmoid(self.w(x))

def check_sample_param(m1: torch.nn.Module, m2: torch.nn.Module):
    for non_parallel_model_parameter, ddp_model_parameter in zip(
            m1.parameters(), m2.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

def naive_ddp_func(rank, world_size, backend_type, train_x_path, train_y_path):
    from torch import optim
    setup(rank=rank, world_size=world_size, backend_type=backend_type)

    # shape [num_samples, dim]
    train_x: torch.Tensor = torch.load(train_x_path)
    train_y: torch.Tensor = torch.load(train_y_path)
    sample_num = train_x.shape[0]
    train_x_dim = train_x.shape[1]
    # currently no mulitple gpu support, running with cpu and gloo for testing
    device = torch.device('cpu')
    
    num_iter = 4
    ddp_model = testModule(train_x_dim).to(device)
    gt_model =  testModule(train_x_dim).to(device)

    # before trainig start
    # all rank should have same 
    check_sample_param(ddp_model, gt_model)


    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(gt_model.parameters(), lr=0.1)


    for i in range(num_iter):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()
        # ddp model forward and backward
        batch_size_per_device = sample_num // world_size
        # shape: batch_size_per_device, dim
        ddp_x = train_x[rank * batch_size_per_device: (rank + 1) * batch_size_per_device, :].to(device)
        ddp_y = train_y[rank * batch_size_per_device: (rank + 1) * batch_size_per_device, :].to(device)
        ddp_out = ddp_model(ddp_x)
        # mse error
        loss = torch.nn.functional.mse_loss(ddp_out, ddp_y)
        loss.backward()

        # reduce all gridient
        for parm in ddp_model.parameters():
            if parm.requires_grad and parm.grad != None:
                dist.all_reduce(
                    parm.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )
                parm.grad /= world_size
        ddp_optimizer.step()

        base_out = gt_model(train_x)
        base_loss = torch.nn.functional.mse_loss(base_out, train_y)
        base_loss.backward()        
        non_parallel_optimizer.step()

        # for all rank, non_parallel_optimizer should have same model
        check_sample_param(ddp_model, gt_model)

def naive_ddp_test():
    dataset_dir = '/home/wuzy/my_code/cs336/assignment2-systems/debug'
    sample_num = 16
    data_dim = 64
    x = torch.randn(sample_num, data_dim)
    y = torch.tensor([[i % 2 for i in range(sample_num)]]).view(sample_num, 1)
    train_x_path = os.path.join(dataset_dir, 'train_x.pt')
    train_y_path = os.path.join(dataset_dir, 'train_y.pt')
    torch.save(x, train_x_path)
    torch.save(y, train_y_path)
    
    
    pass

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

    
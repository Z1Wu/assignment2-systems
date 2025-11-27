
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from multiprocessing import Manager
from cs336_basics.model import BasicsTransformerLM
from torch import nn
from typing import List, Type, Any
from copy import deepcopy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors



class Bucket:
    def __init__(self) -> None:
        self.param_names = []
        self.params_data = []
        self.flatten_tensor: torch.Tensor | None = None
        self.handle: Any = None
        self.cur_bucket_size = 0
    
    def __str__(self) -> str:
        return f"""
[
{self.param_names}
{self.params_data}
{self.cur_bucket_size}
]"""

class DDP(nn.Module):
    r"""
    # sample use case 
    model= ToyModel().to(device)
    ddp_model=DDP(model)
    for_in range(train_steps):
        x,y=get_batch()
        logits= ddp_model(x)
        loss=loss_fn(logits,y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    """
    
    def __init__(self, model: nn.Module, bucket_size: int = 0) -> None:
        super().__init__()
        # 1. broadcast parameter to all device, make sure module on all deivces have same init state
        self.original_model = model

        # FIXME only cares about parameters here, maybe need to consider buffer in module as well
        # run all reduce on model params
        # model.parameters() is stricted ordered
        params = [p.data for p in model.parameters()]
        new_params = self._broadcast_params_one_time(params)
        for p, new_val in zip(model.parameters(), new_params):
            p.data.copy_(new_val)

        self.buckets: List[Bucket] = []
        self.bucket_size = bucket_size
        self.total_hook_count = 0
        self.hook_invoke_count = 0
        for name, params in model.named_parameters():
            if params.requires_grad:
                print(f"register grident hook for {name}")
                self.total_hook_count += 1
                params.register_post_accumulate_grad_hook(
                    self._prams_grad_hook_builder(name)
                )
            
    def _prams_grad_hook_builder(self, parm_name):
        def func(params: nn.Parameter):
            if len(self.buckets) == 0:
                self.buckets.append(Bucket())
            assert params.grad != None
            self.hook_invoke_count -= 1
            cur_bucket = self.buckets[-1]
            cur_bucket.param_names.append(parm_name)
            cur_bucket.params_data.append(params.grad.data)
            cur_bucket.cur_bucket_size += params.grad.nelement() * params.grad.dtype.itemsize
            print(f"grident hook for {parm_name} invoked with bucket {cur_bucket}")
            # bucket size exceed or final hook invoked
            if cur_bucket.cur_bucket_size >= self.bucket_size or self.hook_invoke_count == self.total_hook_count:
                # communicate current bucket
                print(f"launch communication for bucket")
                cur_bucket.flatten_tensor = _flatten_dense_tensors(cur_bucket.params_data)
                cur_bucket.handle = dist.all_reduce(
                    cur_bucket.flatten_tensor,
                    async_op=True # use async here to avoid block
                )
                self.buckets.append(Bucket())
        return func

    def _broadcast_params_one_time(self, params: List[torch.Tensor]):
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        # no zero copy view, a new copied contiguous
        flatted_tensor = _flatten_dense_tensors(params)
        dist.broadcast(
            flatted_tensor,
            src=0,
            async_op=False # use sync op here, reduce with sum op by default
        )
        return _unflatten_dense_tensors(
            flatted_tensor,
            params
        )


    def forward(self, *inputs, **kwargs):
        # dispatch call to underly module
        return self.original_model(*inputs, **kwargs)

    def finish_gradient_synchronization(self): 
        for bucket in self.buckets:
            if bucket.handle != None:
                print("wait graident commnunication finsih")
                bucket.handle.wait()
                new_data = _unflatten_dense_tensors(bucket.flatten_tensor, bucket.params_data)
                for name, new_val in zip(bucket.param_names, new_data):
                    param = self.original_model.get_parameter(name)
                    assert param.grad != None
                    print(f"update gradient for {name} \n old val {param.grad.data} \n new val {new_val / dist.get_world_size()}")
                    param.grad.data.copy_(
                        # average grident
                        new_val / dist.get_world_size()
                    )
        # clear buckets, reset counter
        self.buckets = []
        self.hook_invoke_count = 0


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
        self.w = torch.nn.Linear(dim, 1, bias=False)
        # torch.nn.init.zeros_(self.w.weight)
    
    def forward(self, x):
        return torch.nn.functional.sigmoid(self.w(x))

def check_sample_param(m1: torch.nn.Module, m2: torch.nn.Module):
    for non_parallel_model_parameter, ddp_model_parameter in zip(
            m1.parameters(), m2.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

def naive_ddp_func(rank, world_size, model_class, backend_type, train_x_path, train_y_path):
    from torch import optim
    setup(rank=rank, world_size=world_size, backend_type=backend_type)

    # shape [num_samples, dim]
    train_x: torch.Tensor = torch.load(train_x_path)
    train_y: torch.Tensor = torch.load(train_y_path)
    sample_num = train_x.shape[0]
    # currently no mulitple gpu support, running with cpu and gloo for testing
    device = torch.device('cpu')
    
    num_iter = 4
    # init 一致的话，需要把参数传递到其他 rank 的 model 上
    gt_model =  model_class().to(device)
    ddp_model = deepcopy(gt_model)
    ddp_model = DDP(ddp_model)

    # before trainig start
    # rank zero show have ddp_model and gt_model with same parameters for testing
    if rank == 0:
        check_sample_param(ddp_model, gt_model)

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel m
    # odel
    non_parallel_optimizer = optim.SGD(gt_model.parameters(), lr=0.1)

    sub_proc_times = []
    import time
    # benchmark
    for i in range(num_iter):
        if rank == 0:
            print(f'[iter {i}] running ...')
        log_prefix = f'[iter {i}][rank {rank}]'
        start = time.time() 
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
        finish_forward = time.time()
        loss.backward()
        finish_backward = time.time()
        print(f'{log_prefix} {list(ddp_model.parameters())}')

        # reduce all gridient
        # for parm in ddp_model.parameters():
        #     if parm.requires_grad and parm.grad != None:
        #         print(f'{log_prefix} before: {parm.grad}')
        #         waitable_result = dist.all_reduce(
        #             parm.grad,
        #             op=dist.ReduceOp.SUM,
        #             async_op=False
        #         )
        #         # if using async op = True, we need to wait aysnc op to finish here
        #         if waitable_result != None:
        #             waitable_result.wait()
        #         print(f'{log_prefix} after: {parm.grad}')
        #         parm.grad /= world_size
        ddp_model.finish_gradient_synchronization()
        finish_grident_comm = time.time()
        ddp_optimizer.step()
        finish_optim_update = time.time()
        
        sub_proc_times.append(
            {
                'forward_time': finish_forward - start,
                'backward_time': finish_backward - finish_forward,
                'grident_update_time': finish_grident_comm - finish_backward,
                'optim_step_time': finish_optim_update - finish_grident_comm
            }
        )
        print(f'{log_prefix}: {sub_proc_times[-1]}')
        # only run in rank 0
        if rank == 0:
            base_out = gt_model(train_x)
            base_loss = torch.nn.functional.mse_loss(base_out, train_y)
            base_loss.backward()        
            non_parallel_optimizer.step()

            # for all rank, non_parallel_optimizer should have same model
            check_sample_param(ddp_model, gt_model)

class FactoryProvider:
    # 
    def __init__(self, model_class: Type[nn.Module], *args, **kwargs) -> None:
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> nn.Module:
        return self.model_class(*self.args, **self.kwargs)


def naive_ddp_test():
    dataset_dir = './debug'
    world_size = 2
    backend_type = 'gloo'
    sample_num = 16
    tiny_config = {
        'vocab_size': 1000,
        'context_length': 64,
        'd_model': 64,
        'num_layers': 2,
        'num_heads': 4,
        'd_ff': 256,  # d_model * 4
    }
    train_data_shape = [64]
    train_data_dtype = torch.float32
    train_label_shape = [1]
    train_label_dtype = torch.float32
    model_class = FactoryProvider(testModule, train_data_shape[0])
    # model_class = lambda : BasicsTransformerLM(
    #     **tiny_config
    # )
    # 创建测试的训练数据
    
    x = torch.randn(sample_num, *train_data_shape, dtype=train_data_dtype)
    y = torch.randn(sample_num, *train_label_shape, dtype=train_label_dtype)
    train_x_path = os.path.join(dataset_dir, 'train_x.pt')
    train_y_path = os.path.join(dataset_dir, 'train_y.pt')
    torch.save(x, train_x_path)
    torch.save(y, train_y_path)
    mp.spawn(fn=naive_ddp_func, args=(world_size, model_class, backend_type, train_x_path, train_y_path), nprocs=world_size, join=True) # type: ignore

def naive_mp_benchmark():
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
    # naive_mp_benchmark()
    naive_ddp_test()

    
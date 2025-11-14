import torch
from typing import Callable, Tuple
import math
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
import numpy as np
import timeit
import json
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import os
from jaxtyping import Float, Bool, Int
from torch import Tensor



cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention

shared_configs = {
    'vocab_size': 10000,
    'context_length': 128,
    # 'context_length': 256,
    # 'context_length': 512,
    # 'context_length': 1024,
    'batch_size': 1,
}

test_configs = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32
    }
}

class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, 
            Q: Float[Tensor, " ... queries d_k"],
            K: Float[Tensor, " ... keys    d_k"],
            V: Float[Tensor, " ... keys    d_v"],
            mask: Bool[Tensor, " ... queries keys"] | None = None,
        ) -> Float[Tensor, " ... queries d_v"]:
        return scaled_dot_product_attention(Q, K, V, mask)
    
class AttentionLayerTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, 
            Q: Float[Tensor, " ... queries d_k"],
            K: Float[Tensor, " ... keys    d_k"],
            V: Float[Tensor, " ... keys    d_v"],
            mask: Bool[Tensor, " ... queries keys"] | None = None,
        ) -> Float[Tensor, " ... queries d_v"]:
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, mask)

def get_model_and_input(model_config: dict) -> Tuple[torch.nn.Module, Callable]:
    if model_config['name'] == 'BasicsTransformerLM':
        vocab_size = model_config["vocab_size"]
        context_length = model_config["context_length"]
        d_model = model_config["d_model"]
        num_layers = model_config["num_layers"]
        num_heads = model_config["num_heads"]
        d_ff = model_config["d_ff"]
        device_str = model_config['device']
        rope_theta = 1000
        test_data = np.arange(vocab_size)
        device = torch.device(device_str)
        x, _ = get_batch(
            dataset = test_data,
            batch_size=model_config['batch_size'],
            context_length = model_config['context_length'],
            device = device_str
        )
        model = BasicsTransformerLM(
            vocab_size = vocab_size ,
            context_length = context_length,
            d_model = d_model ,
            num_layers = num_layers ,
            num_heads = num_heads ,
            d_ff = d_ff ,
            rope_theta = rope_theta
        )
        model.to(device)
        if model_config['torch_compile']:
            model.compile()
        return model, lambda : model(x)
    elif model_config['name'] == 'AttentionLayer':
        d_model = model_config['d_model']
        context_length = model_config['context_length']
        batch_size = model_config['batch_size']
        device = model_config['device']
        model = AttentionLayer()
        mask = torch.tril(torch.ones(batch_size, context_length, context_length, dtype=torch.bool, device=torch.device(device)))
        Q = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        K = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        V = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        if model_config['torch_compile']:
            model.compile()
        return model, lambda : model(Q, K, V, mask)
    elif model_config['name'] == 'AttentionLayerTorch':
        d_model = model_config['d_model']
        context_length = model_config['context_length']
        batch_size = model_config['batch_size']
        device = model_config['device']
        model = AttentionLayerTorch()
        mask = torch.tril(torch.ones(batch_size, context_length, context_length, dtype=torch.bool, device=torch.device(device)))
        Q = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        K = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        V = torch.rand((batch_size, context_length, d_model), device=torch.device(device), requires_grad=True)
        if model_config['torch_compile']:
            model.compile()
        return model, lambda : model(Q, K, V, mask)
    else:
        raise ValueError('Invalid model config', model_config)


def model_benchmark(
    model_config: dict,
    runtime_profile_config: dict | None = None,
    with_backward = False,
    mix_precision_config: dict | None = None,
    memory_profile_config: dict | None = None,
    optimizer_config: dict | None = None
):

    model, forward_func = get_model_and_input(model_config)

    run_config = {
        'with_backward': with_backward,
        'model_config': model_config,
        'runtime_profile_config': runtime_profile_config,
        'mix_precision_config': mix_precision_config,
        'optimizer_config': optimizer_config
    }
    print('preparing: ', timeit.default_timer())
    def _get_torch_dtype_from_str(dtype_str: str):
        if dtype_str == 'float32':
            return torch.float32
        elif dtype_str == 'float16':
            return torch.float16
        elif dtype_str == 'bfloat16':
            return torch.bfloat16
        else:
            raise ValueError('invalid dtype str:', dtype_str)
    
    if optimizer_config != None:
        optimizer = AdamW(
            model.parameters(),
            lr = optimizer_config.get('lr', 0.01),
            betas = optimizer_config.get('betas', (0.9, 0.999)),
            eps= optimizer_config.get('eps', 1e-8),
            weight_decay = optimizer_config.get('weight_decay', 0.01)
        )
    else:
        optimizer = None
    def test_run():
        def forward_step():
            # x => [batch, seq_lenth]
            with nvtx.range("forwad pass"):
                model.train()
                if not with_backward:
                    with torch.no_grad():
                        out : torch.Tensor = forward_func()
                else:
                    out : torch.Tensor = forward_func()
                return out

        def backward_step(out: torch.Tensor):
            with nvtx.range("backward pass"):
                torch.sum(out).backward()

        autocast_context = torch.autocast(device_type='cuda', dtype=_get_torch_dtype_from_str(mix_precision_config['dtype'])) \
            if mix_precision_config != None and mix_precision_config['enable'] == True \
            else  nullcontext()
        if optimizer != None:
            optimizer.zero_grad()
        with autocast_context:
            out = forward_step()
        if with_backward:
            backward_step(out)
        if optimizer != None:
            with nvtx.range("optimize step"):
                optimizer.step()
    
    if runtime_profile_config != None:        
        run_config['test_mean_time'] = _benchmark(test_run, num_warmups=runtime_profile_config['num_warmups'], num_trials=runtime_profile_config['num_trials'])

    if memory_profile_config != None :
        print(f'start memory profiling ... ')
        output_file_path = memory_profile_config['result_file_path']
        num_iters = memory_profile_config.get('num_iters', 10)
        with CUDAMemoryProfiler(
            snapshot_path=output_file_path,
            max_entries= memory_profile_config.get('max_entries', 10_000_000)
        ):
            for _ in range(num_iters):
                test_run()
        print(f'finsih memory profiling ...')
    print(f'result: {json.dumps(run_config, indent=2)}')
    return run_config


def _benchmark(test_run: Callable, num_warmups = 3, num_trials = 10) -> float:
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    print('warmup', timeit.default_timer())
    with nvtx.range("warmup runs"):
        for _ in range(num_warmups):
            test_run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    print('start running', timeit.default_timer())
    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    with nvtx.range("formal trials"):
        for trial in range(num_trials):  # Do it multiple times to capture variance
            start_time = timeit.default_timer()
            test_run()  # Actually perform computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
            end_time = timeit.default_timer()
            times.append((end_time - start_time)) # @inspect times
        # print(f'times: {times}')
        print('end', timeit.default_timer())
    mean_time = sum(times) / len(times) # @inspect mean_time
    return mean_time

class CUDAMemoryProfiler:
    """
    用于CUDA内存 profiling的context管理器，自动处理内存记录的开始、停止和快照保存。
    
    用法示例：
    with CUDAMemoryProfiler(snapshot_path="my_snapshot.pickle", max_entries=1_000_000):
        # 执行需要profiling的代码（如模型前向/反向传播）
        model(inputs)
        loss.backward()
    """
    
    def __init__(
        self,
        snapshot_path: str = "cuda_memory_snapshot.pickle",
        max_entries: int = 1_000_000
    ):
        """
        初始化profiler配置
        
        Args:
            snapshot_path: 内存快照保存的文件路径（.pickle格式）
            max_entries: 最大记录的内存事件条目数（防止内存溢出）
        """
        self.snapshot_path = snapshot_path
        self.max_entries = max_entries
        self._is_recording = False

    def __enter__(self):
        """进入context时启动内存记录"""
        # 启动内存历史记录
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self._is_recording = True
        return self  # 可选：返回self供with语句中使用

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出context时停止记录并保存快照（无论是否发生异常）"""
        if self._is_recording:
            try:
                # 保存内存快照到文件（可通过PyTorch在线工具分析）
                torch.cuda.memory._dump_snapshot(self.snapshot_path)
                print(f"CUDA内存快照已保存至: {self.snapshot_path}")
            finally:
                # 确保停止记录，释放资源
                torch.cuda.memory._record_memory_history(enabled=None)
                self._is_recording = False
        
        # 不抑制异常（如果有异常，让其正常传播）
        return False

class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, out_features, bias=False)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        print('input', x.dtype)
        x = self.fc1(x)
        print('fc1', x.dtype)
        x = self.relu(x)
        print('relu', x.dtype)
        x = self.ln(x)
        print('ln1', x.dtype)
        x = self.fc2(x)
        print('fc2', x.dtype)
        return x

def test_mixed_precision():
    x = torch.randn((1, 128))
    model = ToyModel(128, 128)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        x = torch.randn((1, 128))
        model = model.cuda()
        x = x.cuda()
        y = model(x)
        print('y:', y.dtype)
        loss = torch.sum(y)
        print('loss:', loss.dtype)
    loss.backward()
    grad = model.fc1.weight.grad
    if grad != None:
        print('grad fc1 ', grad.dtype)


def test_simple_attention_layer():
    # simple attention profile 
    res_arr = []
    for d_model in [16, 32, 64, 128]:
        for context_length in [256, 1024, 4096, 8192, 16384]:
            try:
                model_config = {
                    'name': 'AttentionLayerTorch',
                    'device': 'cuda',
                    'context_length': context_length,
                    'd_model': d_model,
                    'batch_size': 8,
                    'torch_compile': True
                }
                mix_precision_config={
                        'enable': True,
                        'dtype': 'bfloat16'
                    }
                print(f'start run exp for config: \n {model_config}')
                res = model_benchmark(
                    with_backward=True,
                    model_config = model_config,
                    runtime_profile_config = {
                        'num_warmups': 5,
                        'num_trials': 100
                    },
                    mix_precision_config={
                        'enable': True,
                        'dtype': 'bfloat16'
                    }
                )
                res_arr.append(res)
            except Exception as e:
                print(f'test \n {model_config} fail with error {e}')
                res_arr.append({
                    'model_config':  model_config,
                    'mix_precision_config': mix_precision_config,
                    'test_mean_time': 1e10
                })
            finally:
                with open('profile_results/out_compile_mixed_precision.json', 'w') as f:
                    json.dump(res_arr, f)


def test_compiled_llm():
    res_arr = []
    for compiled in [False, True]:
        for optimizer in [False, True]:
            for backward in [False, True]:
                try:
                    model_config= {
                        'name': 'BasicsTransformerLM',
                        'device': 'cuda',
                        'torch_compile': compiled,
                        **test_configs['medium'],
                        **shared_configs
                    }
                    print(f'start run exp for config: \n {model_config}')
                    res = model_benchmark(
                        with_backward=backward,
                        model_config = model_config,
                        runtime_profile_config = {
                            'num_warmups': 5,
                            'num_trials': 20
                        },
                        optimizer_config= None if not  optimizer else {
                            'lr': 1e-3
                        }
                    )
                    res_arr.append(res)
                except Exception as e:
                    print(f'test \n {model_config} fail with error {e}')
                    res_arr.append({
                        'model_config':  model_config,
                        'test_mean_time': 1e10
                    })
                finally:
                    with open('profile_results/out_llm.json', 'w') as f:
                        json.dump(res_arr, f)

if __name__ == '__main__':
    # runtime profile
    # model_benchmark(
    #     with_backward=False,
    #     runtime_profile_config = {
    #         'num_warmups': 5,
    #         'num_trials': 20
    #     },
    #     mix_precision_config={
    #         'enable': False,
    #         'dtype': 'float16'
    #     },
    #     **test_configs['large'], # type: ignore
    #     **shared_configs,
    # )
    # llm model profile
    # model_benchmark(
    #     with_backward=False,
    #     model_config= {
    #         'name': 'BasicsTransformerLM',
    #         'device': 'cuda',
    #         **test_configs['large'],
    #         **shared_configs
    #     },
    #     # memory_profile_config = {
    #     #     'result_file_path': os.path.join('./profile_results/memory_profile.pickle'),
    #     #     'num_iters': 5
    #     # },
    #     runtime_profile_config = {
    #         'num_warmups': 5,
    #         'num_trials': 20
    #     },
    #     mix_precision_config={
    #         'enable': False,
    #         'dtype': 'float16'
    #     }
    #     # optimizer_config={
    #     #     'lr': 1e-3
    #     # },
    # )

    # test_mixed_precision()
    # test_simple_attention_layer()


    model_benchmark(
        with_backward=True,
        model_config= {
            'name': 'BasicsTransformerLM',
            'device': 'cuda',
            'torch_compile': True,
            **test_configs['small'],
            **shared_configs
        },
        # memory_profile_config = {
        #     'result_file_path': os.path.join('./profile_results/memory_profile.pickle'),
        #     'num_iters': 5
        # },
        runtime_profile_config = {
            'num_warmups': 5,
            'num_trials': 20
        },
        mix_precision_config={
            'enable': False,
            'dtype': 'float16'
        },
        optimizer_config={
            'lr': 1e-3
        },
    )

    # test_compiled_llm()

    # test_simple_attention_layer()
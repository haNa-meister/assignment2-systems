import torch
from cs336_basics.model import BasicsTransformerLM

# Global variable for hyperparameters
HYPERPARAMS = {
    'vocab_size': 50257,
    'context_length': 128,
    'd_model': 768,
    'num_layers': 12,
    'num_heads': 12,
    'd_ff': 3072,
    'rope_theta': 10000.0,

    'batch_size': 32,
    'warmup_steps': 5,
    'benchmark_steps': 20,
    'forward_only': False,
    'device': "cuda",
    'autocast': False
}

def create_batch_data(batch_size, context_length, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    return input_ids

def benchmark(model, batch, hyperparams):
    device = hyperparams['device']
    warmup_steps = hyperparams['warmup_steps']
    benchmark_steps = hyperparams['benchmark_steps']

    model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(batch)

    # Benchmarking phase
    import time
    times = []
    for _ in range(benchmark_steps):
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(batch)
        loss = torch.nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)),
            batch.view(-1)
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    return torch.mean(torch.tensor(times)).item()

def main():
    model = BasicsTransformerLM(
        vocab_size=HYPERPARAMS['vocab_size'],
        context_length=HYPERPARAMS['context_length'],
        d_model=HYPERPARAMS['d_model'],
        num_layers=HYPERPARAMS['num_layers'],
        num_heads=HYPERPARAMS['num_heads'],
        d_ff=HYPERPARAMS['d_ff'],
        rope_theta=HYPERPARAMS['rope_theta']
    )
    print("Model initialized successfully.")

    batch = create_batch_data(
        HYPERPARAMS['batch_size'],
        HYPERPARAMS['context_length'],
        HYPERPARAMS['vocab_size']
    ).to(HYPERPARAMS['device'])
    print("Batch data created successfully.")

    avg_time = benchmark(model, batch, HYPERPARAMS)
    print(f"Average time per step: {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()
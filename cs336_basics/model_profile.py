from __future__ import annotations

from typing import Any
import numpy as np
import torch
import numpy.typing as npt
import timeit
from torch.cuda import nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch

vocab_size = 50256 + 1
context_length = 128
d_model = 512
d_ff = 1314
theta = 10_000
num_layers = 4
num_heads = 16
batch_size = 8
device = "cuda"


SMALL_LLM_CONFIG = dict(
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
)

MEDIUM_LLM_CONFIG = dict(
    d_model=1024,
    d_ff=4096,
    num_layers=24,
    num_heads=16,
)

LARGE_LLM_CONFIG = dict(
    d_model=2048,
    d_ff=5120,
    num_layers=36,
    num_heads=20,
)

XL_LLM_CONFIG = dict(
    d_model=1600,
    d_ff=6400,
    num_layers=48,
    num_heads=25,
)

TWOdot7B_LLM_CONFIG = dict(
    d_model=2560,
    d_ff=10240,
    num_layers=32,
    num_heads=32,
)

def profile_model(
    dataset: Any,
    config: dict[str, int | str],
    n_warmup_steps=5,
    n_profile_steps=10,
    forward_only=False,
):
    llm = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=theta,
    ).to(device)

    optimizer = AdamW(params=llm.parameters())

    with nvtx.range("warmup"):

        # Dummy training loop
        for step in range(n_warmup_steps):
            inputs, outputs = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            optimizer.zero_grad()
            logits = llm(inputs)
            loss = cross_entropy(logits.view(-1, vocab_size), outputs.view(-1))
            loss.backward()
            optimizer.step()

    with nvtx.range("profile"):
        profile_times = []
        # Profiling loop
        for step in range(n_profile_steps):
            inputs, outputs = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            optimizer.zero_grad()
            torch.cuda.synchronize(device=device)
            timer = timeit.default_timer()

            with nvtx.range("forward"):
                logits = llm(inputs)
                loss = cross_entropy(logits.view(-1, vocab_size), outputs.view(-1))

            # if forward_only:
            #     torch.cuda.synchronize(device=device)
            #     elapsed = timeit.default_timer() - timer
            #     print(f"Step {step + 1}/{n_profile_steps}, Forward Time: {elapsed:.6f}")
            #     profile_times.append(elapsed)
            #     continue

            with nvtx.range("backward"):
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize(device=device)
            elapsed = timeit.default_timer() - timer
            print(f"Step {step + 1}/{n_profile_steps}, Forward+Backward Time: {elapsed:.6f}")
            profile_times.append(elapsed)

        print(f"Mean time/step={np.mean(profile_times):.6f} std={np.std(profile_times):.6f}")

if __name__ == "__main__":

    # randomly generate an numpy array of length 50 * context_length.
    # each element is an int in [0, vocab_size)
    dataset = np.random.randint(0, vocab_size, size=(50 * context_length,))

    profile_model(
        dataset=dataset,
        config=SMALL_LLM_CONFIG,
        forward_only=False
    )

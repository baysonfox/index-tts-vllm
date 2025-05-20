import torch
import torchaudio
from indextts.infer_vllm import IndexTTS as IndexTTSVLLM # Alias to avoid confusion
from collections import OrderedDict
import os
import shutil
import asyncio

# Helper function to compare tensors (assuming they are on the same device)
def tensors_equal(t1, t2):
    if t1 is None and t2 is None:
        return True
    if t1 is None or t2 is None:
        return False
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        return False
    return torch.equal(t1, t2)

# Helper function to compare lists of tensors
def list_of_tensors_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for t1, t2 in zip(l1, l2):
        if not tensors_equal(t1, t2):
            return False
    return True

async def main_vllm():
    # Create dummy audio prompt files for testing
    sample_rate = 24000
    if not os.path.exists("tests"):
        os.makedirs("tests")
    
    dummy_audio_data = torch.rand(1, sample_rate * 2) # 2 seconds of audio
    dummy_prompt_vllm_1 = "tests/dummy_prompt_vllm_1.wav"
    dummy_prompt_vllm_2 = "tests/dummy_prompt_vllm_2.wav"
    dummy_prompt_vllm_3 = "tests/dummy_prompt_vllm_3.wav"
    
    torchaudio.save(dummy_prompt_vllm_1, dummy_audio_data, sample_rate)
    torchaudio.save(dummy_prompt_vllm_2, dummy_audio_data, sample_rate)
    torchaudio.save(dummy_prompt_vllm_3, dummy_audio_data, sample_rate)

    # Test Case 1: Basic Caching Functionality (Cache Hit) for VLLM
    print("Running VLLM Test Case 1: Basic Caching Functionality...")
    # Note: VLLM might have specific device requirements, defaulting to CPU for robust testing if possible
    # Adjust gpu_memory_utilization to a low value if running on CPU or limited GPU for tests
    tts_vllm_cache_test = IndexTTSVLLM(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", 
                                       is_fp16=False, device='cpu', cache_size=2, gpu_memory_utilization=0.01)
    
    # First call - should compute and cache
    await tts_vllm_cache_test.infer(audio_prompt=[dummy_prompt_vllm_1], text="Hello VLLM", output_path=None)
    
    # Check cond_mel_cache
    assert len(tts_vllm_cache_test.cond_mel_cache) == 1, "VLLM TC1 Failed: cond_mel_cache should have 1 item"
    cached_mel_vllm_1 = tts_vllm_cache_test.cond_mel_cache.get(dummy_prompt_vllm_1).cpu() if tts_vllm_cache_test.cond_mel_cache.get(dummy_prompt_vllm_1) is not None else None
    
    # Check latent_cache (key is tuple of sorted audio prompts)
    latent_key_1 = tuple(sorted([dummy_prompt_vllm_1]))
    assert len(tts_vllm_cache_test.latent_cache) == 1, "VLLM TC1 Failed: latent_cache should have 1 item"
    cached_latent_vllm_1 = tts_vllm_cache_test.latent_cache.get(latent_key_1).cpu() if tts_vllm_cache_test.latent_cache.get(latent_key_1) is not None else None

    # Second call with the same prompt - should be a cache hit
    await tts_vllm_cache_test.infer(audio_prompt=[dummy_prompt_vllm_1], text="Hello VLLM again", output_path=None)
    
    retrieved_mel_vllm_1 = tts_vllm_cache_test.cond_mel_cache.get(dummy_prompt_vllm_1).cpu() if tts_vllm_cache_test.cond_mel_cache.get(dummy_prompt_vllm_1) is not None else None
    retrieved_latent_vllm_1 = tts_vllm_cache_test.latent_cache.get(latent_key_1).cpu() if tts_vllm_cache_test.latent_cache.get(latent_key_1) is not None else None

    assert len(tts_vllm_cache_test.cond_mel_cache) == 1, "VLLM TC1 Failed: cond_mel_cache should still have 1 item after hit"
    assert tensors_equal(cached_mel_vllm_1, retrieved_mel_vllm_1), "VLLM TC1 Failed: cond_mel was not retrieved correctly"
    assert len(tts_vllm_cache_test.latent_cache) == 1, "VLLM TC1 Failed: latent_cache should still have 1 item after hit"
    assert tensors_equal(cached_latent_vllm_1, retrieved_latent_vllm_1), "VLLM TC1 Failed: latent was not retrieved correctly"
    print("VLLM Test Case 1 Passed!")

    # Test Case 2: LRU Eviction for VLLM
    print("\nRunning VLLM Test Case 2: LRU Eviction...")
    # Current cond_mel_cache: {dummy_prompt_vllm_1: ...}
    # Current latent_cache: {(dummy_prompt_vllm_1,): ...}
    
    # Process dummy_prompt_vllm_2.wav
    await tts_vllm_cache_test.infer(audio_prompt=[dummy_prompt_vllm_2], text="Second VLLM prompt", output_path=None)
    latent_key_2 = tuple(sorted([dummy_prompt_vllm_2]))
    assert len(tts_vllm_cache_test.cond_mel_cache) == 2, "VLLM TC2 Failed: cond_mel_cache should have 2 items"
    assert dummy_prompt_vllm_1 in tts_vllm_cache_test.cond_mel_cache
    assert dummy_prompt_vllm_2 in tts_vllm_cache_test.cond_mel_cache
    assert len(tts_vllm_cache_test.latent_cache) == 2, "VLLM TC2 Failed: latent_cache should have 2 items"
    assert latent_key_1 in tts_vllm_cache_test.latent_cache
    assert latent_key_2 in tts_vllm_cache_test.latent_cache
    
    # Process dummy_prompt_vllm_3.wav - should cache it and evict dummy_prompt_vllm_1 related entries
    await tts_vllm_cache_test.infer(audio_prompt=[dummy_prompt_vllm_3], text="Third VLLM prompt", output_path=None)
    latent_key_3 = tuple(sorted([dummy_prompt_vllm_3]))
    assert len(tts_vllm_cache_test.cond_mel_cache) == 2, "VLLM TC2 Failed: cond_mel_cache should have 2 items after eviction"
    assert dummy_prompt_vllm_1 not in tts_vllm_cache_test.cond_mel_cache, "VLLM TC2 Failed: dummy_prompt_vllm_1 should be evicted from cond_mel_cache"
    assert dummy_prompt_vllm_2 in tts_vllm_cache_test.cond_mel_cache
    assert dummy_prompt_vllm_3 in tts_vllm_cache_test.cond_mel_cache
    
    assert len(tts_vllm_cache_test.latent_cache) == 2, "VLLM TC2 Failed: latent_cache should have 2 items after eviction"
    assert latent_key_1 not in tts_vllm_cache_test.latent_cache, "VLLM TC2 Failed: latent_key_1 should be evicted from latent_cache"
    assert latent_key_2 in tts_vllm_cache_test.latent_cache
    assert latent_key_3 in tts_vllm_cache_test.latent_cache
    
    # Verify LRU order by accessing dummy_prompt_vllm_2 again
    await tts_vllm_cache_test.infer(audio_prompt=[dummy_prompt_vllm_2], text="Second VLLM prompt again", output_path=None)
    # Expected order for cond_mel_cache: dummy_prompt_vllm_3 (oldest), dummy_prompt_vllm_2 (newest)
    # Expected order for latent_cache: latent_key_3 (oldest), latent_key_2 (newest)
    assert next(iter(tts_vllm_cache_test.cond_mel_cache)) == dummy_prompt_vllm_3, "VLLM TC2 Failed: LRU order incorrect for cond_mel_cache"
    assert next(iter(tts_vllm_cache_test.latent_cache)) == latent_key_3, "VLLM TC2 Failed: LRU order incorrect for latent_cache"
    print("VLLM Test Case 2 Passed!")

    # Test Case 3: Multiple audio prompts for conditioning
    print("\nRunning VLLM Test Case 3: Multiple audio prompts...")
    tts_multi_prompt_test = IndexTTSVLLM(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                                         is_fp16=False, device='cpu', cache_size=3, gpu_memory_utilization=0.01)
    prompts_list1 = [dummy_prompt_vllm_1, dummy_prompt_vllm_2]
    latent_key_multi1 = tuple(sorted(prompts_list1))

    await tts_multi_prompt_test.infer(audio_prompt=prompts_list1, text="Multi-prompt test 1", output_path=None)
    assert len(tts_multi_prompt_test.cond_mel_cache) == 2, "VLLM TC3 Failed: cond_mel_cache should have 2 items"
    assert dummy_prompt_vllm_1 in tts_multi_prompt_test.cond_mel_cache
    assert dummy_prompt_vllm_2 in tts_multi_prompt_test.cond_mel_cache
    assert len(tts_multi_prompt_test.latent_cache) == 1, "VLLM TC3 Failed: latent_cache should have 1 item for the combined latent"
    assert latent_key_multi1 in tts_multi_prompt_test.latent_cache
    
    cached_latent_multi1 = tts_multi_prompt_test.latent_cache.get(latent_key_multi1).cpu() if tts_multi_prompt_test.latent_cache.get(latent_key_multi1) is not None else None

    # Second call with same multi-prompts
    await tts_multi_prompt_test.infer(audio_prompt=prompts_list1, text="Multi-prompt test 1 again", output_path=None)
    retrieved_latent_multi1 = tts_multi_prompt_test.latent_cache.get(latent_key_multi1).cpu() if tts_multi_prompt_test.latent_cache.get(latent_key_multi1) is not None else None
    assert len(tts_multi_prompt_test.latent_cache) == 1, "VLLM TC3 Failed: latent_cache should still have 1 item after hit"
    assert tensors_equal(cached_latent_multi1, retrieved_latent_multi1), "VLLM TC3 Failed: Multi-prompt latent was not retrieved correctly"
    
    # Add a new prompt, forcing eviction from cond_mel_cache if cache_size was 2
    # With cache_size=3 for this test instance, add one more
    await tts_multi_prompt_test.infer(audio_prompt=[dummy_prompt_vllm_3], text="Single prompt to fill cache", output_path=None)
    assert len(tts_multi_prompt_test.cond_mel_cache) == 3
    # Now add another multi-prompt that would evict oldest cond_mels if cache was smaller
    prompts_list2 = [dummy_prompt_vllm_1, dummy_prompt_vllm_3] # dummy_prompt_vllm_1 is accessed again
    latent_key_multi2 = tuple(sorted(prompts_list2))
    await tts_multi_prompt_test.infer(audio_prompt=prompts_list2, text="Multi-prompt test 2", output_path=None)
    # dummy_prompt_vllm_2 should be the oldest in cond_mel_cache and potentially evicted if cache overflows
    # Here, cache_size is 3. Prompts are 1,2,3. Accessing 1,3. 2 should be oldest.
    assert next(iter(tts_multi_prompt_test.cond_mel_cache)) == dummy_prompt_vllm_2, "VLLM TC3 Failed: LRU order for cond_mel_cache after multi-prompt access."
    print("VLLM Test Case 3 Passed!")

    # Cleanup dummy files
    os.remove(dummy_prompt_vllm_1)
    os.remove(dummy_prompt_vllm_2)
    os.remove(dummy_prompt_vllm_3)

if __name__ == "__main__":
    if not os.path.exists("outputs"): # Not strictly needed if output_path=None
        os.makedirs("outputs")
    asyncio.run(main_vllm())
    print("\nAll VLLM caching tests completed successfully.")

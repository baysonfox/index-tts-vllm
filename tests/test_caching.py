import torch
import torchaudio
from indextts.infer import IndexTTS
from collections import OrderedDict
import os
import shutil

# Helper function to compare tensors (assuming they are on the same device)
def tensors_equal(t1, t2):
    if t1 is None and t2 is None:
        return True
    if t1 is None or t2 is None:
        return False
    # Ensure both are tensors before calling torch.equal
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        return False # Or raise an error, depending on expected types
    return torch.equal(t1, t2)

def main():
    # Create a dummy audio prompt file for testing
    sample_rate = 24000
    # Ensure tests directory exists
    if not os.path.exists("tests"):
        os.makedirs("tests")
    
    dummy_audio_data = torch.rand(1, sample_rate * 2) # 2 seconds of audio
    torchaudio.save("tests/dummy_prompt_1.wav", dummy_audio_data, sample_rate)
    torchaudio.save("tests/dummy_prompt_2.wav", dummy_audio_data, sample_rate) # Another distinct file
    torchaudio.save("tests/dummy_prompt_3.wav", dummy_audio_data, sample_rate) # A third one

    # Test Case 1: Basic Caching Functionality (Cache Hit)
    print("Running Test Case 1: Basic Caching Functionality...")
    # Using is_fp16=False and device='cpu' for stable testing
    tts_cache_test = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device='cpu', cache_size=2)
    
    # First call - should compute and cache
    _ = tts_cache_test.infer(audio_prompt="tests/dummy_prompt_1.wav", text="Hello world", output_path=None)
    
    # Store the cached values for comparison
    # Need to handle potential device differences if not strictly CPU
    cached_mel_1 = tts_cache_test.cond_mel_cache.get("tests/dummy_prompt_1.wav").cpu() if tts_cache_test.cond_mel_cache.get("tests/dummy_prompt_1.wav") is not None else None
    cached_latent_1 = tts_cache_test.latent_cache.get("tests/dummy_prompt_1.wav").cpu() if tts_cache_test.latent_cache.get("tests/dummy_prompt_1.wav") is not None else None
    
    assert len(tts_cache_test.cond_mel_cache) == 1, "Test Case 1 Failed: cond_mel_cache should have 1 item"
    assert len(tts_cache_test.latent_cache) == 1, "Test Case 1 Failed: latent_cache should have 1 item"

    # Second call with the same prompt - should be a cache hit
    _ = tts_cache_test.infer(audio_prompt="tests/dummy_prompt_1.wav", text="Hello again", output_path=None)
    
    retrieved_mel_1 = tts_cache_test.cond_mel_cache.get("tests/dummy_prompt_1.wav").cpu() if tts_cache_test.cond_mel_cache.get("tests/dummy_prompt_1.wav") is not None else None
    retrieved_latent_1 = tts_cache_test.latent_cache.get("tests/dummy_prompt_1.wav").cpu() if tts_cache_test.latent_cache.get("tests/dummy_prompt_1.wav") is not None else None

    assert len(tts_cache_test.cond_mel_cache) == 1, "Test Case 1 Failed: cond_mel_cache should still have 1 item after hit"
    assert len(tts_cache_test.latent_cache) == 1, "Test Case 1 Failed: latent_cache should still have 1 item after hit"
    assert tensors_equal(cached_mel_1, retrieved_mel_1), f"Test Case 1 Failed: cond_mel was not retrieved correctly. Got {retrieved_mel_1}, expected {cached_mel_1}"
    assert tensors_equal(cached_latent_1, retrieved_latent_1), f"Test Case 1 Failed: latent was not retrieved correctly. Got {retrieved_latent_1}, expected {cached_latent_1}"
    print("Test Case 1 Passed!")

    # Test Case 2: LRU Eviction
    print("\nRunning Test Case 2: LRU Eviction...")
    # Current cache: {"tests/dummy_prompt_1.wav": ...} (Order: dummy_prompt_1 is most recent)
    
    # Process dummy_prompt_2.wav - should cache it
    _ = tts_cache_test.infer(audio_prompt="tests/dummy_prompt_2.wav", text="Second prompt", output_path=None)
    assert len(tts_cache_test.cond_mel_cache) == 2, "Test Case 2 Failed: cond_mel_cache should have 2 items"
    assert "tests/dummy_prompt_1.wav" in tts_cache_test.cond_mel_cache
    assert "tests/dummy_prompt_2.wav" in tts_cache_test.cond_mel_cache
    # Order: dummy_prompt_1, dummy_prompt_2 (most recent)
    
    # Process dummy_prompt_3.wav - should cache it and evict dummy_prompt_1.wav (oldest)
    _ = tts_cache_test.infer(audio_prompt="tests/dummy_prompt_3.wav", text="Third prompt", output_path=None)
    assert len(tts_cache_test.cond_mel_cache) == 2, "Test Case 2 Failed: cond_mel_cache should have 2 items after eviction"
    assert "tests/dummy_prompt_1.wav" not in tts_cache_test.cond_mel_cache, "Test Case 2 Failed: tests/dummy_prompt_1.wav should have been evicted from cond_mel_cache"
    assert "tests/dummy_prompt_2.wav" in tts_cache_test.cond_mel_cache
    assert "tests/dummy_prompt_3.wav" in tts_cache_test.cond_mel_cache
    # Order: dummy_prompt_2, dummy_prompt_3 (most recent)
    
    # Verify LRU order: Access dummy_prompt_2.wav again, making it the most recently used
    _ = tts_cache_test.infer(audio_prompt="tests/dummy_prompt_2.wav", text="Second prompt again", output_path=None)
    # Order: dummy_prompt_3, dummy_prompt_2 (most recent)
    oldest_mel_key = next(iter(tts_cache_test.cond_mel_cache))
    assert oldest_mel_key == "tests/dummy_prompt_3.wav", f"Test Case 2 Failed: LRU order incorrect for cond_mel_cache. Expected oldest: tests/dummy_prompt_3.wav, Got: {oldest_mel_key}"
    
    oldest_latent_key = next(iter(tts_cache_test.latent_cache))
    assert oldest_latent_key == "tests/dummy_prompt_3.wav", f"Test Case 2 Failed: LRU order incorrect for latent_cache. Expected oldest: tests/dummy_prompt_3.wav, Got: {oldest_latent_key}"
    print("Test Case 2 Passed!")

    # Test Case 3: Cache disabled (cache_size = 0)
    print("\nRunning Test Case 3: Cache disabled (cache_size=0)...")
    tts_no_cache = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device='cpu', cache_size=0)
    _ = tts_no_cache.infer(audio_prompt="tests/dummy_prompt_1.wav", text="No cache test", output_path=None)
    assert len(tts_no_cache.cond_mel_cache) == 0, "Test Case 3 Failed: cond_mel_cache should be empty when cache_size is 0"
    assert len(tts_no_cache.latent_cache) == 0, "Test Case 3 Failed: latent_cache should be empty when cache_size is 0"
    print("Test Case 3 Passed!")

    # Test Case 4: Cache with size 1
    print("\nRunning Test Case 4: Cache with size 1...")
    tts_cache_one = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device='cpu', cache_size=1)
    _ = tts_cache_one.infer(audio_prompt="tests/dummy_prompt_1.wav", text="Cache size one test 1", output_path=None)
    assert "tests/dummy_prompt_1.wav" in tts_cache_one.cond_mel_cache
    _ = tts_cache_one.infer(audio_prompt="tests/dummy_prompt_2.wav", text="Cache size one test 2", output_path=None)
    assert "tests/dummy_prompt_1.wav" not in tts_cache_one.cond_mel_cache, "Test Case 4 Failed: Old item not evicted with cache_size=1 for cond_mel_cache"
    assert "tests/dummy_prompt_2.wav" in tts_cache_one.cond_mel_cache
    assert "tests/dummy_prompt_1.wav" not in tts_cache_one.latent_cache, "Test Case 4 Failed: Old item not evicted with cache_size=1 for latent_cache"
    assert "tests/dummy_prompt_2.wav" in tts_cache_one.latent_cache
    print("Test Case 4 Passed!")

    # Cleanup dummy files
    os.remove("tests/dummy_prompt_1.wav")
    os.remove("tests/dummy_prompt_2.wav")
    os.remove("tests/dummy_prompt_3.wav")
    
    # Remove any generated output files if they were created by mistake (output_path=None should prevent this)
    # For safety, this example won't delete recursively.
    # if os.path.exists("outputs") and os.path.isdir("outputs"):
    #     for item in os.listdir("outputs"): # Only remove files generated by this test if possible
    #         if item.startswith("Cache_test_output_"):
    #             os.remove(os.path.join("outputs", item))


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist, as original regression test does
    # This is not strictly needed for these tests if output_path=None is always used
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    main()
    print("\nAll caching tests completed successfully.")

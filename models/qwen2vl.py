import logging
import time
from typing import Union, Optional
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.utils import encode_image_base64
except ImportError:
    logging.warning("VLLM not installed. Please install vllm to use Qwen2VL model.")
    LLM = None
    SamplingParams = None


class Qwen2VL_Model:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        temperature: float = 0.2,
        top_p: float = 0.99,
        max_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        patience: int = 1000,
        sleep_time: float = 0,
    ):
        if LLM is None:
            raise ImportError("VLLM is required for Qwen2VL model. Please install vllm.")
        
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.patience = patience
        self.sleep_time = sleep_time
        
        # Initialize VLLM model
        logging.info(f"Initializing VLLM with model: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192*2,  # Increased back to handle longer prompts
            limit_mm_per_prompt={"image": 1},  # Reduced from 5 to 1 for stability
            enforce_eager=True,  # Disable CUDA graphs to avoid assertion errors
            disable_custom_all_reduce=True,  # Disable custom all-reduce for multi-GPU stability
        )
        
        # Initialize sampling parameters
        # Ensure top_p is not None for VLLM compatibility
        if top_p is None:
            top_p = 0.99
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=None,
        )
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logging.info("Qwen2VL model initialized successfully")

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for VLLM"""
        buffered = BytesIO()
        # Convert to RGB for all non-RGB modes to ensure JPEG compatibility
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def get_response(self, user_prompt: str, decoded_image: Union[Image.Image, None] = None, num_answers: int = 1) -> Union[str, list]:
        """Generate response(s) using VLLM backend
        
        Args:
            user_prompt: The prompt text
            decoded_image: Optional image input
            num_answers: Number of answers to generate (default: 1)
            
        Returns:
            str if num_answers=1, list of str if num_answers>1
        """
        patience = self.patience
        
        # Create sampling parameters for this specific request
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=num_answers,  # Generate multiple outputs
            stop_token_ids=None,
        )
        
        while patience > 0:
            patience -= 1
            try:
                # Prepare the conversation
                if decoded_image is not None:
                    # Encode image for VLLM
                    image_base64 = self._encode_image(decoded_image)
                    #TODO: system prompt / user prompt 对齐baseline
                    # Format the prompt with image for Qwen2VL
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text", 
                                    "text": user_prompt
                                }
                            ]
                        }
                    ]
                else:
                    # Text-only prompt
                    messages = [
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                
                prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # text
                prompt_str += "Let's think step by step."
                
                # For VLLM generate, create proper inputs
                if decoded_image is not None:
                    # For multimodal input with VLLM
                    inputs = {
                        "prompt": prompt_str,
                        "multi_modal_data": {"image": decoded_image}
                    }
                    outputs = self.llm.generate(
                        inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                else:
                    # For text-only input
                    outputs = self.llm.generate(
                        [prompt_str],  # VLLM expects a list of prompts
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                logging.info(f"decoded_image: {decoded_image}\nPrompt : {prompt_str}\nOutputs : {outputs[0].outputs[0].text}")
                
                if outputs and len(outputs) > 0:
                    # Extract all generated responses
                    responses = []
                    for output in outputs[0].outputs:
                        response = output.text.strip()
                        if response and response != "":
                            responses.append(response)
                    
                    if responses:
                        # Return single response if num_answers=1, otherwise return list
                        if num_answers == 1:
                            return responses[0]
                        else:
                            # Pad with empty responses if we got fewer than requested
                            while len(responses) < num_answers:
                                responses.append("")
                            return responses[:num_answers]
                
                logging.warning("Empty response generated, retrying...")
                
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                
                # Handle specific VLLM errors
                if "out of memory" in str(e).lower():
                    logging.error("GPU out of memory. Consider reducing batch size or model size.")
                    break
                elif "too long to fit into the model" in str(e).lower():
                    logging.error("Prompt too long for model context. Skipping this sample.")
                    break
                elif "max_tokens" in str(e).lower():
                    # Reduce max tokens and retry
                    sampling_params.max_tokens = max(32, int(sampling_params.max_tokens * 0.8))
                    logging.warning(f"Reduced max_tokens to {sampling_params.max_tokens}")
                elif "cannot write mode" in str(e).lower() and "as JPEG" in str(e).lower():
                    logging.error("Image format error. This should be fixed by RGB conversion.")
                    break
                
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        
        # Return appropriate empty response based on num_answers
        if num_answers == 1:
            return ""
        else:
            return [""] * num_answers

    def __del__(self):
        """Cleanup VLLM resources"""
        try:
            if hasattr(self, 'llm'):
                del self.llm
        except:
            pass 
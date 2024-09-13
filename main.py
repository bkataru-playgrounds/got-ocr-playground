import cv2
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)

model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0', 
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().cuda()

# input your test image
image_file = './test/one.jpeg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

print(res)
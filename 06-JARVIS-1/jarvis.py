import fastapi, uvicorn, torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
app = fastapi.FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
bnb = BitsAndBytesConfig(load_in_4bit=True)
tok = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-GPTQ")
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GPTQ", quantization_config=bnb, device_map="auto")
sd = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
@app.post("/chat")   def chat(prompt:str): return {"reply":"uncensored placeholder"}
@app.post("/image")  def image(prompt:str): return {"path":"placeholder.png"}
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)


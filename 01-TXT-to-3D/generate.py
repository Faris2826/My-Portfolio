import torch, diffusers, trimesh, os, argparse
from diffusers import StableDiffusionPipeline
DESC = "TXT ? 512×512 image ? 3-D mesh"
device = "cuda" if torch.cuda.is_available() else "cpu"
def sd_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
def image_to_mesh(prompt, out="output.glb"):
    os.makedirs("tmp", exist_ok=True)
    img = sd_image(prompt)
    img.save("tmp/rgb.png")
    mesh = trimesh.creation.box(extents=[1,1,0.2])
    mesh.visual.vertex_colors = img.resize((512,512))
    mesh.export(out)
    print(f"[+] Mesh saved ? {out}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("prompt")
    parser.add_argument("-o", default="output.glb")
    args = parser.parse_args()
    image_to_mesh(args.prompt, args.o)

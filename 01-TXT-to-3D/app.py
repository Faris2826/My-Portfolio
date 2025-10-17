import torch, os, subprocess, sys, shutil, gradio as gr
from diffusers import StableDiffusionPipeline
from pytorch3d.utils import ico_sphere
from pytorch3d.io import save_obj, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)

def generate_4views(prompt):
    views = ["front", "right", "back", "left"]
    images = []
    for v in views:
        img = PIPE(f"{prompt}, {v} view, white background", num_inference_steps=25, guidance_scale=7.5).images[0]
        images.append(img)
    return images

def images_to_mesh(images, out="output.glb"):
    os.makedirs("tmp", exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"tmp/{i}.png")
    # naive depth extrusion for demo
    depth = 255 - np.array(images[0].convert('L'))
    vertices, faces = [], []
    h, w = depth.shape
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            z = depth[y, x] / 255.0 * 2
            vertices.append([x/w*2-1, -y/h*2+1, z])
    # simple quad grid
    verts = torch.tensor(vertices, dtype=torch.float32)
    idx = 0
    face_list = []
    for y in range(0, h//4-1):
        for x in range(0, w//4-1):
            i = y*(w//4) + x
            face_list.append([i, i+1, i+w//4+1, i+w//4])
    faces = torch.tensor(face_list, dtype=torch.int64)
    mesh = Meshes(verts=[verts], faces=[faces])
    save_ply("tmp/mesh.ply", verts, faces)
    shutil.copy("tmp/mesh.ply", out)
    print(f"[+] Mesh saved ? {out}")
    return out

def gradio_fn(prompt):
    imgs = generate_4views(prompt)
    mesh_path = images_to_mesh(imgs)
    return imgs + [mesh_path]

iface = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Textbox(label="Prompt"),
    outputs=[gr.Image(type="pil")]*4 + gr.File(label="Download .ply"),
    title="TXT ? 3-D",
    description="Type a prompt, get 4 views + 3-D mesh in ~2 min on RTX 3060."
)
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

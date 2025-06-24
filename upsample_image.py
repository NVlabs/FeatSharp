import argparse
import os

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.v2.functional import to_image, to_dtype_image, to_pil_image

from featsharp.builder import load_from_file
from featsharp.util import pca, extract_normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsample an image using a specified upsampler.")
    parser.add_argument("--upsampler", type=str, required=True, help="Path to the upsampler model file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the upsampled image.")
    parser.add_argument("--no-upsample", action="store_true",
                        help="If set, the upsampling will not be performed, and the low-res features will be saved.")
    parser.add_argument("--rand-rot-seed", type=int, default=None,
                        help="Random seed for rotation during PCA. If None, no rotation is applied.")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        input_images = [args.input]
        output_images = [args.output]
    elif os.path.isdir(args.input):
        input_images = []
        output_images = []
        for root, _, files in os.walk(args.input):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(root, f)
                    rel_path = os.path.relpath(input_path, args.input)
                    output_path = os.path.join(args.output, rel_path)
                    input_images.append(input_path)
                    output_images.append(output_path)
        # Create all necessary output directories
        for out_path in output_images:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        raise ValueError(f"Input path {args.input} is neither a file nor a directory.")

    ups = load_from_file(args.upsampler)
    ups.eval().cuda()

    input_size = ups.input_size

    conditioner = extract_normalize(ups.input_conditioner)

    for input_image, output_image in zip(input_images, output_images):
        img: Image.Image = Image.open(input_image).convert('RGB')

        if img.size != (input_size, input_size):
            min_dim = min(img.size)
            scale_factor = input_size / min_dim
            new_size = (int(round(img.size[0] * scale_factor)), int(round(img.size[1] * scale_factor)))
            img = img.resize(new_size, Image.BICUBIC)

            left = (img.size[0] - input_size) // 2
            top = (img.size[1] - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))

        img = to_image(img)
        img = to_dtype_image(img, dtype=torch.float32, scale=True)
        img = conditioner(img[None])

        img = img.cuda()

        with torch.no_grad():
            low_res_input = F.interpolate(img, size=(ups.featurizer.input_size,) * 2, mode='bilinear', align_corners=False)
            low_res = ups.featurizer(low_res_input, return_summary=False)

            low_res2, high_res, _ = ups(img, denormalize=False, return_summary=False)

        [low_res], pca_info = pca([low_res.cpu()], rand_rot_seed=args.rand_rot_seed)

        if args.no_upsample:
            [low_res2], _ = pca([low_res2.cpu()], fit_pca=pca_info, rand_rot_seed=args.rand_rot_seed)
            pca_feats = low_res2
        else:
            [high_res], _ = pca([high_res.cpu()], fit_pca=pca_info, rand_rot_seed=args.rand_rot_seed)
            pca_feats = high_res

        pil_out: Image.Image = to_pil_image(pca_feats[0])

        pil_out = pil_out.resize((min_dim,) * 2, Image.NEAREST)

        pil_out.save(output_image)
        print(f"Upsampled image saved to {output_image}")

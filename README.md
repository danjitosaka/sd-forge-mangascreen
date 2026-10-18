# sd-forge-mangascreen

A native **SD WebUI Forge / Forge Neo** port and studio extension of [ComfyUI-MangaScreen](https://github.com/ashukae/ComfyUI-MangaScreen).

Turn any illustration into a clean black-and-white manga or comic page — lines kept crisp, values pushed into deliberate flat tone regions, real 45° AM halftone screentones rendered with 3x supersampling, white fold cutouts inside solid black ("beta"), and comic panel framing.

---

## Key Features

- **100% Model-Agnostic**: Works with any diffusion architecture (**SDXL, Illustrious / NoobAI, Flux, Z-Image / ZIT, SD 1.5, Qwen**) and also runs completely standalone on existing images without loading any checkpoint into VRAM.
- **Dual Mode Integration**:
  1. **Dedicated `MangaScreen Studio` Tab**: Convert single images or entire batch folders interactively, inspect intermediate pipeline stages (`Manga Levels`, `Pen Lines`, `Manga Finish Pre-Panel`), and switch between one-click presets.
  2. **`txt2img` & `img2img` Post-Processing Script**: Automatically converts generated images at the end of sampling, with options to keep the original image alongside the finished Manga page and embed all parameters in PNG info.
- **Built-In Lineart Preprocessor Integration**: Seamlessly hooks into Forge's native `lineart_anime_denoise` (`erika.pth` — *Manga Line Extraction*) and `lineart_anime` (`netG.pth`), or runs in zero-VRAM pure OpenCV mode, or accepts a custom external line map.

---

## Pipeline & Parameters

### Pre-Scaling
- **Working Resolution (`scale_mp`)**: Scales the working image to a target megapixel count using Lanczos interpolation (`2.0 MP` recommended so halftone dots have at least ~5 px per dot; `0` keeps the original image size).

### 1. Manga Levels
- **`levels_black` / `levels_white` (`0.20` / `0.70`)**: Input black and white thresholds. Lower `white` until skin and light hair turn into clean paper white.
- **`levels_gamma` (`1.30`)**: Midtone gamma (> 1.0 brightens midtones).

### 2. Pen Lines
- **`pen_mode`**:
  - `Manga Lineart (erika.pth - Recommended)`
  - `Anime Lineart (netG.pth)`
  - `Internal OpenCV Lines Only (Zero VRAM)`
  - `Custom Line Map (External Image)`
- **`pen_source`**: Extract lines from the `Manga Levels` output (recommended) or from the original image.
- **`pen_resolution`**: Detector resolution (`1280` default; `0` matches image width).

### 3. Manga Finish
- **`screen_lines` (`250`)**: Rows of halftone dots across the image width (`120` coarse, `200` normal, `250-300` fine).
- **`white_point` / `black_point` (`0.80` / `0.30`)**: Paper white and solid black ("beta") cutoffs. Raise `black_point` to `0.45-0.55` to push dark clothing into solid black ink.
- **`light_tone` / `dark_tone` (`0.18` / `0.42`)**: Dot coverage of the two flat screentone levels.
- **`gradation` (`1.00`)**: Smooth gradation screentone for regions containing continuous shadow ramps (`0.0` = flat tone steps only).
- **`pen_threshold` (`0.30`)**: Lower values draw more extracted pen lines, cloth folds, and hatching on top.
- **`white_lines` (`0.40`)**: Cuts out main folds and seams in thin white lines inside solid black ("beta") regions.
- **`min_region` (`600`)**: Minimum tone region area (in pixels at 1024px width); smaller fragments are merged into surrounding tones.
- **`screen_angle` (`45.0°`)**: Halftone screen angle.

### 4. Manga Panel
- **`frame`**: `original`, `4:3 wide`, `16:9 wide`, `3:4 tall`, `1:1`.
- **`crop_x` / `crop_y` (`0.50` / `0.40`)**: Horizontal and vertical panel crop center.
- **`border` / `margin` (`5` / `60`)**: Panel border thickness and outer paper margin (in pixels at 1024px width).
- **`paper`**: `white`, `off-white`, `cream`.
- **`ink` (`0.50`)**: Simulates printed ink spread and organic black density (`0.0` = pure digital black and white).

---

## Installation

Clone or extract into your WebUI `extensions` directory:

```bash
cd sd-webui-forge-neo/extensions
git clone <repo-url> sd-forge-mangascreen
```

Restart the WebUI.

---

## Credits & License

- Original algorithm and ComfyUI nodes by [ashukae — ComfyUI-MangaScreen](https://github.com/ashukae/ComfyUI-MangaScreen) (MIT License).

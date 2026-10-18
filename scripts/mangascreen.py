import os
import sys
import time
import gradio as gr
from PIL import Image

# Ensure extension root is on sys.path so lib_mangascreen is always importable
EXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

from lib_mangascreen import (
    FRAMES,
    PAPERS,
    PEN_MODES,
    PEN_SOURCES,
    PRESETS,
    process_manga_pipeline,
)
from modules import images, paths, script_callbacks, scripts, shared
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion


QWEN_PROMPT_STANDARD = (
    "Redraw <image1> as a printed black-and-white manga page panel, as if inked by the original artist "
    "for serialization. Keep the exact composition, pose, character design, face, expression and drawing "
    "style of <image1>. Use confident ink lines with varied line weight, solid black for dark hair and "
    "deep shadows, white paper for skin and light surfaces, ink hatching lines for shadows on cloth, "
    "and regular screentone only where a mid tone is needed. No color and no smooth gradients. "
    "Do not add anything that is not in <image1>."
)

QWEN_PROMPT_BOLD_BETA = (
    "Redraw <image1> as a printed black-and-white manga page panel, as if inked by the original artist "
    "for serialization. Keep the exact composition, pose, character design, face, expression and drawing "
    "style of <image1>. Use confident ink lines with varied line weight. Decide every area as paper white, "
    "screentone, or solid black ink: anything that is dark or black in color in <image1> (dark clothing, "
    "dark hair, deep shadows) is filled with solid black ink, with its folds and seams left as thin white "
    "lines; skin and light surfaces stay white paper; mid colors become flat screentone. No grey gradients, "
    "no soft airbrushed shading, no color. Do not add anything that is not in <image1>."
)


def _build_parameter_controls(prefix: str = "ms"):
    """Build reusable Gradio parameter controls for both Script and Studio Tab."""
    preset_names = list(PRESETS.keys())
    default_preset = preset_names[0]
    d = PRESETS[default_preset]

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            choices=preset_names,
            value=default_preset,
            label="Quick Preset",
            info="Load a pre-calibrated profile (all sliders below remain freely adjustable)",
            elem_id=f"{prefix}_preset",
        )
        scale_mp = gr.Slider(
            minimum=0.0,
            maximum=4.0,
            value=d["scale_mp"],
            step=0.25,
            label="Working Resolution (Megapixels)",
            info="2.0 MP is recommended for crisp halftone dots (~5px+ per dot). 0 = keep original size",
            elem_id=f"{prefix}_scale_mp",
        )

    with gr.Accordion("1. Manga Levels (Grayscale & Input Levels Pre-Processing)", open=True):
        enable_levels = gr.Checkbox(
            value=d["enable_levels"],
            label="Enable Manga Levels (Recommended for color illustrations; disable if input is already B&W)",
            elem_id=f"{prefix}_enable_levels",
        )
        with gr.Row():
            levels_black = gr.Slider(
                minimum=0.00,
                maximum=0.90,
                value=d["levels_black"],
                step=0.01,
                label="Black Point (black)",
                info="Input darker than this becomes pure black (default: 0.20)",
                elem_id=f"{prefix}_levels_black",
            )
            levels_white = gr.Slider(
                minimum=0.10,
                maximum=1.00,
                value=d["levels_white"],
                step=0.01,
                label="White Point (white)",
                info="Input brighter than this becomes paper white. Lower until skin and light hair lose their tone (default: 0.70)",
                elem_id=f"{prefix}_levels_white",
            )
            levels_gamma = gr.Slider(
                minimum=0.30,
                maximum=3.00,
                value=d["levels_gamma"],
                step=0.05,
                label="Midtone Gamma (gamma)",
                info="Values above 1.0 brighten the midtones (default: 1.30)",
                elem_id=f"{prefix}_levels_gamma",
            )

    with gr.Accordion("2. Pen Lines (Auxiliary Manga Lineart Extraction)", open=True):
        with gr.Row():
            pen_mode = gr.Dropdown(
                choices=PEN_MODES,
                value=d["pen_mode"],
                label="Pen Line Extractor",
                info="erika.pth (Manga Lineart) is the official detector used by ComfyUI-MangaScreen",
                elem_id=f"{prefix}_pen_mode",
            )
            pen_source = gr.Dropdown(
                choices=PEN_SOURCES,
                value=d["pen_source"],
                label="Line Extraction Source",
                info="Official workflow extracts lines from the Manga Levels output",
                elem_id=f"{prefix}_pen_source",
            )
            pen_resolution = gr.Slider(
                minimum=0,
                maximum=2048,
                value=d["pen_resolution"],
                step=64,
                label="Detector Resolution (px)",
                info="1280px recommended; 0 = match image width",
                elem_id=f"{prefix}_pen_resolution",
            )
        custom_pen_image = gr.Image(
            label="Custom Line Map (Used only when 'Custom Line Map' is selected)",
            type="pil",
            sources=["upload"],
            visible=False,
            elem_id=f"{prefix}_custom_pen_image",
        )

        def _toggle_custom_pen(mode_val):
            return gr.update(visible=("Custom" in str(mode_val)))

        pen_mode.change(fn=_toggle_custom_pen, inputs=[pen_mode], outputs=[custom_pen_image], queue=False, show_progress=False)

    with gr.Accordion("3. Manga Finish (45° Halftone Screen, Flat Tones, Gradation & White Cut)", open=True):
        enable_finish = gr.Checkbox(
            value=d["enable_finish"],
            label="Enable Manga Finish (Screentone & Inking Engine)",
            elem_id=f"{prefix}_enable_finish",
        )
        with gr.Row():
            screen_lines = gr.Slider(
                minimum=40,
                maximum=600,
                value=d["screen_lines"],
                step=10,
                label="Screen Lines (screen_lines)",
                info="Rows of dots across the image width. 120 = coarse, 200 = normal, 250-300 = fine",
                elem_id=f"{prefix}_screen_lines",
            )
            white_point = gr.Slider(
                minimum=0.30,
                maximum=1.00,
                value=d["white_point"],
                step=0.01,
                label="Paper White Threshold (white_point)",
                info="Brighter than this becomes paper white. Lower it to clear more tone (default: 0.80)",
                elem_id=f"{prefix}_white_point",
            )
            black_point = gr.Slider(
                minimum=0.00,
                maximum=0.70,
                value=d["black_point"],
                step=0.01,
                label="Solid Black / Beta Threshold (black_point)",
                info="Darker than this becomes solid black ink. Raise to 0.45-0.55 for dark clothes (default: 0.30)",
                elem_id=f"{prefix}_black_point",
            )
        with gr.Row():
            light_tone = gr.Slider(
                minimum=0.02,
                maximum=0.90,
                value=d["light_tone"],
                step=0.01,
                label="Light Tone Coverage (light_tone)",
                info="Dot coverage of the light flat tone (default: 0.18)",
                elem_id=f"{prefix}_light_tone",
            )
            dark_tone = gr.Slider(
                minimum=0.02,
                maximum=0.95,
                value=d["dark_tone"],
                step=0.01,
                label="Dark Tone Coverage (dark_tone)",
                info="Dot coverage of the dark flat tone (default: 0.42)",
                elem_id=f"{prefix}_dark_tone",
            )
            gradation = gr.Slider(
                minimum=0.00,
                maximum=1.00,
                value=d["gradation"],
                step=0.05,
                label="Gradation Tone (gradation)",
                info="1.0 = smooth shadow gradation screen; 0.0 = flat tone steps only",
                elem_id=f"{prefix}_gradation",
            )
        with gr.Row():
            pen_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=d["pen_threshold"],
                step=0.05,
                label="Pen Line Threshold (pen_threshold)",
                info="Lower draws more extracted pen lines, folds, and hatching on top (default: 0.30)",
                elem_id=f"{prefix}_pen_threshold",
            )
            white_lines = gr.Slider(
                minimum=0.00,
                maximum=1.00,
                value=d["white_lines"],
                step=0.05,
                label="White Lines in Solid Black (white_lines)",
                info="Cuts out main folds and seams in white inside solid black areas (0 = off, default: 0.40)",
                elem_id=f"{prefix}_white_lines",
            )
        with gr.Row():
            min_region = gr.Slider(
                minimum=50,
                maximum=2000,
                value=d["min_region"],
                step=50,
                label="Min Tone Region Size (min_region px)",
                info="Merges tone fragments smaller than this area (at 1024px width) into surrounding tones (default: 600)",
                elem_id=f"{prefix}_min_region",
            )
            screen_angle = gr.Slider(
                minimum=0.0,
                maximum=90.0,
                value=d["screen_angle"],
                step=5.0,
                label="Halftone Screen Angle (°)",
                info="Traditional manga screentone angle is 45°",
                elem_id=f"{prefix}_screen_angle",
            )

    with gr.Accordion("4. Manga Panel (Panel Framing, Border, Margin, Paper & Printed Ink)", open=True):
        enable_panel = gr.Checkbox(
            value=d["enable_panel"],
            label="Enable Manga Panel (Aspect Crop, Panel Border, Margin & Paper/Ink Feel)",
            elem_id=f"{prefix}_enable_panel",
        )
        with gr.Row():
            frame = gr.Dropdown(
                choices=list(FRAMES.keys()),
                value=d["frame"],
                label="Panel Shape (frame)",
                info="'original' keeps full image aspect; others crop to panel ratio",
                elem_id=f"{prefix}_frame",
            )
            paper = gr.Dropdown(
                choices=list(PAPERS.keys()),
                value=d["paper"],
                label="Paper Stock (paper)",
                info="Paper background tint and subtle fibre texture",
                elem_id=f"{prefix}_paper",
            )
            ink = gr.Slider(
                minimum=0.00,
                maximum=1.00,
                value=d["ink"],
                step=0.05,
                label="Printed Ink Feel (ink)",
                info="0.5 = soft printed ink edges and organic blacks; 0.0 = pure digital black & white",
                elem_id=f"{prefix}_ink",
            )
        with gr.Row():
            border = gr.Slider(
                minimum=0,
                maximum=40,
                value=d["border"],
                step=1,
                label="Panel Border Thickness (border px)",
                info="Border thickness in pixels at 1024px width (default: 5)",
                elem_id=f"{prefix}_border",
            )
            margin = gr.Slider(
                minimum=0,
                maximum=400,
                value=d["margin"],
                step=5,
                label="Outer Paper Margin (margin px)",
                info="Paper margin around the panel in pixels at 1024px width (default: 60)",
                elem_id=f"{prefix}_margin",
            )
            crop_x = gr.Slider(
                minimum=0.00,
                maximum=1.00,
                value=d["crop_x"],
                step=0.05,
                label="Horizontal Crop Position (crop_x)",
                info="Where the panel sits horizontally when cropped (0 = left, 0.5 = center, 1 = right)",
                elem_id=f"{prefix}_crop_x",
            )
            crop_y = gr.Slider(
                minimum=0.00,
                maximum=1.00,
                value=d["crop_y"],
                step=0.05,
                label="Vertical Crop Position (crop_y)",
                info="Where the panel sits vertically when cropped (0 = top, 0.4 = upper-mid, 1 = bottom)",
                elem_id=f"{prefix}_crop_y",
            )

    all_param_components = [
        scale_mp,
        enable_levels,
        levels_black,
        levels_white,
        levels_gamma,
        pen_mode,
        pen_source,
        pen_resolution,
        custom_pen_image,
        enable_finish,
        screen_lines,
        white_point,
        black_point,
        light_tone,
        dark_tone,
        gradation,
        pen_threshold,
        white_lines,
        min_region,
        screen_angle,
        enable_panel,
        frame,
        crop_x,
        crop_y,
        border,
        margin,
        paper,
        ink,
    ]

    preset_target_components = [
        scale_mp,
        enable_levels,
        levels_black,
        levels_white,
        levels_gamma,
        pen_mode,
        pen_source,
        pen_resolution,
        enable_finish,
        screen_lines,
        white_point,
        black_point,
        light_tone,
        dark_tone,
        gradation,
        pen_threshold,
        white_lines,
        min_region,
        screen_angle,
        enable_panel,
        frame,
        crop_x,
        crop_y,
        border,
        margin,
        paper,
        ink,
    ]

    def _apply_preset(preset_name: str):
        p_cfg = PRESETS.get(preset_name)
        if not p_cfg:
            return [gr.skip()] * len(preset_target_components)
        return [
            gr.update(value=p_cfg["scale_mp"]),
            gr.update(value=p_cfg["enable_levels"]),
            gr.update(value=p_cfg["levels_black"]),
            gr.update(value=p_cfg["levels_white"]),
            gr.update(value=p_cfg["levels_gamma"]),
            gr.update(value=p_cfg["pen_mode"]),
            gr.update(value=p_cfg["pen_source"]),
            gr.update(value=p_cfg["pen_resolution"]),
            gr.update(value=p_cfg["enable_finish"]),
            gr.update(value=p_cfg["screen_lines"]),
            gr.update(value=p_cfg["white_point"]),
            gr.update(value=p_cfg["black_point"]),
            gr.update(value=p_cfg["light_tone"]),
            gr.update(value=p_cfg["dark_tone"]),
            gr.update(value=p_cfg["gradation"]),
            gr.update(value=p_cfg["pen_threshold"]),
            gr.update(value=p_cfg["white_lines"]),
            gr.update(value=p_cfg["min_region"]),
            gr.update(value=p_cfg["screen_angle"]),
            gr.update(value=p_cfg["enable_panel"]),
            gr.update(value=p_cfg["frame"]),
            gr.update(value=p_cfg["crop_x"]),
            gr.update(value=p_cfg["crop_y"]),
            gr.update(value=p_cfg["border"]),
            gr.update(value=p_cfg["margin"]),
            gr.update(value=p_cfg["paper"]),
            gr.update(value=p_cfg["ink"]),
        ]

    preset_dropdown.change(
        fn=_apply_preset,
        inputs=[preset_dropdown],
        outputs=preset_target_components,
        queue=False,
        show_progress=False,
    )

    return all_param_components


class MangaScreenScript(scripts.Script):
    sorting_priority = 1250

    def title(self):
        return "MangaScreen (Manga Page & Halftone Filter)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab_prefix = "ms_i2i" if is_img2img else "ms_t2i"
        with InputAccordion(value=False, label=self.title(), elem_id=f"{tab_prefix}_accordion") as enable:
            gr.HTML(
                "<div style='margin-bottom:6px;opacity:0.9;'>"
                "<b>Model-Agnostic Manga Filter:</b> works with any checkpoint (<b>SDXL, Illustrious, Flux, Z-Image, SD1.5, Qwen</b>). "
                "Turns generated images into crisp B&W manga pages with flat tones, 45° halftone screentone, and panel framing."
                "</div>"
            )
            with gr.Row():
                keep_original = gr.Checkbox(
                    value=True,
                    label="Keep original image in output gallery",
                    elem_id=f"{tab_prefix}_keep_original",
                )
                save_manga = gr.Checkbox(
                    value=True,
                    label="Automatically save Manga page to disk (-manga suffix)",
                    elem_id=f"{tab_prefix}_save_manga",
                )

            param_controls = _build_parameter_controls(prefix=tab_prefix)

        (
            scale_mp,
            enable_levels,
            levels_black,
            levels_white,
            levels_gamma,
            pen_mode,
            pen_source,
            pen_resolution,
            custom_pen_image,
            enable_finish,
            screen_lines,
            white_point,
            black_point,
            light_tone,
            dark_tone,
            gradation,
            pen_threshold,
            white_lines,
            min_region,
            screen_angle,
            enable_panel,
            frame,
            crop_x,
            crop_y,
            border,
            margin,
            paper,
            ink,
        ) = param_controls

        self.infotext_fields = [
            PasteField(enable, "MangaScreen Enable"),
            PasteField(scale_mp, "MangaScreen Scale MP"),
            PasteField(enable_levels, "MangaScreen Levels"),
            PasteField(levels_black, "MangaScreen Levels Black"),
            PasteField(levels_white, "MangaScreen Levels White"),
            PasteField(levels_gamma, "MangaScreen Levels Gamma"),
            PasteField(pen_mode, "MangaScreen Pen Mode"),
            PasteField(pen_source, "MangaScreen Pen Source"),
            PasteField(pen_resolution, "MangaScreen Pen Res"),
            PasteField(enable_finish, "MangaScreen Finish"),
            PasteField(screen_lines, "MangaScreen Screen Lines"),
            PasteField(white_point, "MangaScreen White Point"),
            PasteField(black_point, "MangaScreen Black Point"),
            PasteField(light_tone, "MangaScreen Light Tone"),
            PasteField(dark_tone, "MangaScreen Dark Tone"),
            PasteField(gradation, "MangaScreen Gradation"),
            PasteField(pen_threshold, "MangaScreen Pen Threshold"),
            PasteField(white_lines, "MangaScreen White Lines"),
            PasteField(min_region, "MangaScreen Min Region"),
            PasteField(screen_angle, "MangaScreen Angle"),
            PasteField(enable_panel, "MangaScreen Panel"),
            PasteField(frame, "MangaScreen Frame"),
            PasteField(crop_x, "MangaScreen Crop X"),
            PasteField(crop_y, "MangaScreen Crop Y"),
            PasteField(border, "MangaScreen Border"),
            PasteField(margin, "MangaScreen Margin"),
            PasteField(paper, "MangaScreen Paper"),
            PasteField(ink, "MangaScreen Ink"),
        ]

        return [enable, keep_original, save_manga, *param_controls]

    def postprocess(
        self,
        p,
        processed,
        enable: bool,
        keep_original: bool,
        save_manga: bool,
        scale_mp: float,
        enable_levels: bool,
        levels_black: float,
        levels_white: float,
        levels_gamma: float,
        pen_mode: str,
        pen_source: str,
        pen_resolution: int,
        custom_pen_image: Image.Image | None,
        enable_finish: bool,
        screen_lines: int,
        white_point: float,
        black_point: float,
        light_tone: float,
        dark_tone: float,
        gradation: float,
        pen_threshold: float,
        white_lines: float,
        min_region: int,
        screen_angle: float,
        enable_panel: bool,
        frame: str,
        crop_x: float,
        crop_y: float,
        border: int,
        margin: int,
        paper: str,
        ink: float,
    ):
        if not enable:
            return

        p.extra_generation_params.update(
            {
                "MangaScreen Enable": True,
                "MangaScreen Scale MP": scale_mp,
                "MangaScreen Levels": enable_levels,
                "MangaScreen Levels Black": levels_black,
                "MangaScreen Levels White": levels_white,
                "MangaScreen Levels Gamma": levels_gamma,
                "MangaScreen Pen Mode": pen_mode,
                "MangaScreen Pen Source": pen_source,
                "MangaScreen Pen Res": pen_resolution,
                "MangaScreen Finish": enable_finish,
                "MangaScreen Screen Lines": screen_lines,
                "MangaScreen White Point": white_point,
                "MangaScreen Black Point": black_point,
                "MangaScreen Light Tone": light_tone,
                "MangaScreen Dark Tone": dark_tone,
                "MangaScreen Gradation": gradation,
                "MangaScreen Pen Threshold": pen_threshold,
                "MangaScreen White Lines": white_lines,
                "MangaScreen Min Region": min_region,
                "MangaScreen Angle": screen_angle,
                "MangaScreen Panel": enable_panel,
                "MangaScreen Frame": frame,
                "MangaScreen Crop X": crop_x,
                "MangaScreen Crop Y": crop_y,
                "MangaScreen Border": border,
                "MangaScreen Margin": margin,
                "MangaScreen Paper": paper,
                "MangaScreen Ink": ink,
            }
        )

        first_idx = getattr(processed, "index_of_first_image", 0)
        orig_images = list(processed.images[first_idx:])
        manga_results = []
        manga_infotexts = []

        for idx, img in enumerate(orig_images):
            manga_pil, _ = process_manga_pipeline(
                img,
                scale_mp=scale_mp,
                enable_levels=enable_levels,
                levels_black=levels_black,
                levels_white=levels_white,
                levels_gamma=levels_gamma,
                pen_mode=pen_mode,
                pen_source=pen_source,
                pen_resolution=pen_resolution,
                custom_pen_image=custom_pen_image,
                enable_finish=enable_finish,
                screen_lines=screen_lines,
                white_point=white_point,
                black_point=black_point,
                light_tone=light_tone,
                dark_tone=dark_tone,
                gradation=gradation,
                pen_threshold=pen_threshold,
                white_lines=white_lines,
                min_region=min_region,
                screen_angle=screen_angle,
                enable_panel=enable_panel,
                frame=frame,
                crop_x=crop_x,
                crop_y=crop_y,
                border=border,
                margin=margin,
                paper=paper,
                ink=ink,
                return_intermediates=False,
            )

            info_idx = first_idx + idx
            info_text = (
                processed.infotexts[info_idx]
                if info_idx < len(processed.infotexts)
                else processed.info
            )
            seed = (
                processed.all_seeds[idx]
                if hasattr(processed, "all_seeds") and idx < len(processed.all_seeds)
                else p.seed
            )
            prompt = (
                processed.all_prompts[idx]
                if hasattr(processed, "all_prompts") and idx < len(processed.all_prompts)
                else p.prompt
            )

            if save_manga and p.outpath_samples:
                images.save_image(
                    manga_pil,
                    p.outpath_samples,
                    "",
                    seed,
                    prompt,
                    shared.opts.samples_format,
                    info=info_text,
                    p=p,
                    suffix="-manga",
                )

            manga_results.append(manga_pil)
            manga_infotexts.append(info_text)

        if keep_original:
            processed.images.extend(manga_results)
            processed.infotexts.extend(manga_infotexts)
        else:
            processed.images[first_idx:] = manga_results


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as mangascreen_tab:
        gr.HTML(
            "<div style='padding: 8px 12px; border-radius: 8px; background: rgba(128,128,128,0.1); margin-bottom: 10px;'>"
            "<h3 style='margin: 0 0 4px 0;'>MangaScreen Studio — B&W Manga Page & Halftone Filter</h3>"
            "<p style='margin: 0; opacity: 0.85;'>"
            "Turn any illustration into a clean B&W manga page (crisp ink lines, flat tone areas, "
            "3x supersampled 45° AM halftone screentones, white fold cutouts inside solid black, and panel framing). "
            "<b>Runs standalone without requiring any Diffusion Checkpoint loaded in VRAM.</b>"
            "</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("Single Image"):
                        input_image = gr.Image(
                            label="Input Image (Color or B&W Illustration)",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=420,
                        )
                    with gr.TabItem("Batch Folder"):
                        batch_input_dir = gr.Textbox(
                            label="Input Images Folder",
                            placeholder="e.g. /path/to/input_images",
                        )
                        batch_output_dir = gr.Textbox(
                            label="Output Folder (Leave empty to save in output/mangascreen)",
                            placeholder="Leave empty for default (output/mangascreen)",
                        )

                with gr.Row():
                    run_single_btn = gr.Button("Generate Manga Page", variant="primary", scale=3)
                    run_batch_btn = gr.Button("Process Batch Folder", variant="secondary", scale=2)

                with gr.Row():
                    show_intermediates = gr.Checkbox(
                        value=True,
                        label="Show intermediate stages in gallery (Manga Levels, Pen Lines, Pre-Panel)",
                    )
                    auto_save_studio = gr.Checkbox(
                        value=True,
                        label="Automatically save output to output/mangascreen",
                    )

                studio_params = _build_parameter_controls(prefix="ms_studio")

                with gr.Accordion("Reference Edit Prompts (Optional — for Qwen-Image Edit / Img2Img)", open=False):
                    gr.Markdown(
                        "If you want to run an instruction-based Edit model (such as **Qwen-Image Edit**) "
                        "before passing the image into MangaScreen, here are the official prompts:"
                    )
                    gr.Textbox(
                        value=QWEN_PROMPT_STANDARD,
                        label="Prompt 1 — Standard (QWEN-Manga)",
                        lines=4,
                        interactive=False,
                        show_copy_button=True,
                    )
                    gr.Textbox(
                        value=QWEN_PROMPT_BOLD_BETA,
                        label="Prompt 2 — Bold Beta (Dark Clothes -> Solid Black with White Folds; use black_point = 0.40)",
                        lines=4,
                        interactive=False,
                        show_copy_button=True,
                    )

            with gr.Column(scale=5):
                output_gallery = gr.Gallery(
                    label="MangaScreen Output (Final Page + Intermediate Stages)",
                    show_label=True,
                    columns=2,
                    height=680,
                    object_fit="contain",
                )
                status_info = gr.HTML("")

        def _get_default_out_dir() -> str:
            base_out = getattr(shared.opts, "outdir_extras_samples", "") or os.path.join(paths.data_path, "output", "mangascreen")
            out_dir = os.path.join(os.path.dirname(base_out.rstrip("\\/")), "mangascreen") if "extras" in base_out else base_out
            os.makedirs(out_dir, exist_ok=True)
            return out_dir

        def _run_single(
            img: Image.Image | None,
            show_inter: bool,
            auto_save: bool,
            *params,
        ):
            if img is None:
                return [], "<b style='color:#ff6b6b;'>Please upload or paste an input image first!</b>"

            t0 = time.perf_counter()
            (
                scale_mp,
                enable_levels,
                levels_black,
                levels_white,
                levels_gamma,
                pen_mode,
                pen_source,
                pen_resolution,
                custom_pen_image,
                enable_finish,
                screen_lines,
                white_point,
                black_point,
                light_tone,
                dark_tone,
                gradation,
                pen_threshold,
                white_lines,
                min_region,
                screen_angle,
                enable_panel,
                frame,
                crop_x,
                crop_y,
                border,
                margin,
                paper,
                ink,
            ) = params

            final_pil, intermediates = process_manga_pipeline(
                img,
                scale_mp=scale_mp,
                enable_levels=enable_levels,
                levels_black=levels_black,
                levels_white=levels_white,
                levels_gamma=levels_gamma,
                pen_mode=pen_mode,
                pen_source=pen_source,
                pen_resolution=pen_resolution,
                custom_pen_image=custom_pen_image,
                enable_finish=enable_finish,
                screen_lines=screen_lines,
                white_point=white_point,
                black_point=black_point,
                light_tone=light_tone,
                dark_tone=dark_tone,
                gradation=gradation,
                pen_threshold=pen_threshold,
                white_lines=white_lines,
                min_region=min_region,
                screen_angle=screen_angle,
                enable_panel=enable_panel,
                frame=frame,
                crop_x=crop_x,
                crop_y=crop_y,
                border=border,
                margin=margin,
                paper=paper,
                ink=ink,
                return_intermediates=show_inter,
            )

            saved_msg = ""
            if auto_save:
                out_dir = _get_default_out_dir()
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_file = os.path.join(out_dir, f"manga_{ts}.png")
                final_pil.save(out_file)
                saved_msg = f" | Saved to: <code>{out_file}</code>"

            gallery_items = [(final_pil, "Final Manga Page")]
            if show_inter:
                for label, inter_img in intermediates.items():
                    gallery_items.append((inter_img, label))

            elapsed = time.perf_counter() - t0
            w, h = final_pil.size
            info_html = (
                f"<div style='padding:6px;opacity:0.9;'>"
                f"Completed in <b>{elapsed:.2f}s</b> | Page Resolution: <b>{w}x{h} px</b>{saved_msg}"
                f"</div>"
            )
            return gallery_items, info_html

        def _run_batch(
            in_dir: str,
            out_dir: str,
            *params,
        ):
            in_dir = (in_dir or "").strip().strip('"')
            out_dir = (out_dir or "").strip().strip('"')
            if not in_dir or not os.path.isdir(in_dir):
                return [], "<b style='color:#ff6b6b;'>Please specify a valid input directory!</b>"

            if not out_dir:
                out_dir = _get_default_out_dir()
            os.makedirs(out_dir, exist_ok=True)

            valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            files = [f for f in sorted(os.listdir(in_dir)) if f.lower().endswith(valid_exts)]
            if not files:
                return [], "<b style='color:#ff6b6b;'>No supported images found in the specified folder!</b>"

            (
                scale_mp,
                enable_levels,
                levels_black,
                levels_white,
                levels_gamma,
                pen_mode,
                pen_source,
                pen_resolution,
                custom_pen_image,
                enable_finish,
                screen_lines,
                white_point,
                black_point,
                light_tone,
                dark_tone,
                gradation,
                pen_threshold,
                white_lines,
                min_region,
                screen_angle,
                enable_panel,
                frame,
                crop_x,
                crop_y,
                border,
                margin,
                paper,
                ink,
            ) = params

            t0 = time.perf_counter()
            results = []
            for fname in files:
                fpath = os.path.join(in_dir, fname)
                try:
                    with Image.open(fpath) as pil_in:
                        final_pil, _ = process_manga_pipeline(
                            pil_in,
                            scale_mp=scale_mp,
                            enable_levels=enable_levels,
                            levels_black=levels_black,
                            levels_white=levels_white,
                            levels_gamma=levels_gamma,
                            pen_mode=pen_mode,
                            pen_source=pen_source,
                            pen_resolution=pen_resolution,
                            custom_pen_image=custom_pen_image,
                            enable_finish=enable_finish,
                            screen_lines=screen_lines,
                            white_point=white_point,
                            black_point=black_point,
                            light_tone=light_tone,
                            dark_tone=dark_tone,
                            gradation=gradation,
                            pen_threshold=pen_threshold,
                            white_lines=white_lines,
                            min_region=min_region,
                            screen_angle=screen_angle,
                            enable_panel=enable_panel,
                            frame=frame,
                            crop_x=crop_x,
                            crop_y=crop_y,
                            border=border,
                            margin=margin,
                            paper=paper,
                            ink=ink,
                            return_intermediates=False,
                        )
                    stem, _ = os.path.splitext(fname)
                    out_path = os.path.join(out_dir, f"{stem}_manga.png")
                    final_pil.save(out_path)
                    results.append((final_pil, f"{stem}_manga.png"))
                except Exception as e:
                    print(f"[MangaScreen] Failed processing {fname}: {e}")

            elapsed = time.perf_counter() - t0
            msg = (
                f"<div style='padding:6px;opacity:0.9;'>"
                f"Batch completed: <b>{len(results)}/{len(files)}</b> pages generated in <b>{elapsed:.2f}s</b> | "
                f"Saved to: <code>{out_dir}</code></div>"
            )
            return results, msg

        run_single_btn.click(
            fn=_run_single,
            inputs=[input_image, show_intermediates, auto_save_studio, *studio_params],
            outputs=[output_gallery, status_info],
        )

        run_batch_btn.click(
            fn=_run_batch,
            inputs=[batch_input_dir, batch_output_dir, *studio_params],
            outputs=[output_gallery, status_info],
        )

    return [(mangascreen_tab, "MangaScreen Studio", "mangascreen_studio_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)

from __future__ import annotations

from PIL import Image


def preprocess(img: Image.Image, max_side: int, border: float = 0.0,
               composite_alpha: bool = True) -> Image.Image:
    """Return an RGB copy of ``img`` ready for the vision encoder.

    1. ``composite_alpha``: flatten transparency onto white (otherwise a plain
       RGB conversion, as in Subtask 2 V2).
    2. ``border`` > 0: pad every side with white by that fraction of the image
       width/height (the 10% border of the two-stage Subtask 1 system).
       Padding happens before the size cap, so it cannot enlarge the image
       beyond ``max_side``.
    3. Downscale with Lanczos so the longest side is at most ``max_side``.
    """
    if composite_alpha and (img.mode in ("RGBA", "LA")
                            or (img.mode == "P" and "transparency" in img.info)):
        alpha = img.convert("RGBA").split()[-1]
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=alpha)
        img = bg
    else:
        img = img.convert("RGB")

    if border and border > 0:
        w, h = img.size
        bw, bh = round(w * border), round(h * border)
        if bw > 0 or bh > 0:
            canvas = Image.new("RGB", (w + 2 * bw, h + 2 * bh), (255, 255, 255))
            canvas.paste(img, (bw, bh))
            img = canvas

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def load_image(path, max_side: int, border: float = 0.0,
               composite_alpha: bool = True) -> Image.Image:
    with Image.open(path) as raw:
        return preprocess(raw, max_side, border, composite_alpha)

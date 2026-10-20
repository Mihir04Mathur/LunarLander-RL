"""A state-driven cinematic renderer for Lunar Lander demonstration videos.

This is a presentation layer only. It reads the actual environment observation
and selected action but never feeds pixels or altered rewards back to PPO.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CinematicRenderer:
    """Render an illustrated Moon mission from the real LunarLander state."""

    def __init__(self, width: int, height: int, seed: int = 2026) -> None:
        self.width, self.height, self.seed = width, height, seed
        self.ground_y = int(height * 0.84)
        self.background = self._make_background(seed)
        self.terrain = self._make_terrain(seed)
        self.title_font = self._font("arialbd.ttf", 16)
        self.small_font = self._font("arial.ttf", 12)

    @staticmethod
    def _font(name: str, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()

    def _make_background(self, seed: int) -> Image.Image:
        """Create a stable star field, faint galaxy band, and glowing Earth."""
        pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            t = y / max(self.height - 1, 1)
            pixels[y, :, :] = (2 + int(8 * t), 5 + int(10 * t), 19 + int(25 * t))
        image = Image.fromarray(pixels).convert("RGBA")
        detail = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(detail)
        rng = random.Random(seed)
        for _ in range(self.width * self.height // 180):
            x = rng.randrange(self.width)
            y = int(self.height * 0.35 + x * 0.12 + rng.gauss(0, self.height * 0.13))
            if 0 <= y < int(self.height * 0.72):
                draw.ellipse((x, y, x + 2, y + 2), fill=(145, 177, 255, rng.randrange(8, 35)))
        for _ in range(max(220, self.width * self.height // 700)):
            x, y = rng.randrange(self.width), rng.randrange(max(1, int(self.height * 0.72)))
            brightness = rng.randrange(115, 256)
            radius = 1 if rng.random() < 0.91 else 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(brightness, brightness, min(255, brightness + 15), rng.randrange(120, 256)))

        earth_x, earth_y, radius = int(self.width * 0.82), int(self.height * 0.20), max(30, self.width // 12)
        for ring in range(16, 0, -1):
            extra = ring * 3
            draw.ellipse((earth_x - radius - extra, earth_y - radius - extra, earth_x + radius + extra, earth_y + radius + extra), fill=(55, 134, 255, max(2, 18 - ring)))
        draw.ellipse((earth_x - radius, earth_y - radius, earth_x + radius, earth_y + radius), fill=(45, 107, 166, 255), outline=(190, 232, 255, 220), width=2)
        draw.ellipse((earth_x - radius // 2, earth_y - radius // 3, earth_x + radius // 5, earth_y + radius // 2), fill=(122, 172, 140, 120))
        return Image.alpha_composite(image, detail)

    def _make_terrain(self, seed: int) -> Image.Image:
        """Draw cratered lunar terrain and a clear landing pad."""
        terrain = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(terrain)
        rng = random.Random(seed + 1)
        ridge = [(0, self.ground_y)]
        for x in range(0, self.width + 18, 18):
            wave = math.sin(x / 53) * 5 + math.sin(x / 21) * 2
            ridge.append((x, int(self.ground_y + wave + rng.randint(-3, 3))))
        draw.polygon(ridge + [(self.width, self.height), (0, self.height)], fill=(72, 80, 96, 255))
        draw.line(ridge, fill=(190, 203, 218, 230), width=2)
        for _ in range(24):
            x, y = rng.randrange(-20, self.width), rng.randrange(self.ground_y + 10, self.height)
            radius_x, radius_y = rng.randrange(8, 34), rng.randrange(2, 7)
            draw.ellipse((x, y, x + radius_x, y + radius_y), outline=(33, 39, 52, 160), width=2)
            draw.arc((x + 2, y - 1, x + radius_x, y + radius_y + 3), 185, 350, fill=(178, 190, 204, 80), width=1)
        pad_x, pad_y = self.width // 2, self.ground_y + 5
        draw.polygon([(pad_x - 55, pad_y), (pad_x + 55, pad_y), (pad_x + 42, pad_y + 9), (pad_x - 42, pad_y + 9)], fill=(35, 47, 63, 255), outline=(113, 216, 255, 255))
        draw.rectangle((pad_x - 24, pad_y + 2, pad_x + 24, pad_y + 6), fill=(72, 196, 229, 255))
        for side in (-1, 1):
            beacon_x = pad_x + side * 50
            draw.line((beacon_x, pad_y, beacon_x, pad_y - 26), fill=(125, 145, 164, 255), width=2)
            draw.ellipse((beacon_x - 4, pad_y - 31, beacon_x + 4, pad_y - 23), fill=(255, 185, 84, 255))
        return terrain

    def _rocket(self, observation: np.ndarray, action: int | None, step: int) -> Image.Image:
        """Draw a custom lander whose pose and engine flame follow real state."""
        _, altitude, _, _, angle, _, left_leg, right_leg = observation
        sprite = Image.new("RGBA", (100, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        center = 50
        if action == 2:
            flicker = 7 + (step % 5) * 3
            draw.polygon([(center - 12, 101), (center + 12, 101), (center, 132 + flicker)], fill=(255, 89, 24, 210))
            draw.polygon([(center - 7, 101), (center + 7, 101), (center, 118 + flicker)], fill=(255, 226, 105, 255))
        elif action in (1, 3):
            direction = -1 if action == 1 else 1
            draw.polygon([(center + direction * 23, 71), (center + direction * 41, 75), (center + direction * 23, 81)], fill=(255, 165, 52, 225))
        draw.polygon([(center, 8), (center - 20, 40), (center - 16, 90), (center + 16, 90), (center + 20, 40)], fill=(217, 225, 232, 255), outline=(91, 115, 143, 255))
        draw.polygon([(center, 15), (center - 9, 38), (center + 9, 38)], fill=(117, 212, 246, 255), outline=(225, 250, 255, 255))
        draw.rectangle((center - 18, 84, center + 18, 100), fill=(77, 91, 111, 255), outline=(177, 192, 209, 255))
        leg_spread = 23 if left_leg or right_leg else 18
        draw.line((center - 13, 91, center - leg_spread, 113), fill=(191, 205, 218, 255), width=3)
        draw.line((center + 13, 91, center + leg_spread, 113), fill=(191, 205, 218, 255), width=3)
        draw.line((center - leg_spread - 6, 114, center - leg_spread + 5, 114), fill=(229, 237, 244, 255), width=3)
        draw.line((center + leg_spread - 5, 114, center + leg_spread + 6, 114), fill=(229, 237, 244, 255), width=3)
        rotated = sprite.rotate(-math.degrees(float(angle)), resample=Image.Resampling.BICUBIC, expand=True)
        rocket_x = int(self.width / 2 + float(observation[0]) * self.width * 0.34 - rotated.width / 2)
        rocket_y = int(self.ground_y - float(altitude) * self.height * 0.54 - rotated.height * 0.52)
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        canvas.alpha_composite(rotated, (rocket_x, rocket_y))
        return canvas

    def render(self, raw_frame: np.ndarray, observation: np.ndarray, episode: int, step: int, native_return: float, terminal_status: str | None = None, action: int | None = None) -> np.ndarray:
        """Render an upgraded frame based on the real simulation state."""
        del raw_frame  # State-based art is intentionally more distinct than an overlay.
        image = self.background.copy()
        image.alpha_composite(self.terrain)
        image.alpha_composite(self._rocket(observation, action, step))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        _, altitude, horizontal_speed, vertical_speed, tilt, _, _, _ = observation
        draw.rounded_rectangle((15, 15, 320, 89), radius=10, fill=(4, 12, 31, 228), outline=(100, 216, 255, 220), width=2)
        draw.text((29, 26), "AUTOMATIC LUNAR LANDER", fill=(234, 247, 255, 255), font=self.title_font)
        draw.text((29, 50), f"MISSION AL-01    EPISODE {episode + 1:02d}    T+ {step / 50:05.1f}S", fill=(139, 210, 249, 255), font=self.small_font)
        draw.text((29, 68), f"NATIVE RETURN  {native_return:8.1f}", fill=(255, 204, 111, 255), font=self.small_font)
        draw.rounded_rectangle((width - 210, 15, width - 15, 117), radius=10, fill=(4, 12, 31, 228), outline=(100, 216, 255, 220), width=2)
        telemetry = [f"ALTITUDE  {altitude:7.3f}", f"V-SPEED   {vertical_speed:7.3f}", f"H-SPEED   {horizontal_speed:7.3f}", f"ATTITUDE  {math.degrees(float(tilt)):7.2f} DEG"]
        for line, y in zip(telemetry, range(27, 99, 18)):
            draw.text((width - 196, y), line, fill=(226, 241, 255, 255), font=self.small_font)
        status = terminal_status or "AUTOPILOT - DESCENT GUIDANCE"
        status_color = (255, 204, 100, 255) if terminal_status is None else (116, 246, 171, 255)
        draw.rounded_rectangle((15, height - 47, min(width - 15, 355), height - 15), radius=8, fill=(4, 12, 31, 232), outline=(93, 161, 205, 170))
        draw.text((28, height - 37), status, fill=status_color, font=self.title_font)
        draw.rectangle((0, 0, width, 7), fill=(0, 0, 0, 230))
        draw.rectangle((0, height - 7, width, height), fill=(0, 0, 0, 230))
        image = Image.alpha_composite(image, overlay)
        return np.asarray(image.convert("RGB"))

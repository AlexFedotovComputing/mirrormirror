from pathlib import Path
import re

text = Path(__file__).parents[1].joinpath("mirror_config.txt").read_text("utf-8")
D_MM = float(re.search(r"d\s*=\s*([\d.]+)", text).group(1))



if __name__ == "__main__":
    print(f"Micromirror diameter: {D_MM} mm")


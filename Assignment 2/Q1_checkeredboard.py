import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Rectangle

# Where the PDF is saved
HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "Q1" / "checkerboard_9x6.pdf"

# Board layout: number of squares (the number of INNER corners is one less in each direction)
SQUARES_X = 10
SQUARES_Y = 7
SQUARE_SIZE_MM = 25

# Paper size in millimeters (A4 landscape)
PAGE_WIDTH_MM = 297
PAGE_HEIGHT_MM = 210

MILLIMETERS_PER_INCH = 25.4

# Center the board on the page; the white margin around it is needed for reliable corner detection
boardWidthMm = SQUARES_X * SQUARE_SIZE_MM
boardHeightMm = SQUARES_Y * SQUARE_SIZE_MM
offsetXMm = (PAGE_WIDTH_MM - boardWidthMm) / 2
offsetYMm = (PAGE_HEIGHT_MM - boardHeightMm) / 2
assert offsetXMm >= 10 and offsetYMm >= 10, "Board is too large for the page, reduce SQUARE_SIZE_MM"

# One axes covering the entire page, with the axis units set to millimeters
figure = plt.figure(figsize=(PAGE_WIDTH_MM / MILLIMETERS_PER_INCH, PAGE_HEIGHT_MM / MILLIMETERS_PER_INCH))
axes = figure.add_axes([0, 0, 1, 1])
axes.set_xlim(0, PAGE_WIDTH_MM)
axes.set_ylim(PAGE_HEIGHT_MM, 0)       # y axis pointing down, like image coordinates
axes.axis("off")

# Draw every second square in black (checkerboard pattern)
for row in range(SQUARES_Y):
    for column in range(SQUARES_X):
        if (row + column) % 2 == 0:
            axes.add_patch(Rectangle((offsetXMm + column * SQUARE_SIZE_MM, offsetYMm + row * SQUARE_SIZE_MM),
                                     SQUARE_SIZE_MM, SQUARE_SIZE_MM, facecolor="black", edgecolor="none"))

figure.savefig(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")
print(f"Inner corners to use in the calibration script: INNER_CORNERS = ({SQUARES_X - 1}, {SQUARES_Y - 1})")
print(f"SQUARE_SIZE_M = {SQUARE_SIZE_MM / 1000}")
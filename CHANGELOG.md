# Changelog

## 1.1.0

**Text removal**
- Detection: PP-OCRv5-mobile joins PP-OCRv4; both run at several scales and their boxes are merged per line, which finds
  edge letters and letter-spaced type that a single pass missed.
- Stroke mask: several candidate ink colours per box (histogram peaks, brightest / darkest pixels), ranked by how well they
  separate glyphs from their surroundings. In *thorough mode* the detector re-checks the picture with the mask painted
  over and only accepts a mask after which no text is left (else the whole padded box is used).
- Fill: native-resolution 512 × 512 windows with context, feathered blending and grain matching instead of a down-scaled
  pass, which smeared patterns.
- New fill engine **MI-GAN** (~28 MB, optional) next to **LaMa** and the built-in fast fill. The new *Fill engine* card
  chooses between *Auto*, *Fast*, *MI-GAN* and *LaMa*; models can be downloaded, imported and removed one by one.
  Settings of 1.0 (`use AI` / `always AI`) are migrated.

**Interface**
- Redesigned in black, white and one red after the 1975 NASA Graphics Standards Manual: Liberation Sans, strict
  rectangles, heavy section rules, no gradients, shadows, textures or rounded corners. The wipe-away title card, the
  folding tabs, the queue rows and all dialogs were redrawn; the app icon is now a white eraser on a red square.
- Keyboard focus is visible on all controls and the segmented switches can be driven with the arrow keys.

**Other**
- Fonts: Liberation Sans is the only interface typeface; Archivo Black and Space Mono remain only for the sample picture.

## 1.0.0

First release: batch text removal, single-picture eraser / repair / watermark / crop tools, Windows build.

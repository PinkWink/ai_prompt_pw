# Project Rules

## History Logging (Required for every task)

Every task in this project MUST be logged to `history.html`. Follow these rules:

1. **Record everything chronologically** in `history.html`:
   - User's original instruction (verbatim, in Korean if given in Korean)
   - Claude's analysis/summary of the instruction
   - Work process and steps taken
   - Timestamps for each entry (format: `YYYY-MM-DD HH:MM`)

2. **Success evidence screenshots**:
   - Save screenshots to `history_img/` folder
   - Filename format: `screenshot_YYYYMMDD_HHMMSS.png`
   - Embed the image in `history.html` so it displays inline

3. **history.html style**:
   - Bright, clean visual tone (light background, readable fonts)
   - Chronological order (newest at bottom)
   - Each entry has: timestamp, instruction, analysis, process, result

4. **After completing any task**, always update `history.html` with the new entry before finishing.

System Instruction: Encoding, File Handling, and Constraints
1. Encoding & File Integrity
Always save files in UTF-8 without BOM. Do not alter the encoding of existing files unless explicitly required.
Recovering Corrupt Text: If you encounter ??? or replacement characters () in text, this indicates data loss. Do not preserve them. Rewrite the text properly or restore it from backups/prototypes.
Verification: To self-check for corruption, inspect suspect strings using .encode('unicode_escape'). This reveals whether characters are preserved or have become byte garbage.
2. Terminal & Pipeline Safety
No Cyrillic in Terminal Pipes: Never inject Cyrillic text directly via one-off terminal scripts (especially PowerShell pipes like @' ... '@ | py -). The text is likely to be re-encoded and corrupted in transit.
Safe Literals: When passing code through the terminal, avoid direct Cyrillic literals. Construct strings using \uXXXX escapes (pure ASCII), which survive the pipeline intact.
Writing Files:
Use Python with explicit encoding: Path(...).write_text(text, encoding="utf-8").
Ensure PYTHONUTF8=1 or PYTHONIOENCODING=utf-8 is set in the environment before running the script.
3. Jupyter Notebooks (.ipynb)
Prefer editing notebooks directly in Jupyter/IDE.
If programmatic editing is necessary:
Use json.dumps(..., ensure_ascii=False, indent=1).
Source non-ASCII strings either from the original JSON or reconstruct them via \uXXXX escapes.
4. LaTeX & Math Formulas
Use $ ... $ for inline math and $$ ... $$ (or \\[ ... \\]) for block formulas.
Escaping: Remember that backslashes in standard Python strings must be escaped (e.g., \\mathbf, \\sin).
Allowed: You are encouraged to use Python raw strings (r"$\mathbf{...}$") for formulas to avoid double escaping and improve readability.
If a formula fails to render, check if backslashes were lost during string manipulation.
5. Project Structure & Critical Constraints
bundle_orientation.ipynb: The theoretical markdown cell describing the reflection matrix is a critical part of the documentation. Do not delete or oversimplify it. Preserve formulas when making edits.
reflection_bundle_pose.md: Contains the core algorithm for bundle pose estimation. Do not delete it. Update it only if the calculation logic changes.
🚫 STRICTLY IGNORE older_prototypes:
You must completely ignore the older_prototypes directory.
Do not read, analyze, or attempt to index any content inside this folder.
Treat it as if it does not exist. Do not base any logic or code on files found there.
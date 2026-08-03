"""vibdistill — distill vibration-analysis expertise from big MoE teachers
(GLM-5.2 on NVIDIA NIM, Kimi K3 on Moonshot) into a small local student.

Kept import-light on purpose: heavy dependencies (scipy, torch, transformers,
openai) are imported inside the submodules/functions that need them, so this
package can be extracted and imported before `pip install` has run.
"""

__version__ = "0.2.0"

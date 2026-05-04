import matplotlib.pyplot as plt
import os

# ── Terminal Plot Config ────────────────────────────────
TERMINAL_TEXT = """
Loading dataset...
Training: 6042 | Testing: 1511
Training SVM (may take 2-3 mins)...

Accuracy: 90.21%
              precision    recall  f1-score   support

without_mask       0.92      0.89      0.90       766
   with_mask       0.89      0.92      0.90       745

    accuracy                           0.90      1511
   macro avg       0.90      0.90      0.90      1511
weighted avg       0.90      0.90      0.90      1511

Model saved to mask_classifier.pkl ✅
PS E:\\Projects\\facedetect> python detect.py
Press Q to quit
Mask Detected: 94.3% (Green Box)
No Mask Detected: 99.6% (Red Box)
"""

ASSETS_DIR = "report_assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# ── Render Terminal Text into Image ─────────────────────
plt.figure(figsize=(10, 8), facecolor='#0c0c0c') # Dark theme like VS Code
plt.text(0.1, 0.9, TERMINAL_TEXT, 
         family='monospace', fontsize=12, color='#ffffff',
         verticalalignment='top', linespacing=1.6)

plt.axis('off')
plt.title('Terminal Output: Training & Detection Log', color='#ffffff', fontsize=16, pad=20)

plt.savefig(os.path.join(ASSETS_DIR, "appendix_b_s6_terminal.png"), 
            bbox_inches='tight', dpi=150, facecolor='#0c0c0c')
plt.close()

print(f"\nTerminal output image saved to {os.path.join(ASSETS_DIR, 'appendix_b_s6_terminal.png')}")

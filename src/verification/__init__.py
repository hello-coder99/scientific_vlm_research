# Verification package
from .comparator import verify_chart, compare_text, compare_points
from .report import generate_report
from .visual_verifier import (
    verify_chart_visually,
    VerificationResult,
    VisualProbeResult,
    TextGroundingResult,
    detect_visual_markers,
)
from .visual_debugger import (
    render_verification_overlay,
    extract_evidence_crops,
    generate_html_certificate,
    generate_verification_package,
)

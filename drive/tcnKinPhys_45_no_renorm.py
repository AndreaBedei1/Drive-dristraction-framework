"""PhysioKinTCN using personal calibration only (no P1/P2 renormalisation).

This experiment is identical to ``tcnKinPhys_45.py`` except that kinematic
signals retain their first-50-second personal normalization throughout LOPO,
validation, audit folds, and final training.  HR already follows that personal
calibration principle.  The original scripts retain their historical default.
"""

from pathlib import Path

import tcnKinPhys_45 as experiment


if __name__ == "__main__":
    experiment.configure_experiment()
    experiment.core.USE_ROUTE_RENORM = False
    experiment.core.OUT_DIR = Path(__file__).parent / "physio_no_renorm_results"
    experiment.core.OUT_DIR.mkdir(exist_ok=True)

    print("\nPhysioKinTCN personal-calibration experiment")
    print("  Target       : unchanged KinTCN safety-event weak supervision")
    print("  Calibration  : first 50 s, personal/session-specific")
    print("  Route renorm : disabled (P1/P2 never affect preprocessing)")
    print("  Inputs       : four kinematic channels + HR")
    print(f"  Output dir   : {experiment.core.OUT_DIR}\n")
    experiment.core.main()

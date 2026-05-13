"""run_01_smoke: 0.1% smoke variant of run_01.

Runs the same wiring as run_01 but at 0.001 sample_frac to verify the
end-to-end pipeline completes in under a minute before committing to the
full 5% comparison run.
"""

from __future__ import annotations

from projects.civic_shout_action_rate_increase.runs.run_01 import main

if __name__ == "__main__":
    main(run_id="run_01_smoke", sample_frac=0.001, sample_seed=42, scope="smoke")

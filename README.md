### Context

The current state of the smalldata_tools repo as it is being shuttled from beamtime to beamtime at MFX is reflected in the sfx_mfx branch, including my edits but also a ton of untracked changes by other folks. I don't know what belongs to who so I can't give anyone proper credit. I do want to keep everything in a more permanent place though, so that's what this is.

I merged sfx_mfx back into master preserving everything in master that didn't directly conflict with my changes. Some of my changes remove plots in BeamlineSummaries_mfx.py that I don't find useful and/or haven't been rendering correctly for me, and others update existing plots (detector damage) or add new ones (FEE and Ebeam related). I would be happy to submit a PR to merge in the subset of my changes that don't remove functionality if it is of interest to anyone.

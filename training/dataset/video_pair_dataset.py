"""
Same-source-video paired dataset for GenD / LNCLIP-DF (arXiv:2508.06248).

The paper pairs each fake with the REAL video it was generated from ("each fake video is
generated from its real counterpart") to force the model onto low-level manipulation
artifacts and prevent identity/background/compression shortcuts.

FaceForensics++ naming makes this recoverable: a fake frame lives under
    .../manipulated_sequences/<Method>/c23/frames/<TARGET>_<SOURCE>/<n>.png   e.g. 001_870
and its source real lives under
    .../original_sequences/youtube/c23/frames/<TARGET>/<n>.png                e.g. 001
so the paired real is the TARGET id (the part before the underscore).

This subclasses pairDataset and only overrides which real is chosen; the load/augment/
normalize path and collate_fn (balanced real-then-fake batch) are inherited unchanged.
"""

import random
from dataset.pair_dataset import pairDataset


def _video_id(frame_path):
    """'.../frames/001_870/000.png' -> '001_870'  (real: '.../frames/001/000.png' -> '001')."""
    return frame_path.replace('\\', '/').rsplit('/', 2)[-2]


class videoPairDataset(pairDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)

        # Index real frames by their source video id (e.g. '001').
        self.real_by_vid = {}
        for item in self.real_imglist:                     # item = (img_path, spe_label, 0)
            self.real_by_vid.setdefault(_video_id(item[0]), []).append(item)

        # Source (target) video id for each fake: '001_870' -> '001'.
        self.fake_source_vid = [_video_id(img).split('_')[0] for img, _, _ in self.fake_imglist]

        missing = sum(1 for v in self.fake_source_vid if v not in self.real_by_vid)
        print(f"[videoPair] {len(self.fake_imglist)} fakes across {len(self.real_by_vid)} source "
              f"videos; {missing} fakes have no same-source real (fall back to random real).")

    def _select_real(self, index):
        """Pick a random frame from the fake's SOURCE real video; fall back to a random
        real only if that source video is absent from this split (should not happen for FF++)."""
        pool = self.real_by_vid.get(self.fake_source_vid[index])
        if not pool:
            return self.real_imglist[random.randint(0, len(self.real_imglist) - 1)]
        return pool[random.randint(0, len(pool) - 1)]

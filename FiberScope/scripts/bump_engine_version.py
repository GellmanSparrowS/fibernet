"""Keep the cache identity in sync with the numeric core.

Run after changing fslab/engine2.py, fslab/simcache.py or fslab/structure.py:

    python scripts/bump_engine_version.py            # refresh hash only
    python scripts/bump_engine_version.py --version e2v5   # + new version tag
"""
import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TARGET = os.path.join(ROOT, "fslab", "simcache.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None,
                    help="new ENGINE_VERSION string (bump on numeric change)")
    args = ap.parse_args()
    from fslab.simcache import engine_src_hash, ENGINE_SRC_HASH, ENGINE_VERSION
    new_hash = engine_src_hash()
    if new_hash is None:
        print("cannot read sources (frozen?); nothing to do")
        return 1
    text = io.open(TARGET, encoding="utf-8").read()
    old_ver = ENGINE_VERSION
    new_ver = args.version or old_ver
    text2 = re.sub(r'^ENGINE_VERSION = "[^"]*"',
                   'ENGINE_VERSION = "%s"' % new_ver, text,
                   count=1, flags=re.M)
    text2 = re.sub(r'^ENGINE_SRC_HASH = "[^"]*"',
                   'ENGINE_SRC_HASH = "%s"' % new_hash, text2,
                   count=1, flags=re.M)
    assert text2 != text or (new_hash == ENGINE_SRC_HASH and
                             new_ver == old_ver), "constants not found"
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(text2)
    print("ENGINE_VERSION %s -> %s" % (old_ver, new_ver))
    print("ENGINE_SRC_HASH %s -> %s" % (ENGINE_SRC_HASH, new_hash))
    return 0


if __name__ == "__main__":
    sys.exit(main())
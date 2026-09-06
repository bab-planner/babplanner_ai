"""C1 — crowd_cctv 에서 **라이선스가 명확한 부분만** 내려받는다.

재배포자가 붙인 apache-2.0 태그를 믿지 않는다. 이 zip은 다섯 개의 원본
데이터셋을 합친 것이고, 그중 절반 이상이 비상업 조건이다. 원본 라이선스가
재배포 태그보다 우선한다.

**필터를 설정값으로 두지 않고 다운로드 계층에 건다.** zip 중앙 디렉터리를
읽어 허용된 항목만 HTTP Range로 받으면, 비상업 데이터는 디스크에 아예
존재하지 않는다. 설정 한 줄을 잘못 바꿔서 섞이는 일이 생기지 않는다.

3.29 GB 전체를 받지 않는다. 허용 subset만 받으므로 훨씬 작다.

    .venv-waste\\Scripts\\python.exe research\\cctv_counting\\scripts\\c1_fetch_crowd_cctv.py

허용되지 않은 출처도 레지스트리에는 **왜 제외했는지와 함께** 남긴다.
지워 버리면 나중에 누군가 같은 데이터를 다시 후보로 올린다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import http.client
import struct
import sys
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from datetime import date
from pathlib import Path

REPO = "KienNgyuen/crowd-detection-dataset"
REVISION = "22188f5aecd1d6263e800243ab2fdb9b2b6354d6"
ARCHIVE = "crowd_cctv.zip"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{ARCHIVE}"

#: 출처별 라이선스 판정.
#:
#: `commercial` 이 True인 것만 코드·평가 경로에 들어간다. 나머지는 내려받지
#: 않는다. "불명확"은 허용하지 않는다 — 확인되지 않은 것을 쓰는 것과 쓸 수
#: 있다고 확인한 것은 다르다.
SOURCES = {
    "coco": {
        "name": "COCO",
        "license": "annotations CC BY 4.0",
        "commercial": True,
        "note": (
            "주석은 CC BY 4.0. 이미지는 Flickr 출처라 개별 이용 조건이 따로 있고, "
            "이 재배포본은 그 개별 출처를 싣고 있지 않다. 상업 이용이 관행적으로 "
            "허용되는 유일한 부분이라 포함하되, 이 한계를 기록해 둔다."
        ),
    },
    "crowdhuman": {
        "name": "CrowdHuman",
        "license": "학술 연구 조건 (원 배포처 약관)",
        "commercial": False,
        "note": "상업 이용 가부가 확인되지 않았다. 불명확한 것은 쓰지 않는다.",
    },
    "mot": {
        "name": "MOT17 / MOT20",
        "license": "CC BY-NC-SA 계열",
        "commercial": False,
        "note": "NC — 비상업. 이 zip 이미지의 40%를 차지한다.",
    },
    "visdrone": {
        "name": "VisDrone",
        "license": "연구 목적 한정",
        "commercial": False,
        "note": "드론 시점이라 급식소 조건과도 멀다.",
    },
}

PERMISSIVE = frozenset(key for key, meta in SOURCES.items() if meta["commercial"])


#: 리다이렉트를 한 번만 따라가고 그 뒤로는 같은 연결을 다시 쓴다.
#: 멤버 2000개를 매번 새 TLS 핸드셰이크로 받으면 그것이 시간의 대부분이다.
_SESSION: dict[str, object] = {}


def _resolved() -> tuple[str, str]:
    if "host" not in _SESSION:
        response = urllib.request.urlopen(
            urllib.request.Request(URL, method="HEAD"), timeout=60
        )
        _SESSION["length"] = int(response.headers["Content-Length"])
        parts = urllib.parse.urlsplit(response.url)
        _SESSION["host"] = parts.netloc
        _SESSION["path"] = parts.path + (f"?{parts.query}" if parts.query else "")
    return str(_SESSION["host"]), str(_SESSION["path"])


def _get(a: int, b: int) -> bytes:
    """바이트 구간 하나. 연결을 살려 두고 끊어지면 한 번 다시 연다."""
    host, path = _resolved()
    for attempt in range(2):
        connection = _SESSION.get("connection")
        if connection is None:
            connection = http.client.HTTPSConnection(host, timeout=120)
            _SESSION["connection"] = connection
        try:
            connection.request("GET", path, headers={"Range": f"bytes={a}-{b}"})
            response = connection.getresponse()
            body = response.read()
            if response.status not in (200, 206):
                raise RuntimeError(f"HTTP {response.status} for bytes={a}-{b}")
            return body
        except (http.client.HTTPException, OSError):
            # CDN이 연결을 닫았다. 새로 열어 한 번만 다시 시도한다.
            try:
                connection.close()
            except Exception:
                pass
            _SESSION["connection"] = None
            if attempt == 1:
                raise
    raise RuntimeError("도달할 수 없는 경로")


def _size() -> int:
    _resolved()
    return int(_SESSION["length"])  # type: ignore[arg-type]


#: 32비트 칸에 안 들어가는 값을 ZIP64 extra 필드로 옮겼다는 표시.
ZIP64_SENTINEL = 0xFFFFFFFF


def _zip64_fields(extra: bytes, wanted: list[str]) -> dict[str, int]:
    """ZIP64 extra 필드(0x0001)에서 옮겨진 값을 꺼낸다.

    센티널이 붙은 칸만, **중앙 디렉터리에 적힌 순서대로** 8바이트씩 들어
    있다. 순서를 지키지 않으면 압축 크기를 오프셋으로 읽게 된다.
    """
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[cursor : cursor + 4])
        body = extra[cursor + 4 : cursor + 4 + size]
        if header_id == 0x0001:
            values = {}
            at = 0
            for name in wanted:
                if at + 8 > len(body):
                    break
                values[name] = struct.unpack("<Q", body[at : at + 8])[0]
                at += 8
            return values
        cursor += 4 + size
    return {}


def central_directory() -> dict[str, tuple[int, int, int]]:
    """zip 항목 이름 -> (로컬 헤더 오프셋, 압축 크기, 압축 방식).

    전체를 받지 않고 끝부분만 읽어 목록을 만든다. 이 아카이브는 ZIP64라
    일부 항목의 오프셋이 32비트 칸에 없다.
    """
    total = _size()
    tail = _get(total - 65557, total - 1)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0:
        raise RuntimeError("zip 중앙 디렉터리를 찾지 못했습니다.")
    cd_size = struct.unpack("<I", tail[marker + 12 : marker + 16])[0]
    cd_offset = struct.unpack("<I", tail[marker + 16 : marker + 20])[0]

    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:
        # ZIP64 end-of-central-directory. 32비트 EOCD의 값이 잘려 있을 수 있다.
        eocd64_at = struct.unpack("<Q", tail[locator + 8 : locator + 16])[0]
        record = _get(eocd64_at, eocd64_at + 55)
        if record[:4] == b"PK\x06\x06":
            cd_size = struct.unpack("<Q", record[40:48])[0]
            cd_offset = struct.unpack("<Q", record[48:56])[0]

    blob = _get(cd_offset, cd_offset + cd_size - 1)
    entries: dict[str, tuple[int, int, int]] = {}
    cursor = 0
    while cursor < len(blob) - 4 and blob[cursor : cursor + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", blob[cursor + 10 : cursor + 12])[0]
        compressed = struct.unpack("<I", blob[cursor + 20 : cursor + 24])[0]
        uncompressed = struct.unpack("<I", blob[cursor + 24 : cursor + 28])[0]
        name_len, extra_len, comment_len = struct.unpack(
            "<HHH", blob[cursor + 28 : cursor + 34]
        )
        header_offset = struct.unpack("<I", blob[cursor + 42 : cursor + 46])[0]
        name = blob[cursor + 46 : cursor + 46 + name_len].decode("utf-8", "replace")

        if ZIP64_SENTINEL in (compressed, uncompressed, header_offset):
            extra = blob[cursor + 46 + name_len : cursor + 46 + name_len + extra_len]
            # 순서가 정해져 있다: 원본 크기, 압축 크기, 로컬 헤더 오프셋.
            # 센티널이 붙은 칸만 실린다.
            wanted = [
                field
                for field, value in (
                    ("uncompressed", uncompressed),
                    ("compressed", compressed),
                    ("offset", header_offset),
                )
                if value == ZIP64_SENTINEL
            ]
            found = _zip64_fields(extra, wanted)
            compressed = found.get("compressed", compressed)
            header_offset = found.get("offset", header_offset)

        entries[name] = (header_offset, compressed, method)
        cursor += 46 + name_len + extra_len + comment_len
    return entries


#: 로컬 헤더의 extra 필드 여유분. 헤더와 데이터를 한 번에 받으려고 넉넉히
#: 더 읽는다. 요청을 절반으로 줄이는 값이고, 남는 바이트는 버린다.
_LOCAL_EXTRA_SLACK = 256


def read_member(entries: dict, name: str) -> bytes:
    """멤버 하나를 **요청 한 번으로** 읽는다.

    로컬 헤더와 데이터를 따로 받으면 멤버마다 왕복이 두 번이다. 2000개면
    4000번이고, 그 왕복이 시간의 대부분을 차지한다. 헤더 길이를 모르므로
    넉넉히 더 읽어서 같은 버퍼에서 잘라 쓴다.
    """
    offset, compressed, method = entries[name]
    if compressed == 0:
        # 빈 파일이다. 사람이 하나도 없는 프레임의 라벨이 이렇게 나온다.
        # 빈 범위를 요청하면 416이 돌아온다.
        return b""

    head_room = 30 + len(name.encode("utf-8")) + _LOCAL_EXTRA_SLACK
    buffer = _get(offset, offset + head_room + compressed - 1)
    name_len, extra_len = struct.unpack("<HH", buffer[26:30])
    start = 30 + name_len + extra_len
    blob = buffer[start : start + compressed]
    if len(blob) != compressed:
        # 여유분이 모자랐다. 드문 경우라 그때만 정확히 다시 받는다.
        absolute = offset + start
        blob = _get(absolute, absolute + compressed - 1)
    return zlib.decompress(blob, -15) if method == 8 else blob


def source_of(member: str) -> str:
    """파일 이름 앞부분이 출처다. `mot_mot20_MOT20-03_f000468.jpg` 처럼 붙어 있다."""
    return Path(member).name.split("_")[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research/cctv_counting/datasets/raw/crowd_cctv"),
        help="원본을 둘 곳. gitignore 아래여야 한다",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/cctv_counting/datasets/c1_crowd_cctv_registry.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="시험용. 앞의 N장만")
    args = parser.parse_args(argv)

    print(f"{REPO} @ {REVISION[:12]} / {ARCHIVE}")
    print("중앙 디렉터리 읽는 중 (전체 다운로드 아님)...")
    entries = central_directory()

    images = [n for n in entries if n.startswith("images/") and n.endswith(".jpg")]
    counts_all = Counter(source_of(n) for n in images)
    unknown = sorted(set(counts_all) - set(SOURCES))
    if unknown:
        # 모르는 출처는 라이선스도 모른다. 조용히 포함하지 않는다.
        print(f"라이선스를 모르는 출처가 있습니다: {unknown}")
        return 1

    keep = sorted(n for n in images if source_of(n) in PERMISSIVE)
    skip_counts = {k: v for k, v in counts_all.items() if k not in PERMISSIVE}
    print(f"\n전체 이미지 {len(images)}장")
    for key in sorted(counts_all):
        meta = SOURCES[key]
        mark = "포함" if meta["commercial"] else "제외"
        print(f"  {meta['name']:16} {counts_all[key]:5}장  {meta['license']:28} {mark}")
    print(f"\n내려받을 대상 {len(keep)}장 (+ 라벨 {len(keep)}개)")

    if args.limit:
        keep = keep[: args.limit]

    downloaded = 0
    total_bytes = 0
    fingerprint = hashlib.sha256()
    for index, member in enumerate(sorted(keep), start=1):
        label = member.replace("images/", "labels/").replace(".jpg", ".txt")
        for name in (member, label):
            dest = args.root / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(read_member(entries, name))
            total_bytes += dest.stat().st_size
            fingerprint.update(hashlib.sha256(dest.read_bytes()).hexdigest().encode())
        downloaded += 1
        if index % 100 == 0:
            print(f"  {index}/{len(keep)}")

    registry = {
        "datasetKey": "crowd_cctv_permissive",
        "source": f"https://huggingface.co/datasets/{REPO}",
        "archive": ARCHIVE,
        "revision": REVISION,
        "downloadedOn": date.today().isoformat(),
        "redistributorLicenseTag": "apache-2.0",
        "redistributorTagAccepted": False,
        "redistributorTagNote": (
            "재배포자가 붙인 태그다. 원본 다섯 개 중 셋이 비상업 조건이라 이 태그가 "
            "전체를 덮을 수 없다. 원본 라이선스를 우선한다."
        ),
        "sourceLicenses": SOURCES,
        "archiveComposition": dict(sorted(counts_all.items())),
        "included": {
            "sources": sorted(PERMISSIVE),
            "images": downloaded,
            "totalBytes": total_bytes,
            "sha256OfSortedFileDigests": fingerprint.hexdigest(),
        },
        "excluded": {
            "sources": sorted(skip_counts),
            "images": sum(skip_counts.values()),
            "reason": "비상업 또는 조건 불명확. 내려받지 않았으므로 디스크에 없다.",
            "researchOnlyNote": (
                "내부 연구 참고가 필요하면 별도 경로에서 research-only로 명시해 다루고, "
                "학습·가중치 생성·제품 성능 근거에는 쓰지 않는다."
            ),
        },
        "annotationFormat": {
            "layout": "YOLO. images/{split}/*.jpg 와 labels/{split}/*.txt 가 1:1",
            "line": "class cx cy w h (이미지 크기로 정규화)",
            "classes": {"0": "person", "1": "pedestrian"},
            "classNote": (
                "두 클래스의 의미 차이가 문서화돼 있지 않다. 우리 과제는 사람 수이므로 "
                "어댑터에서 한 클래스로 합친다."
            ),
            "trackIds": False,
            "trackIdNote": "track id가 없어 tracking·ID switch·선 통과 평가를 할 수 없다.",
        },
        "providedSplit": {
            "counts": {"train": 8382, "val": 985, "test": 492},
            "usable": False,
            "reason": (
                "MOT 시퀀스 11개 전부가 train·val·test에 동시에 들어 있고 프레임 간격 "
                "최빈값이 1이다. 연속 프레임이 갈렸다. C2에서 시퀀스 단위로 다시 나눈다."
            ),
        },
        "usage": {
            "benchmark": "external-public",
            "role": "person detector / occupancy baseline",
            "limitation": (
                "허용 subset이 COCO뿐이라 감시 카메라 영상이 아니다. 웹 사진에서 잰 "
                "검출 성능이며 급식소 천장 카메라로의 전이를 보장하지 않는다."
            ),
        },
        "note": "원본 파일은 커밋하지 않는다. 지문만 남긴다.",
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n받은 이미지 {downloaded}장, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"레지스트리: {args.registry}")
    print(
        f"\n제외 {sum(skip_counts.values())}장은 내려받지 않았다 — "
        "디스크에 없으므로 실수로 섞일 수 없다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

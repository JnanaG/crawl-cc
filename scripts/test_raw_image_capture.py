from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.dcd_scraper import DongchediScraper  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeBinaryResponse:
    def __init__(self, payload: bytes, content_type: str = "image/jpeg", status_code: int = 200):
        self.payload = payload
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        for idx in range(0, len(self.payload), chunk_size):
            yield self.payload[idx : idx + chunk_size]


class FakeScraper(DongchediScraper):
    def __init__(self, raw_dir: Path):
        super().__init__(min_interval_sec=0.0, max_retry=1)
        self.raw_dir = str(raw_dir)
        self.raw_image_dir = str(raw_dir / "images")
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.raw_image_dir).mkdir(parents=True, exist_ok=True)

    def _request_binary_with_retry(self, url: str, timeout: int = 20):
        payload = f"binary:{url}".encode("utf-8")
        return True, FakeBinaryResponse(payload), 0, 200, ""


def build_raw_payload() -> dict:
    return {
        "props": {
            "pageProps": {
                "seriesId": "888001",
                "seriesHomeHead": {
                    "series_id": 888001,
                    "cover_url": "https://example.com/cover.jpg",
                    "series_image_info_list": [],
                },
                "selectedImageModules": {
                    "wg": {
                        "module_key": "wg",
                        "category_name": "外观",
                        "required_view": "正面前脸图",
                        "selected_images": [
                            {"car_id": 1, "car_name": "2026款 A", "image_url": "https://example.com/gallery-1.jpg"}
                        ],
                    },
                    "ns": {
                        "module_key": "ns",
                        "category_name": "内饰",
                        "required_view": "正面主副驾拍摄图",
                        "selected_images": [
                            {"car_id": 1, "car_name": "2026款 A", "image_url": "https://example.com/interior-1.jpg"}
                        ],
                    },
                },
                "imageFloorData": {"floor_head_list": [], "floor_image_list": []},
            }
        }
    }


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "raw_image_capture"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        scraper = FakeScraper(raw_dir=artifact_dir / "raw")
        result = scraper.save_series_images("888001", build_raw_payload(), include_contextual_images=False)

        manifest_path = Path(result["image_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert_true(manifest["series_id"] == "888001", "manifest series_id 不正确")
        assert_true(manifest["success_count"] == 3, f"图片保存成功数不正确: {manifest['success_count']}")
        assert_true(all(item["status"] == "success" for item in manifest["items"]), "存在保存失败图片")
        assert_true(all(Path(item["local_path"]).exists() for item in manifest["items"]), "本地原图文件不存在")

        print("[PASS] 原始图片保存测试通过")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 原始图片保存测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

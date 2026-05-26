import requests
import json
import re
import os
import random
import time
from urllib.parse import quote
from loguru import logger
from bs4 import BeautifulSoup


class DongchediScraper:
    def __init__(self, min_interval_sec: float = 0.8, max_retry: int = 3):
        self.base_url = "https://www.dongchedi.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9"
        })
        self.min_interval_sec = min_interval_sec
        self.max_retry = max_retry
        self.last_request_ts = 0.0
        # 确保原始数据存储目录存在
        self.raw_dir = os.path.join("data", "raw", "dongchedi")
        os.makedirs(self.raw_dir, exist_ok=True)

    def _throttle(self):
        now = time.monotonic()
        elapsed = now - self.last_request_ts
        if elapsed < self.min_interval_sec:
            sleep_sec = self.min_interval_sec - elapsed + random.uniform(0.05, 0.2)
            time.sleep(sleep_sec)
        self.last_request_ts = time.monotonic()

    def _request_with_retry(self, url: str, timeout: int = 10):
        last_error = ""
        last_status = None
        attempts = 0
        for attempt in range(1, self.max_retry + 1):
            attempts = attempt
            try:
                self._throttle()
                resp = self.session.get(url, timeout=timeout)
                last_status = resp.status_code
                # 4xx 通常是路径或权限问题，除 429 外不做重试，避免浪费请求预算。
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return False, None, attempts - 1, last_status, f"HTTP {resp.status_code}"
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return True, resp, attempts - 1, last_status, ""
            except Exception as e:
                last_error = str(e)
                logger.warning(f"请求失败({attempt}/{self.max_retry}) url={url}, error={e}")
                if attempt < self.max_retry:
                    backoff = min(2 ** (attempt - 1), 8) + random.uniform(0.1, 0.4)
                    time.sleep(backoff)
        return False, None, attempts - 1, last_status, last_error

    def get_homepage_series_ids(self, limit: int = 5) -> list[str]:
        """从首页获取热门车系的 ID 列表"""
        logger.info("正在从懂车帝首页获取车系 ID...")
        try:
            ok, resp, _, _, err = self._request_with_retry(self.base_url, timeout=10)
            if not ok or resp is None:
                raise RuntimeError(err or "请求首页失败")
            # 使用正则快速提取主页上的所有 /auto/series/xxxx 链接
            ids = list(set(re.findall(r'/auto/series/(\d+)', resp.text)))
            logger.info(f"首页共发现 {len(ids)} 个车系 ID")
            return ids[:limit]
        except Exception as e:
            logger.error(f"获取首页车系 ID 失败: {e}")
            return []

    def _extract_series_ids_from_html(self, html_text: str) -> list[str]:
        return list(set(re.findall(r"/auto/series/(\d+)", html_text or "")))

    def _extract_series_ids_from_local_cache(self) -> list[str]:
        """从本地 raw 目录回收历史抓取过的车系 ID，作为额外入口。"""
        if not os.path.isdir(self.raw_dir):
            return []
        ids = set()
        for name in os.listdir(self.raw_dir):
            m = re.match(r"series_(\d+)\.(json|html)$", name)
            if m:
                ids.add(m.group(1))
        return list(ids)

    def _collect_series_ids_from_entry_urls(self, entry_urls: list[str], max_pages_per_entry: int = 3) -> list[str]:
        """
        从多个公共入口页收集车系ID。
        - 如果 URL 包含 {page}，会按分页尝试。
        """
        seen = set()
        for template in entry_urls:
            if "{page}" in template:
                page_urls = [
                    template.format(page=page)
                    for page in range(1, max(1, max_pages_per_entry) + 1)
                ]
            else:
                page_urls = [template]
            for url in page_urls:
                ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
                if not ok or resp is None:
                    logger.debug(f"入口页抓取失败: url={url}, error={err}")
                    continue
                for sid in self._extract_series_ids_from_html(resp.text):
                    seen.add(sid)
        return list(seen)

    def _collect_series_ids_from_search_keywords(self, keywords: list[str], max_requests: int = 20) -> list[str]:
        """
        通过站内搜索页扩展车系池。
        该入口天然会覆盖更多冷门车系，通常比首页更全。
        """
        seen = set()
        used = 0
        for kw in keywords:
            if used >= max_requests:
                break
            url = f"{self.base_url}/search?keyword={quote(kw)}"
            ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
            used += 1
            if not ok or resp is None:
                logger.debug(f"搜索入口抓取失败: keyword={kw}, error={err}")
                continue
            for sid in self._extract_series_ids_from_html(resp.text):
                seen.add(sid)
        return list(seen)

    def collect_series_ids(self, target_count: int = 300, max_expand_requests: int = 30) -> list[str]:
        """多入口收集车系ID：本地缓存 + 首页 + 公共入口页 + 搜索页，再做关联扩展。"""
        seen = set()
        queue = []

        def _push(ids: list[str], source: str):
            before = len(seen)
            for sid in ids:
                if sid in seen:
                    continue
                seen.add(sid)
                queue.append(sid)
                if len(seen) >= target_count:
                    break
            added = len(seen) - before
            if added > 0:
                logger.info(f"{source} 新增车系ID: +{added}, 当前总数: {len(seen)}")

        # 入口1：本地历史抓取缓存（真实数据）
        _push(self._extract_series_ids_from_local_cache(), "本地缓存入口")

        # 入口2：首页
        _push(self.get_homepage_series_ids(limit=max(target_count, 200)), "首页入口")

        # 入口3：公共入口页（榜单/库页等）
        entry_urls = [
            f"{self.base_url}/auto/library/x-x-x-x-x-x-x-x-x-x",
            f"{self.base_url}/auto/library/x-x-x-x-x-x-x-x-x-x?page={{page}}",
        ]
        page_budget = max(2, min(8, max_expand_requests // 5))
        _push(
            self._collect_series_ids_from_entry_urls(
                entry_urls=entry_urls,
                max_pages_per_entry=page_budget,
            ),
            "公共入口页",
        )

        # 入口4：搜索页关键词（品牌+级别+热门词）
        keywords = [
            "比亚迪", "吉利", "长安", "奇瑞", "理想", "蔚来", "小鹏", "问界",
            "丰田", "本田", "大众", "宝马", "奔驰", "奥迪", "日产", "福特",
            "SUV", "轿车", "MPV", "混动", "纯电", "增程", "皮卡", "轻客",
        ]
        search_budget = max(8, min(30, max_expand_requests))
        _push(
            self._collect_series_ids_from_search_keywords(
                keywords=keywords,
                max_requests=search_budget,
            ),
            "搜索入口",
        )

        if not seen:
            return []

        expand_count = 0

        while queue and len(seen) < target_count and expand_count < max_expand_requests:
            current = queue.pop(0)
            url = f"{self.base_url}/auto/series/{current}"
            ok, resp, _, _, err = self._request_with_retry(url, timeout=10)
            if not ok or resp is None:
                logger.debug(f"扩展车系池失败: series_id={current}, error={err}")
                continue

            expand_count += 1
            discovered = self._extract_series_ids_from_html(resp.text)
            for sid in discovered:
                if sid in seen:
                    continue
                seen.add(sid)
                queue.append(sid)
                if len(seen) >= target_count:
                    break

        ids = list(seen)
        logger.info(
            f"车系池收集完成: 当前 {len(ids)} 个ID, 目标 {target_count}, 扩展请求 {expand_count}/{max_expand_requests}"
        )
        return ids[:target_count]

    def fetch_series_data(self, series_id: str) -> tuple[dict, dict]:
        """抓取指定车系详情页，并提取出 __NEXT_DATA__ JSON，同时保存原始数据"""
        url = f"{self.base_url}/auto/series/{series_id}"
        logger.info(f"正在抓取车系页: {url}")
        fetch_meta = {
            "url": url,
            "retry_count": 0,
            "http_status": None,
            "error": "",
        }
        try:
            ok, resp, retry_count, http_status, err = self._request_with_retry(url, timeout=10)
            fetch_meta["retry_count"] = retry_count
            fetch_meta["http_status"] = http_status
            if not ok or resp is None:
                fetch_meta["error"] = err or "请求失败"
                return {}, fetch_meta
            
            # 保存原始 HTML (用于存档或后续备用分析)
            html_path = os.path.join(self.raw_dir, f"series_{series_id}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(resp.text)
                
            # 解析页面中的 __NEXT_DATA__
            soup = BeautifulSoup(resp.text, 'html.parser')
            script_tag = soup.find('script', id='__NEXT_DATA__')
            
            if not script_tag:
                logger.warning(f"车系 {series_id} 页面未找到 __NEXT_DATA__ 节点")
                fetch_meta["error"] = "__NEXT_DATA__ missing"
                return {}, fetch_meta
                
            data = json.loads(script_tag.string)
            
            # 同样将提取出来的核心 JSON 保存一份原始数据
            json_path = os.path.join(self.raw_dir, f"series_{series_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            return data, fetch_meta
            
        except Exception as e:
            logger.error(f"抓取车系 {series_id} 失败: {e}")
            fetch_meta["error"] = str(e)
            return {}, fetch_meta

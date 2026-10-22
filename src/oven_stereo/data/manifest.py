"""可共享的相对路径清单与数据获取；原始文件不覆盖。"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PureWindowsPath
from urllib.request import Request, urlopen


def read_manifest(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("数据清单必须是 schema_version=1 的 JSON 对象")
    return value


def asset_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError("清单只允许相对于仓库根的路径")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("清单路径不能超出仓库根目录")
    return target


def verify_asset(path: str | Path, asset: dict) -> None:
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != asset["size_bytes"] or digest != asset["sha256"]:
        raise ValueError(f"原始文件大小或 SHA256 不符：{path}")


def acquire_assets(manifest: dict, root: str | Path = ".") -> None:
    """下载缺少的原始文件，已有文件只核验；下载失败保留明确异常。"""
    for asset in manifest["assets"]:
        target = asset_path(root, asset["path"])
        if target.exists():
            verify_asset(target, asset)
            continue
        if not asset["url"].startswith("https://"):
            raise ValueError("数据下载仅允许 HTTPS 地址")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        # 独占创建，不覆盖之前中断的下载或其他进程的文件。
        with temporary.open("xb") as output:
            try:
                # 本机 Windows 的 urllib/系统代理组合会重置连接；优先使用自带 curl。
                # 不修改系统代理或关闭证书校验；其他平台使用标准库。
                curl = shutil.which("curl.exe")
                if curl is not None:
                    subprocess.run(
                        [curl, "--disable", "--fail", "--location", "--silent", "--show-error",
                         "--proto", "=https", "--proto-redir", "=https",
                         "--connect-timeout", "10", "--max-time", "60", asset["url"]],
                        stdout=output, check=True,
                    )
                else:
                    request = Request(asset["url"], headers={"User-Agent": "oven-stereo/B-01"})
                    with urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, output)
            except Exception:
                output.close()
                temporary.unlink()
                raise
        try:
            verify_asset(temporary, asset)
            temporary.rename(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

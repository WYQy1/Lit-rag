"""基础冒烟测试。

设计原则：**不依赖任何 API Key**，因此在 GitHub Actions 的公开 CI 中也能全部通过。
需要真实调用大模型的测试请放在 tests/integration/ 下并标记为 integration。
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

import pytest


def test_main_module_importable() -> None:
    """main 模块应能被正常导入，且暴露 main 函数。"""
    module = importlib.import_module("main")
    assert callable(module.main)


def test_main_runs_and_prints(capsys: pytest.CaptureFixture[str]) -> None:
    """main() 应可执行，并打印项目就绪提示与 Python 版本。"""
    module = importlib.import_module("main")
    module.main()

    out = capsys.readouterr().out
    assert "Lit-rag" in out
    assert sys.version.split(".")[0] in out


def test_python_version_supported() -> None:
    """运行时 Python 版本应满足 pyproject.toml 中声明的 >=3.11。"""
    assert sys.version_info >= (3, 11)


@pytest.mark.skipif(shutil.which("git") is None, reason="未安装 git，跳过")
def test_secret_files_are_gitignored() -> None:
    """安全回归测试：.env 必须被 git 忽略，防止 API Key 被误提交。

    这是本项目最重要的一条测试 —— 一旦有人改坏 .gitignore，CI 会立刻失败。
    """
    for path in (".env", "hello-llm/.env"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            capture_output=True,
            check=False,
        )
        # 返回码 0 表示该路径确实被忽略规则命中
        assert result.returncode == 0, f"{path} 没有被 .gitignore 忽略，存在密钥泄露风险！"

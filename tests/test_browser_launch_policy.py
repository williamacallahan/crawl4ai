from crawl4ai.async_configs import BrowserConfig
from crawl4ai.browser_manager import BrowserManager, ManagedBrowser

TLS_BYPASS_FLAGS = {
    "--ignore-certificate-errors",
    "--ignore-certificate-errors-spki-list",
}


def test_managed_browser_respects_tls_verification_policy():
    verified = ManagedBrowser.build_browser_flags(
        BrowserConfig(ignore_https_errors=False)
    )
    insecure = ManagedBrowser.build_browser_flags(
        BrowserConfig(ignore_https_errors=True)
    )

    assert TLS_BYPASS_FLAGS.isdisjoint(verified)
    assert TLS_BYPASS_FLAGS.issubset(insecure)


def test_playwright_launch_respects_tls_verification_policy():
    verified = BrowserManager(
        BrowserConfig(ignore_https_errors=False)
    )._build_browser_args()["args"]
    insecure = BrowserManager(
        BrowserConfig(ignore_https_errors=True)
    )._build_browser_args()["args"]

    assert TLS_BYPASS_FLAGS.isdisjoint(verified)
    assert TLS_BYPASS_FLAGS.issubset(insecure)


def test_no_sandbox_is_only_present_when_operator_configures_it():
    default_launch = BrowserManager(BrowserConfig())._build_browser_args()
    configured_launch = BrowserManager(
        BrowserConfig(extra_args=["--no-sandbox"])
    )._build_browser_args()

    assert "--no-sandbox" not in default_launch["args"]
    assert default_launch["chromium_sandbox"] is True
    assert "--no-sandbox" in configured_launch["args"]
    assert configured_launch["chromium_sandbox"] is False
    assert "--no-sandbox" not in ManagedBrowser.build_browser_flags(BrowserConfig())

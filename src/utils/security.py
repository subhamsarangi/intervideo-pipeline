"""Security utilities for input validation and protection"""

import re
import socket
import unicodedata
from typing import List
from urllib.parse import urlparse

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore previous.*instructions",
    r"forget.*instructions",
    r"pretend.*you.*are",
    r"system.*prompt",
    r"jailbreak",
    r"as if you were",
    r"override.*settings",
    r"bypass.*security",
    r"execute.*code",
]


def detect_prompt_injection(text: str) -> bool:
    """
    Simple pattern-based prompt injection detector

    IMPORTANT: this is a best-effort noise filter, not a security boundary.
    It only catches literal, English-language phrasings of known patterns.
    It will NOT catch rephrasing, other languages, split/obfuscated phrases,
    or encoded payloads (base64, etc). Do not rely on this function alone --
    pair it with wrap_content_delimiters(), an explicit "treat this as data,
    not instructions" system prompt, least-privilege tool access for any
    LLM step that consumes this text, and constrained/structured output.

    Args:
        text: Text to check

    Returns:
        True if potential injection detected
    """
    # Normalize unicode (NFKC) and collapse whitespace so trivial evasions
    # like full-width characters or irregular spacing don't slip past the
    # patterns below. This still does nothing against rephrasing or
    # non-English text.
    normalized = unicodedata.normalize("NFKC", text)
    text_lower = re.sub(r"\s+", " ", normalized).lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    return False


def is_private_ip(ip: str) -> bool:
    """Check if IP is in private range"""
    private_ranges = [
        "127.0.0.0/8",  # Loopback
        "10.0.0.0/8",  # Private
        "172.16.0.0/12",  # Private
        "192.168.0.0/16",  # Private
        "169.254.0.0/16",  # Link-local
        "100.64.0.0/10",  # Carrier-grade NAT / shared address space (used by some cloud metadata setups)
        "0.0.0.0/8",  # "This" network
        "::1/128",  # IPv6 loopback
        "fc00::/7",  # IPv6 private
        "fe80::/10",  # IPv6 link-local
        "::ffff:0:0/96",  # IPv4-mapped IPv6 (must be unwrapped before checking, see below)
    ]

    # Convert IP to integer for range checking
    try:
        from ipaddress import ip_address, IPv4Network, IPv6Network, IPv6Address

        addr = ip_address(ip)

        # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) so they get
        # checked against the IPv4 private ranges too, not just the IPv6 ones
        if isinstance(addr, IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped

        for range_str in private_ranges:
            if "/" in range_str:
                network = (
                    IPv4Network(range_str)
                    if "." in range_str
                    else IPv6Network(range_str)
                )
                if addr in network:
                    return True
    except ValueError:
        pass

    return False


def validate_jd_url(url: str) -> bool:
    """
    Validate URL for SSRF attacks

    Args:
        url: URL to validate

    Returns:
        True if URL is safe

    Raises:
        ValueError: If URL fails validation
    """
    # Check scheme
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Must be http or https")

    # Check for no hostname
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    # Check for localhost
    if parsed.hostname in ("localhost", "127.0.0.1"):
        raise ValueError("URL points to localhost")

    # Check for private IPs. is_private_ip() only understands literal IP
    # addresses, so if the hostname is a literal IP we can check it directly.
    # But if it's a domain name (the common case), checking the string itself
    # does nothing -- a domain like "evil.com" that resolves to 127.0.0.1 or
    # a cloud metadata IP (169.254.169.254) would pass right through. So we
    # resolve DNS here and validate every IP the hostname resolves to.
    if is_private_ip(parsed.hostname):
        raise ValueError(f"URL points to private IP: {parsed.hostname}")

    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname: {parsed.hostname} ({e})")

    resolved_ips = {info[4][0] for info in resolved}
    for resolved_ip in resolved_ips:
        if is_private_ip(resolved_ip):
            raise ValueError(
                f"URL hostname {parsed.hostname} resolves to private IP: {resolved_ip}"
            )

    # NOTE: this closes the "domain resolves to a private IP" gap, but it does
    # NOT fully close DNS rebinding: DNS could change between this check and
    # the actual request. Whatever code makes the real HTTP request should
    # pin the connection to one of the IPs validated here (e.g. via a custom
    # requests HTTPAdapter that connects to the IP directly and sets the
    # Host header) and should disable automatic redirect-following, re-running
    # validate_jd_url() on any redirect target before following it.

    return True


def wrap_content_delimiters(content: str, source_name: str = "content") -> str:
    """
    Wrap content in delimiters for safe LLM processing

    Args:
        content: Content to wrap
        source_name: Name of the content source

    Returns:
        Wrapped content with delimiters
    """
    delimiter = f"<<<{source_name.upper()}>>>"
    end_delimiter = f"<<</END {source_name.upper()}>>>"

    return f"{delimiter}\n{content}\n{end_delimiter}"


def validate_cv_text(text: str) -> bool:
    """
    Validate CV text for injection attempts

    Args:
        text: CV text to validate

    Returns:
        True if CV appears safe

    Raises:
        ValueError: If injection detected
    """
    if detect_prompt_injection(text):
        raise ValueError("Potential prompt injection detected in CV text")

    return True


def validate_jd_text(text: str) -> bool:
    """
    Validate JD text for injection attempts

    Args:
        text: JD text to validate

    Returns:
        True if JD appears safe

    Raises:
        ValueError: If injection detected
    """
    if detect_prompt_injection(text):
        raise ValueError("Potential prompt injection detected in JD text")

    return True

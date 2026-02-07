# Security Policy (Automated)

## Supported Versions

Security updates and support are scoped to the versions referenced in the
project documentation and dependency list.

| Component | Supported Version |
| --- | --- |
| Python | 3.11.14 |
| scikit-learn | 1.3.0 |
| pandas | 2.0.3 |
| numpy | 1.25.2 |
| matplotlib | 3.7.2 |
| seaborn | 0.13.2 |
| web3 | 6.5.0 |
| Jupyter Notebook/Lab | Not installed in this environment |
| Node.js | 20.19.2 |
| npm | 9.2.0 |
| Ganache CLI | 7.9.2 |

## Known Vulnerabilities (Unfixed)

None currently tracked in this repository's environment.

## Previously Reported Vulnerabilities (Remediated)

The following issues were reported by a vulnerability scan and have been
remediated in the devcontainer setup:

| CVE | Severity | Fixable | Present In | Affected Package(s) | Remediation |
| --- | --- | --- | --- | --- | --- |
| CVE-2025-12840 | 7.8 (High) | Yes | Debian | openexr 3.1.13-2 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2025-12495 | 7.8 (High) | Yes | Debian | openexr 3.1.13-2 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2025-12839 | 7.8 (High) | Yes | Debian | openexr 3.1.13-2 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2017-14988 | N/A (Low) | Yes | Debian | openexr 3.1.13-2 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2025-45582 | 4.1 (Medium) | Yes | Debian | tar 1.35+dfsg-3.1 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2009-3546 | N/A (Medium) | Yes | Debian | libwmf 0.2.13-1.1 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2007-3996 | N/A (Medium) | Yes | Debian | libwmf 0.2.13-1.1 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2007-3477 | N/A (Low) | Yes | Debian | libwmf 0.2.13-1.1 | Upgrade base packages via `apt-get dist-upgrade` in the devcontainer. |
| CVE-2025-64756 | 7.5 (High) | Yes | npm | glob 10.4.5 | Upgrade npm to a patched release via `npm install -g npm@latest`. |
| CVE-2025-8869 | 5.9 (Medium) | Yes | PyPI | pip 24.0 | Upgrade pip to `>=24.2` via `python -m pip install --upgrade 'pip>=24.2'`. |

## Reporting a Vulnerability

Please report security issues by opening a GitHub issue:
/Blockchain-enabled-Adversarial-Threat-Intelligence-Sharing-for-Robust-Ransomware-Detection-air-gaps

To help us triage quickly, include:
- A clear description of the issue and potential impact
- Steps to reproduce (proof of concept if possible)
- Affected components and versions
- Any suggested remediation or mitigations

We will acknowledge valid reports within 72 hours and provide status updates
as we investigate.

## Disclosure Policy

We follow coordinated disclosure best practices. Please avoid publicly
disclosing details until a fix or mitigation is available. Our goal is to
resolve critical issues within 90 days when feasible, with shorter timelines
for actively exploited vulnerabilities.

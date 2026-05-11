"""
Shared HR knowledge base used across demos 04-07.
Import this module wherever you need the canonical document corpus.
"""

DOCUMENTS: list[dict] = [
    {
        "id": "doc_001",
        "title": "Remote Work Policy",
        "content": """Remote Work Policy

Employees are authorized to work remotely up to 3 days per week with manager approval.
Remote work requires maintaining availability during core hours (10 AM - 3 PM).

Equipment: The company provides a laptop and headset for remote work. Employees are
responsible for a stable internet connection (minimum 10 Mbps). VPN is required for
accessing all company systems remotely.

Eligibility: Employees must have completed their 90-day probationary period and have
a performance rating of "Meets Expectations" or higher to be eligible for remote work."""
    },
    {
        "id": "doc_002",
        "title": "Employee Benefits and Leave Policy",
        "content": """Employee Benefits:
- Health insurance: medical, dental, and vision coverage for employee and dependents
- 401(k) retirement plan with 5% company match (vesting after 1 year)
- Flexible spending accounts (FSA) for healthcare and dependent care
- Life insurance: 2x annual salary
- Short-term and long-term disability insurance
- Employee assistance program (EAP)

Leave Policy:
- Vacation: 15 days per year (increases to 20 days after 5 years, 25 days after 10 years)
- Sick leave: 10 days per year (non-rollover)
- Personal days: 3 days per year
- Parental leave: 12 weeks paid for primary caregiver, 4 weeks paid for secondary caregiver
- Bereavement leave: 5 days for immediate family, 3 days for extended family

Professional Development:
- Annual training budget: $2,000 per employee
- Conference attendance: 1-2 per year with manager approval
- Certification reimbursement: up to $500 per certification
- Mentorship program available to all employees"""
    },
    {
        "id": "doc_003",
        "title": "Code Review and Engineering Standards",
        "content": """Code Review Policy:
All code changes must undergo peer review before merging into any protected branch.
Reviews should focus on:
- Code quality and maintainability
- Test coverage (minimum 80% for new code)
- Documentation completeness
- Security considerations and vulnerability scanning

Pull Request Process:
1. Create feature branch from main
2. Submit pull request with description of changes
3. Minimum 2 reviewer approvals required (1 senior engineer)
4. All automated CI checks must pass
5. Squash commits before merging

Code Style:
- Follow language-specific style guides (PEP 8 for Python, Google style for Java)
- Use meaningful variable and function names
- Maximum function length: 50 lines
- Maximum file length: 500 lines"""
    },
    {
        "id": "doc_004",
        "title": "IT Security Standards",
        "content": """Password Policy:
- Minimum 12 characters
- Must include uppercase, lowercase, numbers, and special characters
- Password rotation every 90 days
- No reuse of last 10 passwords

Multi-Factor Authentication (MFA):
- Required for all company accounts
- Use authenticator app (Google Authenticator or Microsoft Authenticator)
- Hardware tokens available for executives and high-privilege accounts

Data Security:
- Never share credentials under any circumstances
- Lock your workstation when stepping away (Windows+L or Cmd+Ctrl+Q)
- Encrypt all sensitive data at rest and in transit
- Report security incidents immediately to security@company.com or call IT Security (ext. 5555)

Acceptable Use:
- Company devices are for business use; personal use is permitted in moderation
- Do not install unauthorized software
- Do not connect to unsecured public Wi-Fi without VPN"""
    },
    {
        "id": "doc_005",
        "title": "Performance Review Process",
        "content": """Performance Review Cycle:
Reviews are conducted twice per year: mid-year (June) and year-end (December).

Rating Scale:
- Exceeds Expectations (EE): Consistently surpasses goals
- Meets Expectations (ME): Achieves all primary goals
- Partially Meets Expectations (PME): Achieves some goals; improvement needed
- Does Not Meet Expectations (DNE): Significant improvement required

Process:
1. Employee completes self-assessment (2 weeks before review)
2. Manager completes assessment independently
3. 1:1 review meeting to discuss ratings and development plan
4. Goals set for next review period (SMART goals required)
5. HR sign-off for ratings of EE or DNE

Compensation:
- Merit increases tied to performance rating
- EE: 5-8% increase | ME: 3-5% | PME: 0-2% | DNE: No increase
- Bonus eligibility: ME and above"""
    },
    {
        "id": "doc_006",
        "title": "Expense Reimbursement Policy",
        "content": """Eligible Expenses:
- Business travel (flights, hotels, ground transport)
- Client meals and entertainment (with prior approval)
- Home office equipment (up to $500/year with manager approval)
- Professional memberships and subscriptions

Submission Process:
1. Submit expenses within 30 days of incurring the expense
2. All receipts required for expenses over $25
3. Submit via Concur expense management system
4. Manager approval required for all submissions
5. Reimbursement processed within 2 business days of approval

Limits:
- Meals: $75 per person per meal (client meals)
- Hotels: $250 per night (US domestic)
- Flights: Economy class for trips under 6 hours; business class eligible over 6 hours
- Per diem: $50/day for incidentals during travel"""
    },
]
